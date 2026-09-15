# -*- coding: utf-8 -*-
"""관문 — **대기 층의 전력과 지문이 같은 상태를 가리키나** (14.173). 다섯 줄.

땜질 둘이 서로를 무효화한 자리다.

```
  12.163  `standby_operating_signatures` 가 오븐 `standby_sig` 를 **FAN_LIGHT 로 덮어썼다**.
          근거: "y_standby 15.0W 인데 standby_sig 는 OFF_STANDBY 의 6.44mA — **10배 어긋난다**"
  13.40   `carrier_apps` 가 FAN_LIGHT 를 **활성 층으로** 옮겼다 (`sb_p = 0`).
          => y_standby 15.0W -> **0W**.  12.163 의 전제가 사라졌다.
  지금    라벨의 대기 전력 **1.126W**(~5.2mA) 대 지문 **67.91mA**  ->  **13배** (반대 방향)
```

그리고 그 덮어쓰기가 **축퇴를 만든다** — 오븐 팬·조명의 68mA 를 두 갈래로 낼 수 있다:

```
  A 켜짐  sig_state[oven][1] x 14.57W = 67.97 mA      <- 라벨이 말하는 길
  B 대기  idle x standby_sig[oven]    = 67.91 mA      <- 12.163 이 만든 길
```

`run_train_cnn` 이 `--standby-operating session` 과 `carrier_apps`(캐시) 를 **같이** 쓰면
이 충돌이 산다. 이 관문이 그것을 잡는다.

```
  [1] `carrier_apps` 가 든 캐시인지 읽는다 (meta.json)
  [2] 그 기기의 라벨 대기 전력이 정말 0 근처인가 (홀드아웃)
  [3] 덮어쓴 지문이 그 대기 전력과 몇 배 어긋나나
  [4] ★ **축퇴** — 켜짐 갈래와 대기 갈래가 같은 전류를 내나
  [5] 덮어쓰기를 끄면(`--standby-operating off`) 지문이 라벨과 맞나
```

    python -X utf8 -m src.run_gate_standby --cache cache/train60_v32h \\
        --holdout processed_data/holdout60_v32h
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache/train60_v32h")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32h")
    ap.add_argument("--tol", type=float, default=3.0, help="몇 배까지 봐주나")
    a = ap.parse_args()
    ok = True

    # [1] carrier_apps
    car = []
    mp = Path(a.cache) / "meta.json"
    if mp.exists():
        car = list(json.loads(mp.read_text(encoding="utf-8")).get("carrier_apps") or [])
    print("[1] 캐시 `%s` 의 carrier_apps = %s" % (a.cache, car or "(없음)"))
    if not car:
        print("    carrier_apps 가 없으면 12.163 의 전제가 그대로다 — 이 관문은 통과")
        print("\n전부 통과")
        return 0

    from src.model.net import standby_signatures, harmonic_signatures_by_state
    from src.model.companion import standby_operating_signatures
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import SESSION_PLUGGED_APPS
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sb0 = standby_signatures(pool, APPS)
    sb1, pw1, used = standby_operating_signatures(pool, APPS, only=SESSION_PLUGGED_APPS)
    gs, us = harmonic_signatures_by_state(pool, APPS)

    hp = Path(a.holdout)
    ha = json.loads((hp / "meta.json").read_text(encoding="utf-8"))["appliances"]

    for app in car:
        if app not in used:
            print("\n  %s — 12.163 이 안 덮어썼다 (건너뜀)" % app)
            continue
        k, hk = APPS.index(app), ha.index(app)
        sbw = np.asarray(np.load(hp / "y_standby.npy", mmap_mode="r"))[:, hk]
        onw = np.asarray(np.load(hp / "y_on.npy", mmap_mode="r"))[:, hk]
        pww = np.asarray(np.load(hp / "y_power.npy", mmap_mode="r"))[:, hk]
        live = sbw > 0.01
        sb_med = float(np.median(sbw[live])) if live.any() else 0.0
        a0 = float(np.hypot(sb0[k, 0, 0], sb0[k, 0, 1]) * 1000)
        a1 = float(np.hypot(sb1[k, 0, 0], sb1[k, 0, 1]) * 1000)

        print("\n■ %s" % app)
        print("[2] 라벨의 대기 전력 중앙 **%.3fW** (0 아닌 창 %d/%d) · 12.163 이 본 값 %.2fW"
              % (sb_med, int(live.sum()), len(sbw), pw1[k]))
        drop = pw1[k] / max(sb_med, 1e-6)
        print("    => 13.40 이 대기 전력을 **%.0f배** 줄였다 %s"
              % (drop, "  <- ⚠ 12.163 의 전제가 사라졌다" if drop > 3 else ""))

        # [3] 지문이 그 전력과 맞나 (와트당은 그 기기 sig 로 환산)
        c1 = gs[k, 1, :, 0] + 1j * gs[k, 1, :, 1]
        mA_per_W = abs(c1[0]) * 1000
        want = sb_med * mA_per_W
        print("[3] 대기 전력 %.3fW 에 맞는 전류 ~**%.2f mA** | 지금 지문 **%.2f mA** (12.163) "
              "· 끄면 %.2f mA"
              % (sb_med, want, a1, a0))
        bad = a1 / max(want, 1e-6)
        print("    어긋남 **x%.1f** (끄면 x%.1f)  %s"
              % (bad, a0 / max(want, 1e-6), "**FAIL**" if bad > a.tol else "OK"))
        ok &= bad <= a.tol

        # [4] ★ 축퇴
        onmask = (onw > 0.5) & (pww < 100)
        p_fan = float(np.median(pww[onmask])) if onmask.any() else pw1[k]
        viaA = mA_per_W * p_fan
        print("[4] ★ **축퇴** — 같은 전류를 두 갈래로 낼 수 있나")
        print("      A 켜짐  sig_state[%s][1] x %.2fW = **%.2f mA**" % (app, p_fan, viaA))
        print("      B 대기  idle x standby_sig       = **%.2f mA**" % a1)
        deg = abs(viaA - a1) / max(viaA, 1e-6)
        print("      차 %.1f%%  %s" % (deg * 100,
              "<- ⚠ **사실상 같다. 손실이 두 길을 구별 못 한다**" if deg < 0.10 else "OK"))
        ok &= deg >= 0.10

        # [5] 끄면 맞나
        fix = a0 / max(want, 1e-6)
        print("[5] `--standby-operating off` 로 두면 어긋남 **x%.1f** %s"
              % (fix, "(고쳐진다)" if fix < bad else "(안 고쳐진다)"))

    print("\n%s" % ("전부 통과" if ok else
                   "**실패 — 대기 층의 전력과 지문이 다른 상태를 가리킨다.**\n"
                   "  `carrier_apps` 를 쓰는 캐시에서는 `--standby-operating off` 가 맞다.\n"
                   "  (12.163 은 `carrier_apps` 가 없던 시절의 고침이다.)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
