# -*- coding: utf-8 -*-
"""**SMPS 표류** 자 — ② 를 ① 과 **갈라서** 재기 위한 것 (14.165).

14.159 가 병을 셋으로 갈랐는데 판정 표에 ② 를 재는 칸이 없었다.

```
  ①  수백 W 치환   오븐<->포트 — `run_diag_flipedge` 의 띠 길이로 잰다
  ②  수십 W 표류   빔·충전·미니PC 가 서로 밀고 당긴다 — **이 도구**
  ③  스위칭 과도   유령 — `run_diag_ghost` · `run_diag_spike` 로 잰다
```

**참이 안 변하는 구간**만 골라서(모든 기기의 전이에서 `--guard` 초 이상 떨어진 창)
거기서 무엇이 흔들리는지 잰다.

```
  [1] 관측이 얼마나 흔들리나   그 구간의 P관측 표준편차   <- **자연의 폭**
  [2] 모델의 SMPS 합이 흔들리나  Σ(빔+충전+미니PC) 표준편차  <- 이게 [1] 보다 크면 **표류**
  [3] 기기별                  각자의 표준편차와 평균
  [4] 쌍 상관                 −1 에 가까우면 **2자 축퇴**(서로 주고받음),
                             0 근처면 축퇴가 아니라 그냥 **각자 떠는 것**
```

⚠ 이 자는 **δ=100W 죽은구역 안**이다. `cnn_dzn` 이 이 줄을 **안 움직이는 것이
정상**이다 — 움직이면 죽은구역이 뜻대로 안 걸린 것이다.

    python -X utf8 -m src.run_diag_smpsdrift --stem test_4 \\
        --ckpt results/cnn_pcap_s0.pt results/cnn_pcap_s1.pt results/cnn_pcap_s2.pt
"""
import argparse
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.run_gate_check import load_model, sync_even_median  # noqa: E402

SMPS = ("beam_projector", "laptop_charger", "minipc")
SH = {"oven": "오븐", "electiric_kettle": "포트", "hotplate": "핫플", "hair_dryer": "드라이",
      "beam_projector": "빔", "laptop_charger": "충전", "minipc": "미니PC",
      "fan": "선풍", "air_conditioner": "에어컨"}


def steady_mask(ev, t, guard):
    """모든 기기의 on/off 전이에서 `guard` 초 이상 떨어진 창만 True."""
    ok = np.ones(len(t), bool)
    for v, d in ev["intervals"].items():
        for key in ("on", "uncertain"):
            for t0, t1 in d.get(key, []):
                for edge in (t0, t1):
                    ok &= np.abs(t - edge) > guard
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--stem", nargs="+", default=["test_4"])
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--guard", type=float, default=8.0,
                    help="전이에서 이만큼 떨어진 창만 본다 (초)")
    ap.add_argument("--even-median", type=int, default=0,
                    help="짝수차 중앙값을 **강제**한다 — 분포 밖 시험용")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sync_even_median(a.ckpt, a.even_median)
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    from src.run_train_seq import real_windows
    cache = real_windows(apps, a.grid_s, dev)
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    ks = [apps.index(v) for v in SMPS if v in apps]
    names = [v for v in SMPS if v in apps]

    sel, P = {}, {}
    for stem in a.stem:
        d = cache[stem]
        t = np.asarray(d["t"], float)
        ok = steady_mask(ev[stem], t, a.guard)
        # 참으로 켜져 있는 SMPS 가 하나라도 있어야 "나눌 것"이 있다
        ok &= d["y"][:, ks].any(1)
        sel[stem] = ok
        P[stem] = np.asarray(d["p_obs"], float)
        print("%-8s 창 %4d 중 **안정** %4d개 (전이에서 %.0f초 밖 · SMPS 켜짐)"
              % (stem, len(t), ok.sum(), a.guard))
    if not any(v.any() for v in sel.values()):
        print("안정 구간이 없다 — `--guard` 를 줄여라")
        return 1

    print("\n[1] **관측이 흔들리는 폭** — 이것이 자연의 폭이다")
    for stem in a.stem:
        if sel[stem].any():
            print("    %-8s P관측 sd **%5.1fW** (평균 %6.1fW · n=%d)"
                  % (stem, P[stem][sel[stem]].std(), P[stem][sel[stem]].mean(),
                     sel[stem].sum()))

    for ck in a.ckpt:
        m = load_model(ck, dev)[0]
        m.eval()
        print("\n" + "=" * 88)
        print("■ %s" % ck.split("/")[-1].replace(".pt", ""))
        print("=" * 88)
        for stem in a.stem:
            ok = sel[stem]
            if not ok.any():
                continue
            d = cache[stem]
            PW = []
            with torch.no_grad():
                idx = np.nonzero(ok)[0]
                for i in range(0, len(idx), 256):
                    j = idx[i:i + 256]
                    o = m(torch.from_numpy(np.ascontiguousarray(d["fine"][j])).to(dev),
                          torch.from_numpy(np.ascontiguousarray(d["wide"][j])).to(dev))
                    PW.append(o["power"].float().cpu().numpy().astype(np.float64))
            PW = np.concatenate(PW)[:, ks]
            S = PW.sum(1)
            obs = P[stem][ok].std()
            print("  %s" % stem)
            print("    [2] SMPS **합** sd **%5.1fW** (평균 %6.1fW)  —  관측 sd %.1fW 의 "
                  "**%.1f배**%s"
                  % (S.std(), S.mean(), obs, S.std() / max(obs, 1e-9),
                     "   <- ⚠ 표류" if S.std() > 2 * obs else ""))
            print("    [3] 기기별   " + " · ".join(
                "%s %.1f±%.1fW" % (SH.get(v, v), PW[:, i].mean(), PW[:, i].std())
                for i, v in enumerate(names)))
            cs = []
            for i, j in itertools.combinations(range(len(names)), 2):
                if PW[:, i].std() > 1e-6 and PW[:, j].std() > 1e-6:
                    cs.append("%s<->%s %+.2f" % (SH.get(names[i], names[i]),
                                                 SH.get(names[j], names[j]),
                                                 np.corrcoef(PW[:, i], PW[:, j])[0, 1]))
            print("    [4] 쌍 상관   " + (" · ".join(cs) if cs else "(못 잼)")
                  + "      −1 에 가까우면 2자 축퇴 · 0 근처면 각자 떠는 것")
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("\n⚠ 이 줄은 **δ=100W 죽은구역 안**이다. `cnn_dzn` 이 여기를 안 움직이는 것이 정상.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
