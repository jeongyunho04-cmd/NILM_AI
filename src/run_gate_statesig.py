# -*- coding: utf-8 -*-
"""관문 — **상태별 지문의 분모** (14.167). 여덟 줄.

고친 것: `harmonic_signatures_by_state` 가 `target_power_w` 만 분모로 써서, 라벨이
`is_on=0` 으로 적는 상태(오븐 FAN_LIGHT)를 **한 사이클도 못 모았다**. 그 칸은 조용히
**기기 전체 지문**(= 히터, 순저항)으로 되돌아가 있었다.

```
  [1] measured_fallback=False 면 옛 동작과 **비트 동일**
  [2] 켜도 **이미 맞춰져 있던 칸은 전부 비트 동일** (라벨 전력이 먼저다)
  [3] 오븐 s1 이 **새로 맞춰진다** — 그리고 오븐 s2 와 **다르다**
  [4] 핫플 s1 은 **여전히 안 맞춰진다** (표본 39개 < 200). 잡음을 배우면 안 된다
  [5] 새로 맞춘 오븐 s1 이 **직접 계산한 중앙값과 일치**한다
  [6] `last_source` 가 어느 칸이 어느 분모에서 왔는지 정확히 적는다
  [7] 기기 전체 지문(`harmonic_signatures`)과 대기 지문은 **안 움직인다**
  [8] **진짜 `NILMLoss` 를 지어서** `sig_state` 버퍼에 그 값이 실려 있는지 본다
```

    python -X utf8 -m src.run_gate_statesig
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.net import (harmonic_signatures, harmonic_signatures_by_state,  # noqa: E402
                           standby_signatures, state_power_w)
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]


def ratios(sig_khs):
    """(H,2) -> (|h1| mA/W, h2/h1, h3/h1)."""
    c = sig_khs[:, 0] + 1j * sig_khs[:, 1]
    a1 = max(abs(c[0]), 1e-12)
    return abs(c[0]) * 1000, abs(c[1]) / a1, abs(c[2]) / a1


def main() -> int:
    ok = True
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    ko, kh = APPS.index("oven"), APPS.index("hotplate")

    old, used_old = harmonic_signatures_by_state(pool, APPS, measured_fallback=False)
    src_old = harmonic_signatures_by_state.last_source.copy()
    new, used_new = harmonic_signatures_by_state(pool, APPS)
    src_new = harmonic_signatures_by_state.last_source.copy()
    base = harmonic_signatures(pool, APPS)

    # [1] 끄면 옛 동작 — 기기 전체로 되돌아간 칸이 그대로 base 여야 한다
    back = np.array_equal(old[ko, 1], base[ko]) and np.array_equal(old[kh, 1], base[kh])
    print("[1] measured_fallback=False -> 오븐·핫플 s1 이 **기기 전체 지문 그대로**  %s"
          % ("OK" if back else "FAIL"))
    ok &= back

    # [2] 이미 맞춰져 있던 칸은 전부 비트 동일
    same = True
    diff = []
    for k in range(len(APPS)):
        for s in range(old.shape[1]):
            if used_old[k, s]:
                if not np.array_equal(old[k, s], new[k, s]):
                    same = False
                    diff.append("%s s%d" % (APPS[k], s))
    print("[2] 옛 %d칸이 전부 **비트 동일**  %s%s"
          % (int(used_old.sum()), "OK" if same else "FAIL",
             "" if same else "  <- " + " · ".join(diff)))
    ok &= same

    # [3] 오븐 s1 이 새로 맞춰지고 s2 와 다르다
    grew = bool(used_new[ko, 1]) and not bool(used_old[ko, 1])
    apart = not np.array_equal(new[ko, 1], new[ko, 2])
    r1, r2 = ratios(new[ko, 1]), ratios(new[ko, 2])
    print("[3] 오븐 s1 새로 맞춤 %s · s2 와 다름 %s  %s"
          % (grew, apart, "OK" if (grew and apart) else "FAIL"))
    print("      s1  |h1| %.3f mA/W · h2/h1 %.4f · h3/h1 %.4f" % r1)
    print("      s2  |h1| %.3f mA/W · h2/h1 %.4f · h3/h1 %.4f" % r2)
    print("      h2 비 **%.1f배** · h3 비 **%.1f배**"
          % (r1[1] / max(r2[1], 1e-12), r1[2] / max(r2[2], 1e-12)))
    ok &= (grew and apart)

    # [4] 핫플 s1 은 여전히 안 맞춘다 (표본 39 < 200)
    n_h = 0
    for a in pool.appliance_activations.get("hotplate", []):
        st = np.asarray(a.state_id)
        n_h += int(((st == 1) & (state_power_w(a, True) > 1.0)).sum())
    kept = (not bool(used_new[kh, 1])) and np.array_equal(new[kh, 1], base[kh])
    print("[4] 핫플 s1 표본 %d개 (<200) 이라 **여전히 안 맞춘다**  %s"
          % (n_h, "OK" if kept else "FAIL"))
    ok &= kept

    # [5] 직접 계산과 일치
    cs, ps = [], []
    for a in pool.appliance_activations.get("oven", []):
        st = np.asarray(a.state_id)
        pw = state_power_w(a, True)
        m = (st == 1) & (pw > 1.0)
        if m.any():
            cs.append(a.net_harmonics_complex[m]); ps.append(pw[m])
    c = np.concatenate(cs); p = np.concatenate(ps)[:, None]
    per = c / np.maximum(p, 1e-6)
    want = np.stack([np.median(per.real, 0), np.median(per.imag, 0)], -1).astype(np.float32)
    d = float(np.abs(want[:15] - new[ko, 1]).max())
    print("[5] 직접 계산한 중앙값과 일치 (n=%d)  최대차 %.3e  %s"
          % (len(c), d, "OK" if d < 1e-7 else "FAIL"))
    ok &= (d < 1e-7)

    # [6] last_source 가 정확하다
    s_ok = (src_new[ko, 1] == 2 and src_new[ko, 2] == 1 and src_new[kh, 1] == 0
            and src_new[kh, 2] == 1 and int((src_old == 2).sum()) == 0)
    print("[6] last_source  오븐 s1=%d(실측) s2=%d(라벨) · 핫플 s1=%d(없음) · 옛판 실측칸 %d개  %s"
          % (src_new[ko, 1], src_new[ko, 2], src_new[kh, 1], int((src_old == 2).sum()),
             "OK" if s_ok else "FAIL"))
    ok &= s_ok

    # [7] 기기 전체 지문·대기 지문은 안 움직인다
    base2 = harmonic_signatures(pool, APPS)
    sb1 = standby_signatures(pool, APPS)
    sb2 = standby_signatures(pool, APPS)
    untouched = np.array_equal(base, base2) and np.array_equal(sb1, sb2)
    print("[7] 기기 전체 지문·대기 지문 **안 건드림**  %s" % ("OK" if untouched else "FAIL"))
    ok &= untouched

    # [8] 진짜 NILMLoss 를 지어서 버퍼를 본다
    from src.model.lossbuild import build_loss
    L = build_loss(APPS, "cpu", verbose=False)
    buf = dict(L.named_buffers())["sig_state"].cpu().numpy()
    landed = np.allclose(buf[ko, 1], new[ko, 1], atol=0, rtol=0)
    r = ratios(buf[ko, 1])
    print("[8] 진짜 `NILMLoss.sig_state` 에 실렸다 — 오븐 s1 h2/h1 %.4f  %s"
          % (r[1], "OK" if landed else "FAIL"))
    ok &= landed

    print("\n%s" % ("전부 통과" if ok else "**실패한 줄이 있다**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
