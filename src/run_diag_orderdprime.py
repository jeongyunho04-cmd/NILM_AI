# -*- coding: utf-8 -*-
"""**차수별 판별력 `d'_h`** — 그 차수가 기기를 실제로 얼마나 가르나 (14.411).

왜 이것인가
----------
`L_harm` 의 분모는 `harm_scale_h` = 차수별 **신호 크기**의 중앙값이다. 그래서 크기가
작은 차수가 폭증한다 — §64.3 이 h2 하나로 손실의 60% 를 쟀다.

14.411 이 처음 제안한 고침(잔차로 백색화, `den = harm_scale·profile`)은 **반대 방향**
이었다. 관문 `run_gate_residscale` [5] 가 반증했다: profile 은 이미 `harm_scale` 로
나눈 **상대** 단위이고 h2 의 상대 바닥이 표에서 제일 작아(0.376), z-점수는 h2 에
h1 의 **80배** 가중을 준다. **조용한 차수에 상을 주는데 h2 는 조용한 동시에 쓸모없다**
(12.72: 계측 인공물).

나눠야 할 것은 "크기"도 "조용함"도 아니라 **쓸모**다:

```
  신호   1W 를 기기 i 에서 j 로 잘못 보내면 차수 h 에 얼마가 나타나나
           S_h = median_{i<j} ‖sig_i,h − sig_j,h‖ / harm_scale_h     [손실 단위 / W]
         ⚠ 짝수차는 손실이 **크기 공간**에서 재므로(`--harm-even-magnitude`)
           차이도 `| |sig_i,h| − |sig_j,h| |` 로 잡는다. 안 그러면 위상차가
           d' 에 새는데 손실은 그 위상을 보지도 않는다.
  잡음   그 차수의 **정답 배분 잔차** (손실 단위) = `HARM_DEADZONE_PROFILE[h]`
  d'_h = S_h / profile_h
```

    python -X utf8 -m src.run_diag_orderdprime
    python -X utf8 -m src.run_diag_orderdprime --group smps
"""
from typing import List
import argparse
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.model.losses import HARM_DEADZONE_PROFILE as PROF  # noqa: E402
from src.model.lossbuild import build_loss  # noqa: E402

APPS = ("air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven")
SMPS = ("minipc", "laptop_charger", "beam_projector")


def order_dprime(sig: np.ndarray, hs: np.ndarray, prof: np.ndarray,
                 idx: List[int], even_mag: bool = True):
    """(S_h, d'_h). `sig` (K,H,2) · `hs` (H,) · `prof` (H,)."""
    H = sig.shape[1]
    even = np.array([(h + 1) % 2 == 0 for h in range(H)])
    S = np.zeros(H)
    for h in range(H):
        vals = []
        for i, j in itertools.combinations(idx, 2):
            a, b = sig[i, h], sig[j, h]
            if even_mag and even[h]:
                d = abs(np.hypot(*a) - np.hypot(*b))     # 크기 공간
            else:
                d = float(np.hypot(*(a - b)))            # 복소
            vals.append(d)
        S[h] = np.median(vals) / max(hs[h], 1e-12)
    return S, S / np.maximum(prof[:H], 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=("all", "smps"), default="all")
    a = ap.parse_args()

    loss = build_loss(list(APPS), "cpu", harm_even_magnitude=True,
                      state_signatures=True, standby_operating="session")
    hs = loss.harm_scale.numpy().astype(np.float64)
    sig = loss.sig.numpy().astype(np.float64)
    H = sig.shape[1]
    prof = np.asarray(PROF[:H], float)
    idx_all = list(range(len(APPS)))
    idx_sm = [APPS.index(x) for x in SMPS]

    S_a, D_a = order_dprime(sig, hs, prof, idx_all)
    S_s, D_s = order_dprime(sig, hs, prof, idx_sm)
    D = D_s if a.group == "smps" else D_a

    print("  차수 | harm_scale(mA) | profile |  S_h(9종)  d'_h  |  S_h(SMPS)  **d'_h**")
    for h in range(H):
        print("   h%-3d| %13.2f | %7.3f | %8.4f %6.2f |  %8.4f %8.2f"
              % (h + 1, hs[h] * 1000, prof[h], S_a[h], D_a[h], S_s[h], D_s[h]))
    ev = np.array([(h + 1) % 2 == 0 for h in range(H)])
    print("\n  d' 중앙   9종 홀수 **%.2f** 짝수 **%.2f**  |  SMPS 홀수 **%.2f** 짝수 **%.2f**"
          % (np.median(D_a[~ev]), np.median(D_a[ev]),
             np.median(D_s[~ev]), np.median(D_s[ev])))

    #: 가중 = d' 에 비례. `den_h = harm_scale_h · (d_ref/d_h)^w`, `d_ref` 는 기하평균이라
    #: w=1 에서 전체 규모가 보존된다 (한쪽으로 통째로 밀리지 않는다).
    g = float(np.exp(np.mean(np.log(np.maximum(D, 1e-9)))))
    r = g / np.maximum(D, 1e-9)
    print("\n  분모비 (w=1)  den/harm_scale = d_ref/d'_h   [d_ref = 기하평균 %.3f]" % g)
    print("   " + " ".join("h%d %.2f" % (h + 1, r[h]) for h in range(H)))
    print("\n  ⇒ **눌리는 차수** (비 > 2): %s"
          % (", ".join("h%d(x%.1f)" % (h + 1, r[h]) for h in range(H) if r[h] > 2) or "없음"))
    print("  ⇒ **세지는 차수** (비 < 0.5): %s"
          % (", ".join("h%d(x%.2f)" % (h + 1, r[h]) for h in range(H) if r[h] < 0.5) or "없음"))
    print("\n  상수로 박을 꼴 (%s 기준):" % a.group)
    print("  HARM_ORDER_DPRIME = [" + ", ".join("%.4f" % x for x in D) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
