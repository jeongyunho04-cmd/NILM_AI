# -*- coding: utf-8 -*-
"""**잔차 채널**이 값싼 흡수처를 없애는가 — 모델 없이 CPU 로 (13.84.29).

사용자: *"값싼 흡수처가 왜 자꾸 생기지? 프로젝터·포트를 고치면 또 다른 기기로 옮겨가던데."*

진단(13.84.29): `L_harm` 이 관측을 **전부** 기기에 배정하도록 강제하는데(`noise_sig` 는 상수라
자유 항이 기기뿐이다), 실측에서 설명 못 하는 잔차가 관측 |I1| 의 **8~14%**(46~321mA)다.
미니PC 지문 전체가 47mA 이므로 잔차가 찾으려는 신호보다 크다. 그래서 어느 기기를 고치든
잔차는 **두 번째로 싼 기기**로 옮겨 간다 — 크기는 순방향 모형 오차가 정하지 흡수처가 정하지 않는다.

**처방 가설:** "모르겠다" 고 말할 채널을 주면 기기가 그 일을 안 해도 된다.
```
지금      min_{w>=0}      || D w − obs ||²
가설      min_{w>=0, r}   || D w + r − obs ||² + λ|r|₁      (L1 이어야 한다)
```
⚠ **L2 대가는 아무것도 안 바꾼다.** min_w min_r [||Aw+r−b||² + λ²||r||²] = min_w [λ²/(1+λ²)·||Aw−b||²]
라 손실의 상수배일 뿐이다 (2026-09-12 에 실제로 그렇게 재고 λ 전 구간에서 배분이 **한 자리도**
안 움직이는 것을 봤다). L1 잔차는 w 에 대해 **Huber** 손실과 같다 — 큰 잔차 성분이 이차가 아니라
일차로 벌점되어 어긋난 차수가 배분을 끌고 가지 못한다. 그래서 IRLS 로 푼다.
**결정적 대조:** λ 를 올릴 때
  · 미니PC 가 **참 OFF** 인 구간의 배분(유령)이 빨리 죽고
  · 미니PC 가 **참 ON** 인 구간의 배분(신호)은 남으면  -> 처방이 산다
  · 둘이 같이 죽으면 미니PC 는 잔차와 구별이 안 되는 것이고 이 처방으로는 못 고친다

⚠ 이것은 **모델이 아니라 사전 분해**다. 여기서 살아야 GPU 를 쓴다.

    python -X utf8 src/run_diag_nnls_resid.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from scipy.optimize import nnls
from scipy.stats import trim_mean

from src.preprocessing import load_nilm_npz
from src.run_diag_nnls import APPS, FS, ODD, SEG_S, build_dict

FILES = ["test_4", "test_3", "test_2", "test_1"]
#: Huber 문턱 δ. 관측 규모(||b||/√n)에 대한 배수다. δ -> ∞ 면 지금의 최소제곱과 같고,
#: δ 가 작을수록 "큰 잔차는 그냥 모르는 것으로 두라" 가 된다.
LAMBDAS = [np.inf, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02]


def solve_resid(D, obs, orders, delta, iters=25):
    """비음수 **Huber** 회귀 — IRLS 로 푼다.

    L1 잔차 대가는 w 에 대해 Huber 손실과 같다. 반복 재가중 최소제곱으로 푼다:
    잔차가 δ 보다 큰 성분은 가중치를 δ/|e| 로 낮춰 **이차 끌림을 끊는다**.
    남는 `r = b − Aw` 가 "설명 못 한 것" 이고, 그것이 기기로 안 가는지를 본다.
    """
    A = np.concatenate([D[orders].real, D[orders].imag], axis=0)      # (2H, C)
    b = np.concatenate([obs[orders].real, obs[orders].imag])          # (2H,)
    nb = np.linalg.norm(b) + 1e-12
    w, _ = nnls(A, b)
    if np.isfinite(delta):
        d = delta * nb / max(np.sqrt(len(b)), 1e-12)
        for _ in range(iters):
            e = b - A @ w
            u = np.minimum(1.0, d / np.maximum(np.abs(e), 1e-12))     # Huber 가중
            sw = np.sqrt(u)[:, None]
            w_new, _ = nnls(A * sw, b * np.sqrt(u))
            if np.max(np.abs(w_new - w)) < 1e-9 * (1 + np.max(w)):
                w = w_new
                break
            w = w_new
    r = b - A @ w
    return w, r, float(np.linalg.norm(r) / nb)


def main():
    D, names = build_dict()
    print("사전 %d 열 · %d 차수 · λ 훑기 %s" % (D.shape[1], D.shape[0],
                                              ", ".join("%g" % x for x in LAMBDAS)))
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    km = APPS.index("minipc")

    # 조건마다: (미니PC 참 ON 인가, 그 조건의 구간들)
    buckets = {True: [], False: []}
    for stem in FILES:
        r_ = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r_["harmonics_complex"])
        n = len(np.asarray(r_["power_features"])[:, 0])
        spec = ev[stem]
        if "minipc" not in spec["appliances_present"]:
            continue
        mk = {}
        for a in APPS:
            m = np.zeros(n, bool)
            for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                m[int(t0 * FS):int(min(t1 * FS, n))] = True
            mk[a] = m
        lab = np.stack([mk[a] for a in APPS])
        chg = np.r_[0, np.nonzero(np.abs(np.diff(lab.astype(np.int8), axis=1)).sum(0))[0] + 1, n]
        for s0, s1 in zip(chg[:-1], chg[1:]):
            s0, s1 = s0 + 5 * FS, s1 - 5 * FS
            for t in range(s0, s1 - SEG_S * FS, SEG_S * FS):
                buckets[bool(mk["minipc"][t])].append(trim_mean(H[t:t + SEG_S * FS], 0.2, axis=0))
    print("정상 구간: 미니PC 참ON %d개 · 참OFF %d개"
          % (len(buckets[True]), len(buckets[False])))

    print("\n%-8s | %-28s | %-28s | %s"
          % ("λ", "미니PC 참ON 배분 (신호)", "미니PC 참OFF 배분 (유령)", "잔차"))
    print("%-8s | %8s %8s %8s | %8s %8s %8s | %5s %5s"
          % ("", "중앙W", "p25", "p75", "중앙W", "p75", "p90", "ON", "OFF"))
    base = {}
    for lam in LAMBDAS:
        out = {}
        for on in (True, False):
            ws, rs = [], []
            for obs in buckets[on]:
                w, r, res = solve_resid(D, obs, ODD, lam)
                per = 0.0
                for c, nm in enumerate(names):
                    if nm.split(":")[0] == "minipc":
                        per += w[c]
                ws.append(per); rs.append(res)
            out[on] = (np.asarray(ws), np.asarray(rs))
        a, b = out[True], out[False]
        if not base:
            base = {True: np.median(a[0]), False: np.median(b[0])}
        print("%-8s | %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f | %4.1f%% %4.1f%%"
              % ("최소제곱" if not np.isfinite(lam) else "%g" % lam,
                 np.median(a[0]), np.percentile(a[0], 25), np.percentile(a[0], 75),
                 np.median(b[0]), np.percentile(b[0], 75), np.percentile(b[0], 90),
                 100 * np.median(a[1]), 100 * np.median(b[1])))
    print("\n판정: 유령(참OFF)이 신호(참ON)보다 **빨리** 줄면 잔차 채널이 산다.")
    print("      둘이 같은 속도로 줄면 미니PC 는 잔차와 구별이 안 되는 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
