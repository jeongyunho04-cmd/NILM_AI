# -*- coding: utf-8 -*-
"""**캐시에도 그 법칙이 남아 있나** (14.388, 사용자 지적).

§48 의 계획은 `sig(P)` 를 학습 목표에 거는 것인데, 학습이 맞추는 목표는 **캐시**다.
법칙은 **풀(격리 조각)** 에서 쟀다. 증강기가 사이에 끼어 있으므로 캐시에서 다시 재야 한다.

증강기가 하는 일 중 이 법칙에 닿는 것 둘:
```
  ① `rescale_to_power` — s(p) 곡선이 **복소**(S 가 (K,15) complex)라 위상을 담는다 ✔
  ② ⚠⚠ `phase_jitter_max_deg = 4.0` — `exp(1j·h·θ)`, θ~U(−4,4)도
     **내가 찾은 법칙과 정확히 같은 모양**의 회전을 **무작위로** 주입한다
     (미니PC 의 실제 폭이 ±3.5, 충전기가 ±6 도/차수인데 지터가 ±4 다)
```
"""
import numpy as np
from src import env_guard  # noqa: F401
from src.synthesis.augmentor import DataAugmentor  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

HS = np.array([3, 5, 7, 9, 11, 13], float)
DEV = ["minipc", "laptop_charger"]
KO = {"minipc": "미니PC", "laptop_charger": "충전기"}
TRUE = {"minipc": 3.90, "laptop_charger": 9.33}     # 풀에서 잰 a_k


def k_of(c, hs=HS):
    """(n,15) 복소 -> 차수 비례 회전 k [도/차수]."""
    ph = np.array([np.angle(np.exp(1j * np.angle(c[:, int(h) - 1])).mean()) for h in hs])
    ph = np.unwrap(ph)
    return float(np.degrees((hs @ ph) / (hs @ hs)))


pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
print("캐시(증강 후)에서 k = a·ln(P) 가 살아남나\n")
print("%-8s %-22s %8s %8s %9s %9s"
      % ("기기", "판", "a", "R²", "잔차σ", "풀 대비"))
for dev in DEV:
    acts = pool.appliance_activations.get(dev, [])
    if not acts:
        continue
    for tag, jit in (("증강 없음 (풀 그대로)", None),
                     ("증강, 지터 **끔**", 0.0),
                     ("증강, 지터 **4도**(현행)", 4.0)):
        rng = np.random.default_rng(0)
        aug = DataAugmentor(phase_jitter_max_deg=(jit if jit else 0.0))
        xs, ys = [], []
        for rep in range(40):
            a = acts[rep % len(acts)]
            if jit is None:
                c = np.asarray(a.net_harmonics_complex)
                p = np.asarray(a.target_power_w, float)
            else:
                np.random.seed(1000 + rep)
                g = aug.augment_activation(a, phase_jitter_deg=(
                    float(np.random.uniform(-jit, jit)) if jit > 0 else 0.0))
                c = np.asarray(g.net_harmonics_complex)
                p = np.asarray(g.target_power_w, float)
            m = p > 2.0
            if m.sum() < 300:
                continue
            c, p = c[m], p[m]
            qs = np.percentile(p, np.linspace(0, 100, 5))
            for i in range(4):
                s = (p >= qs[i]) & (p <= qs[i + 1])
                if s.sum() < 100:
                    continue
                xs.append(np.log(np.median(p[s])))
                ys.append(k_of(c[s]))
        if len(xs) < 8:
            print("%-8s %-22s 표본 부족" % (KO[dev], tag))
            continue
        xs, ys = np.asarray(xs), np.asarray(ys)
        A = np.polyfit(xs, ys, 1)
        res = ys - np.polyval(A, xs)
        r2 = 1 - (res ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-12)
        print("%-8s %-22s %8.2f %8.3f %9.2f %9s"
              % (KO[dev] if tag.startswith("증강 없음") else "", tag, A[0], r2, res.std(),
                 "%.0f%%" % (100 * A[0] / TRUE[dev])))
    print()
