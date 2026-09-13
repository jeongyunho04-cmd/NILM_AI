# -*- coding: utf-8 -*-
"""**완벽한 예측기**를 넣어도 순방향 모형이 못 맞추는 양 (13.84.35).

사용자: *"회로모델로 SMPS 가 환경에 따라 어떻게 바뀌는지 대부분 시뮬레이션했는데
어디서 세션 이동이 생기는지 모르겠다."*

`L_harm` 이 쓰는 식은 이것뿐이다:
    pred = Σ_k sig_k·power_k + Σ_k idle_k·standby_sig_k + noise_sig(**상수**)
여기에 **참값 라벨과 참값 와트**를 그대로 넣는다. 그래도 남는 것이 모델이 원리적으로
표현할 수 없는 양이다. 그 양의 **기록 간 산포**가 세션 이동의 크기다.

세 가지를 낸다:
  ① 차수별 남는 전류 대 관측 — 얼마나 못 맞추나
  ② 기록별 남는 전류의 산포 — 그것이 세션마다 얼마나 움직이나
  ③ 그 산포를 환경(R,X,V,vh)이 설명하나 — 회로모델이 닿는 곳인가

    python -X utf8 src/run_diag_fwd.py [--records 1500]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

ORD = [1, 3, 5, 7, 9, 11, 13, 15]


def env_r2(X, yc):
    num = den = 0.0
    for y in (yc.real, yc.imag):
        p = cross_val_predict(LinearRegression(), X, y, cv=5)
        num += float(np.sum((y - p) ** 2)); den += float(np.sum((y - y.mean()) ** 2))
    return max(1.0 - num / max(den, 1e-30), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_v1")
    ap.add_argument("--records", type=int, default=1500)
    a = ap.parse_args()
    C = Path(a.cache)
    meta = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    N = min(a.records, meta["records"])

    obs = np.asarray(np.load(C / "obs_harm.npy", mmap_mode="r")[:N])       # (N,T,15,2)
    yon = np.asarray(np.load(C / "y_on.npy", mmap_mode="r")[:N]).astype(bool)
    ypl = np.asarray(np.load(C / "y_plugged.npy", mmap_mode="r")[:N]).astype(bool)
    ypw = np.asarray(np.load(C / "y_power.npy", mmap_mode="r")[:N])
    zg = np.asarray(np.load(C / "z_grid.npy", mmap_mode="r")[:N])

    # `build_loss` 와 **같은 순서·같은 값**으로 순방향 모형을 세운다
    from src.model.net import harmonic_signatures, standby_signatures, noise_signature
    from src.model.companion import standby_operating_signatures
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import SESSION_PLUGGED_APPS
    from src.synthesis.sp_curves import background_signature, background_power
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, apps)
    sb = standby_signatures(pool, apps)
    sb_op, sb_pw, sb_used = standby_operating_signatures(pool, apps, only=SESSION_PLUGGED_APPS)
    for x in sb_used:
        sb[apps.index(x)] = sb_op[apps.index(x)]
    nzs = noise_signature(pool) + background_signature()
    del pool
    print("순방향 모형: 지문 %d기기 · 대기 지문 · 잡음+상시배경(%.2fW) — `build_loss` 와 동일"
          % (len(apps), background_power()))

    cx = lambda A: A[..., 0] + 1j * A[..., 1]
    O = cx(obs)                                                  # (N,T,15)
    idle = (ypl & ~yon).astype(np.float32)
    P = np.einsum("ntk,kh->nth", ypw, cx(sig))
    S = np.einsum("ntk,kh->nth", idle, cx(sb))
    pred = P + S + cx(nzs)[None, None]
    R = O - pred
    oi = [o - 1 for o in ORD]

    print("\n① 참값을 그대로 넣어도 남는 전류 (전 표본 중앙, mA)")
    print("  %-5s %10s %10s %10s %10s %10s"
          % ("차수", "관측", "활성기기", "대기", "잡음상수", "**남음**"))
    for o in ORD:
        j = o - 1
        print("  h%-4d %9.1f %10.1f %10.1f %10.1f %10.1f  (관측의 %4.0f%%)"
              % (o, 1000 * np.median(np.abs(O[:, :, j])), 1000 * np.median(np.abs(P[:, :, j])),
                 1000 * np.median(np.abs(S[:, :, j])), 1000 * abs(cx(nzs)[j]),
                 1000 * np.median(np.abs(R[:, :, j])),
                 100 * np.median(np.abs(R[:, :, j])) / max(np.median(np.abs(O[:, :, j])), 1e-12)))

    print("\n② 그 남는 전류가 **기록마다** 얼마나 다른가 (기록별 중앙의 산포)")
    M = np.median(R.real, 1) + 1j * np.median(R.imag, 1)          # (N,15)
    G = np.concatenate([zg[:, 0, :], np.zeros((N, 0))], 1)        # (N,2) R,X
    print("  %-5s %12s %12s %12s | %10s"
          % ("차수", "기록별 |중앙|", "기록간 sd", "기록간 CV", "환경 R²"))
    for o in ORD:
        j = o - 1
        m = M[:, j]
        cv = float(np.std(np.abs(m)) / (np.mean(np.abs(m)) + 1e-12))
        r2 = env_r2((G - G.mean(0)) / (G.std(0) + 1e-9), m)
        print("  h%-4d %11.1f %11.1f %11.1f%% | %9.3f"
              % (o, 1000 * np.median(np.abs(m)), 1000 * np.std(np.abs(m)), 100 * cv, r2))

    print("\n③ 견줌 — 미니PC 12W 가 내는 전류 (이것보다 크면 배분을 가린다)")
    km = apps.index("minipc")
    mp = cx(sig)[km] * 12.0
    print("  %-5s %12s %12s %12s" % ("차수", "미니PC 12W", "남는 전류", "배"))
    for o in ORD:
        j = o - 1
        v = 1000 * np.median(np.abs(M[:, j]))
        print("  h%-4d %11.2f %11.2f %11.1f" % (o, 1000 * abs(mp[j]), v, v / max(1000 * abs(mp[j]), 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
