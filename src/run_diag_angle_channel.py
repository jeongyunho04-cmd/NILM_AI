# -*- coding: utf-8 -*-
"""∠I1 이 '채널에 없다' 는 주장을 채널로 검정한다.

세밀 ch0 = asinh(Re I1 · 20), ch8 = asinh(Im I1 · 20). ∠I1 은 이 둘의 함수다.
조건창(충전기+프로젝터 ON)에서 미니PC ON/OFF 의 Fisher d 를
  ① ∠I1 (원 각도)  ② ch0 단독  ③ ch8 단독  ④ (ch0,ch8) 최적 선형결합(LDA)
로 잰다. ④ 가 ① 만큼 나오면 정보는 이미 채널에 있다.
또 구간별 σ 대 전력 기울기(진단의 'σ 0.3°' 표)도 같이 낸다.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src.preprocessing import load_nilm_npz

FS, W, STRIDE, TGT = 60, 60 * 60, 5 * 60, 10 * 60
CURRENT_SCALE = 20.0


def fisher1(a, b):
    s = np.sqrt(0.5 * (a.var() + b.var())) + 1e-12
    return abs(a.mean() - b.mean()) / s


def fisher_lda(A, B):
    """다변량 Fisher 거리 (합동 공분산)."""
    mu = A.mean(0) - B.mean(0)
    S = 0.5 * (np.cov(A.T) + np.cov(B.T)) + 1e-12 * np.eye(A.shape[1])
    return float(np.sqrt(mu @ np.linalg.solve(S, mu)))


for stem in ["test_4", "test_1", "test_3"]:
    r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"][stem]
    H = r["harmonics_complex"]
    n = H.shape[0]
    valid = np.asarray(r["is_valid"]).astype(bool)
    P = np.asarray(r["power_features"])[:, 0]

    def mask(a):
        m = np.zeros(n, bool)
        for t0, t1 in ev["intervals"].get(a, {}).get("on", []):
            m[int(t0 * FS):int(min(t1 * FS, n))] = True
        return m

    y = mask("minipc")
    cond = mask("laptop_charger")
    if "beam_projector" in ev["intervals"]:
        cond &= mask("beam_projector")
    ch0 = np.arcsinh(H[:, 0].real * CURRENT_SCALE)
    ch8 = np.arcsinh(H[:, 0].imag * CURRENT_SCALE)
    ang = np.degrees(np.angle(H[:, 0]))
    mag = np.abs(H[:, 0])

    idx = [t0 for t0 in range(0, n - W + 1, STRIDE) if valid[t0:t0 + W].all()]
    tgt = [slice(t0 + W - TGT, t0 + W) for t0 in idx]
    keep = [k for k, sl in enumerate(tgt) if cond[sl].all()]
    lab = np.array([bool(y[tgt[k]].all()) for k in keep])
    neg = np.array([not y[tgt[k]].any() for k in keep])
    if lab.sum() < 4 or neg.sum() < 4:
        print("%-7s 표본 부족 ON %d OFF %d" % (stem, lab.sum(), neg.sum()))
        continue
    g = lambda a: np.array([a[tgt[k]].mean() for k in keep])
    v_ang, v0, v8, v_mag = g(ang), g(ch0), g(ch8), g(mag)
    A = np.stack([v0, v8], 1)
    print("\n=== %s  조건창 ON %d / OFF %d" % (stem, lab.sum(), neg.sum()))
    print("   ∠I1 (각도)            d = %.2f" % fisher1(v_ang[lab], v_ang[neg]))
    print("   ch0 = asinh(ReI1·20)  d = %.2f" % fisher1(v0[lab], v0[neg]))
    print("   ch8 = asinh(ImI1·20)  d = %.2f" % fisher1(v8[lab], v8[neg]))
    print("   (ch0,ch8) 최적 선형   d = %.2f   <- 채널만으로 얻을 수 있는 최대"
          % fisher_lda(A[lab], A[neg]))
    print("   |I1| (크기만)         d = %.2f" % fisher1(v_mag[lab], v_mag[neg]))

    if stem == "test_4":
        print("   -- 구간별 ∠I1 σ 대 전력 기울기 (진단의 'σ 0.3°' 표 검정)")
        ts = sorted({0.0} | {t for a, d in ev["intervals"].items() for iv in d.get("on", []) for t in iv} | {n / FS})
        for a_, b_ in zip(ts[:-1], ts[1:]):
            i0, i1 = int(a_ * FS), int(b_ * FS)
            if i1 - i0 < 20 * FS or not valid[i0:i1].all():
                continue
            sl = slice(i0, i1)
            slope = np.polyfit(np.arange(i1 - i0) / FS, P[sl], 1)[0] * (b_ - a_)
            print("      %5.0f-%-5.0f s  ∠I1 σ %6.3f°   구간 ΔP %+7.1fW   Pσ %6.2fW   기기 %s"
                  % (a_, b_, ang[sl].std(), slope, P[sl].std(),
                     "+".join(k[:4] for k in ev["intervals"] if mask(k)[(i0 + i1) // 2])))
