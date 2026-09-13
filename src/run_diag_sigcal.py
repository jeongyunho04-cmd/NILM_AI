"""자리별 **기기 지문 보정** — 라벨 없이 그 파일에서 추정한다 (13.84.31 후보 B).

상수 r 은 상시 부하와 축퇴됐다. 형제 편차는 **그 형제가 켜진 만큼만** 생기므로
보정은 지문에 붙어야 한다:  C_k -> C_k · t_k^((h-1)/14) · exp(i·c_k·h)
2개 스칼라/기기라 과적합이 안 난다. `sibling_rotate smps_dev1` 이 쓰는 것과 같은 형태다.
"""
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from scipy.optimize import nnls, minimize
from scipy.stats import trim_mean
from src.preprocessing import load_nilm_npz
from src.run_diag_nnls import APPS, ODD, FS, SEG_S, build_dict

D, names = build_dict()
ev = json.load(open('processed_data/real_events.json', encoding='utf-8'))['files']
# ⚠ 미니PC 는 **보정하지 않는다** — 자기 지문을 기울이면 와트가 그만큼 나뉘는
#   자명한 축퇴가 생긴다 (넣었더니 test_2 미니PC 가 23.3 -> 2.0W 로 무너졌다).
#   보정은 **형제만**. 기준을 고정해야 미니PC 와트가 뜻을 갖는다.
SM = ['laptop_charger', 'beam_projector']
COLS = {a: [c for c, nm in enumerate(names) if nm.split(':')[0] == a] for a in APPS}
KM = COLS['minipc']
ORD = np.array([1, 3, 5, 7, 9, 11, 13, 15])

def warp(Dc, cols, c_deg, t):
    """그 기기 열에 회전 c(°/차수) + 크기 기울기 t 를 건다."""
    out = Dc.copy()
    ph = np.exp(1j * np.deg2rad(c_deg) * ORD)
    mg = t ** ((ORD - 1) / 14.0)
    for j in cols:
        out[ODD, j] = Dc[ODD, j] * ph * mg
    return out

def mat(Dc):
    return np.concatenate([Dc[ODD].real, Dc[ODD].imag], axis=0)

def load(stem):
    r = load_nilm_npz('processed_data/composite_eval/%s.npz' % stem)
    H = np.asarray(r['harmonics_complex']); n = len(H)
    mk = {}
    for a in APPS:
        m = np.zeros(n, bool)
        for t0, t1 in ev[stem]['intervals'].get(a, {}).get('on', []):
            m[int(t0 * FS):int(min(t1 * FS, n))] = True
        mk[a] = m
    lab = np.stack([mk[a] for a in APPS])
    chg = np.r_[0, np.nonzero(np.abs(np.diff(lab.astype(np.int8), axis=1)).sum(0))[0] + 1, n]
    B, Y = [], []
    for s0, s1 in zip(chg[:-1], chg[1:]):
        s0, s1 = s0 + 5 * FS, s1 - 5 * FS
        for t in range(s0, s1 - SEG_S * FS, SEG_S * FS):
            v = trim_mean(H[t:t + SEG_S * FS], 0.2, axis=0)
            B.append(np.concatenate([v[ODD].real, v[ODD].imag])); Y.append(bool(mk['minipc'][t]))
    return (np.stack(B), np.array(Y)) if B else (None, None)

def fit(B, x):
    Dc = D.copy()
    for i, a in enumerate(SM):
        Dc = warp(Dc, COLS[a], x[2 * i], np.clip(x[2 * i + 1], 0.6, 1.6))
    A = mat(Dc)
    W = np.stack([nnls(A, b)[0] for b in B])
    return W, A, np.linalg.norm(B - W @ A.T, axis=1)

def obj(x, B):
    # 경계 밖은 평평해지지 않게 벌점으로 막는다 (Powell 이 밖으로 튀어 무의미한 값을 냈다)
    pen = 0.0
    for i in range(len(x) // 2):
        pen += max(0.0, abs(x[2 * i]) - 12.0) ** 2          # 회전 |c| <= 12°/h
        pen += 100.0 * max(0.0, abs(np.log(max(x[2 * i + 1], 1e-3))) - np.log(1.35)) ** 2
    r = fit(B, x)[2]
    return float(np.mean(r ** 2)) * (1.0 + pen) + 1e-4 * float(np.sum(np.asarray(x[0::2]) ** 2))

print('%-8s %5s %9s %9s | %-26s | %s'
      % ('파일', '구간', '잔차 전', '잔차 후', '미니PC 참ON/참OFF (차=분리도)', '추정한 보정 (회전°/h, 기울기)'))
for stem in ['test_1', 'test_2', 'test_3', 'test_4']:
    B, Y = load(stem)
    if B is None or len(B) < 8:
        print('%-8s 표본 부족' % stem); continue
    W0, A0, e0 = fit(B, np.zeros(4))
    res = minimize(obj, np.array([0., 1., 0., 1.]), args=(B,),
                   method='Powell', options={'maxiter': 2500, 'xtol': 1e-2, 'ftol': 1e-4})
    W1, A1, e1 = fit(B, res.x)
    m0, m1 = W0[:, KM].sum(1), W1[:, KM].sum(1)
    f = lambda v: (np.median(v[Y]) if Y.any() else np.nan, np.median(v[~Y]) if (~Y).any() else np.nan)
    a0, b0 = f(m0); a1, b1 = f(m1)
    par = " · ".join("%s %+.1f°/h x%.2f" % (a[:6], res.x[2 * i], res.x[2 * i + 1]) for i, a in enumerate(SM))
    print('%-8s %5d %8.1f %8.1f | 전 %4.1f/%4.1f (차 %+4.1f) -> 후 %4.1f/%4.1f (차 %+4.1f) | %s'
          % (stem, len(B), 1000 * np.median(e0) / np.sqrt(2), 1000 * np.median(e1) / np.sqrt(2),
             a0, b0, (a0 - b0) if np.isfinite(a0) else np.nan,
             a1, b1, (a1 - b1) if np.isfinite(a1) else np.nan, par))
