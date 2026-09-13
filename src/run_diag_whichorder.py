"""유령 구간에서 미니PC 가 **어느 차수를** 고치는가 (13.84.32)."""
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from scipy.optimize import nnls
from scipy.stats import trim_mean
from src.preprocessing import load_nilm_npz
from src.run_diag_nnls import APPS, ODD, FS, SEG_S, build_dict

D, names = build_dict()
ev = json.load(open('processed_data/real_events.json', encoding='utf-8'))['files']
ORD = [1, 3, 5, 7, 9, 11, 13, 15]
KM = [c for c, nm in enumerate(names) if nm.split(':')[0] == 'minipc']
KEEP = [c for c in range(D.shape[1]) if c not in KM]
A = np.concatenate([D[ODD].real, D[ODD].imag], axis=0)
h = len(ODD)
mag = lambda v: np.hypot(v[:h], v[h:])

ph, tr = [], []
for stem in ['test_1', 'test_2', 'test_3', 'test_4']:
    r = load_nilm_npz('processed_data/composite_eval/%s.npz' % stem)
    H = np.asarray(r['harmonics_complex']); n = len(H)
    if 'minipc' not in ev[stem]['appliances_present']:
        continue
    mk = {}
    for a in APPS:
        m = np.zeros(n, bool)
        for t0, t1 in ev[stem]['intervals'].get(a, {}).get('on', []):
            m[int(t0 * FS):int(min(t1 * FS, n))] = True
        mk[a] = m
    lab = np.stack([mk[a] for a in APPS])
    chg = np.r_[0, np.nonzero(np.abs(np.diff(lab.astype(np.int8), axis=1)).sum(0))[0] + 1, n]
    for s0, s1 in zip(chg[:-1], chg[1:]):
        s0, s1 = s0 + 5 * FS, s1 - 5 * FS
        for t in range(s0, s1 - SEG_S * FS, SEG_S * FS):
            obs = trim_mean(H[t:t + SEG_S * FS], 0.2, axis=0)
            b = np.concatenate([obs[ODD].real, obs[ODD].imag])
            wf, _ = nnls(A, b); mp = sum(wf[c] for c in KM)
            wo, _ = nnls(A[:, KEEP], b)
            e_with = mag(b - A @ wf); e_wo = mag(b - A[:, KEEP] @ wo)
            gain = (e_wo - e_with) / np.maximum(mag(b), 1e-9)      # 미니PC 가 줄인 오차 (관측 대비)
            (ph if (not mk['minipc'][t] and mp > 5) else
             (tr if (mk['minipc'][t] and mp > 5) else [])).append((gain, mp))

print('미니PC 가 차수별로 줄이는 오차 (관측 크기 대비 %%) — 클수록 그 차수를 고치는 것')
print('%-22s %5s %7s | %s' % ('구간', 'n', '미니PC W', ' '.join('h%-4d' % x for x in ORD)))
for nm, L in (('유령 (참 OFF · >5W)', ph), ('참 ON (>5W)', tr)):
    if len(L) < 4:
        print('%-22s %5d  표본 부족' % (nm, len(L))); continue
    G = np.stack([x[0] for x in L]); W = np.array([x[1] for x in L])
    print('%-22s %5d %7.1f | %s' % (nm, len(L), np.median(W),
                                    ' '.join('%+5.1f' % (100 * x) for x in np.median(G, 0))))
