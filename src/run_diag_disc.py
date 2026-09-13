"""**재구성** 틀 대 **판별** 틀 — 같은 구간, 같은 자료 (13.84.33).

오늘 NNLS(재구성)로 미니PC AUC 0.51 을 내고 "정보가 없다" 고 했다. 13.84.15 는 특징 셋
로지스틱으로 0.929 를 냈다. 같은 구간에서 둘을 나란히 잰다. 파일 하나 빼기로 채점한다 —
그 파일은 학습에 안 들어간다.
"""
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from scipy.optimize import nnls
from scipy.stats import trim_mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.preprocessing import load_nilm_npz
from src.run_diag_nnls import APPS, ODD, FS, SEG_S, build_dict

D, names = build_dict()
A = np.concatenate([D[ODD].real, D[ODD].imag], axis=0)
KM = [c for c, nm in enumerate(names) if nm.split(':')[0] == 'minipc']
ev = json.load(open('processed_data/real_events.json', encoding='utf-8'))['files']
ORD = [1, 3, 5, 7, 9, 11, 13, 15]

X, Y, G, NN = [], [], [], []
for stem in ['test_1', 'test_2', 'test_3', 'test_4']:
    if 'minipc' not in ev[stem]['appliances_present']:
        continue
    r = load_nilm_npz('processed_data/composite_eval/%s.npz' % stem)
    H = np.asarray(r['harmonics_complex']); P = np.asarray(r['power_features']); n = len(H)
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
            v = trim_mean(H[t:t + SEG_S * FS], 0.2, axis=0)
            pf = np.median(P[t:t + SEG_S * FS], 0)
            i1 = abs(v[0]) + 1e-9
            f = [np.log(i1), np.angle(v[0])]
            for o in ORD[1:]:
                f += [abs(v[o - 1]) / i1, np.angle(v[o - 1] * np.conj(v[0]))]
            f += [np.log(max(pf[0], 1e-3)), pf[3]]          # P, 역률
            X.append(f); Y.append(int(mk['minipc'][t])); G.append(stem)
            b = np.concatenate([v[ODD].real, v[ODD].imag])
            w, _ = nnls(A, b); NN.append(sum(w[c] for c in KM))
X = np.array(X); Y = np.array(Y); G = np.array(G); NN = np.array(NN)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
print('구간 %d (미니PC 참ON %d · 참OFF %d) · 특징 %d개' % (len(Y), Y.sum(), (1 - Y).sum(), X.shape[1]))

print('\n%-22s %8s | %s' % ('방법', '전체 AUC', '파일 하나 빼기 AUC'))
print('%-22s %8s | %s' % ('', '', ' '.join('%-9s' % s for s in sorted(set(G)))))
# ① 재구성 (NNLS 미니PC 와트)
au = roc_auc_score(Y, NN)
per = []
for s in sorted(set(G)):
    m = G == s
    per.append(roc_auc_score(Y[m], NN[m]) if len(set(Y[m])) > 1 else np.nan)
print('%-22s %8.3f | %s' % ('① 재구성 NNLS', au, ' '.join('%-9.3f' % x for x in per)))
# ② 판별 (로지스틱, 파일 하나 빼기)
per, allp = [], np.zeros(len(Y))
for s in sorted(set(G)):
    te = G == s; tr = ~te
    if len(set(Y[tr])) < 2 or len(set(Y[te])) < 2:
        per.append(np.nan); continue
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], Y[tr])
    p = lr.predict_proba(X[te])[:, 1]; allp[te] = p
    per.append(roc_auc_score(Y[te], p))
print('%-22s %8.3f | %s' % ('② 판별 로지스틱', roc_auc_score(Y, allp), ' '.join('%-9.3f' % x for x in per)))
# ③ 13.84.15 의 특징 셋만
idx = [0, 1, 2]
per, allp = [], np.zeros(len(Y))
for s in sorted(set(G)):
    te = G == s; tr = ~te
    if len(set(Y[tr])) < 2 or len(set(Y[te])) < 2:
        per.append(np.nan); continue
    lr = LogisticRegression(max_iter=3000).fit(X[tr][:, idx], Y[tr])
    p = lr.predict_proba(X[te][:, idx])[:, 1]; allp[te] = p
    per.append(roc_auc_score(Y[te], p))
print('%-22s %8.3f | %s' % ('③ ∠I1·|I1|·|I3|/|I1|', roc_auc_score(Y, allp), ' '.join('%-9.3f' % x for x in per)))
