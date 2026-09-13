"""세션 경계 이동 — 보정 문제인가 정보 부족인가 (13.84.33).

세션 **안에서** 되면 보정 문제다 (경계만 옮기면 된다).
세션 안에서도 안 되면 그 파일에는 정보가 없는 것이다.
⚠ 세션 안 채점은 **시간 블록**으로 가른다 — 이웃 구간은 거의 같은 창이다
  ([[overlapping-windows-need-time-block-split]]).
"""
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from scipy.stats import trim_mean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.preprocessing import load_nilm_npz
from src.run_diag_nnls import APPS, FS, SEG_S

ev = json.load(open('processed_data/real_events.json', encoding='utf-8'))['files']
ORD = [1, 3, 5, 7, 9, 11, 13, 15]
FILES = ['test_1', 'test_2', 'test_3', 'test_4']

def feats(stem):
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
    X, Y, T, V = [], [], [], []
    for s0, s1 in zip(chg[:-1], chg[1:]):
        s0, s1 = s0 + 5 * FS, s1 - 5 * FS
        for t in range(s0, s1 - SEG_S * FS, SEG_S * FS):
            v = trim_mean(H[t:t + SEG_S * FS], 0.2, axis=0)
            pf = np.median(P[t:t + SEG_S * FS], 0)
            i1 = abs(v[0]) + 1e-9
            X.append([np.log(i1), np.angle(v[0]), abs(v[2]) / i1])   # 13.84.15 의 셋
            Y.append(int(mk['minipc'][t])); T.append(t); V.append(pf[4])
    return np.array(X), np.array(Y), np.array(T), np.array(V)

DAT = {s: feats(s) for s in FILES}
print('%-8s %5s %6s | %9s %9s | %s' % ('파일', '구간', '참ON%', '세션안 AUC', '세션밖 AUC', '평균 V'))
W = {}
for s in FILES:
    X, Y, T, V = DAT[s]
    if len(set(Y)) < 2:
        print('%-8s %5d  한 부류만' % (s, len(Y))); continue
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # 세션 안: 시간 앞뒤 블록으로 가른다
    mid = np.median(T); a, b = T < mid, T >= mid
    au = []
    for tr, te in ((a, b), (b, a)):
        if len(set(Y[tr])) > 1 and len(set(Y[te])) > 1:
            lr = LogisticRegression(max_iter=3000).fit(Xs[tr], Y[tr])
            au.append(roc_auc_score(Y[te], lr.predict_proba(Xs[te])[:, 1]))
    inn = np.mean(au) if au else np.nan
    # 세션 밖: 나머지 파일로 학습
    Xo = np.concatenate([(DAT[o][0] - X.mean(0)) / (X.std(0) + 1e-9) for o in FILES if o != s])
    Yo = np.concatenate([DAT[o][1] for o in FILES if o != s])
    out = np.nan
    if len(set(Yo)) > 1:
        lr = LogisticRegression(max_iter=3000).fit(Xo, Yo)
        out = roc_auc_score(Y, lr.predict_proba(Xs)[:, 1])
    lr = LogisticRegression(max_iter=3000).fit(Xs, Y); W[s] = lr.coef_[0]
    print('%-8s %5d %5.0f%% | %9.3f %9.3f | %6.1f V' % (s, len(Y), 100 * Y.mean(), inn, out, V.mean()))

print('\n세션마다 맞춘 경계가 서로 같은가 (계수 벡터 cos)')
ks = [k for k in FILES if k in W]
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = W[ks[i]], W[ks[j]]
        print('   %-8s 대 %-8s  cos %+.3f   | 계수 %s  대  %s'
              % (ks[i], ks[j], float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                 np.array2string(a, precision=2), np.array2string(b, precision=2)))
