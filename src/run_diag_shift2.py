"""세션 이동을 **합성에서** 잰다 — 표본이 무제한이다 (13.84.33).

실측은 참OFF 구간이 파일당 6~19개라 AUC 오차가 ±0.15~0.25 다. 합성 캐시는 기록마다
환경(R, X)을 새로 뽑으므로 "환경 안" 대 "환경 밖" 을 제대로 가를 수 있다.
미니PC 판별을 **형제가 켜져 있는 구간**에서만 본다 (그게 어려운 자리다).
"""
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

C = Path('cache/seqraw_v1')
meta = json.loads((C / 'meta.json').read_text(encoding='utf-8'))
apps = meta['appliances']
km = apps.index('minipc')
sib = [apps.index(x) for x in ('laptop_charger', 'beam_projector')]
raw = np.load(C / 'raw.npy', mmap_mode='r')
yon = np.load(C / 'y_on.npy', mmap_mode='r')
zg = np.load(C / 'z_grid.npy', mmap_mode='r')
grid = np.arange(meta['target_offset'], int(meta['record_s'] * 60) - 13 * 60 - 1, int(meta['grid_s'] * 60))

N = 1200
X, Y, E = [], [], []
for i in range(N):
    yo = np.asarray(yon[i])
    m = (yo[:, sib].sum(1) > 0)                    # 형제가 하나라도 켜진 단계만
    if m.sum() < 6:
        continue
    r = np.asarray(raw[i])
    idx = np.nonzero(m)[0][::6]
    for t in idx:
        c = grid[t]
        seg = slice(max(0, c - 600), c)
        h = r[0:15, seg] + 1j * r[15:30, seg]
        v = np.median(h.real, 1) + 1j * np.median(h.imag, 1)
        i1 = abs(v[0]) + 1e-9
        X.append([np.log(i1), np.angle(v[0]), abs(v[2]) / i1,
                  abs(v[4]) / i1, abs(v[6]) / i1, np.angle(v[2] * np.conj(v[0]))])
        Y.append(int(yo[t, km])); E.append(i)
X = np.array(X); Y = np.array(Y); E = np.array(E)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
print('표본 %d (미니PC ON %d / OFF %d) · 환경(기록) %d개' % (len(Y), Y.sum(), (1 - Y).sum(), len(set(E))))

recs = np.array(sorted(set(E)))
rng = np.random.RandomState(0); rng.shuffle(recs)
tr_r, te_r = set(recs[:len(recs) * 3 // 4]), set(recs[len(recs) * 3 // 4:])
tr = np.array([e in tr_r for e in E]); te = ~tr
lr = LogisticRegression(max_iter=4000).fit(X[tr], Y[tr])
p = lr.predict_proba(X)[:, 1]
print('\n① 환경 밖 (기록을 갈라 학습/채점)   AUC %.3f' % roc_auc_score(Y[te], p[te]))

# ② 환경 안 — 기록마다 그 기록 안에서만 채점 (경계는 전체에서 배운 것)
per = []
for e in list(te_r)[:400]:
    m = E == e
    if len(set(Y[m])) > 1 and m.sum() >= 6:
        per.append(roc_auc_score(Y[m], p[m]))
print('② 같은 경계, **기록 안에서만** 채점  AUC %.3f (중앙, 기록 %d개)'
      % (np.median(per), len(per)))

# ③ 기록마다 경계를 다시 맞추면 (상한 — 그 기록 안 절반으로 학습)
per2 = []
for e in list(te_r)[:400]:
    m = np.nonzero(E == e)[0]
    if len(m) < 12 or len(set(Y[m])) < 2:
        continue
    h = len(m) // 2
    a, b = m[:h], m[h:]
    if len(set(Y[a])) < 2 or len(set(Y[b])) < 2:
        continue
    l2 = LogisticRegression(max_iter=3000).fit(X[a], Y[a])
    per2.append(roc_auc_score(Y[b], l2.predict_proba(X[b])[:, 1]))
if per2:
    print('③ 기록마다 경계를 새로 맞추면        AUC %.3f (중앙, 기록 %d개)'
          % (np.median(per2), len(per2)))
print('\n   ②와 ① 의 차이 = **순위를 뭉개는 환경 간 이동**')
print('   ③과 ② 의 차이 = 경계를 옮겨서 얻을 수 있는 것')
