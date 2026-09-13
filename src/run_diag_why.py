"""세션 이동의 원인 — 관측 가능한가 (13.84.33).

합성이므로 환경(R, X, 전압)을 안다. 그것을 특징에 넣었을 때 AUC 가 회복되면
**이동은 관측 가능**하고 모델이 안 쓴 것이다. 안 되면 관측 밖(어느 녹화·동작점을 뽑았나)이다.
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
apps = meta['appliances']; km = apps.index('minipc')
sib = [apps.index(x) for x in ('laptop_charger', 'beam_projector')]
raw = np.load(C / 'raw.npy', mmap_mode='r'); yon = np.load(C / 'y_on.npy', mmap_mode='r')
zg = np.load(C / 'z_grid.npy', mmap_mode='r')
grid = np.arange(meta['target_offset'], int(meta['record_s'] * 60) - 13 * 60 - 1, int(meta['grid_s'] * 60))

BASE, ZF, VF, Y, E = [], [], [], [], []
for i in range(1200):
    yo = np.asarray(yon[i]); m = yo[:, sib].sum(1) > 0
    if m.sum() < 6:
        continue
    r = np.asarray(raw[i]); z = np.asarray(zg[i])[0]
    for t in np.nonzero(m)[0][::6]:
        c = grid[t]; seg = slice(max(0, c - 600), c)
        h = r[0:15, seg] + 1j * r[15:30, seg]
        v = np.median(h.real, 1) + 1j * np.median(h.imag, 1)
        i1 = abs(v[0]) + 1e-9
        BASE.append([np.log(i1), np.angle(v[0]), abs(v[2]) / i1,
                     abs(v[4]) / i1, abs(v[6]) / i1, np.angle(v[2] * np.conj(v[0]))])
        ZF.append([z[0], z[1]])                                   # 선로 임피던스 R, X
        vh = r[33:45, seg]                                        # 전압 고조파 채널
        vr = np.median(r[32, seg])                                # V 실효 대리
        VF.append([vr] + list(np.median(np.abs(vh), 1)[:6]))
        Y.append(int(yo[t, km])); E.append(i)
BASE = np.array(BASE); ZF = np.array(ZF); VF = np.array(VF); Y = np.array(Y); E = np.array(E)
nz = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-9)
BASE, ZF, VF = nz(BASE), nz(ZF), nz(VF)
recs = np.array(sorted(set(E))); np.random.RandomState(0).shuffle(recs)
tr_r = set(recs[:len(recs) * 3 // 4])
tr = np.array([e in tr_r for e in E]); te = ~tr
print('표본 %d · 환경 %d · 학습 %d / 채점 %d' % (len(Y), len(recs), tr.sum(), te.sum()))

def run(nm, Xs, inter=False):
    X = np.concatenate(Xs, 1)
    if inter:                       # 기본특징 x 환경특징 교차항 — 환경에 따라 경계가 **기울게**
        A, B = Xs[0], np.concatenate(Xs[1:], 1)
        X = np.concatenate([X] + [A * B[:, [j]] for j in range(B.shape[1])], 1)
    lr = LogisticRegression(max_iter=5000, C=0.5).fit(X[tr], Y[tr])
    print('   %-40s AUC %.3f' % (nm, roc_auc_score(Y[te], lr.predict_proba(X[te])[:, 1])))

print('\n환경 정보를 주면 회복되나 (환경 밖 채점)')
run('① 기본 특징만 (지금)', [BASE])
run('② + 선로 임피던스 R,X', [BASE, ZF])
run('③ + 전압(실효·고조파)', [BASE, VF])
run('④ + 둘 다', [BASE, ZF, VF])
run('⑤ 둘 다 + **교차항** (경계가 환경에 따라 기움)', [BASE, ZF, VF], inter=True)
print('\n참고: 기록마다 경계를 새로 맞춘 상한 0.879 · 기록 안 채점 0.767')
