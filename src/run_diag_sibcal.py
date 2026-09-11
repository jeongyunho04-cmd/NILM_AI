"""형제 전이로 **그 세션의 형제 지문**을 뽑아 빼면 미니PC 판별이 오르는가 (13.84.34).

세션 이동의 정체는 "그 기록에 뽑힌 형제 활성화가 어느 것이냐" 였다(13.84.33). 그렇다면
형제가 켜지는 순간의 Δ 가 **그 세션의 형제 지문**이므로, 그것으로 빼면 이동이 사라져야 한다.

짝지은 대조 — 둘 다 형제의 **참 전력**을 준다. 다른 것은 **어느 지문으로 빼느냐** 하나다.
   ① 풀 평균 지문 (지금)      ② 그 기록의 전이에서 뽑은 지문
⚠ 형제 전이 시각은 참값을 준다 (13.84.21 과 같은 '천장' 채점). 배치에서는 전력 계단으로 찾는다.
"""
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.run_diag_nnls import build_dict

C = Path('cache/seqraw_v1')
meta = json.loads((C / 'meta.json').read_text(encoding='utf-8'))
apps = meta['appliances']; km = apps.index('minipc')
SIB = ['laptop_charger', 'beam_projector']
ks = [apps.index(x) for x in SIB]
raw = np.load(C / 'raw.npy', mmap_mode='r')
yon = np.load(C / 'y_on.npy', mmap_mode='r'); ypw = np.load(C / 'y_power.npy', mmap_mode='r')
grid = np.arange(meta['target_offset'], int(meta['record_s'] * 60) - 13 * 60 - 1, int(meta['grid_s'] * 60))
D, names = build_dict()
POOL = {}
for a in SIB:
    cs = [c for c, nm in enumerate(names) if nm.split(':')[0] == a]
    POOL[a] = D[:, cs].mean(1)                       # (15,) 와트당 페이저

MINSTEP = 15.0
NTR = []

def phasor(r, c, w=300):
    h = r[0:15, max(0, c - w):c]
    return np.median(h.real, 1) + 1j * np.median(h.imag, 1)

FE = {'pool': [], 'insitu': []}; RS = {'pool': [], 'insitu': []}; Y, E = [], []
n_ok = 0
for i in range(1500):
    yo = np.asarray(yon[i]); yp = np.asarray(ypw[i]); r = np.asarray(raw[i])
    # 그 기록 안에서 형제마다 전이 하나를 찾아 제자리 지문을 뽑는다
    sig = {}
    for a, k in zip(SIB, ks):
        d = np.diff(yo[:, k].astype(np.int8))
        got = []
        for j in np.nonzero(d != 0)[0]:
            if j < 4 or j > len(grid) - 5:
                continue
            if np.any(np.diff(yo[max(0, j - 4):j + 5], axis=0) != 0, axis=0).sum() > 1:
                continue                              # 그 순간 다른 기기도 바뀌면 버린다
            dw = yp[j + 4, k] - yp[j - 4, k]
            if abs(dw) < MINSTEP:                     # 작은 계단은 나누기에서 잡음이 커진다
                continue
            got.append((phasor(r, grid[j + 4]) - phasor(r, grid[j - 4])) / dw)
        if got:                                       # 그 기록의 전이를 **모두** 평균한다
            sig[a] = np.mean(np.stack(got), 0)
        NTR.append(len(got))
    if len(sig) < len([a for a, k in zip(SIB, ks) if yo[:, k].any()]):
        continue
    n_ok += 1
    m = yo[:, ks].sum(1) > 0
    for t in np.nonzero(m)[0][::6]:
        v = phasor(r, grid[t])
        for tag, S in (('pool', POOL), ('insitu', {**POOL, **sig})):
            res = v.copy()
            for a, k in zip(SIB, ks):
                res = res - S[a] * yp[t, k]           # 형제의 **참 전력** x 지문
            i1 = abs(res[0]) + 1e-9
            FE[tag].append([np.log(i1), np.angle(res[0]), abs(res[2]) / i1,
                            abs(res[4]) / i1, abs(res[6]) / i1,
                            np.angle(res[2] * np.conj(res[0]))])
            RS[tag].append(np.linalg.norm(np.abs(res)))
        Y.append(int(yo[t, km])); E.append(i)
Y = np.array(Y); E = np.array(E)
print('기록 %d개에서 형제 전이를 찾음 · 표본 %d (미니PC ON %d / OFF %d)'
      % (n_ok, len(Y), Y.sum(), (1 - Y).sum()))
recs = np.array(sorted(set(E))); np.random.RandomState(0).shuffle(recs)
tr_r = set(recs[:len(recs) * 3 // 4])
tr = np.array([e in tr_r for e in E]); te = ~tr
print('\n%-34s %s' % ('형제를 어느 지문으로 빼는가', '환경 밖 AUC'))
for tag, nm in (('pool', '① 풀 평균 지문 (지금)'), ('insitu', '② 그 기록의 전이에서 뽑은 지문')):
    X = np.array(FE[tag]); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    lr = LogisticRegression(max_iter=5000).fit(X[tr], Y[tr])
    print('%-34s %11.3f' % (nm, roc_auc_score(Y[te], lr.predict_proba(X[te])[:, 1])))
print('\n참고: 형제를 안 빼고 원시 특징만 0.648 · 기록마다 경계 재적합 상한 0.879')
