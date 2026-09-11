"""전력 구간별 지문이 h11 결손과 유령을 고치는가 — CPU (13.84.32).

손실은 `power x 고정지문` 이라 전력에 선형인데, 와트당 고차 함량이 동작점에 따라 47~109% 변한다.
전력 구간별 사전으로 바꾸고 **반복**한다 (전력을 알아야 구간을 고르므로):
    w <- nnls(D(bin(w)), b)   를 몇 번.
"""
import sys, json
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from scipy.optimize import nnls
from scipy.stats import trim_mean
from src.preprocessing import load_nilm_npz
from src.run_diag_nnls import APPS, ODD, FS, SEG_S, build_dict
from src.synthesis.segment_pool import SegmentPool

D0, names = build_dict()
pool = SegmentPool(npz_dir='processed_data/npz', time_split='all', carrier_apps=('oven',))
SM = ['laptop_charger', 'beam_projector', 'minipc']
ORD = [1, 3, 5, 7, 9, 11, 13, 15]
h = len(ODD)

def pbins(a, nb=4):
    """그 기기의 전력대별 와트당 페이저."""
    rows = []
    for act in pool.appliance_activations[a]:
        p = np.asarray(act.target_power_w, np.float64)
        H = np.asarray(act.net_harmonics_complex)
        m = p > 2
        if m.sum() < 120:
            continue
        idx = np.nonzero(m)[0]
        for s in range(0, len(idx) - 60, 60):
            j = idx[s:s + 60]
            w = float(np.median(p[j]))
            if w > 1:
                rows.append((w, (np.median(H[j].real, 0) + 1j * np.median(H[j].imag, 0)) / w))
    if len(rows) < 40:
        return None
    W = np.array([r[0] for r in rows]); V = np.stack([r[1] for r in rows])
    q = np.quantile(W, np.linspace(0, 1, nb + 1))
    out = []
    for i in range(nb):
        m = (W >= q[i]) & (W <= q[i + 1])
        if m.sum() >= 10:
            out.append((q[i], q[i + 1], (np.median(V[m].real, 0) + 1j * np.median(V[m].imag, 0))[ODD]))
    return out

BINS = {a: pbins(a) for a in SM}
for a in SM:
    print('  %-16s 구간 %s' % (a, ' · '.join('%.0f~%.0fW' % (b[0], b[1]) for b in BINS[a]) if BINS[a] else '없음'))

COLS = {a: [c for c, nm in enumerate(names) if nm.split(':')[0] == a] for a in APPS}
def build(ws):
    """기기별 추정 전력 ws 에 맞는 구간 지문으로 사전을 바꾼다."""
    D = D0.copy()
    for a in SM:
        if not BINS[a]:
            continue
        w = ws.get(a, 0.0)
        b = min(BINS[a], key=lambda x: abs(0.5 * (x[0] + x[1]) - w)) if w > 1 else BINS[a][len(BINS[a]) // 2]
        for j in COLS[a]:
            D[ODD, j] = b[2]
    return np.concatenate([D[ODD].real, D[ODD].imag], axis=0)

KM = [c for c, nm in enumerate(names) if nm.split(':')[0] == 'minipc']
KEEP = [c for c in range(D0.shape[1]) if c not in KM]
A0 = np.concatenate([D0[ODD].real, D0[ODD].imag], axis=0)
mag = lambda v: np.hypot(v[:h], v[h:])

ev = json.load(open('processed_data/real_events.json', encoding='utf-8'))['files']
res = {'avg': {'on': [], 'off': [], 'r': [], 'ratio': []}, 'bin': {'on': [], 'off': [], 'r': [], 'ratio': []}}
for stem in ['test_1', 'test_2', 'test_3', 'test_4']:
    r_ = load_nilm_npz('processed_data/composite_eval/%s.npz' % stem)
    H = np.asarray(r_['harmonics_complex']); n = len(H)
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
            w, _ = nnls(A0, b)
            res['avg']['r'].append(np.linalg.norm(A0 @ w - b) / np.linalg.norm(b))
            res['avg']['ratio'].append(mag(A0 @ w) / np.maximum(mag(b), 1e-9))
            res['avg']['on' if mk['minipc'][t] else 'off'].append(sum(w[c] for c in KM))
            for _ in range(4):
                ws = {a: sum(w[c] for c in COLS[a]) for a in SM}
                A = build(ws); w, _ = nnls(A, b)
            res['bin']['r'].append(np.linalg.norm(A @ w - b) / np.linalg.norm(b))
            res['bin']['ratio'].append(mag(A @ w) / np.maximum(mag(b), 1e-9))
            res['bin']['on' if mk['minipc'][t] else 'off'].append(sum(w[c] for c in KM))

print('\n%-10s %8s | %s' % ('사전', '잔차', '차수별 예측/관측 중앙 (1 이면 맞음)'))
print('%-10s %8s | %s' % ('', '', ' '.join('h%-5d' % x for x in ORD)))
for k in ('avg', 'bin'):
    R = np.median(np.stack(res[k]['ratio']), 0)
    print('%-10s %7.1f%% | %s' % ('평균(지금)' if k == 'avg' else '전력구간별',
                                  100 * np.median(res[k]['r']),
                                  ' '.join('%6.3f' % x for x in R)))
print('\n%-10s | 미니PC 참ON 중앙  참OFF 중앙  **분리도**  참OFF p90(유령)' % '사전')
for k in ('avg', 'bin'):
    on = np.array(res[k]['on']); off = np.array(res[k]['off'])
    print('%-10s | %13.1f %11.1f %11.1f %14.1f'
          % ('평균(지금)' if k == 'avg' else '전력구간별', np.median(on), np.median(off),
             np.median(on) - np.median(off), np.percentile(off, 90)))

print('\n판정 — 수준이 밀린 것인가 분리가 된 것인가')
print('%-10s | %s' % ('사전', '참ON p25/p50/p75   참OFF p50/p75/p90   >5W ON/OFF     AUC'))
for k in ('avg', 'bin'):
    on = np.array(res[k]['on']); off = np.array(res[k]['off'])
    auc = float(np.mean([[(a > b) + 0.5 * (a == b) for b in off] for a in on]))
    print('%-10s | %5.1f/%5.1f/%5.1f  %5.1f/%5.1f/%5.1f  %4.0f%%/%4.0f%%   **%.3f**'
          % ('평균(지금)' if k == 'avg' else '전력구간별',
             np.percentile(on, 25), np.median(on), np.percentile(on, 75),
             np.median(off), np.percentile(off, 75), np.percentile(off, 90),
             100 * (on > 5).mean(), 100 * (off > 5).mean(), auc))
print('\n   AUC 는 순위만 보므로 **전체가 위로 밀리면 안 변한다**. 이것이 진짜 판정이다.')
