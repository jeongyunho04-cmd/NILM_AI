"""h11·h13 결손의 원인 — 와트당 고차 함량이 동작점에 따라 변하는가 (13.84.32)."""
import sys
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from src.synthesis.segment_pool import SegmentPool

pool = SegmentPool(npz_dir='processed_data/npz', time_split='all', carrier_apps=('oven',))
ORD = {1: 0, 3: 2, 5: 4, 7: 6, 9: 8, 11: 10, 13: 12, 15: 14}
print('와트당 |I_h| (mA/W) 이 전력에 따라 어떻게 변하나 — 사전은 이것을 **평균**해서 쓴다')
for a in ['laptop_charger', 'beam_projector', 'minipc']:
    rows = []
    for act in pool.appliance_activations[a]:
        p = np.asarray(act.target_power_w, np.float64)
        H = np.abs(np.asarray(act.net_harmonics_complex))
        m = p > 2
        if m.sum() < 120:
            continue
        idx = np.nonzero(m)[0]
        for s in range(0, len(idx) - 60, 60):
            j = idx[s:s + 60]
            w = float(np.median(p[j]))
            if w <= 1:
                continue
            rows.append((w, np.median(H[j], 0) / w))
    if len(rows) < 20:
        print('  %-16s 표본 부족 %d' % (a, len(rows))); continue
    W = np.array([r[0] for r in rows]); V = np.stack([r[1] for r in rows])
    q = np.quantile(W, [0, .25, .5, .75, 1.])
    print('  %-16s (전력 %.0f~%.0fW, 표본 %d)' % (a, W.min(), W.max(), len(W)))
    print('     %-14s %s' % ('전력대', ' '.join('h%-5d' % h for h in ORD)))
    band = []
    for i in range(4):
        m = (W >= q[i]) & (W <= q[i + 1])
        if m.sum() < 5:
            continue
        med = np.median(V[m], 0)
        band.append(med)
        print('     %5.0f~%-8.0f %s' % (q[i], q[i + 1],
              ' '.join('%6.2f' % (1000 * med[ORD[h]]) for h in ORD)))
    if len(band) >= 2:
        B = np.stack(band)
        rng = (B.max(0) - B.min(0)) / np.maximum(B.mean(0), 1e-12)
        print('     %-14s %s   <- 전력대 사이 변동폭' % ('변동/평균',
              ' '.join('%5.0f%%' % (100 * rng[ORD[h]]) for h in ORD)))
