"""지문 재현성의 바닥 — 같은 기기를 다시 녹화하면 얼마나 다른가 (13.84.31)."""
import sys
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from src.synthesis.segment_pool import SegmentPool

ODD = [0, 2, 4, 6, 8, 10, 12, 14]
pool = SegmentPool(npz_dir='processed_data/npz', time_split='all', carrier_apps=('oven',))
SM = ['laptop_charger', 'beam_projector', 'minipc']
print('기기별: 활성화마다 **와트당 페이저**를 뽑아 녹화 안/녹화 간 산포를 잰다')
print('%-16s %6s %6s %12s %12s %12s' % ('기기', '녹화', '활성화', '녹화내 산포', '녹화간 산포', '전체'))
for a in SM:
    acts = pool.appliance_activations[a]
    byrec = {}
    for act in acts:
        p = np.asarray(act.target_power_w, np.float64)
        H = np.asarray(act.net_harmonics_complex)
        m = p > max(2.0, 0.3 * np.median(p[p > 0]) if (p > 0).any() else 2.0)
        if m.sum() < 120:
            continue
        w = np.median(p[m])
        v = np.median(H[m][:, ODD].real, 0) + 1j * np.median(H[m][:, ODD].imag, 0)
        if w <= 0 or not np.isfinite(v).all():
            continue
        byrec.setdefault(getattr(act, 'source_file', getattr(act, 'recording', '?')), []).append(v / w)
    recs = {k: np.stack(v) for k, v in byrec.items() if len(v) >= 2}
    if len(recs) < 2:
        recs = {k: np.stack(v) for k, v in byrec.items() if len(v) >= 1}
    allv = np.concatenate([v for v in recs.values()]) if recs else None
    if allv is None or len(allv) < 3:
        print('%-16s  표본 부족' % a); continue
    ri = lambda Z: np.concatenate([Z.real, Z.imag], axis=-1)
    def spread(Z):
        M = Z.mean(0)
        return float(np.median(np.linalg.norm(ri(Z) - ri(M), axis=1)) / (np.linalg.norm(ri(M)) + 1e-12))
    within = np.median([spread(v) for v in recs.values() if len(v) >= 2]) if any(len(v) >= 2 for v in recs.values()) else np.nan
    means = np.stack([v.mean(0) for v in recs.values()])
    between = spread(means) if len(means) >= 2 else np.nan
    print('%-16s %6d %6d %11.1f%% %11.1f%% %11.1f%%'
          % (a, len(recs), len(allv), 100 * within, 100 * between, 100 * spread(allv)))
print()
print('필요 정밀도: 미니PC 12W 를 ±3W 로 읽으려면 형제 지문 오차 < 5.5%')
