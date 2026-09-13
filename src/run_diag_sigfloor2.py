"""산포에서 **동작점·전압**을 걷어내면 얼마가 남나 (13.84.32).

사용자: *"이 산포가 Z 나 전압 고조파에 의한 영향을 배제한 거야?"* — 배제 안 했다. 여기서 한다.
  ① 동작점(W) 을 맞춘 뒤의 산포  — sp_curves 가 겨냥하는 것
  ② V_rms 를 맞춘 뒤            — kappa 환산이 겨냥하는 것
  ③ 둘 다
남는 것이 진짜 바닥이다.
"""
import sys
sys.path.insert(0, '.')
from src import env_guard
import numpy as np
from src.synthesis.segment_pool import SegmentPool

ODD = [0, 2, 4, 6, 8, 10, 12, 14]
pool = SegmentPool(npz_dir='processed_data/npz', time_split='all', carrier_apps=('oven',))
ri = lambda Z: np.concatenate([Z.real, Z.imag], axis=-1)

def spread(V):
    if len(V) < 3:
        return np.nan
    M = V.mean(0)
    return float(np.median(np.linalg.norm(ri(V) - ri(M), axis=1)) / (np.linalg.norm(ri(M)) + 1e-12))

print('%-16s %6s | %7s %8s %8s %8s' % ('기기', '표본', '통제없음', '동작점맞춤', 'V맞춤', '둘다'))
for a in ['laptop_charger', 'beam_projector', 'minipc']:
    rows = []
    for act in pool.appliance_activations[a]:
        p = np.asarray(act.target_power_w, np.float64)
        H = np.asarray(act.net_harmonics_complex)
        pf = np.asarray(act.net_power_features, np.float64)
        v = pf[:, 4] if pf.shape[1] > 4 else np.full(len(p), act.v_ref_v)
        m = p > max(2.0, 0.3 * np.median(p[p > 0])) if (p > 0).any() else np.zeros(len(p), bool)
        if m.sum() < 60:
            continue
        idx = np.nonzero(m)[0]
        # 활성화 안에서도 1초(60사이클)씩 쪼개 표본을 늘린다 — 동작점이 움직이는 것을 살린다
        for s in range(0, len(idx) - 60, 60):
            j = idx[s:s + 60]
            w = float(np.median(p[j]))
            if w <= 1:
                continue
            ph = np.median(H[j][:, ODD].real, 0) + 1j * np.median(H[j][:, ODD].imag, 0)
            rows.append((w, float(np.median(v[j])), ph / w))
    if len(rows) < 12:
        print('%-16s %6d  표본 부족' % (a, len(rows))); continue
    W = np.array([r[0] for r in rows]); V = np.array([r[1] for r in rows])
    Z = np.stack([r[2] for r in rows])
    raw = spread(Z)
    # ① 동작점 맞춤: 전력 5분위 안에서만 산포를 재고 중앙값
    qs = np.quantile(W, np.linspace(0, 1, 6))
    s_w = np.nanmedian([spread(Z[(W >= qs[i]) & (W <= qs[i + 1])]) for i in range(5)])
    # ② V 맞춤: V 3분위 안에서
    qv = np.quantile(V, np.linspace(0, 1, 4))
    s_v = np.nanmedian([spread(Z[(V >= qv[i]) & (V <= qv[i + 1])]) for i in range(3)])
    # ③ 둘 다: 전력 3분위 x V 2분위
    qw3 = np.quantile(W, np.linspace(0, 1, 4)); qv2 = np.quantile(V, np.linspace(0, 1, 3))
    both = [spread(Z[(W >= qw3[i]) & (W <= qw3[i + 1]) & (V >= qv2[j]) & (V <= qv2[j + 1])])
            for i in range(3) for j in range(2)]
    s_b = np.nanmedian(both)
    print('%-16s %6d | %6.1f%% %7.1f%% %7.1f%% %7.1f%%'
          % (a, len(rows), 100 * raw, 100 * s_w, 100 * s_v, 100 * s_b))
print()
print('전력 범위 · V 범위 (통제가 실제로 무엇을 걷어내는지)')
for a in ['laptop_charger', 'beam_projector', 'minipc']:
    W, V = [], []
    for act in pool.appliance_activations[a]:
        p = np.asarray(act.target_power_w, np.float64)
        pf = np.asarray(act.net_power_features, np.float64)
        m = p > 2
        if m.sum() < 60:
            continue
        W += list(p[m]); V += list(pf[m, 4] if pf.shape[1] > 4 else [act.v_ref_v] * int(m.sum()))
    if W:
        print('   %-16s W %.0f~%.0f (중앙 %.0f) · V %.1f~%.1f (중앙 %.1f)'
              % (a, np.percentile(W, 5), np.percentile(W, 95), np.median(W),
                 np.percentile(V, 5), np.percentile(V, 95), np.median(V)))
print()
print('필요 정밀도 5.5%')
