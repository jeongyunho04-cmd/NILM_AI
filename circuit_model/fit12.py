"""
fit12.py — 원시 스냅샷(단일 입력 ADC 포맷) → v12g 회로 파라미터
    python fit12.py laptop_charger raw_laptop_charger_2.csv raw_laptop_charger_3.csv --bg raw_noise_selfpower_1.csv
    python fit12.py minipc raw_minipc_*.csv --bg raw_noise_selfpower_1.csv --fit-g
옵션:
    --bg FILE        빈 회로(자체전원만) 스냅샷. 주기 평균 파형을 전압 위상 정렬해 뺀다.
    --fit-g          선형 덧셈 경로 G 와 병렬 캡 보정 dCx 를 같이 푼다 (미니PC·프로젝터처럼 대기 경로가 있는 기기)
    --r-per-file     R 을 파일별로 (NTC 온도 상태가 다른 세션)
    --tau 60e-6      계측 RC
    --nbins 3        요동 파일을 전력 분위로 쪼개는 수
    --site-ratio W   **자리 비를 목적함수에 넣는다** (설계 13.20). 0 이면 끔(옛 동작).
                     파형만 맞추면 자리 감도가 모자란다 — 파형은 한 자리에서 맞고 두 자리 사이의
                     고조파 비는 10~25% 짧다 (13.15.5 충전기 · 13.18.5 미니PC). 전압으로 두 자리를
                     가르고 **전력이 맞는 짝**마다 `|I_h(E)|/|I_h(D)|` 를 실측과 맞춘다.
                     짝의 두 파일에 **같은 R** 을 써서 비가 R 로 흡수되는 것을 막는다.
출력: circ12_<dev>.pkl  {'device','topo':'v12g','params':(C,R,L0,Isat,Cx,rd,G),'R_files','tau','validation','files','Cx_meas'}
"""
import numpy as np, pandas as pd, argparse, pickle, time
from scipy.optimize import least_squares
try:                                   # 패키지(circuit_model.circuit12)로 먼저 — fcm12 와 같은 규약
    from .circuit12 import sim_wave, rc_periodic, harmonics_from_wave, F
except ImportError:                    # `python circuit_model/fit12.py` 로 직접 돌릴 때
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from circuit_model.circuit12 import sim_wave, rc_periodic, harmonics_from_wave, F

# ⚠ **`circuit12` 를 최상위로 임포트하지 말 것** (2026-09-07, 13.23.5). `_core12` 가
# `@njit(cache=True)` 라 numba 가 디스크 캐시에 **모듈 이름을 박아** 둔다. 같은 파일을 한 번은
# `circuit12`, 한 번은 `circuit_model.circuit12` 로 임포트하면 캐시 항목이 섞이고, 그 뒤 파이프라인이
# 부를 때 `ModuleNotFoundError: No module named 'circuit12'` 로 **조용히 전부 실패**한다
# (`SmpsCircuit.current` 가 예외를 삼켜 `failures` 만 올라간다 — 혼합검증이 "SMPS 창이 없다" 로 나온다).
# 걸렸으면 `circuit_model/__pycache__` 를 지우면 된다.

#: 자리 비를 맞출 차수 (홀수 h3~h15). h1 은 1/V 라 자리 정보가 없다.
SITE_ORDERS = [3, 5, 7, 9, 11, 13, 15]

#: **고차 크기**를 맞출 차수 (13.67). `SITE_ORDERS` 와 같은 목록이지만 하는 일이 다르다 —
#: 저쪽은 두 자리의 **비**를 묶고 이쪽은 **크기 자체**를 묶는다.
#:
#: 왜 필요한가: 목적함수에 고차 크기를 직접 묶는 항이 **없었다.**
#:   · 파형 항은 `‖Is − I‖ / RMS(I)` 라 시간영역 에너지 기준인데 h1 이 그 99% 다.
#:     고차는 사실상 안 보인다 (격리 파형 적합 오차 2.6~10.8% 는 h1 이 만든 값이다)
#:   · 자리 비 항은 `log(|I_h(E)|/|I_h(D)|)` — 두 자리가 같은 배율로 틀리면 0 이다
#: 그래서 혼합검증에서 **기기 한 대만 켜진 창의 h13 이 44~116% 틀렸다** (13.67.1).
#: 로그비로 잰다 — 차수마다 크기가 100배 다르므로 절대 잔차를 쓰면 h3 이 독식한다.
MAG_ORDERS = [3, 5, 7, 9, 11, 13, 15]

NPC = 256; DT = 1.0 / (F * NPC)


def load_bins(files, nbins=3, min_cyc=6, fluct_thr=2.0):
    out = []
    for f in files:
        d = pd.read_csv(f); nc = len(d) // NPC; d = d.iloc[:nc * NPC]
        v = d.v_v.values.reshape(nc, NPC); i = d.i_a.values.reshape(nc, NPC); rg = d['range'].values.reshape(nc, NPC)
        Pc = (v * i).mean(1); ok = (rg == 0).all(1)          # 순수 LOW 주기만 (HIGH 스플라이스는 별도 처리 필요)
        nb = nbins if Pc[ok].std() > fluct_thr else 1
        edges = np.quantile(Pc[ok], np.linspace(0, 1, nb + 1))
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (Pc >= lo) & (Pc <= hi)
            if m.sum() < min_cyc:
                continue
            out.append(dict(V=v[m].mean(0), I=i[m].mean(0), n=int(m.sum()), src=f.split('/')[-1]))
    for b in out:
        b['P'] = float(np.mean(b['V'] * b['I']))
    return sorted(out, key=lambda b: b['P'])


def subtract_bg(bins, bgfile):
    d = pd.read_csv(bgfile); nc = len(d) // NPC
    Vb = d.v_v.values[:nc * NPC].reshape(nc, NPC).mean(0); Ib = d.i_a.values[:nc * NPC].reshape(nc, NPC).mean(0)
    phb = np.angle(np.fft.rfft(Vb)[1]); X = np.fft.rfft(Ib); h = np.arange(len(X))
    for b in bins:
        ph = np.angle(np.fft.rfft(b['V'])[1])
        b['I'] = b['I'] - np.fft.irfft(X * np.exp(1j * h * (ph - phb)), NPC); b['P'] = float(np.mean(b['V'] * b['I']))
    return float(np.mean(Vb * Ib))


def measure_cx(bins, off_thr=0.03, dv_thr=0.5):
    """비도통 구간 i = Cx·dv/dt (fit_raw.measure_cx 와 동일). 덧셈 G 가 있으면 과소추정 → --fit-g 의 dCx 로 보정."""
    best = None
    for b in bins[::-1]:
        V, I = b['V'], b['I']; dV = np.gradient(V, DT)
        off = np.abs(I) < off_thr * np.abs(I).max(); sel = off & (np.abs(dV) > dv_thr * np.abs(dV).max())
        if sel.sum() < 10:
            continue
        cx = np.polyfit(dV[sel], I[sel], 1)[0]; r = np.corrcoef(dV[sel], I[sel])[0, 1]
        if best is None or r > best[1]:
            best = (cx, r)
    if best is None or best[0] <= 0:
        b = bins[-1]; V, I = b['V'], b['I']; th = np.arange(NPC) * 360 / NPC
        rel = ((th - th[np.argmax(V)]) + 180) % 180 - 180; far = (rel < -45) | (rel > 20); dV = np.gradient(V, DT)
        cx = np.polyfit(dV[far], I[far], 1)[0]; r = np.corrcoef(dV[far], I[far])[0, 1]; best = (cx, r)
    return float(best[0]), float(best[1])


def site_pairs(bins, max_dp_frac=0.15, min_gap_v=5.0):
    """전압으로 두 자리를 가르고 **전력이 맞는 짝**을 만든다.

    자리 표는 안 쓴다 — 이 스크립트는 등록부와 독립이다. 대신 파일들의 Vrms 를 정렬해
    **가장 큰 틈**에서 자르고, 그 틈이 `min_gap_v` 보다 좁으면 자리가 하나라고 보고 포기한다
    (실측은 D 214~217V / E 229~232V 로 12V 넘게 벌어져 있다).
    """
    v = np.array([float(np.sqrt(np.mean(b['V'] ** 2))) for b in bins])
    if len(v) < 2:
        return [], None
    o = np.argsort(v)
    gaps = np.diff(v[o])
    k = int(np.argmax(gaps))
    if gaps[k] < min_gap_v:
        return [], None
    lo, hi = list(o[:k + 1]), list(o[k + 1:])
    pairs = []
    for i in lo:
        for j in hi:
            pi, pj = bins[i]['P'], bins[j]['P']
            if abs(pi - pj) <= max_dp_frac * max(pi, pj):
                pairs.append((int(i), int(j)))
    return pairs, (float(v[o[k]]), float(v[o[k + 1]]))


def fit(bins, cx, tau, fit_g=False, r_per_file=False, verbose=True, site_w=0.0,
        mag_w=0.0):
    srcs = sorted({b['src'] for b in bins}); nf = len(srcs) if r_per_file else 1
    names = ['C', 'R', 'L0', 'rd', 'Isat'] + (['G', 'dCx'] if fit_g else []) + (['R%d' % k for k in range(1, nf)] if nf > 1 else [])
    lb = np.log10([5e-6, 0.1, 60e-6, 0.05, 0.1] + ([1e-6, 1e-12] if fit_g else []) + [0.1] * (nf - 1))
    ub = np.log10([500e-6, 60, 20e-3, 10, 50] + ([2e-3, 1e-6] if fit_g else []) + [60] * (nf - 1))
    n = sum(len(b['I']) for b in bins)

    def unpack(x):
        p = 10.0 ** x; C, R, L0, rd, Isat = p[:5]; k = 5
        G = 0.0; dCx = 0.0
        if fit_g:
            G, dCx = p[5], p[6] - 1e-12; k = 7
        Rs = {srcs[0]: R}
        for j, s in enumerate(srcs[1:]):
            Rs[s] = p[k + j] if nf > 1 else R
        return C, L0, rd, Isat, G, dCx, Rs

    pairs, cut = site_pairs(bins) if site_w > 0 else ([], None)
    hs = [h - 1 for h in SITE_ORDERS]
    meas_ratio = {}
    for i, j in pairs:
        a = np.abs(harmonics_from_wave(bins[i]['I'], bins[i]['V']))[hs]
        b_ = np.abs(harmonics_from_wave(bins[j]['I'], bins[j]['V']))[hs]
        meas_ratio[(i, j)] = np.log(np.maximum(b_, 1e-12) / np.maximum(a, 1e-12))
    if verbose and site_w > 0:
        if pairs:
            print('  자리 비 %d짝 (자름 %.1f/%.1fV), 가중 %.2f: ' % (len(pairs), cut[0], cut[1], site_w)
                  + ' '.join('%.0f/%.0fW' % (bins[i]['P'], bins[j]['P']) for i, j in pairs))
        else:
            print('  ⚠ 자리 비: 짝을 못 만들었다 (한 자리뿐이거나 전력이 안 맞는다)')
    nsite = max(1, len(pairs) * len(hs))

    # ── 고차 크기 항 (13.67) ────────────────────────────────────────────────
    hm = [h - 1 for h in MAG_ORDERS]
    meas_mag = [np.log(np.maximum(
        np.abs(harmonics_from_wave(b['I'], b['V']))[hm], 1e-12)) for b in bins]
    nmag = max(1, len(bins) * len(hm))
    if verbose and mag_w > 0:
        print('  고차 크기 %d창 x %d차수, 가중 %.2f (h%s)'
              % (len(bins), len(hm), mag_w, ','.join(str(h) for h in MAG_ORDERS)))

    def resid(x):
        C, L0, rd, Isat, G, dCx, Rs = unpack(x); out = []
        sims = {}
        for k_, b in enumerate(bins):
            Is = sim_wave(b['P'], b['V'], (C, Rs[b['src']], L0, Isat, cx + dCx, rd, G), NPC, tau)
            if Is is None:
                return np.full(n + nsite * bool(pairs) + nmag * (mag_w > 0), 1e3)
            sims[k_] = Is
            out.append((Is - b['I']) / np.sqrt(np.mean(b['I'] ** 2) * len(b['I'])))
            if mag_w > 0:
                a = np.log(np.maximum(
                    np.abs(harmonics_from_wave(Is, b['V']))[hm], 1e-12))
                out.append(mag_w * (a - meas_mag[k_]) / np.sqrt(nmag))
        for i, j in pairs:
            # **짝의 두 파일에 같은 R** — 그래야 비가 R 로 흡수되지 않는다 (13.18.5)
            Ri = Rs[bins[i]['src']]
            si = sim_wave(bins[i]['P'], bins[i]['V'], (C, Ri, L0, Isat, cx + dCx, rd, G), NPC, tau)
            sj = sim_wave(bins[j]['P'], bins[j]['V'], (C, Ri, L0, Isat, cx + dCx, rd, G), NPC, tau)
            if si is None or sj is None:
                return np.full(n + nsite + nmag * (mag_w > 0), 1e3)
            a = np.abs(harmonics_from_wave(si, bins[i]['V']))[hs]
            b_ = np.abs(harmonics_from_wave(sj, bins[j]['V']))[hs]
            r = np.log(np.maximum(b_, 1e-12) / np.maximum(a, 1e-12))
            out.append(site_w * (r - meas_ratio[(i, j)]) / np.sqrt(nsite))
        return np.concatenate(out)

    best = None; t0 = time.time()
    starts = [(80e-6, 4, L0_, 0.2, Is0) for L0_ in (600e-6, 1000e-6, 1600e-6, 2500e-6) for Is0 in (0.5, 0.8, 1.3, 2.0)]
    starts += [(100e-6, 3.2, 600e-6, 0.12, 1.45), (70e-6, 7.4, 1000e-6, 0.37, 0.7), (67e-6, 5.5, 975e-6, 0.25, 0.82)]
    for C0, R0, L0_, rd0, Is0 in starts:
        x0 = [C0, R0, L0_, rd0, Is0] + ([5e-5, 1e-9] if fit_g else []) + [R0] * (nf - 1)
        r = least_squares(resid, np.clip(np.log10(x0), lb + 1e-6, ub - 1e-6), bounds=(lb, ub), method='trf',
                          diff_step=1e-3, xtol=1e-7, ftol=1e-9, max_nfev=250)
        if best is None or r.cost < best.cost:
            best = r
    C, L0, rd, Isat, G, dCx, Rs = unpack(best.x)
    rows = []
    for b in bins:
        Is = sim_wave(b['P'], b['V'], (C, Rs[b['src']], L0, Isat, cx + dCx, rd, G), NPC, tau)
        rows.append((b['src'], b['P'], float(np.sqrt(np.mean((Is - b['I']) ** 2) / np.mean(b['I'] ** 2)))))
    if verbose:
        print('  C=%.1fµF R=%s L0=%.0fµH Isat=%.2fA rd=%.2fΩ G=%.3fmS Cx=%.3f(+%.3f)µF  (%.0fs)' % (
            C * 1e6, '/'.join('%.2f' % Rs[s] for s in srcs), L0 * 1e6, Isat, rd, G * 1e3, cx * 1e6, dCx * 1e6, time.time() - t0))
        print('  잔차: ' + '  '.join('%.1fW %.1f%%' % (P, 100 * e) for _, P, e in rows))
    R_nom = Rs[srcs[-1]]                     # 마지막 파일(가장 더운 상태) 를 명목값으로
    return dict(params=(C, R_nom, L0, Isat, cx + dCx, rd, G), R_files=Rs, validation=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('device'); ap.add_argument('files', nargs='+')
    ap.add_argument('--bg'); ap.add_argument('--fit-g', action='store_true'); ap.add_argument('--r-per-file', action='store_true')
    ap.add_argument('--tau', type=float, default=60e-6); ap.add_argument('--nbins', type=int, default=3); ap.add_argument('-o')
    ap.add_argument('--harm-mag', type=float, default=0.0, metavar='W',
                    help='**고차 크기**를 목적함수에 넣는다 (13.67). 파형 항은 시간영역 '
                         'RMS 정규화라 h1 이 에너지의 99%%를 차지해 고차가 안 보이고, '
                         '자리 비 항은 비만 묶는다. h3~h15 의 log|I_h| 를 직접 맞춘다. '
                         '0 이면 끔(옛 동작).')
    ap.add_argument('--site-ratio', type=float, default=0.0, metavar='W',
                    help='자리 비를 목적함수에 넣는 가중 (0 = 끔). 설계 13.20')
    a = ap.parse_args()
    bins = load_bins(a.files, a.nbins); Pbg = subtract_bg(bins, a.bg) if a.bg else None
    cx, r = measure_cx(bins); print('%s: 동작점 %s  Cx=%.3fµF(r=%.2f)  배경 %s' % (a.device, [round(b['P'], 1) for b in bins], cx * 1e6, r, None if Pbg is None else '%.2fW' % Pbg))
    res = fit(bins, cx, a.tau, a.fit_g, a.r_per_file, site_w=a.site_ratio,
              mag_w=a.harm_mag)
    out = dict(device=a.device, topo='v12g', params=res['params'], R_files=res['R_files'], tau=a.tau, Cx_meas=cx,
               site_ratio_w=a.site_ratio, harm_mag_w=a.harm_mag,
               validation=res['validation'], files=[f.split('/')[-1] for f in a.files], bg=a.bg, adc='single-ended', date='2026-09-06')
    path = a.o or 'circ12_%s.pkl' % a.device; pickle.dump(out, open(path, 'wb')); print('->', path)
