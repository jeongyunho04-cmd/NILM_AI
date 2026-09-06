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
출력: circ12_<dev>.pkl  {'device','topo':'v12g','params':(C,R,L0,Isat,Cx,rd,G),'R_files','tau','validation','files','Cx_meas'}
"""
import numpy as np, pandas as pd, argparse, pickle, time
from scipy.optimize import least_squares
from circuit12 import sim_wave, rc_periodic, F

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


def fit(bins, cx, tau, fit_g=False, r_per_file=False, verbose=True):
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

    def resid(x):
        C, L0, rd, Isat, G, dCx, Rs = unpack(x); out = []
        for b in bins:
            Is = sim_wave(b['P'], b['V'], (C, Rs[b['src']], L0, Isat, cx + dCx, rd, G), NPC, tau)
            if Is is None:
                return np.full(n, 1e3)
            out.append((Is - b['I']) / np.sqrt(np.mean(b['I'] ** 2) * len(b['I'])))
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
    a = ap.parse_args()
    bins = load_bins(a.files, a.nbins); Pbg = subtract_bg(bins, a.bg) if a.bg else None
    cx, r = measure_cx(bins); print('%s: 동작점 %s  Cx=%.3fµF(r=%.2f)  배경 %s' % (a.device, [round(b['P'], 1) for b in bins], cx * 1e6, r, None if Pbg is None else '%.2fW' % Pbg))
    res = fit(bins, cx, a.tau, a.fit_g, a.r_per_file)
    out = dict(device=a.device, topo='v12g', params=res['params'], R_files=res['R_files'], tau=a.tau, Cx_meas=cx,
               validation=res['validation'], files=[f.split('/')[-1] for f in a.files], bg=a.bg, adc='single-ended', date='2026-09-06')
    path = a.o or 'circ12_%s.pkl' % a.device; pickle.dump(out, open(path, 'wb')); print('->', path)
