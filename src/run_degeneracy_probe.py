"""축퇴 쌍을 무엇이 가르는가 — 상태별 실측 지문으로 (12.165.5)."""
import sys, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from src.synthesis.segment_pool import SegmentPool
from src.model.postproc import RESISTIVE_OHM, HALFWAVE_OHM, HALFWAVE_ABS_MIN

V = 222.0
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")

def state_sig(app, want_half=None):
    """그 기기(그리고 반파 여부)의 **와트당** 페이저 중앙값을 실측에서 뽑는다."""
    cs, ps = [], []
    for a in pool.appliance_activations.get(app, []):
        m = np.asarray(a.is_on).astype(bool)
        if not m.any(): continue
        c = a.net_harmonics_complex[m]; p = np.asarray(a.net_power_features)[m, 0]
        ok = p > 50.0
        if not ok.any(): continue
        c, p = c[ok], p[ok]
        if want_half is not None:
            h = (np.abs(c[:, 1]) - np.abs(c[:, 3])) > HALFWAVE_ABS_MIN
            sel = h if want_half else ~h
            if not sel.any(): continue
            c, p = c[sel], p[sel]
        cs.append(c / p[:, None]); ps.append(p)
    if not cs: return None, np.nan
    c = np.concatenate(cs)
    return np.median(c.real, 0) + 1j*np.median(c.imag, 0), float(np.median(np.concatenate(ps)))

SIG = {
    "electiric_kettle": state_sig("electiric_kettle")[0],
    "oven":             state_sig("oven")[0],
    "hotplate":         state_sig("hotplate")[0],
    "hair_dryer":       state_sig("hair_dryer", want_half=False)[0],   # 강풍(전파)
    "hair_dryer_반파":   state_sig("hair_dryer", want_half=True)[0],    # 약풍(반파)
}
R = dict(RESISTIVE_OHM); R["hair_dryer_반파"] = HALFWAVE_OHM["hair_dryer"]
NAME = {"electiric_kettle":"포트","oven":"오븐","hair_dryer":"드라이기강",
        "hotplate":"핫플","hair_dryer_반파":"드라이기약"}
print("상태별 와트당 지문 검산 (그 기기를 V²/R 로 켰을 때)")
print(f"{'':14s}{'W':>7s}{'|I1| A':>9s}{'h2/h1 %':>9s}{'h3/h1 %':>9s}{'|I2|−|I4| A':>13s}")
for k, c in SIG.items():
    p = V*V/R[k]; cc = c*p
    print(f"{NAME[k]:14s}{p:7.0f}{abs(cc[0]):9.3f}{abs(cc[1])/abs(cc[0])*100:9.2f}"
          f"{abs(cc[2])/abs(cc[0])*100:9.2f}{abs(cc[1])-abs(cc[3]):13.3f}")

def phasor(c): return sum(SIG[x]*(V*V/R[x]) for x in c)
keys = list(R); combos = []
for r in range(1, len(keys)+1):
    for c in itertools.combinations(keys, r):
        if "hair_dryer" in c and "hair_dryer_반파" in c: continue
        combos.append((c, sum(1/R[x] for x in c)))
combos.sort(key=lambda x: x[1])
print(f"\n{'축퇴 쌍':50s}{'ΔW':>6s}{'h3배수':>8s}{'ch50 A':>9s}{'ch50 B':>9s}{'가르는 축':>11s}")
for i in range(len(combos)):
    for j in range(i+1, len(combos)):
        a, ga = combos[i]; b, gb = combos[j]
        if abs(gb-ga)/ga > 0.02: break
        ca, cb = phasor(a), phasor(b)
        ra = abs(ca[2])/abs(ca[0])*100; rb = abs(cb[2])/abs(cb[0])*100
        h3x = max(ra,rb)/max(min(ra,rb),1e-9)
        d50a = abs(ca[1])-abs(ca[3]); d50b = abs(cb[1])-abs(cb[3])
        gate = (d50a > HALFWAVE_ABS_MIN) != (d50b > HALFWAVE_ABS_MIN)
        axis = "반파 ch50" if gate else ("h3" if h3x >= 2.5 else "**없다**")
        print(f"{'+'.join(NAME[x] for x in a)+' ↔ '+'+'.join(NAME[x] for x in b):50s}"
              f"{V*V*(gb-ga):6.0f}{h3x:8.1f}{d50a:9.3f}{d50b:9.3f}{axis:>11s}")
