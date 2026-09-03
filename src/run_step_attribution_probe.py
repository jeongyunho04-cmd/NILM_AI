"""ê³ë¨ íëë¥¼ ìë°©í¥ ëª¨íì¼ë¡ ì§ì  íì´ ë³¸ë¤ â ìµìê° ì ëµì ìëê° (12.171.1).

    python -m src.run_step_attribution_probe
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from src.synthesis.segment_pool import SegmentPool
from src.model.net import harmonic_signatures
from src.synthesis.sp_curves import load_curves

NPZ = "processed_data/composite_eval/{}.npz"
SMPS = ("minipc", "laptop_charger", "beam_projector")
ev = json.load(open("processed_data/real_events_refined.json", encoding="utf-8"))["files"]
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
apps = pool.get_appliance_types()
SIG = harmonic_signatures(pool, apps)          # 와트당 (K,15,2)
M = load_curves()
del pool

H = 15
def obs_step(stem, t0, pre=(1, 10), post=(1, 10)):
    z = np.load(NPZ.format(stem))
    C = z["harmonics_complex"]; T = z["t_rel_s"].astype(float)
    P = z["power_features"][:, 0].astype(float); V = z["power_features"][:, 4].astype(float)
    ok = z["is_valid"].astype(bool)
    a = (T > t0 - pre[1]) & (T < t0 - pre[0]) & ok
    b = (T > t0 + post[0]) & (T < t0 + post[1]) & ok
    dC = C[b].mean(0) - C[a].mean(0)
    return dC[:H], float(P[b].mean() - P[a].mean()), float(V[b].mean())

def fixed_current(name, p):
    j = apps.index(name)
    return (SIG[j, :H, 0] + 1j * SIG[j, :H, 1]) * p

def sp_current(name, p, v):
    m = M.get(name)
    if m is None: return None
    return m.current(p, v)[:H]

print(f"{'파일/t':16s}{'ΔP 관측':>9s}{'|ΔI1|':>9s}{'|ΔI3|':>9s}")
CASES = [("test_18", 307.4), ("test_18", 564.7)]
for stem, t0 in CASES:
    dC, dP, V = obs_step(stem, t0)
    print(f"{stem+' '+str(t0):16s}{dP:9.2f}{abs(dC[0])*1000:9.2f}{abs(dC[2])*1000:9.2f}")

    for lab, fn in (("고정 주입", fixed_current), ("s(p)", sp_current)):
        print(f"    [{lab}] 각 기기 단독으로 풀면")
        rows = []
        for nm in SMPS:
            best = None
            for p in np.linspace(0.5, 80.0, 400):
                c = fn(nm, p) if fn is fixed_current else fn(nm, p, V)
                if c is None: break
                r = float(np.linalg.norm(c - dC))
                if best is None or r < best[1]: best = (p, r)
            if best is None:
                rows.append((nm, np.nan, np.nan)); continue
            rows.append((nm, best[0], best[1] / max(np.linalg.norm(dC), 1e-12)))
        rows.sort(key=lambda x: (np.nan_to_num(x[2], nan=9e9)))
        for i, (nm, p, r) in enumerate(rows, 1):
            mark = "  <- 최소" if i == 1 else ""
            good = " ✓정답" if nm == "minipc" else ""
            print(f"       {nm:16s} 최적 {p:6.2f}W  상대잔차 {r:6.3f}{mark}{good}")
