"""실측 test_4 에서 FAN_LIGHT 이 |I2| 에 계단을 만드는가 (12.164.16)."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401
import numpy as np
from src.model.realdata import dense_targets

def mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs: m[int(s*60):min(int(e*60), n)] = True
    return m

STEM = sys.argv[1] if len(sys.argv) > 1 else "test_4"
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"][STEM]
n = int(ev["cycles"]); iv = ev["intervals"]
sess = mask(iv["oven"]["on"], n)
heat = mask(iv["oven"].get("_heater_pulses", []), n)
other = np.zeros(n, bool)
for a in ("electiric_kettle", "hair_dryer", "hotplate"):
    if a in iv: other |= mask(iv[a].get("on", []), n)

rw = dense_targets(STEM, stride=30)
H = np.concatenate([rw.batch(np.arange(i, min(i+512, len(rw))))[3]
                    for i in range(0, len(rw), 512)])
P = np.concatenate([rw.batch(np.arange(i, min(i+512, len(rw))))[2]
                    for i in range(0, len(rw), 512)])
t = rw.target_cycle
mag = np.hypot(H[:,:,0], H[:,:,1]) * 1000     # mA

lowp  = P < 300.0 if "P" in dir() else None
fan   = sess[t] & ~heat[t] & ~other[t]
offw  = ~sess[t] & ~other[t]
print(f"[{STEM}]")
print(f"{'구간':22s}{'n':>6s}{'관측W':>9s}" + "".join(f"{'|I'+str(h)+'|':>9s}" for h in (1,2,3,4,5)))
for lab, m in (("오븐 세션 밖(조용)", offw), ("FAN_LIGHT(조용)", fan)):
    if m.sum() < 5: print(f"{lab:22s} n={m.sum()}"); continue
    print(f"{lab:22s}{m.sum():6d}{np.median(P[m]):9.1f}"
          + "".join(f"{np.median(mag[m,h-1]):9.2f}" for h in (1,2,3,4,5)))
if fan.sum() >= 5 and offw.sum() >= 5:
    d = [np.median(mag[fan,h-1]) - np.median(mag[offw,h-1]) for h in (1,2,3,4,5)]
    sd = [1.4826*np.median(np.abs(mag[offw,h-1]-np.median(mag[offw,h-1]))) for h in (1,2,3,4,5)]
    print(f"{'계단 Δ':22s}{'':6s}{np.median(P[fan])-np.median(P[offw]):9.1f}"
          + "".join(f"{x:9.2f}" for x in d))
    print(f"{'  Δ/흩어짐(σ)':22s}{'':6s}{'':9s}"
          + "".join(f"{d[i]/max(sd[i],1e-9):9.1f}" for i in range(5)))
    print(f"\n합성이 말하는 FAN_LIGHT: |I1| 67.3  |I2| 6.28  |I3| 7.01  |I4| 4.31  |I5| 5.23 mA")
