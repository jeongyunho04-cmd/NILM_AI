"""합성이 실측의 SMPS 파형을 만들어 내는가 — 전력을 맞춰 h3/h1 을 비교한다 (12.173).

    python -m src.run_synth_gap_probe

저항 기기가 하나도 안 켜진 창(SMPS 전용)만 골라, 전력 구간별로 `h3/h1` 을
실측과 합성에서 나란히 잰다. 조건을 안 맞추면 저항 부하가 섞여 비가 흐려진다.
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

NPZ = "processed_data/composite_eval/{}.npz"
RES = ("electiric_kettle","oven","hair_dryer","hotplate","air_conditioner","fan")
ev = json.load(open("processed_data/real_events_refined.json", encoding="utf-8"))["files"]
def mask(pairs, n):
    m = np.zeros(n, bool)
    for s,e in pairs: m[int(s*60):min(int(e*60),n)] = True
    return m
RP, RI1, RI3 = [], [], []
for stem in ("test_15","test_16","test_17","test_18","test_13","test_8","test_7"):
    if stem not in ev: continue
    n = int(ev[stem]["cycles"]); iv = ev[stem]["intervals"]
    big = np.zeros(n, bool)
    for a in RES:
        if a in iv: big |= mask(iv[a].get("on", []), n)
    z = np.load(NPZ.format(stem)); C = z["harmonics_complex"]
    ok = z["is_valid"].astype(bool) & ~big
    RP.append(z["power_features"][ok,0].astype(float))
    RI1.append(np.abs(C[ok,0])); RI3.append(np.abs(C[ok,2]))
RP, RI1, RI3 = map(np.concatenate, (RP, RI1, RI3))

CACHE = {}
for cache in ("cache/train60_h2", "cache/train60_sp"):
    m = json.load(open(cache+"/meta.json", encoding="utf-8")); apps = m["appliances"]
    yo = np.asarray(np.load(cache+"/y_on.npy", mmap_mode="r")[:120000])
    po = np.asarray(np.load(cache+"/p_observed.npy", mmap_mode="r")[:120000])
    oh = np.load(cache+"/obs_harm.npy", mmap_mode="r")
    jb = [apps.index(a) for a in RES if a in apps]
    sel = np.where(yo[:, jb].sum(1) == 0)[0]
    h = np.asarray(oh[sel])
    CACHE[cache] = (po[sel], np.hypot(h[:,0,0],h[:,0,1]), np.hypot(h[:,2,0],h[:,2,1]))

BINS = [(20,40),(40,60),(60,80),(80,110),(110,150)]
print(f"{'전력구간':>12s}{'실측 n':>8s}{'실측 h3/h1':>12s}"
      f"{'h2 n':>8s}{'h2 h3/h1':>11s}{'sp n':>8s}{'sp h3/h1':>11s}")
for lo, hi in BINS:
    m = (RP>=lo)&(RP<hi)
    row = f"{lo:4d}~{hi:<4d}W{'':2s}{int(m.sum()):8d}"
    row += f"{np.median(RI3[m]/RI1[m]):12.3f}" if m.sum()>50 else f"{'—':>12s}"
    for c in ("cache/train60_h2","cache/train60_sp"):
        p,i1,i3 = CACHE[c]; s = (p>=lo)&(p<hi)
        row += f"{int(s.sum()):8d}"
        row += f"{np.median(i3[s]/i1[s]):11.3f}" if s.sum()>50 else f"{'—':>11s}"
    print(row)
print()
print(f"실측 SMPS전용 전력 분포  p10/50/90 = "
      + "/".join(f"{v:.0f}" for v in np.percentile(RP,[10,50,90])) + " W")
for c in ("cache/train60_h2","cache/train60_sp"):
    p,_,_ = CACHE[c]
    print(f"합성 {c.split('/')[-1]:14s} p10/50/90 = "
          + "/".join(f"{v:.0f}" for v in np.percentile(p,[10,50,90])) + " W")
