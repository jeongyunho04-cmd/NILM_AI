"""미니PC 예측을 동반 SMPS 로 가른다 (12.178).

    python -m src.run_minipc_company_probe results/adapt_ac_s0.pt

⚠ **저항 6종이 하나도 안 켜진 창만** 본다. 안 빼면 드라이기가 도는 창이 섞여
관측 평균이 89 -> 360W 로 튄다 (내가 처음에 그렇게 잘못 봤다).
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401
import numpy as np, torch
from src.run_gate_check import forward_file, load_model
RES = ("electiric_kettle","oven","hair_dryer","hotplate","air_conditioner","fan")
ev = json.load(open("processed_data/real_events_refined.json", encoding="utf-8"))["files"]
def mask(p,n):
    m=np.zeros(n,bool)
    for s,e in p: m[int(s*60):min(int(e*60),n)]=True
    return m
dev="cuda" if torch.cuda.is_available() else "cpu"
for ck in sys.argv[1:]:
    model, apps, _ = load_model(ck, dev); j=apps.index("minipc")
    jc, jp = apps.index("laptop_charger"), apps.index("beam_projector")
    print(f"\n=== {ck.split('/')[-1]} ===  (저항 6종이 하나도 안 켜진 창만)")
    print(f"{'파일':9s}{'동반':>11s}{'창':>6s}{'관측W':>8s}{'미니PC':>8s}{'게이트':>8s}"
          f"{'p_raw':>8s}{'충전기':>8s}{'프로젝터':>9s}{'잔차':>8s}")
    for stem in ("test_15","test_18"):
        if stem not in ev or "minipc" not in ev[stem]["intervals"]: continue
        n=int(ev[stem]["cycles"]); iv=ev[stem]["intervals"]
        on=mask(iv["minipc"].get("on",[]),n)
        pj=mask(iv.get("beam_projector",{}).get("on",[]),n)
        ch=mask(iv.get("laptop_charger",{}).get("on",[]),n)
        big=np.zeros(n,bool)
        for a in RES:
            if a in iv: big |= mask(iv[a].get("on",[]),n)
        d=forward_file(model, stem, dev, stride=30); t=d["targets"]
        P=d["gate"]*d["p_raw"]
        base = on[t] & ~big[t]
        for lab, sub in (("미니PC만", ~pj[t] & ~ch[t]), ("+프로젝터", pj[t] & ~ch[t]),
                         ("+충전기", ~pj[t] & ch[t]), ("3종 다", pj[t] & ch[t])):
            m = base & sub
            if m.sum() < 20: continue
            r = d['p_observed'][m]-P[m].sum(1)-d['standby'][m].sum(1)-d['p_noise'][m]
            print(f"{stem:9s}{lab:>11s}{int(m.sum()):6d}{P[m].sum(1).mean()+r.mean()+d['standby'][m].sum(1).mean()+d['p_noise'][m].mean():8.1f}"
                  f"{P[m,j].mean():8.2f}{d['gate'][m,j].mean():8.3f}{d['p_raw'][m,j].mean():8.2f}"
                  f"{P[m,jc].mean():8.1f}{P[m,jp].mean():9.1f}{r.mean():8.1f}")
