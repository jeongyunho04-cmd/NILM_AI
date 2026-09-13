"""ìì¤ì´ ê·¸ 12W ì ì£¼ì¸ì ì ë ì°½ìì ê°ë¦´ ì ìëê° (12.171.2).

    python -m src.run_harm_landscape_probe results/adapt_sp_s0.pt
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401
import numpy as np, torch
from src.run_gate_check import forward_file, load_model, _signatures
from src.synthesis.segment_pool import SegmentPool
from src.model.net import harmonic_scales

ev = json.load(open("processed_data/real_events_refined.json", encoding="utf-8"))["files"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, apps, _ = load_model(sys.argv[1], dev)
sig, sb_sig, nz_sig = _signatures(apps)
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
hsc = harmonic_scales(pool, apps); del pool
jm, jc, jp = apps.index("minipc"), apps.index("laptop_charger"), apps.index("beam_projector")

def mask(pairs, n):
    m = np.zeros(n, bool)
    for s,e in pairs: m[int(s*60):min(int(e*60),n)] = True
    return m

stem = "test_18"; n = int(ev[stem]["cycles"])
on = mask(ev[stem]["intervals"]["minipc"].get("on", []), n)
d = forward_file(model, stem, dev, stride=30)
P = d["gate"] * d["p_raw"]
m = on[d["targets"]]
print(f"{stem}  미니PC ON 창 {int(m.sum())}개")

def err(Pmat):
    pred = np.einsum("bk,khc->bhc", Pmat, sig)
    pred = pred + np.einsum("bk,khc->bhc", d["idle"], sb_sig) + nz_sig[None]
    return (np.abs(pred - d["obs_harm"]) / hsc[None,:,None]).mean((1,2))

base = err(P)
print(f"  기준 재구성 오차 (harm_scale 정규화) 중앙 {np.median(base[m]):.4f}")
print(f"{'옮긴 양':>9s}{'미니PC<-충전기':>16s}{'미니PC<-프로젝터':>18s}{'기준 대비':>11s}")
for dw in (5.0, 10.0, 20.0):
    for src, lab in ((jc, "충전기"), (jp, "프로젝터")):
        Q = P.copy()
        take = np.minimum(Q[:, src], dw)
        Q[:, src] -= take; Q[:, jm] += take
        e = err(Q)
        rel = (np.median(e[m]) - np.median(base[m])) / np.median(base[m])
        print(f"{dw:9.1f}W {lab:>10s} -> 오차 {np.median(e[m]):.4f}  변화 {rel:+.2%}")
print()
print("  비교: 계단 델타에서 잰 판별 마진은 8~18% 였다 (12.171.1)")
print(f"  창 안 SMPS 예측 전력 중앙: 미니PC {np.median(P[m,jm]):.2f}W  "
      f"충전기 {np.median(P[m,jc]):.2f}W  프로젝터 {np.median(P[m,jp]):.2f}W")
print(f"  관측 |I3| 중앙 {np.median(np.hypot(d['obs_harm'][m,2,0], d['obs_harm'][m,2,1]))*1000:.1f} mA"
      f"  vs 미니PC 10W 의 |I3| {abs((sig[jm,2,0]+1j*sig[jm,2,1])*10)*1000:.1f} mA")
