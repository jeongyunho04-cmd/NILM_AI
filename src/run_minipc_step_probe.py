"""미니PC 켜짐 계단에서 모델은 그 12W 를 누구에게 주는가 (12.170).

    python -m src.run_minipc_step_probe results/adapt_sp_s0.pt

`test_18` 의 미니PC on 이벤트 두 개는 라벨이 확실하다 — 관측 계단이 +11.9 /
+12.1W, Δ|I3| +42~45mA 로 깨끗하다. 그 앞뒤 10초로 **모델 예측의 계단**을
기기별로 갈라 보면 그 전력이 어디로 가는지 바로 보인다.
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401
import numpy as np
from src.run_gate_check import forward_file, load_model
import torch

ev = json.load(open("processed_data/real_events_refined.json", encoding="utf-8"))["files"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = {"test_18": (307.4, 564.7)}
for ck in sys.argv[1:]:
    model, apps, _ = load_model(ck, dev)
    print(f"\n=== {ck.split('/')[-1]} ===")
    for stem, ts in STEPS.items():
        d = forward_file(model, stem, dev, stride=30)
        t = d["targets"] / 60.0
        P = d["gate"] * d["p_raw"]
        for t0 in ts:
            a = (t > t0-10) & (t < t0-1); b = (t > t0+1) & (t < t0+10)
            if a.sum() < 3 or b.sum() < 3: continue
            dobs = np.median(d["p_observed"][b]) - np.median(d["p_observed"][a])
            dpred = np.median(P[b].sum(1)) - np.median(P[a].sum(1))
            dsb = np.median(d["standby"][b].sum(1)) - np.median(d["standby"][a].sum(1))
            print(f"  t={t0:.1f}s  관측 계단 {dobs:+7.2f}W | 예측 합 계단 {dpred:+7.2f}W"
                  f" | standby 계단 {dsb:+6.2f}W")
            per = [(apps[j], float(np.median(P[b,j]) - np.median(P[a,j]))) for j in range(len(apps))]
            per.sort(key=lambda x: -abs(x[1]))
            print("      기기별 예측 계단: " + "  ".join(f"{a_} {v:+.2f}" for a_, v in per[:5]))
            j = apps.index("minipc")
            print(f"      미니PC  게이트 {np.median(d['gate'][a,j]):.3f} -> {np.median(d['gate'][b,j]):.3f}"
                  f"   p_raw {np.median(d['p_raw'][a,j]):.2f} -> {np.median(d['p_raw'][b,j]):.2f}W")
            # 관측 |I3| 계단
            h = d["obs_harm"]
            i3 = np.hypot(h[:,2,0], h[:,2,1])
            print(f"      관측 |I3| {np.median(i3[a])*1000:.1f} -> {np.median(i3[b])*1000:.1f} mA"
                  f"   (계단 {np.median(i3[b])*1000-np.median(i3[a])*1000:+.1f})")
