"""FAN_LIGHT 을 고조파로 보는가, 맥락(세션)으로 보는가 (12.164.15)."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401
import numpy as np
from src.run_gate_check import forward_file, load_model
from src.model.inputs import FINE_CYCLES

def mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs: m[int(s*60):min(int(e*60), n)] = True
    return m

ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]["test_4"]
n = int(ev["cycles"])
sess = mask(ev["intervals"]["oven"]["on"], n)
heat = mask(ev["intervals"]["oven"]["_heater_pulses"], n)
# 각 사이클에서 가장 가까운 히터 펄스까지의 거리 (초)
idx = np.where(heat)[0]
allc = np.arange(n)
dist = np.abs(allc[:, None] - idx[None, :]).min(1) / 60.0 if len(idx) else np.full(n, 1e9)

dev = "cuda"
print(f"세밀 가지 {FINE_CYCLES/60:.0f}초 / 광역 가지 60초")
print(f"{'':22s}{'가까운 펄스까지':>16s}{'n':>7s}{'σ(plug)':>10s}{'standby W':>11s}")
for ck in sys.argv[1:]:
    model, apps, _ = load_model(ck, dev)
    jo = apps.index("oven")
    d = forward_file(model, "test_4", dev, stride=30)
    t = d["targets"]
    fan = sess[t] & ~heat[t]
    g = d["gate"][:, jo]
    plug = d["idle"][:, jo] / np.clip(1.0 - g, 1e-4, None)
    tag = ck.split("/")[-1].replace(".pt", "")
    P = d["p_observed"]
    bins = [("펄스 0~10초", fan & (dist[t] < 10)),
            ("펄스 10~30초", fan & (dist[t] >= 10)),
            ("관측 <60W", fan & (P < 60)), ("관측 60~150W", fan & (P >= 60) & (P < 150)),
            ("관측 150~400W", fan & (P >= 150) & (P < 400)), ("관측 >400W", fan & (P >= 400))]
    for lab, m in bins:
        if m.sum() < 5: continue
        print(f"{tag:22s}{lab:>16s}{m.sum():7d}{np.median(plug[m]):10.3f}"
              f"{np.median(d['standby'][m, jo]):11.2f}")
