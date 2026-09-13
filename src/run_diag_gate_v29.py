# -*- coding: utf-8 -*-
"""v29 굽기 전 관문 — 다섯 축을 **한 번에** 잰다.

13.83.13 의 교훈: 믹스를 바꾸면 목적함수에 없는 축이 조용히 희생된다. 그때는
`positive_rate` 를 굽고 나서 봤다. 이번에는 굽기 전에 본다.

  ① 기기별 양성률          에어컨 하한 0.0272 (v27 수준)
  ② 동시성 phi 미↔충·미↔프  |phi| <= 0.10
  ③ SMPS>=2 몫             실측 79% 를 겨냥, 0.45 이상
  ④ 창 안 σ|I_h|           차수별로 실측과 견줌 (전이 유무로 갈라서)
  ⑤ 창당 전이 개수          실측 중앙 2.0 · 부하 있는 창의 전이0 몫 11.7%
"""
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.preprocessing.file_registry import is_periodic_duty
from src.run_recipe_mix_probe import PRESETS
from src.synthesis.dataset import NILMBatchGenerator
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

MA, W = 1000.0, 3600
ORD = (3, 5, 7, 9, 11, 13, 15)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 500
NAME = sys.argv[1] if len(sys.argv) > 1 else "steady"
FOCUS_OFF = 0.4
SMPS = ("laptop_charger", "beam_projector", "minipc")
# 실측 기준값 (13.83.19)
REAL = {"tr0_share": 0.117, "edges_med": 2.0, "smps2": 0.79,
        "sig_tr0": np.array([33.0, 17.2, 26.1, 8.9, 8.5, 7.6, 6.8]),
        "sig_tr12": np.array([45.3, 35.4, 28.9, 18.5, 11.6, 10.0, 8.8])}
V27_AC = 0.0272

mix = PRESETS[NAME]
assert abs(sum(mix.values()) - 1.0) < 1e-9, "믹스 합이 1 이 아니다: %.6f" % sum(mix.values())
np.random.seed(0)
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", holdout_frac=0.2)
syn = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
gen = NILMBatchGenerator(segment_pool=pool, window_size_cycles=W, recipe_mix=mix,
                         synthesizer=syn, compute_gt_harmonics=False,
                         smps_focus_off_p=FOCUS_OFF)
apps = gen.appliance_list
ti = gen.target_index
print("믹스 %s (합 %.6f) · focus_off %.2f · 창 %d" % (NAME, sum(mix.values()), FOCUS_OFF, N))

pos = {a: 0 for a in apps}
rows, on_tgt, steady_ok, steady_n = [], [], 0, 0
for _ in range(N):
    r, rec = gen._synthesize_window()
    I = np.abs(np.asarray(r.harmonics_complex)) * MA
    on = {a: np.asarray(r.gt_is_on.get(a, np.zeros(W, np.int8))) for a in apps}
    t = {a: int(on[a][min(ti, len(on[a]) - 1)]) for a in apps}
    for a in apps:
        pos[a] += t[a]
    e = sum(int(np.abs(np.diff(on[a].astype(np.int8))).sum())
            for a in apps if not is_periodic_duty(a))
    rows.append((e, I.std(0), I.mean(0), float(np.median(np.asarray(r.power_features)[:, 0]))))
    on_tgt.append([t[a] for a in SMPS])
    if rec == "steady_loaded":
        steady_n += 1
        steady_ok += bool(r.metadata.get("steady_loaded"))
on_tgt = np.array(on_tgt)

print("\n① 기기별 양성률 (타깃 시점)")
worst = None
for a in apps:
    p = pos[a] / N
    flag = ""
    if a == "air_conditioner":
        flag = "  <- 하한 %.4f %s" % (V27_AC, "**통과**" if p >= V27_AC else "**미달**")
        worst = p
    print("  %-18s %.4f  (1:%.0f)%s" % (a, p, (1 - p) / max(p, 1e-9), flag))

print("\n② 동시성 phi · ③ SMPS>=2")
def phi(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
mi, ch, pr = on_tgt[:, 2], on_tgt[:, 0], on_tgt[:, 1]
print("  phi 미↔충 %+.3f · 미↔프 %+.3f   (|phi| <= 0.10 목표)" % (phi(mi, ch), phi(mi, pr)))
print("  SMPS>=2 몫 %.3f  (실측 0.79)" % float((on_tgt.sum(1) >= 2).mean()))

print("\n④⑤ 창 안 σ 와 전이 (부하 있는 창 |I1|>500mA 만)")
ld = [r for r in rows if r[2][0] > 500]
tr0 = [r for r in ld if r[0] == 0]
tr12 = [r for r in ld if 1 <= r[0] <= 2]
print("  부하 있는 창 %d · 전이0 몫 **%.3f** (실측 %.3f) · 창당 전이 중앙 **%.1f** (실측 %.1f)"
      % (len(ld), len(tr0) / max(len(ld), 1), REAL["tr0_share"],
         np.median([r[0] for r in ld]) if ld else -1, REAL["edges_med"]))
print("  %-14s %5s %8s | %s | 실측 대비" % ("무리", "창", "|I1|", " ".join("h%-4d" % h for h in ORD)))
for nm, sel, ref in (("전이 0", tr0, REAL["sig_tr0"]), ("전이 1~2", tr12, REAL["sig_tr12"])):
    if len(sel) < 8:
        print("  %-14s %5d (부족)" % (nm, len(sel)))
        continue
    s = np.median(np.array([r[1] for r in sel]), 0)
    v = np.array([s[h - 1] for h in ORD])
    print("  %-14s %5d %8.0f | %s | %.2f"
          % (nm, len(sel), np.median([r[2][0] for r in sel]),
             " ".join("%5.1f" % x for x in v), float(np.median(v / ref))))
print("  실측 전이0   %s" % " ".join("%5.1f" % x for x in REAL["sig_tr0"]))
print("  실측 전이1~2 %s" % " ".join("%5.1f" % x for x in REAL["sig_tr12"]))
if steady_n:
    print("\n  steady_loaded 창 %d개 · 검사 통과 %.1f%%" % (steady_n, 100 * steady_ok / steady_n))
