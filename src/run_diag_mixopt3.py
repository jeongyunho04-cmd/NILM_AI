# -*- coding: utf-8 -*-
"""레시피별 **주변 + 결합** 을 한 번 재고, 그 위에서 믹스를 푼다 (굽지 않고).

사용자 지적: *"smps 비율 더 낮춰도 되지 않나? 이러면 저항성 학습이 제대로 되나?"*
맞다. 세 판에 걸쳐 SMPS 문제를 고치며 저항성이 계속 깎였다:

    양성률        v27     v28    steady(안)
    oven         0.247  0.160   **0.112**   <- v27 의 45%
    hotplate     0.172  0.095   **0.098**
    charger      0.468  0.522     0.532
    projector    0.452  0.514     0.538

에어컨만 하한을 걸고 저항성은 안 걸었다 ([[optimizing-a-mix-can-starve-a-class]]
를 또 반복했다). 이번에는 **모든 부류에 하한**을 걸고 푼다.

재는 것 (레시피당 N창, 전부 타깃 시점):
  · 기기별 양성률          -> 믹스의 양성률은 이것의 **선형 결합**이다
  · (미니PC, 충전기) 2x2 · (미니PC, 프로젝터) 2x2 -> phi 를 정확히 계산
  · SMPS 동시 ON 수 히스토그램
그 위에서 무작위 탐색으로 제약을 만족하는 믹스를 찾는다.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.synthesis.dataset import NILMBatchGenerator
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

W = 3600
N_PER = int(sys.argv[1]) if len(sys.argv) > 1 else 150
FOCUS_OFF = 0.4
RECIPES = ["steady_loaded", "smps_overlap", "high_low_mixed", "low_load_among_standby",
           "random_uniform", "random_realistic", "high_power_resistive",
           "resistive_overlap", "standby_only", "unplugged_baseline"]
np.random.seed(0)
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", holdout_frac=0.2)
syn = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)

marg, jmc, jmp, smh, apps = {}, {}, {}, {}, None
for rec in RECIPES:
    gen = NILMBatchGenerator(segment_pool=pool, window_size_cycles=W, recipe_mix={rec: 1.0},
                             synthesizer=syn, compute_gt_harmonics=False,
                             smps_focus_off_p=FOCUS_OFF)
    apps = gen.appliance_list
    ti = gen.target_index
    iM, iC, iP = apps.index("minipc"), apps.index("laptop_charger"), apps.index("beam_projector")
    c = np.zeros(len(apps))
    jc = np.zeros((2, 2))
    jp = np.zeros((2, 2))
    sh = np.zeros(4)
    for _ in range(N_PER):
        r, _ = gen._synthesize_window()
        t = np.array([int(np.asarray(r.gt_is_on.get(a, np.zeros(W, np.int8)))
                          [min(ti, W - 1)]) for a in apps])
        c += t
        jc[t[iM], t[iC]] += 1
        jp[t[iM], t[iP]] += 1
        sh[t[iM] + t[iC] + t[iP]] += 1
    marg[rec], jmc[rec], jmp[rec], smh[rec] = c / N_PER, jc / N_PER, jp / N_PER, sh / N_PER
    print("  %-24s 쟀다" % rec, flush=True)

json.dump({"recipes": RECIPES, "appliances": list(apps), "n_per": N_PER,
           "focus_off": FOCUS_OFF,
           "marginal": {r: list(map(float, marg[r])) for r in RECIPES},
           "joint_mc": {r: jmc[r].tolist() for r in RECIPES},
           "joint_mp": {r: jmp[r].tolist() for r in RECIPES},
           "smps_hist": {r: list(map(float, smh[r])) for r in RECIPES}},
          open("results/_recipe_app_table.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n=== 레시피 x 기기 양성률 (타깃 시점, 레시피당 %d창) ===" % N_PER)
print("  %-24s %s" % ("레시피", " ".join("%-7s" % a[:7] for a in apps)))
for r in RECIPES:
    print("  %-24s %s" % (r, " ".join("%7.3f" % x for x in marg[r])))


def phi_of(j):
    n11, n10, n01, n00 = j[1, 1], j[1, 0], j[0, 1], j[0, 0]
    d = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    return float((n11 * n00 - n10 * n01) / np.sqrt(d)) if d > 0 else 0.0


def score(mx):
    p = sum(mx[i] * marg[RECIPES[i]] for i in range(len(RECIPES)))
    jc = sum(mx[i] * jmc[RECIPES[i]] for i in range(len(RECIPES)))
    jp = sum(mx[i] * jmp[RECIPES[i]] for i in range(len(RECIPES)))
    sh = sum(mx[i] * smh[RECIPES[i]] for i in range(len(RECIPES)))
    return p, phi_of(jc), phi_of(jp), float(sh[2] + sh[3])


def show(nm, mx):
    p, pc, pp, s2 = score(np.asarray(mx, float))
    print("  %-14s phi 미↔충 %+.3f · 미↔프 %+.3f · SMPS>=2 %.3f · steady %.2f"
          % (nm, pc, pp, s2, mx[RECIPES.index("steady_loaded")]))
    print("  %-14s %s" % ("", " ".join("%7.3f" % x for x in p)))
    return p, pc, pp, s2


print("\n=== 후보 믹스 (굽지 않고 계산) ===")
print("  %-14s %s" % ("", " ".join("%-7s" % a[:7] for a in apps)))
V27 = {"smps_overlap": 0.26, "standby_only": 0.10, "high_low_mixed": 0.14,
       "random_uniform": 0.11, "high_power_resistive": 0.10, "random_realistic": 0.08,
       "unplugged_baseline": 0.05, "resistive_overlap": 0.12,
       "low_load_among_standby": 0.04}
STEADY = {"steady_loaded": 0.15, "smps_overlap": 0.26, "high_low_mixed": 0.17,
          "low_load_among_standby": 0.155, "random_uniform": 0.095,
          "random_realistic": 0.07, "high_power_resistive": 0.03,
          "resistive_overlap": 0.03, "standby_only": 0.02, "unplugged_baseline": 0.02}
for nm, d in (("v27", V27), ("steady(안)", STEADY)):
    show(nm, [d.get(r, 0.0) for r in RECIPES])

# ── 제약을 걸고 무작위 탐색 ────────────────────────────────────────────────
iA = apps.index("air_conditioner")
iO = apps.index("oven")
iH = apps.index("hotplate")
iK = apps.index("electiric_kettle")
iD = apps.index("hair_dryer")
FLOOR = {iA: 0.027, iO: 0.20, iH: 0.15, iK: 0.15, iD: 0.12}   # v27 수준을 겨냥
LO = np.array([0.12 if r == "steady_loaded" else 0.02 for r in RECIPES])
HI = np.array([0.20 if r == "steady_loaded" else 0.35 for r in RECIPES])
rng = np.random.default_rng(0)
best = None
for _ in range(200000):
    x = LO + rng.random(len(RECIPES)) * (HI - LO)
    x = x / x.sum()
    if np.any(x < LO - 1e-9) or np.any(x > HI + 1e-9):
        continue
    p, pc, pp, s2 = score(x)
    if any(p[i] < f for i, f in FLOOR.items()):
        continue
    if abs(pc) > 0.10 or abs(pp) > 0.10:
        continue
    obj = s2 - 2.0 * (abs(pc) + abs(pp))          # SMPS>=2 를 키우고 phi 를 0 으로
    if best is None or obj > best[0]:
        best = (obj, x.copy())
if best is None:
    print("\n  ⚠ 제약을 만족하는 믹스를 못 찾았다 — 하한이 서로 충돌한다")
else:
    x = best[1]
    print("\n=== 제약 만족 최적 (에어컨>=0.027 · 오븐>=0.20 · 핫플>=0.15 ·"
          " 포트>=0.15 · 드라이기>=0.12 · |phi|<=0.10 · steady>=0.12) ===")
    show("steady2", x)
    print("\n  믹스:")
    for r, v in sorted(zip(RECIPES, x), key=lambda t: -t[1]):
        print('        "%s": %.3f,' % (r, round(v, 3)))
    print("  합 %.6f (반올림 뒤 %.6f)" % (x.sum(), sum(round(v, 3) for v in x)))
