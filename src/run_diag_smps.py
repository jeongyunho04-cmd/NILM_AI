# -*- coding: utf-8 -*-
"""통제가 뒤집은 것 — 미니PC 게이트는 **'이 SMPS'가 아니라 'SMPS 전류'** 에 켜지는가.

`openworld2.py` 40W 주입, 같은 씨앗:
    조건            미니PC   충전기   프로젝터  선풍기   오븐   핫플
    B 원본(아는 기기) -0.0782  -0.0367  -0.0338  -0.0502 -0.0227 -0.0016
    A 회전(낯선 지문) -0.0570  -0.0105  -0.0187  -0.1053 -0.0365 -0.0019
**SMPS 셋만 B 가 A 보다 더 아프다.** 낯선 지문이 문제가 아니라 **아는 SMPS 지문이
미니PC 게이트를 켠다**는 뜻이다. 미니PC(9.9W)는 그 셋 중 가장 작다.

재는 것 (합성·실측 같은 자):
  ① 다른 SMPS(충전기·프로젝터)가 켜진 창 대 안 켜진 창에서 미니PC AUC
  ② 미니PC OFF 인 창에서, 다른 SMPS 가 켜졌을 때 게이트가 얼마나 올라가는가
  ③ 대조로 저항성 기기(핫플·포트)가 켜졌을 때도 같은 일이 있는가
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
from sklearn.metrics import roc_auc_score

from src.evaluation.holdout import load_holdout
from src.run_gate_check import load_model, forward_file
from src.run_train_cnn import evaluate, prepare_holdout_inputs

CK = sys.argv[1] if len(sys.argv) > 1 else "results/gs_base_s0.pt"
HD = "processed_data/holdout60_v22"
SMPS = ["laptop_charger", "beam_projector"]
RES = ["hotplate", "electiric_kettle", "hair_dryer"]
CYC = 60.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, apps, _ = load_model(CK, dev)
J = apps.index("minipc")
print("모델 %s · 기기열 %s" % (CK.split("/")[-1], apps))


def block(g, y, cond, name):
    """cond 참/거짓 두 무리에서 AUC 와 OFF 창 게이트 중앙값."""
    out = []
    for sel, t in ((~cond, "없음"), (cond, "있음")):
        yy, gg = y[sel], g[sel]
        a = roc_auc_score(yy, gg) if sel.sum() > 30 and 0 < yy.mean() < 1 else float("nan")
        off = gg[yy == 0]
        out.append((int(sel.sum()), a, float(np.median(off)) if len(off) else float("nan"),
                    float((off > 0.5).mean()) if len(off) else float("nan")))
    print("  %-22s %s" % (name, " | ".join(
        "%s 창%5d AUC %s OFF중앙 %.3f 오탐 %5.1f%%"
        % (t, n, "  -  " if np.isnan(a) else "%.3f" % a, m, 100 * f)
        for (n, a, m, f), t in zip(out, ("없음", "있음")))))


print("\n=== 합성 (holdout60_v22) ===")
hs = load_holdout(HD)
_, on_prob = evaluate(model, prepare_holdout_inputs(hs), dev)
on_prob = np.asarray(on_prob)
Y = np.asarray(hs.y_on)
g, y = on_prob[:, J], Y[:, J].astype(np.int8)
sm = np.zeros(len(y), bool)
for a in SMPS:
    sm |= Y[:, apps.index(a)].astype(bool)
rs = np.zeros(len(y), bool)
for a in RES:
    if a in apps:
        rs |= Y[:, apps.index(a)].astype(bool)
block(g, y, sm, "다른 SMPS 가")
block(g, y, rs, "저항성 기기가")

print("\n=== 실측 (test_1~4) ===")
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
G, YY, SM, RS = [], [], [], []
for stem in ["test_1", "test_2", "test_3", "test_4"]:
    spec = ev.get(stem)
    if spec is None or "minipc" not in spec.get("appliances_present", []):
        continue
    d = forward_file(model, stem, dev, stride=30)
    tc = d["targets"]; n = len(tc)

    def mask(app, key="on"):
        m = np.zeros(n, bool)
        for t0, t1 in spec["intervals"].get(app, {}).get(key, []):
            m[(tc >= t0 * CYC) & (tc < t1 * CYC)] = True
        return m
    keep = ~mask("minipc", "uncertain")
    G.append(d["gate"][keep, J]); YY.append(mask("minipc")[keep].astype(np.int8))
    s = np.zeros(n, bool); r = np.zeros(n, bool)
    for a in SMPS:
        if a in spec.get("appliances_present", []):
            s |= mask(a)
    for a in RES:
        if a in spec.get("appliances_present", []):
            r |= mask(a)
    SM.append(s[keep]); RS.append(r[keep])
G = np.concatenate(G); YY = np.concatenate(YY)
SM = np.concatenate(SM); RS = np.concatenate(RS)
print("  창 %d · 미니PC ON %.1f%% · 다른 SMPS ON %.1f%% · 저항성 ON %.1f%%"
      % (len(YY), 100 * YY.mean(), 100 * SM.mean(), 100 * RS.mean()))
print("  전체 AUC %.3f" % roc_auc_score(YY, G))
block(G, YY, SM, "다른 SMPS 가")
block(G, YY, RS, "저항성 기기가")
