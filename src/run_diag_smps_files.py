# -*- coding: utf-8 -*-
"""형제-ON 미니PC 게이트를 **파일별로** 갈라 채점한다 (13.83.22 ⑥).

    python -X utf8 src/run_diag_smps_files.py cnn_v29 cnn_v25

`run_diag_smps.py` 는 test_1~4 를 합쳐 하나의 AUC 를 낸다. 그 한 숫자 안에 **서로 다른 실패 둘**이
섞여 있었다 — test_1 의 부동 충전기(14W) 와 test_3/4 의 충전기+프로젝터. 판마다 어느 쪽이
나쁜지가 반대라, 합친 AUC 는 판 사이 비교에서 뜻이 없다.

타깃 시점 라벨 · stride 30 · 형제 SMPS(충전기·프로젝터) ON 창만. OFF 창의 충전기 게이트도 같이 낸다
(형제를 제대로 보면서 미니PC 를 덧붙이는지, 형제를 미니PC 로 바꿔 읽는지가 갈린다).
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch  # noqa: F401
from sklearn.metrics import roc_auc_score

from src.run_gate_check import forward_file, load_model

MODELS = sys.argv[1:] or ["cnn_v29", "cnn_v25"]
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
for name in MODELS:
    model, apps, ck = load_model("results/%s.pt" % name, dev)
    J, KC, KP = apps.index("minipc"), apps.index("laptop_charger"), apps.index("beam_projector")
    print("== %s  (형제-ON 창 · 타깃 시점 라벨 · stride 30)" % name)
    print("   %-7s %6s %6s %8s %8s %9s %9s %10s %10s" % ("파일", "ON", "OFF", "AUC", "오탐%", "게이트ON", "게이트OFF", "chg@OFF", "proj@OFF"))
    allG, allY = [], []
    for stem in ["test_1", "test_2", "test_3", "test_4"]:
        spec = ev.get(stem)
        if spec is None or "minipc" not in spec["appliances_present"]:
            continue
        d = forward_file(model, stem, dev, stride=30)
        t = d["targets"]; n = int(t.max()) + 1

        def mask(app, key="on"):
            m = np.zeros(n + 3600, bool)
            for a, b in spec["intervals"].get(app, {}).get(key, []):
                m[int(a * 60):int(b * 60)] = True
            return m[t]

        y = mask("minipc"); unc = mask("minipc", "uncertain")
        sib = np.zeros(len(t), bool)
        for a in ("laptop_charger", "beam_projector"):
            if a in spec["appliances_present"]:
                sib |= mask(a)
        m = sib & ~unc
        yy, gg = y[m], d["gate"][m, J]
        gc, gp = d["gate"][m, KC], d["gate"][m, KP]
        off = yy == 0
        auc = roc_auc_score(yy, gg) if len(set(yy)) == 2 else float("nan")
        print("   %-7s %6d %6d %8.3f %8.1f %9.3f %9.3f %10.3f %10.3f"
              % (stem, (yy == 1).sum(), off.sum(), auc, 100 * (gg[off] > 0.5).mean() if off.any() else float("nan"),
                 np.median(gg[yy == 1]) if (yy == 1).any() else float("nan"),
                 np.median(gg[off]) if off.any() else float("nan"),
                 np.median(gc[off]) if off.any() else float("nan"), np.median(gp[off]) if off.any() else float("nan")))
        allG.append(gg); allY.append(yy)
    G, Y = np.concatenate(allG), np.concatenate(allY)
    print("   %-7s %6d %6d %8.3f %8.1f" % ("전체", (Y == 1).sum(), (Y == 0).sum(), roc_auc_score(Y, G), 100 * (G[Y == 0] > 0.5).mean()))
