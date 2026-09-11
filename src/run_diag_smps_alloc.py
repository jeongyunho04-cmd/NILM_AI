# -*- coding: utf-8 -*-
"""SMPS 3종 배분 — **참 구성별로** 얼마를 누구에게 주는가 (13.84.30).

이 과제의 본진이다. 실패 ②(유령)는 "충전기+프로젝터만 켜져 있는데 미니PC 를 준다",
실패 ③(미탐)은 "미니PC 가 켜져 있는데 안 준다" 다. 둘 다 **조건부**라 조건을 갈라야 보인다
([[split-by-class-before-taking-a-median]] · [[which-unit-you-measure-in-flips-the-answer]]).

⚠ 실측에는 기기별 참값 와트가 없다(13.84.18). 그래서 내는 것은 **배분**이지 오차가 아니다.
  참고 크기: 미니PC IDLE 10~13W · 프로젝터 30~45W · 충전기 14~63W(테이퍼).

    python -X utf8 src/run_diag_smps_alloc.py results/hpc/seq_v37_ep12.pt
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

SMPS = ["laptop_charger", "beam_projector", "minipc"]
SHORT = {"laptop_charger": "충전기", "beam_projector": "프로젝터", "minipc": "미니PC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="results/hpc/seq_v37_ep12.pt")
    ap.add_argument("--baseline", default="results/cnn_v37.pt")
    ap.add_argument("--min-n", type=int, default=8, help="이만큼 안 되는 조건은 안 낸다")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    base = load_model(a.baseline, dev)[0]
    real = real_windows(apps, ck["meta"]["grid_s"], dev)
    ks = [apps.index(x) for x in SMPS]

    rows = defaultdict(lambda: {"chain": [], "win": []})
    for stem in FILES:
        d = real[stem]
        if not all(d["present"][k] for k in ks):
            pass                                            # 일부만 있어도 그 조건으로 센다
        with torch.no_grad():
            P, GL, Z, BP, BG = [], [], [], [], []
            for i in range(0, len(d["t"]), 512):
                f = torch.from_numpy(d["fine"][i:i + 512]).to(dev)
                w = torch.from_numpy(d["wide"][i:i + 512]).to(dev)
                o = model(f, w)
                P.append(o["power"].float()); GL.append(o["on_logit"].float()); Z.append(o["z"].float())
                ob = base(f, w); BP.append(ob["power"].float()); BG.append(ob["on_logit"].float())
            p = torch.cat(P).cpu().numpy()
            bp = torch.cat(BP).cpu().numpy(); bg = (torch.cat(BG) > 0).cpu().numpy()
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                     torch.cat(GL)[None])
            path = viterbi(em, on, off, ini)[0].cpu().numpy().astype(bool)
        y = d["y"].astype(bool)
        for t in range(len(d["t"])):
            cfg = tuple(SHORT[x] for x, k in zip(SMPS, ks) if d["present"][k] and y[t, k])
            key = (stem, cfg)
            rows[key]["chain"].append([p[t, k] * path[t, k] for k in ks])
            rows[key]["win"].append([bp[t, k] * bg[t, k] for k in ks])

    print("%s  ·  참 구성별 SMPS 배분 (중앙 W)" % Path(a.ckpt).name)
    print("  참고: 미니PC IDLE 10~13W · 프로젝터 30~45W · 충전기 14~63W(테이퍼)\n")
    print("%-8s %-22s %5s | %-26s | %-26s"
          % ("파일", "참 ON 인 SMPS", "n", "사슬  충전기/프로젝터/미니PC", "v37창별  충전기/프로젝터/미니PC"))
    for (stem, cfg), v in sorted(rows.items()):
        n = len(v["chain"])
        if n < a.min_n:
            continue
        c = np.median(np.asarray(v["chain"]), axis=0)
        w = np.median(np.asarray(v["win"]), axis=0)
        mark = ""
        if "미니PC" not in cfg and cfg:
            mark = "  <- 유령" if c[2] > 3 else "  (유령 없음)"
        elif "미니PC" in cfg:
            mark = "  <- 미탐" if c[2] < 5 else ""
        print("%-8s %-22s %5d | %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f%s"
              % (stem, "+".join(cfg) or "없음", n, c[0], c[1], c[2], w[0], w[1], w[2], mark))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
