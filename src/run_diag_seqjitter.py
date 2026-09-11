# -*- coding: utf-8 -*-
"""실측 값이 epoch 마다 널뛰는 것이 **최적화 잡음인가 디코딩 이산성인가** (13.84.27).

사용자: *"학습률이 너무 커서 안착을 못 하는 것 아닌가?"*

두 가설이 다른 자국을 남긴다.
  ① 최적화 잡음 — 점수(방출·전이)가 epoch 마다 크게 흔들린다. 학습률을 줄이면 잡힌다.
  ② 디코딩 이산성 — 점수는 매끄럽게 움직이는데 **Viterbi 가 고른 열이 통째로 뒤집힌다**.
     test_2 의 미니PC 는 참값 ON 이 85% 라 "내내 꺼짐"(0.15)과 "내내 켜짐"(0.99) 사이에
     중간이 거의 없다. 이러면 학습률을 줄여도 안 잡히고, 고칠 곳은 전환 벌점·방출 눈금이다.

    python -X utf8 src/run_diag_seqjitter.py results/hpc [--app minipc]
"""
import argparse
import re
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/hpc")
    ap.add_argument("--app", default="minipc")
    ap.add_argument("--files", nargs="*", default=["test_2", "test_4"])
    a = ap.parse_args()

    runs = defaultdict(dict)
    for p in sorted(Path(a.dir).glob("seq_*_ep*.pt")):
        m = re.match(r"(.+)_ep(\d+)$", p.stem)
        if m:
            runs[m.group(1)][int(m.group(2))] = str(p)
    if not runs:
        raise SystemExit("seq_*_ep*.pt 를 못 찾았습니다: " + a.dir)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    real = None
    for name in sorted(runs):
        eps = sorted(runs[name])
        print("\n=== %s · %s ===" % (name, a.app))
        print("  파일     ep   방출 p50(참ON)  방출 p50(참OFF)  전이on p99   벌점   예측ON%  참ON%  정확도")
        for stem in a.files:
            prev = None
            for ep in eps:
                ck = torch.load(runs[name][ep], map_location=dev, weights_only=False)
                apps = ck["appliances"]
                k = apps.index(a.app)
                model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False)[0]
                model.load_state_dict(ck["model"]); model.eval()
                heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                                   score_norm=int(ck.get("score_norm", 0))).to(dev)
                heads.load_state_dict(ck["heads"]); heads.eval()
                if real is None:
                    real = real_windows(apps, ck["meta"]["grid_s"], dev)
                d = real[stem]
                with torch.no_grad():
                    Z, GL = [], []
                    for i in range(0, len(d["t"]), 512):
                        o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                                  torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                        Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                    em, on, off, ini = heads(torch.cat(Z)[None],
                                             torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                             torch.cat(GL)[None])
                    path = viterbi(em, on, off, ini)[0, :, k].cpu().numpy()
                e = em[0, :, k].cpu().numpy()
                so = on[0, :, k].cpu().numpy()
                y = d["y"][:, k].astype(bool)
                eon = float(np.median(e[y])) if y.any() else np.nan
                eoff = float(np.median(e[~y])) if (~y).any() else np.nan
                mark = ""
                if prev is not None:
                    de = abs(eon - prev[0])
                    dp = abs(path.mean() - prev[1])
                    # 점수는 조금 움직였는데 열이 통째로 뒤집혔으면 ② 다.
                    if dp > 0.3 and de < 0.15:
                        mark = "  <- 점수 %.3f 변화에 열이 %.0f%% 뒤집힘 (이산)" % (de, 100 * dp)
                print("  %-8s %2d   %+12.3f  %+14.3f  %+10.2f  %+6.2f  %6.0f%% %5.0f%%  %6.3f%s"
                      % (stem, ep, eon, eoff, float(np.percentile(so, 99)),
                         float(heads.switch_bias[k]), 100 * path.mean(), 100 * y.mean(),
                         float((path == y).mean()), mark))
                prev = (eon, path.mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
