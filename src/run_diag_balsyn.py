# -*- coding: utf-8 -*-
"""방출 배율 α 를 **합성에서도** 훑는다 — 균형이 영역에 따라 달라지는가 (13.84.44 ②b).

실측에서 α=0.25 가 α=1 보다 나았다 (전체 0.9530 대 0.9493, 미니PC test_2 는 0.657 -> 0.990).
그렇다면 학습이 왜 α=1 을 골랐나? **합성에서는 방출이 쓸 만하기 때문**이라는 가설이 선다
(합성 홀드아웃 창별이 0.956 이다). 그렇다면 균형은 특징이 아니라 **영역에 따라 달라지는
하이퍼파라미터**이고, 합성만으로는 못 맞춘다.

합성에서 같은 훑기를 해서 최적 α 가 1 근처면 가설이 서고, 0.25 근처면 기각된다.

    python -X utf8 src/run_diag_balsyn.py --cache <seqraw> --ck results/seq_h38_base.pt
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.run_gate_check import load_model
from src.run_train_seq import FS, PRE, SeqChunks, collate

ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ck", default="results/seq_h38_base.pt")
    ap.add_argument("--records", type=int, default=200)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.cache, "meta.json"), encoding="utf-8"))
    apps = meta["appliances"]
    raw_mm = np.load(os.path.join(a.cache, "raw.npy"), mmap_mode="r")
    grid = np.arange(meta["target_offset"], raw_mm.shape[-1] - PRE * FS - 1,
                     int(float(meta["grid_s"]) * FS))
    nrec = min(a.records, raw_mm.shape[0])
    print("캐시 %s · 생성기 %s · 기록 %d · 단계 %d" % (a.cache, meta.get("gen"), nrec, len(grid)))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ck, map_location=dev, weights_only=False)
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    ds = SeqChunks(a.cache, np.arange(nrec), len(grid), meta["target_offset"],
                   meta["window_cycles"], grid, fixed_start=0)
    km = apps.index("minipc")

    acc = {al: [] for al in ALPHAS}
    mp = {al: [] for al in ALPHAS}
    wac, wmp = [], []
    with torch.no_grad():
        for r in range(nrec):
            b = collate([ds[r]])
            fine = b["fine"].to(dev).flatten(0, 1)
            wide = b["wide"].to(dev).flatten(0, 1)
            Z, GL = [], []
            for i in range(0, len(fine), 512):
                o = model(fine[i:i + 512], wide[i:i + 512])
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
            gl = torch.cat(GL)[None]
            em, on, off, ini = heads(torch.cat(Z)[None], b["dfeat"].to(dev), gl)
            y = b["y_on"][0].numpy().astype(bool)
            # ⚠ 실측 규약과 맞춘다 — **그 기록에 있는 기기만** 센다
            pres = np.nonzero(y.any(0))[0]
            w = (gl[0] > 0).cpu().numpy()
            for k in pres:
                wac.append(float((w[:, k] == y[:, k]).mean()))
            if km in pres:
                wmp.append(float((w[:, km] == y[:, km]).mean()))
            for al in ALPHAS:
                p = viterbi(em * al, on, off, ini * al)[0].cpu().numpy()
                for k in pres:
                    acc[al].append(float((p[:, k] == y[:, k]).mean()))
                if km in pres:
                    mp[al].append(float((p[:, km] == y[:, km]).mean()))
            if (r + 1) % 50 == 0:
                print("  %d/%d" % (r + 1, nrec), flush=True)

    print("\n합성에서의 α 훑기 (기록 %d · 있는 기기만)" % nrec)
    print("   %-8s %10s %10s" % ("α", "전체평균", "미니PC"))
    for al in ALPHAS:
        print("   %-8.2f %10.4f %10.4f"
              % (al, float(np.mean(acc[al])), float(np.mean(mp[al])) if mp[al] else np.nan))
    print("   %-8s %10.4f %10.4f" % ("창별", float(np.mean(wac)),
                                     float(np.mean(wmp)) if wmp else np.nan))
    bm = max(ALPHAS, key=lambda x: float(np.mean(acc[x])))
    bmm = max(ALPHAS, key=lambda x: float(np.mean(mp[x])) if mp[x] else -1)
    print("   -> 합성 최고 α = %.2f (전체) · %.2f (미니PC)" % (bm, bmm))
    print("   견줌: **실측** 최고 α = 0.25 (전체 0.9530, α=1 은 0.9493)")
    print("        가설 — 합성이 1 근처를 고르면 '균형은 영역 의존' 이 선다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
