# -*- coding: utf-8 -*-
"""시퀀스 체크포인트를 **실측 5파일**에 채점한다 (13.84.27).

HPC 는 공용 계정이라 실측 npz 를 두지 않는다 — 거기서는 `--no-real` 로 합성 홀드아웃만 보고,
체크포인트(2.5MB)를 받아 **여기서** 실측을 잰다. 실측은 원래 학습에 안 들어가므로 결과는 같다.

    python -X utf8 src/run_score_seq.py results/seq_v37.pt results/seq_scratch.pt
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.model.transition import N_FEAT
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows


def score(ck_path, dev, real_cache):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    # 몸통은 체크포인트가 통째로 담고 있다. 구조·가림은 `--init`/`--ref` 가 가리키던 판에서 온다.
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False)[0]
    model.load_state_dict(ck["model"])
    model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"])
    heads.eval()

    if not real_cache:
        real_cache.update(real_windows(apps, ck["meta"]["grid_s"], dev))
    km = apps.index("minipc")
    rows, ca, wa = {}, [], []
    with torch.no_grad():
        for stem, d in real_cache.items():
            Z, GL = [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
            z = torch.cat(Z)[None]
            gl = torch.cat(GL)[None]
            em, on, off, ini = heads(z, torch.from_numpy(d["dfeat"]).float()[None].to(dev), gl)
            path = viterbi(em, on, off, ini)[0].cpu().numpy()
            w = (gl[0] > 0).cpu().numpy()
            y = d["y"].astype(bool)
            # ⚠ `run_train_seq.score_real` 과 **같은 규약**이어야 한다 — 그 파일에 있는 기기만
            #   세고, 기기별 정확도를 평균한다. 전 기기를 세면 없는 기기가 공짜로 맞아
            #   0.929 가 0.961 로 보인다 ([[match-the-scoring-convention-before-comparing]]).
            for k in np.nonzero(d["present"])[0]:
                ca.append(float((path[:, k] == y[:, k]).mean()))
                wa.append(float((w[:, k] == y[:, k]).mean()))
            rows[stem] = (float((w[:, km] == y[:, km]).mean()),
                          float((path[:, km] == y[:, km]).mean()))
    return float(np.mean(ca)), float(np.mean(wa)), rows


def main():
    cks = sys.argv[1:] or ["results/seq_e2e.pt"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache, out = {}, {}
    for c in cks:
        out[c] = score(c, dev, cache)
        print("%-28s 실측 사슬 %.4f / 창별 %.4f" % (c, out[c][0], out[c][1]), flush=True)
    print("\n미니PC 시간 정확도 (실측)")
    print("  파일     " + "".join("  %-14s" % c.split("/")[-1][:-3] for c in cks) + "   창별")
    for stem in FILES:
        if stem not in out[cks[0]][2]:
            continue
        line = "  %-8s" % stem
        for c in cks:
            line += "  %14.3f" % out[c][2][stem][1]
        print(line + "  %6.3f" % out[cks[0]][2][stem][0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
