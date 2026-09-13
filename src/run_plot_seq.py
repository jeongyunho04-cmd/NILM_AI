# -*- coding: utf-8 -*-
"""사슬 예측을 실측 5파일의 **시간축**에 그린다 (13.84.28).

지표는 표로 내고 그림은 **모양이 정보일 때만** 그린다 — 타임라인이 그 자리다.
파일마다 그 파일에 있는 기기를 줄로 놓고, 기기마다 띠 셋을 그린다.
    참값 · 창별(게이트 > 0) · 사슬(Viterbi)
틀린 구간은 빨강으로 덮는다 — 미탐(참 ON 인데 껐다)과 오탐(참 OFF 인데 켰다)을 나눠 칠한다.

    python -X utf8 src/run_plot_seq.py results/hpc/seq_v37_ep12.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch

from src.model.chain import ChainHeads, viterbi
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

C_TRUE, C_PRED, C_MISS, C_FALSE = "#3b6ea5", "#4a4a4a", "#d94f4f", "#e8a33d"


def bands(ax, y, m, t, color, h=0.26):
    """불리언 열을 구간 띠로."""
    m = np.asarray(m, bool)
    if not m.any():
        return
    d = np.diff(np.r_[0, m.view(np.int8), 0])
    for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        ax.add_patch(plt.Rectangle((t[s], y - h / 2), t[min(e, len(t) - 1)] - t[s], h,
                                   facecolor=color, edgecolor="none"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="results/hpc/seq_v37_ep12.pt")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--baseline", default="results/cnn_v37.pt")
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
    base = load_model(a.baseline, dev)[0]              # 고정 v37 창별 기준선
    real = real_windows(apps, ck["meta"]["grid_s"], dev)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_rows = sum(int(real[s]["present"].sum()) for s in FILES)
    fig, axes = plt.subplots(len(FILES), 1, figsize=(13, 2.2 + 0.42 * n_rows + 0.9 * len(FILES)),
                             gridspec_kw={"height_ratios": [max(1, int(real[s]["present"].sum()))
                                                            for s in FILES]})
    for ax, stem in zip(np.atleast_1d(axes), FILES):
        d = real[stem]
        with torch.no_grad():
            Z, GL, BL = [], [], []
            for i in range(0, len(d["t"]), 512):
                f = torch.from_numpy(d["fine"][i:i + 512]).to(dev)
                w = torch.from_numpy(d["wide"][i:i + 512]).to(dev)
                o = model(f, w); Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                BL.append(base(f, w)["on_logit"].float())
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                     torch.cat(GL)[None])
            path = viterbi(em, on, off, ini)[0].cpu().numpy().astype(bool)
            win = (torch.cat(BL) > 0).cpu().numpy()
        t, y = d["t"], d["y"].astype(bool)
        ks = np.nonzero(d["present"])[0]
        for r, k in enumerate(ks):
            yy = len(ks) - 1 - r
            bands(ax, yy + 0.30, y[:, k], t, C_TRUE)                       # 참값
            bands(ax, yy + 0.00, win[:, k], t, C_PRED)                     # 창별
            bands(ax, yy + 0.00, win[:, k] & ~y[:, k], t, C_FALSE)
            bands(ax, yy + 0.00, ~win[:, k] & y[:, k], t, C_MISS)
            bands(ax, yy - 0.30, path[:, k], t, C_PRED)                    # 사슬
            bands(ax, yy - 0.30, path[:, k] & ~y[:, k], t, C_FALSE)
            bands(ax, yy - 0.30, ~path[:, k] & y[:, k], t, C_MISS)
        ax.set_yticks(np.arange(len(ks)))
        ax.set_yticklabels([apps[k] for k in ks[::-1]], fontsize=8)
        ax.set_ylim(-0.6, len(ks) - 0.4); ax.set_xlim(t[0], t[-1])
        acc_w = np.mean([(win[:, k] == y[:, k]).mean() for k in ks])
        acc_c = np.mean([(path[:, k] == y[:, k]).mean() for k in ks])
        ttl = "%s — 그 파일 전 기기 평균  창별 %.3f -> 사슬 %.3f" % (stem, acc_w, acc_c)
        if "minipc" in [apps[k] for k in ks]:
            km = apps.index("minipc")
            ttl += "   ·   미니PC  %.3f -> %.3f" % ((win[:, km] == y[:, km]).mean(),
                                                    (path[:, km] == y[:, km]).mean())
        ax.set_title(ttl, fontsize=9, loc="left", pad=4)
        ax.tick_params(labelsize=8); ax.grid(axis="x", alpha=0.25)
    np.atleast_1d(axes)[-1].set_xlabel("시간 (초)", fontsize=9)
    fig.suptitle("%s — 실측 5파일 타임라인   (실측은 학습에 안 들어간다)" % Path(a.ckpt).name,
                 fontsize=12, y=0.995)
    fig.legend(handles=[Patch(color=C_TRUE, label="참값 ON"), Patch(color=C_PRED, label="예측 맞음"),
                        Patch(color=C_MISS, label="미탐 — 참 ON 인데 껐다"),
                        Patch(color=C_FALSE, label="오탐 — 참 OFF 인데 켰다")],
               loc="upper center", bbox_to_anchor=(0.5, 0.972), ncol=4, fontsize=9.5,
               frameon=False)
    fig.text(0.5, 0.948, "기기마다 띠 셋:  위 = 참값  ·  가운데 = 창별(v37 게이트)  ·  아래 = 사슬(Viterbi)",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.938])
    p = out / ("timeline_%s.png" % Path(a.ckpt).stem)
    fig.savefig(p, dpi=130); plt.close(fig)
    print("저장 %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
