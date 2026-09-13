# -*- coding: utf-8 -*-
"""방출 배율 α 와 3차원 자기보정이 **독립인가 겹치는가** (13.84.54).

둘 다 test_2 를 열었다 (α=0.25 로 0.657 -> 0.990 · 보정으로 0.657 -> 1.000). 같은 것을
고치는 중일 수 있다. 가장 날카로운 물음은 **보정을 걸면 최적 α 가 움직이는가** 다 —
13.84.44 의 진단이 "실측에서 미니PC 방출이 틀린 쪽을 가리키니 약화가 이득" 이었으므로,
보정이 그 방출을 고쳤다면 **더 이상 약화가 필요 없어야 한다** (최적 α 가 1 쪽으로 돌아온다).

    python -X utf8 src/run_diag_combine.py [results/seq_h38_base.pt] [--k 3]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.model.lossbuild import build_loss
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

FS = 60
MIN_OFF, MIN_OFF_FRAC, ITERS = 30, 0.10, 4
ALPHAS = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ck", nargs="?", default="results/seq_h38_base.pt")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--basis", default="results/drift_basis.npz")
    a = ap.parse_args()

    B = np.load(a.basis, allow_pickle=True)
    V = np.asarray(B["V"], float)[:a.k]
    ORD = [int(x) for x in B["orders"]]
    H = len(ORD); oi = [o - 1 for o in ORD]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ck, map_location=dev, weights_only=False)
    apps = list(ck["appliances"])
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    crit = build_loss(apps, "cpu", verbose=False)
    sg = crit.sig.numpy(); sigc = sg[..., 0] + 1j * sg[..., 1]
    nzc = crit.noise_sig.numpy()[:, 0] + 1j * crit.noise_sig.numpy()[:, 1]
    km = apps.index("minipc")
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)

    S = {}
    with torch.no_grad():
        for stem in FILES:
            d = cache.get(stem)
            if d is None or not d["present"][km]:
                continue
            Z, GL, PW = [], [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float()); PW.append(o["power"].float())
            em0, on, off, ini = heads(torch.cat(Z)[None],
                                      torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                      torch.cat(GL)[None])
            pw = torch.cat(PW).cpu().numpy()
            r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
            Hh = np.asarray(r["harmonics_complex"])
            ti = np.clip((d["t"] * FS).astype(int), 0, len(Hh) - 1)
            S[stem] = dict(em0=em0, on=on, off=off, ini=ini, pw=pw,
                           obs=Hh[ti][:, oi], y=d["y"].astype(bool), T=len(d["t"]))

    def calib_emission(s):
        """3차원 보정 EM 을 수렴까지 돌려 미니PC 방출을 만든다. 관문에 막히면 None."""
        path = viterbi(s["em0"], s["on"], s["off"], s["ini"])[0].cpu().numpy()
        noff = int((~path[:, km]).sum())
        if noff < MIN_OFF or noff < MIN_OFF_FRAC * s["T"]:
            return None
        rest = np.zeros_like(s["obs"])
        for j in range(len(apps)):
            if j == km:
                continue
            rest += s["pw"][:, j, None] * sigc[j][oi][None, :]
        rest = rest + nzc[oi][None, :]
        sm = sigc[km][oi]
        emn = None
        for _ in range(ITERS):
            m = ~path[:, km]
            if m.sum() < MIN_OFF:
                break
            tgt = s["obs"][m] - rest[m]
            b = np.concatenate([tgt.real, tgt.imag], 1).mean(0)
            corr = V.T @ (V @ b)
            cc = corr[:H] + 1j * corr[H:]
            resid = s["obs"] - rest - cc[None, :]
            w = (resid @ np.conj(sm)).real / (np.abs(sm) ** 2).sum()
            emn = ((w - 6.0) / 6.0).astype(np.float32)
            em = s["em0"].clone(); em[0, :, km] = torch.from_numpy(emn).to(dev)
            path = viterbi(em, s["on"], s["off"], s["ini"])[0].cpu().numpy()
        return emn

    CAL = {stem: calib_emission(s) for stem, s in S.items()}
    print("보정 관문: %s" % "  ".join("%s %s" % (k, "통과" if v is not None else "막힘")
                                     for k, v in CAL.items()))

    print("\n미니PC 시간 정확도 — 행은 방출 배율 α, 열은 파일")
    for use_cal in (False, True):
        print("\n   [%s]" % ("보정 **걸고**" if use_cal else "보정 없이"))
        print("   %-7s %8s | %s" % ("α", "평균", "".join("  %-8s" % s for s in FILES if s in S)))
        best = (None, -1)
        for al in ALPHAS:
            row, accs = [], []
            for stem, s in S.items():
                em = s["em0"].clone()
                if use_cal and CAL[stem] is not None:
                    em[0, :, km] = torch.from_numpy(CAL[stem]).to(dev)
                p = viterbi(em * al, s["on"], s["off"], s["ini"] * al)[0].cpu().numpy()
                v = float((p[:, km] == s["y"][:, km]).mean())
                accs.append(v); row.append(v)
            m = float(np.mean(accs))
            if m > best[1]:
                best = (al, m)
            print("   %-7.3f %8.4f | %s" % (al, m, "".join("  %-8.3f" % v for v in row)))
        print("   -> 최적 α = %.3f (%.4f)" % best)

    print("\n읽는 법 — 보정을 걸었을 때 최적 α 가 **1 쪽으로 돌아오면** 둘은 같은 것을 고친 것이고")
    print("  (보정이 방출을 믿을 만하게 만들었다는 뜻), 여전히 작은 α 가 최적이면 **독립**이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
