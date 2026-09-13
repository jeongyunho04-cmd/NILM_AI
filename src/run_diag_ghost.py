# -*- coding: utf-8 -*-
"""실패 ② (유령) — 잔차의 **크기·전력·정렬도** 축을 잰다 (13.84.58).

13.84.53 의 자기보정은 실패 ③ 을 열었지만 유령(test_3·test_4)은 **원리적으로** 못 연다:
보정이 서려면 "미니PC 가 꺼진 구간" 이 필요한데 유령의 정의가 모델이 그 구간을 못 알아보는
것이다 (13.84.55). 그래서 다른 축이 필요하다. 아직 안 본 것이 있다 —

자기보정은 잔차를 미니PC 지문에 사영해 **와트**를 쓴다. 그런데 사영은 **정렬도를 버린다.**
진짜 미니PC 가 켜지면 잔차가 지문 **방향**을 가리켜야 하고, 유령은 "어쩌다 그 방향 성분이
있는 잔차" 이므로 정렬도가 낮아야 한다. 크기가 같아도 방향이 다르면 갈린다.

    resid(t) = obs(t) − 잡음 − Σ_{j≠미니PC} P̂_j·sig_j
    w(t)   = <resid, sig_미니PC> / |sig_미니PC|²        [W]   <- 지금 쓰는 것
    cos(t) = <resid, sig_미니PC> / (|resid|·|sig_미니PC|)     <- **안 쓰는 것**

재는 것: 참ON · 유령(참OFF인데 모델 ON) · 참OFF 세 무리에서 w 와 cos 의 분포와 AUC.
⚠ 라벨은 **채점에만** 쓴다. 판별기를 라벨로 맞추지 않는다.

    python -X utf8 src/run_diag_ghost.py [results/seq_h38_base.pt]
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
ORD = [1, 3, 5, 7, 9, 11, 13, 15]
OI = [o - 1 for o in ORD]


def auc(pos, neg):
    """순위 통계 AUC (sklearn 없이)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.concatenate([pos, neg])
    r = np.empty(len(a), float)
    r[np.argsort(a, kind="mergesort")] = np.arange(len(a))
    # 동점 평균순위
    s = np.sort(a); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            m = np.nonzero(a == s[i])[0]
            r[m] = r[m].mean()
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def q(x, *ps):
    return "  ".join("%7.2f" % np.quantile(x, p) for p in ps) if len(x) else "      —" * len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ck", nargs="?", default="results/seq_h38_base.pt")
    a = ap.parse_args()

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
    sm = sigc[km][OI]
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)

    print("미니PC 지문 |sig| %.1f mA/W · 잡음 |noise| %.1f mA"
          % (1000 * np.abs(sm).sum() / len(sm) * len(sm) / len(sm) * np.linalg.norm(sm) / np.linalg.norm(sm)
             * 1000 / 1000 * np.linalg.norm(sm), 1000 * np.linalg.norm(nzc[OI])))
    print("\n   %-8s %8s %8s %8s | %-24s | %-24s"
          % ("파일", "참ON", "유령", "참OFF", "w [W] 25/50/75 분위", "cos 25/50/75 분위"))

    POS, GH, NEG = {"w": [], "c": []}, {"w": [], "c": []}, {"w": [], "c": []}
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
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                     torch.cat(GL)[None])
            pw = torch.cat(PW).cpu().numpy()
            path = viterbi(em, on, off, ini)[0].cpu().numpy()
            y = d["y"].astype(bool)
            r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
            Hh = np.asarray(r["harmonics_complex"])
            ti = np.clip((d["t"] * FS).astype(int), 0, len(Hh) - 1)
            obs = Hh[ti][:, OI]

            rest = np.zeros_like(obs)
            for j in range(len(apps)):
                if j == km:
                    continue
                rest += pw[:, j, None] * sigc[j][OI][None, :]
            resid = obs - rest - nzc[OI][None, :]
            ip = (resid @ np.conj(sm)).real
            w = ip / (np.abs(sm) ** 2).sum()
            cs = ip / np.maximum(np.linalg.norm(resid, axis=1) * np.linalg.norm(sm), 1e-12)

            mon, mgh = y[:, km], (~y[:, km]) & path[:, km]
            mof = (~y[:, km]) & (~path[:, km])
            for M, S in ((mon, POS), (mgh, GH), (mof, NEG)):
                S["w"].append(w[M]); S["c"].append(cs[M])
            print("   %-8s %8d %8d %8d | %s | %s"
                  % (stem, mon.sum(), mgh.sum(), mof.sum(),
                     q(w[mgh], .25, .5, .75), q(cs[mgh], .25, .5, .75)))
            print("   %-8s %8s %8s %8s |   참ON %s |   참ON %s"
                  % ("", "", "", "", q(w[mon], .25, .5, .75), q(cs[mon], .25, .5, .75)))

    for S in (POS, GH, NEG):
        for k in S:
            S[k] = np.concatenate(S[k]) if S[k] else np.zeros(0)
    print("\n   전체 무리별 (참ON %d · 유령 %d · 참OFF %d 단계)"
          % (len(POS["w"]), len(GH["w"]), len(NEG["w"])))
    print("   %-8s %-28s %-28s" % ("", "w [W]  25 / 50 / 75", "cos  25 / 50 / 75"))
    for nm, S in (("참ON", POS), ("유령", GH), ("참OFF", NEG)):
        print("   %-8s %-28s %-28s" % (nm, q(S["w"], .25, .5, .75), q(S["c"], .25, .5, .75)))
    print("\n   AUC 참ON 대 유령   w %.3f   cos %.3f" % (auc(POS["w"], GH["w"]), auc(POS["c"], GH["c"])))
    print("   AUC 참ON 대 참OFF   w %.3f   cos %.3f" % (auc(POS["w"], NEG["w"]), auc(POS["c"], NEG["c"])))
    print("\n   읽는 법 — cos 의 AUC 가 w 보다 뚜렷이 높으면 **정렬도가 안 쓰인 축**이고")
    print("   방출이나 사슬에 넣을 값어치가 있다. 둘 다 0.5 근처면 이 축으로는 못 연다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
