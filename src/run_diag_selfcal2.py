# -*- coding: utf-8 -*-
"""채점 시점 자기보정을 **표류 부분공간 안으로 묶는다** (13.84.53 = 13.84.52 ⓑ).

13.84.46 의 EM 은 차수마다 복소 이득을 자유롭게 뒀다 — **자유도 16**. 그래서 1회는
+0.098/+0.135 로 듣고 2회부터 미니PC 를 빨아들여 붕괴했다 (test_2 0.792 -> 0.496).

13.84.52 가 잰 것: 형제 지문 표류는 **3차원이 94%** 이고 으뜸 모드는 미니PC 와 **89° 직교**다.
그러면 보정을 그 부분공간 안으로 제한하면 자유도가 16 -> k (1~3) 가 되어 축퇴가 막힌다.
천장은 13.84.35 ⑥ 의 신탁 0.747 -> 0.937 (+0.19).

모형:  obs ≈ noise + Σ_{j≠미니PC} P̂_j·sig_j + **Vᵀa**        (a ∈ R^k, 파일당 하나)
E단계: resid = obs − 위 → 미니PC 지문에 사영 → 와트 추정 → 방출로 갈아 끼우고 Viterbi
M단계: 미니PC 가 **OFF 라고 풀린** 단계에서만 a 를 최소제곱

⚠ 기저는 `results/drift_basis.npz` 이고 **형제 녹화로만** 만든다 (미니PC 를 넣으면 자기 신호를 지운다).
⚠ 13.84.31 ⑤ 의 축퇴 — OFF 구간이 모자란 파일은 아예 안 건다 (test_2 는 미니PC 가 92% 켜져 있다).

    python -X utf8 src/run_diag_selfcal2.py [results/seq_h38_base.pt] [--k 1]
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
from src.run_diag_chainprop import marginals
from src.model.lossbuild import build_loss
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

FS = 60
SIB = ["laptop_charger", "beam_projector"]
MIN_OFF, MIN_OFF_FRAC = 30, 0.10
ITERS = 4
#: 13.84.46 의 자유도 16 판 결과 — 견줌용
FREE16 = {"test_1": [0.834, 0.932, 0.891, 0.902, 0.891],
          "test_2": [0.657, 0.792, 0.499, 0.496, 0.496]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ck", nargs="?", default="results/seq_h38_base.pt")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--basis", default="results/drift_basis.npz")
    ap.add_argument("--anchor", default="hard", choices=("hard", "soft", "quant"),
                    help="M단계 앵커 — hard: 경로가 OFF 라 한 단계만 (13.84.53) · "
                         "soft: 사후확률 1-P(ON) 가중 (전 단계) · "
                         "quant: 미니PC 방출 하위 q%% 단계 (고정 크기)")
    ap.add_argument("--q", type=float, default=0.25, help="quant 앵커의 몫")
    a = ap.parse_args()

    B = np.load(a.basis, allow_pickle=True)
    V = np.asarray(B["V"], float)[:a.k]                 # (k, 2H)
    ORD = [int(x) for x in B["orders"]]
    H = len(ORD)
    oi = [o - 1 for o in ORD]
    print("표류 기저 %s · k=%d · 차수 %s · 블록 %d사이클"
          % (a.basis, a.k, ORD, int(B["block"])))
    print("   (기저는 형제 녹화 %s 로만 만들었다)" % " · ".join(str(x) for x in B["sources"]))

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
    nz = crit.noise_sig.numpy(); nzc = nz[:, 0] + 1j * nz[:, 1]
    km = apps.index("minipc")
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)

    print("\n   %-8s %7s %8s | %s" % ("파일", "OFF단계", "관문", "반복별 미니PC 시간 정확도"))
    first, last = [], []
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
            gl = torch.cat(GL)[None]
            em0, on, off, ini = heads(torch.cat(Z)[None],
                                      torch.from_numpy(d["dfeat"]).float()[None].to(dev), gl)
            pw = torch.cat(PW).cpu().numpy()
            y = d["y"].astype(bool)
            r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
            Hh = np.asarray(r["harmonics_complex"])
            ti = np.clip((d["t"] * FS).astype(int), 0, len(Hh) - 1)
            obs = Hh[ti][:, oi]                                   # (T,H) complex

            path = viterbi(em0, on, off, ini)[0].cpu().numpy()
            accs = [float((path[:, km] == y[:, km]).mean())]
            first.append(accs[0])
            noff = int((~path[:, km]).sum())
            if a.anchor == "hard":
                gate = noff >= MIN_OFF and noff >= MIN_OFF_FRAC * len(path)
            else:
                # soft/quant 는 딱딱한 OFF 집합이 필요 없다 — 실패 ②(유령) 파일을 열려는 것이다
                gate = len(path) >= MIN_OFF

            if gate:
                # 미니PC 를 뺀 나머지 기기 + 계측 잡음 (반복 중 안 변한다)
                rest = np.zeros_like(obs)
                for j in range(len(apps)):
                    if j == km:
                        continue
                    rest += pw[:, j, None] * sigc[j][oi][None, :]
                rest = rest + nzc[oi][None, :]
                sm = sigc[km][oi]
                coef = []
                emcur = em0
                for _ in range(ITERS):
                    # ── 앵커: 미니PC 가 **없을 법한** 단계에 가중 ──────────────
                    if a.anchor == "hard":
                        w = (~path[:, km]).astype(float)
                    elif a.anchor == "soft":
                        pon = marginals(emcur.double(), on.double(), off.double(),
                                        ini.double())[0, :, km].cpu().numpy()
                        w = 1.0 - pon
                    else:
                        e = emcur[0, :, km].cpu().numpy()
                        w = (e <= np.quantile(e, a.q)).astype(float)
                    ess = float(w.sum())
                    if ess < MIN_OFF:
                        break
                    # ── M: 표류 부분공간 안에서 a (k개 실수) 만 맞춘다 ────────
                    tgt = obs - rest                              # (T,H) complex
                    bb = np.concatenate([tgt.real, tgt.imag], 1)  # (T,2H)
                    b = (w[:, None] * bb).sum(0) / ess
                    aa = V @ b                                    # 정규직교라 사영이 곧 해다
                    coef.append(aa.copy())
                    corr = V.T @ aa                               # (2H,)
                    cc = corr[:H] + 1j * corr[H:]
                    # ── E: 보정 뒤 잔차를 미니PC 지문에 사영 -> 와트 ──────────
                    resid = obs - rest - cc[None, :]
                    w_hat = (resid @ np.conj(sm)).real / (np.abs(sm) ** 2).sum()
                    emn = torch.from_numpy(((w_hat - 6.0) / 6.0).astype(np.float32)).to(dev)
                    em = em0.clone(); em[0, :, km] = emn
                    emcur = em
                    path = viterbi(em, on, off, ini)[0].cpu().numpy()
                    accs.append(float((path[:, km] == y[:, km]).mean()))
            last.append(accs[-1])
            print("   %-8s %7d %8s | %s" % (stem, noff, "통과" if gate else "**막음**",
                                            "  ".join("%.3f" % v for v in accs)))
            if gate and coef:
                print("   %-8s %7s %8s |   보정 크기 |a| %s (mA)"
                      % ("", "", "", "  ".join("%.1f" % (1000 * np.linalg.norm(c)) for c in coef)))
            if stem in FREE16:
                print("   %-8s %7s %8s |   견줌 자유도16 (13.84.46): %s"
                      % ("", "", "", "  ".join("%.3f" % v for v in FREE16[stem])))

    print("\n   미니PC 평균  처음 %.4f -> 마지막 %.4f" % (np.mean(first), np.mean(last)))
    print("   견줌 13.84.46 (자유도 16): 0.8046 -> 0.7788 (붕괴)")
    print("   ⚠ 신탁 없음 — 라벨은 채점에만. 보정은 모델이 OFF 라고 푼 구간에서만 맞췄다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
