# -*- coding: utf-8 -*-
"""잔차 **정렬도**를 방출에 더해 유령을 막는다 (13.84.59 = 13.84.58 의 처방).

13.84.58 이 잰 것: 미니PC 잔차의 정렬도 `cos` 가 참ON 대 유령을 **AUC 0.807** 로 가르는데
(지금 쓰는 와트 w 는 0.687) 모델은 그 축을 **안 쓴다.** 기전은 분해로 나온다 —

    |resid|      참ON 0.181 · 유령 0.223   AUC 0.293  (뒤집으면 0.707: 유령에서 잔차가 **크다**)
    |미설명|      참ON 0.128 · 유령 0.211   AUC 0.231  (뒤집으면 0.769)
    cos          참ON 0.64  · 유령 0.28    AUC 0.807

**방출 머리는 일치도만 재고 불일치는 안 잰다.** 유령은 "형제 모형이 크게 틀려서 생긴 큰 잔차의
일부가 어쩌다 미니PC 쪽을 가리키는" 것이고, 사영은 그 '어쩌다' 를 못 본다.

처방: 미니PC 방출에 `lam * (cos(t) − c0)` 를 더한다. 신탁 없다 — cos 는 obs·P̂·지문만 쓴다.
13.84.53 의 3차원 보정과 **직교한 처방**이다 (그쪽은 실패 ③, 이쪽은 실패 ②).

    python -X utf8 src/run_diag_ghostgate.py [results/seq_h38_base.pt] [--calib]
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
from src.run_diag_ghost import OI, FS
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

MIN_OFF, MIN_OFF_FRAC, ITERS = 30, 0.10, 4
LAMS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
C0 = 0.45           #: 관문 중심 — 13.84.58 의 분위수에서 참ON 73% / 유령 14% 지점


def build(ck_path, dev):
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    apps = list(ck["appliances"])
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    return ck, apps, model, heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ck", nargs="?", default="results/seq_h38_base.pt")
    ap.add_argument("--c0", type=float, default=C0)
    ap.add_argument("--calib", action="store_true", help="13.84.53 의 3차원 보정도 같이 건다")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--basis", default="results/drift_basis.npz")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck, apps, model, heads = build(a.ck, dev)
    crit = build_loss(apps, "cpu", verbose=False)
    sg = crit.sig.numpy(); sigc = sg[..., 0] + 1j * sg[..., 1]
    nzc = crit.noise_sig.numpy()[:, 0] + 1j * crit.noise_sig.numpy()[:, 1]
    km = apps.index("minipc"); sm = sigc[km][OI]
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)
    V = np.asarray(np.load(a.basis, allow_pickle=True)["V"], float)[:a.k] if a.calib else None
    H = len(OI)

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
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                     torch.cat(GL)[None])
            pw = torch.cat(PW).cpu().numpy()
            r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
            Hh = np.asarray(r["harmonics_complex"])
            ti = np.clip((d["t"] * FS).astype(int), 0, len(Hh) - 1)
            rest = np.zeros_like(Hh[ti][:, OI])
            for j in range(len(apps)):
                if j != km:
                    rest += pw[:, j, None] * sigc[j][OI][None, :]
            S[stem] = dict(em=em, on=on, off=off, ini=ini, y=d["y"].astype(bool),
                           obs=Hh[ti][:, OI], rest=rest + nzc[OI][None, :], T=len(d["t"]))

    def cosw(s, corr=None):
        """잔차의 정렬도 (신탁 없음)."""
        resid = s["obs"] - s["rest"] - (corr[None, :] if corr is not None else 0.0)
        ip = (resid @ np.conj(sm)).real
        return ip / np.maximum(np.linalg.norm(resid, axis=1) * np.linalg.norm(sm), 1e-12)

    def calib(s):
        """13.84.53 의 3차원 EM. 막히면 None."""
        path = viterbi(s["em"], s["on"], s["off"], s["ini"])[0].cpu().numpy()
        noff = int((~path[:, km]).sum())
        if noff < MIN_OFF or noff < MIN_OFF_FRAC * s["T"]:
            return None, None
        cc = np.zeros(H, complex); emn = None
        for _ in range(ITERS):
            m = ~path[:, km]
            if m.sum() < MIN_OFF:
                break
            tgt = s["obs"][m] - s["rest"][m]
            b = np.concatenate([tgt.real, tgt.imag], 1).mean(0)
            c = V.T @ (V @ b); cc = c[:H] + 1j * c[H:]
            w = ((s["obs"] - s["rest"] - cc[None, :]) @ np.conj(sm)).real / (np.abs(sm) ** 2).sum()
            emn = ((w - 6.0) / 6.0).astype(np.float32)
            e = s["em"].clone(); e[0, :, km] = torch.from_numpy(emn).to(dev)
            path = viterbi(e, s["on"], s["off"], s["ini"])[0].cpu().numpy()
        return emn, cc

    CAL = {k: (calib(v) if a.calib else (None, None)) for k, v in S.items()}
    if a.calib:
        print("보정 관문: %s" % "  ".join("%s %s" % (k, "통과" if v[0] is not None else "막힘")
                                         for k, v in CAL.items()))
    print("\n미니PC 시간 정확도 — 행은 정렬도 세기 lam (c0=%.2f)%s"
          % (a.c0, " · 3차원 보정 함께" if a.calib else ""))
    names = [s for s in FILES if s in S]
    print("   %-7s %8s | %s" % ("lam", "평균", "".join("  %-8s" % s for s in names)))
    best = (None, -1)
    for lam in LAMS:
        row = []
        for stem in names:
            s = S[stem]
            emn, cc = CAL[stem]
            em = s["em"].clone()
            if emn is not None:
                em[0, :, km] = torch.from_numpy(emn).to(dev)
            c = cosw(s, cc)
            em[0, :, km] = em[0, :, km] + lam * torch.from_numpy((c - a.c0).astype(np.float32)).to(dev)
            p = viterbi(em, s["on"], s["off"], s["ini"])[0].cpu().numpy()
            row.append(float((p[:, km] == s["y"][:, km]).mean()))
        m = float(np.mean(row))
        if m > best[1]:
            best = (lam, m)
        print("   %-7.2f %8.4f | %s" % (lam, m, "".join("  %-8.3f" % v for v in row)))
    print("   -> 최적 lam = %.2f (%.4f)" % best)
    print("\n   견줌  손 안 댐 0.8046  ·  3차원 보정만 0.9170 (13.84.53)")
    print("   ⚠ 신탁 없음 — cos 는 obs·P̂·지문만 쓴다. 라벨은 채점에만.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
