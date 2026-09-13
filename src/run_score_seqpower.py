# -*- coding: utf-8 -*-
"""**전력 헤드**를 채점한다 — 게이트만 보면 헤드가 죽은 것을 못 본다 (13.84.28).

[[score-the-power-head-not-just-the-gate]] — AUC 최고인데 전력 헤드가 죽어 있던 전례가 있다.

⚠ **실측 파일에는 기기별 참값 와트가 없다**(13.84.18). 그래서 실측에서 잴 수 있는 것은 둘뿐이다.
  ① **합계 대 계측 총전력** — 기기별 예측 + 잡음/대기 가 `p_observed` 와 맞아야 한다
  ② **참 ON / 참 OFF 구간의 예측 와트** — OFF 에서 0 근처인가, ON 에서 그 기기 크기인가
기기별 오차 귀속은 합성에서만 된다 (`--synth` 로 같이 낸다. `gt_target_power_w` 가 있다).

    python -X utf8 src/run_score_seqpower.py results/hpc/seq_v37_ep12.pt [--baseline results/cnn_v37.pt]
"""
import argparse
import sys
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


def forward(model, heads, d, dev, apps):
    """창별 전력 · 게이트 · 사슬 경로를 한 파일에서."""
    with torch.no_grad():
        P, GL, Z, SB, NZ = [], [], [], [], []
        for i in range(0, len(d["t"]), 512):
            o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                      torch.from_numpy(d["wide"][i:i + 512]).to(dev))
            P.append(o["power"].float()); GL.append(o["on_logit"].float())
            Z.append(o["z"].float()); SB.append(o["standby"].float())
        p = torch.cat(P).cpu().numpy()
        gl = torch.cat(GL)
        sb = torch.cat(SB).cpu().numpy()
        path = None
        if heads is not None:
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev), gl[None])
            path = viterbi(em, on, off, ini)[0].cpu().numpy().astype(bool)
    return p, (gl > 0).cpu().numpy(), sb, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="results/hpc/seq_v37_ep12.pt")
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
    base = load_model(a.baseline, dev)[0]
    real = real_windows(apps, ck["meta"]["grid_s"], dev)

    from src.preprocessing import load_nilm_npz
    FS = 60

    print("① 합계 대 계측 총전력 — 단계마다 (예측합 − 관측) 의 중앙값")
    print("     '문지름' = 게이트/사슬로 끄지 않고 **전력 헤드를 통째로** 더한 것.")
    print("     둘을 갈라야 오차가 헤드 크기 탓인지 껐다 켰다 탓인지 안다.")
    print()
    print("  파일       계측중앙   사슬오차    %%   |  창별오차(v37)   %%   |  문지름(사슬몸통)  %%")
    tot = {"new": [], "old": [], "raw": []}
    for stem in FILES:
        d = real[stem]
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        pobs = np.asarray(r["power_features"])[:, 0]
        # 격자 시각의 관측 전력. real_windows 의 t 는 초 단위 타깃 시각이다.
        idx = np.clip((d["t"] * FS).astype(int), 0, len(pobs) - 1)
        po = pobs[idx]
        pn, wn, sbn, path = forward(model, heads, d, dev, apps)
        pb, wb, sbb, _ = forward(base, None, d, dev, apps)
        # 사슬이 끈 기기는 전력도 0 으로 본다 (사슬이 정하는 것은 상태다)
        sn = (pn * path).sum(1)
        sb_ = (pb * wb).sum(1)
        sr = pn.sum(1)                     # 게이팅 없이 헤드만
        m = max(np.median(po), 1.0)
        en, eo, er = np.median(sn - po), np.median(sb_ - po), np.median(sr - po)
        tot["new"].append(en); tot["old"].append(eo); tot["raw"].append(er)
        print("  %-9s %8.0f  %+8.0f %+5.0f%%  | %+9.0f %+6.0f%%  | %+10.0f %+6.0f%%"
              % (stem, np.median(po), en, 100 * en / m, eo, 100 * eo / m, er, 100 * er / m))
    print("  %-9s %8s  %+8.0f         | %+9.0f          | %+10.0f"
          % ("중앙", "", np.median(tot["new"]), np.median(tot["old"]), np.median(tot["raw"])))

    print("\n② 참 ON / 참 OFF 구간의 예측 와트 (전 파일 모음, 그 기기가 있는 파일만)")
    print("  기기                참OFF 중앙   참ON 중앙   참ON p10~p90    OFF 에서 새는 양")
    acc = {k: {"off": [], "on": []} for k in range(len(apps))}
    for stem in FILES:
        d = real[stem]
        pn, wn, sbn, path = forward(model, heads, d, dev, apps)
        y = d["y"].astype(bool)
        for k in np.nonzero(d["present"])[0]:
            acc[k]["off"].append(pn[~y[:, k], k])
            acc[k]["on"].append(pn[y[:, k], k])
    for k, nm in enumerate(apps):
        off = np.concatenate(acc[k]["off"]) if acc[k]["off"] else np.array([])
        on = np.concatenate(acc[k]["on"]) if acc[k]["on"] else np.array([])
        if not len(on) and not len(off):
            continue
        print("  %-18s %10.1f %11.1f  %6.1f~%-7.1f %10.1f W"
              % (nm, np.median(off) if len(off) else np.nan,
                 np.median(on) if len(on) else np.nan,
                 np.percentile(on, 10) if len(on) else np.nan,
                 np.percentile(on, 90) if len(on) else np.nan,
                 np.median(off) if len(off) else np.nan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
