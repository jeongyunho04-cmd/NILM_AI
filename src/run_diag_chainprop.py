# -*- coding: utf-8 -*-
"""사슬이 **한번 틀리면 끌고 가는가** — 오류의 길이 구조를 재고, 복호를 바꿔 본다 (13.84.41).

사용자 지적: "한번 기기의 꺼짐이나 켜짐을 놓치면 이게 계속 이어진다."

구조상 그럴 수밖에 없다. 방출 `em` 은 정상상태 판별인데 SMPS 배분은 정상상태에서 정보가 없다
(13.84.31: 요구 5.5% 대 재현성 바닥 9~30%). 그러면 Viterbi 가 상태를 바꿀 유일한 근거는
전이 점수가 `switch_bias` 를 넘는 것뿐이고, 못 넘으면 **다음 전이까지 틀린 채로 간다**.
13.84.25 가 이미 그 모양을 봤다 — test_2 의 참 켜짐이 파일 상위 0.0% 인데 절대값 −0.75 가
전환 벌점 −3.97 을 못 넘었다.

재는 것 셋:
  ① 정확도    창별(독립) · Viterbi · **주변확률**(forward-backward) 셋을 같은 체크포인트로
  ② 길이 구조 오류를 이어진 덩어리로 묶어 **덩어리 수와 길이 분포**. 전파면 덩어리가 길고 적다
  ③ 그림      파일별 타임라인 (참값 · 창별 · Viterbi · 주변확률 · 전이점수)

주변확률 복호는 **재학습이 필요 없다.** Viterbi 는 경로 하나에 전부를 걸지만 주변확률은
시점마다 따로 정하므로, 전파가 원인이라면 여기서 바로 갈린다.

    python -X utf8 src/run_diag_chainprop.py results/seq_h38_base.pt [--plot]
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
from src.model.transition import N_FEAT
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows


def marginals(em, sw_on, sw_off, init=None):
    """2상태 사슬의 시점별 사후확률 P(y_t=1) — forward-backward. (B,T,K) -> (B,T,K).

    점수 규약은 `chain.crf_nll` 과 **같아야 한다**: `em[t]` 는 시점 t 에 상태 1 일 때 더한다.
    """
    B, T, K = em.shape
    dt, dev = em.dtype, em.device
    a0 = torch.zeros(B, T, K, dtype=dt, device=dev)
    a1 = torch.zeros(B, T, K, dtype=dt, device=dev)
    a1[:, 0] = em[:, 0] + (init if init is not None else 0.0)
    for t in range(1, T):
        a0[:, t] = torch.logaddexp(a0[:, t - 1], a1[:, t - 1] + sw_off[:, t])
        a1[:, t] = torch.logaddexp(a1[:, t - 1], a0[:, t - 1] + sw_on[:, t]) + em[:, t]
    b0 = torch.zeros(B, T, K, dtype=dt, device=dev)
    b1 = torch.zeros(B, T, K, dtype=dt, device=dev)
    for t in range(T - 2, -1, -1):
        b0[:, t] = torch.logaddexp(b0[:, t + 1], b1[:, t + 1] + sw_on[:, t + 1] + em[:, t + 1])
        b1[:, t] = torch.logaddexp(b0[:, t + 1] + sw_off[:, t + 1], b1[:, t + 1] + em[:, t + 1])
    # ⚠ 자 검정: 어느 시점에서 묶어도 logZ 가 같아야 한다 ([[check-the-ruler-against-a-known-value]]).
    lz = torch.logaddexp(a0 + b0, a1 + b1)
    err = float((lz - lz[:, -1:]).abs().max())
    assert err < 1e-2, "forward-backward 가 안 맞는다 (logZ 편차 %.3g)" % err
    return torch.sigmoid((a1 + b1) - (a0 + b0))


def runs(bad):
    """틀린 시점 bool (T,) -> 이어진 덩어리 길이 목록."""
    if not bad.any():
        return np.zeros(0, int)
    d = np.diff(np.r_[0, bad.astype(np.int8), 0])
    return np.nonzero(d == -1)[0] - np.nonzero(d == 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ck", nargs="?", default="results/seq_h38_base.pt")
    ap.add_argument("--plot", action="store_true", help="results/plots/chainprop_*.png 로 그린다")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ck, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)
    dt_s = float(ck["meta"]["grid_s"])

    acc = {"창별": [], "Viterbi": [], "주변확률": []}
    rl = {"창별": [], "Viterbi": [], "주변확률": []}
    per_file = {}
    print("체크포인트 %s · 격자 %.1f초" % (a.ck, dt_s))
    with torch.no_grad():
        for stem, d in cache.items():
            Z, GL = [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
            z = torch.cat(Z)[None]
            gl = torch.cat(GL)[None]
            em, on, off, ini = heads(z, torch.from_numpy(d["dfeat"]).float()[None].to(dev), gl)
            vit = viterbi(em, on, off, ini)[0].cpu().numpy()
            mar = (marginals(em.double(), on.double(), off.double(),
                             ini.double())[0] > 0.5).cpu().numpy()
            win = (gl[0] > 0).cpu().numpy()
            y = d["y"].astype(bool)
            pk = {}
            for k in np.nonzero(d["present"])[0]:
                for name, p in (("창별", win), ("Viterbi", vit), ("주변확률", mar)):
                    bad = p[:, k] != y[:, k]
                    acc[name].append(1.0 - float(bad.mean()))
                    rl[name].append(runs(bad))
                    pk.setdefault(name, {})[apps[k]] = 1.0 - float(bad.mean())
            per_file[stem] = (pk, y, win, vit, mar, em[0].cpu().numpy(),
                              on[0].cpu().numpy(), off[0].cpu().numpy(), d["t"])

    print("\n① 정확도 — 그 파일에 **있는 기기만** 센 평균 (run_score_seq 와 같은 규약)")
    for name in ("창별", "Viterbi", "주변확률"):
        print("   %-8s %.4f" % (name, float(np.mean(acc[name]))))

    print("\n② 오류의 길이 구조 — 전파면 덩어리가 **길고 적다**")
    print("   %-8s %8s %9s %9s %9s %9s" % ("", "덩어리", "중앙(초)", "p90(초)", "최장(초)", ">60초 몫"))
    for name in ("창별", "Viterbi", "주변확률"):
        allr = np.concatenate([r for r in rl[name] if len(r)]) if any(len(r) for r in rl[name]) else np.zeros(0)
        if not len(allr):
            print("   %-8s   (오류 없음)" % name); continue
        s = allr * dt_s
        print("   %-8s %8d %9.0f %9.0f %9.0f %8.0f%%"
              % (name, len(s), np.median(s), np.percentile(s, 90), s.max(),
                 100 * s[s > 60].sum() / s.sum()))

    print("\n③ 미니PC 시간 정확도 (파일별)")
    km = apps.index("minipc")
    print("   %-8s %9s %9s %9s" % ("파일", "창별", "Viterbi", "주변확률"))
    for stem in FILES:
        if stem not in per_file:
            continue
        pk = per_file[stem][0]
        if "minipc" not in pk.get("Viterbi", {}):
            continue
        print("   %-8s %9.3f %9.3f %9.3f"
              % (stem, pk["창별"]["minipc"], pk["Viterbi"]["minipc"], pk["주변확률"]["minipc"]))

    # ── ④ 되돌릴 힘 — 방출이 전환 벌점을 언제 이길 수 있나 ─────────────────
    # 사슬이 틀린 상태를 스스로 빠져나오려면, 그 구간에서 **옳은 상태의 방출 이득**이
    # **전환 두 번(들어가고 나오는)의 값**을 넘어야 한다. 손익분기 길이를 잰다.
    print("\n④ 되돌릴 힘 — 방출이 전환 벌점을 이기는 데 필요한 길이")
    sb = heads.switch_bias.detach().cpu().numpy()
    print("   %-18s %10s %12s %12s %12s"
          % ("기기", "전환벌점", "|방출|/단계", "손익분기(초)", "실제 최장(초)"))
    for stem, (pk, y, win, vit, mar, emv, onv, offv, t) in per_file.items():
        pass
    allem = {k: [] for k in range(len(apps))}
    allrun = {k: [] for k in range(len(apps))}
    for stem, (pk, y, win, vit, mar, emv, onv, offv, t) in per_file.items():
        for k in range(len(apps)):
            if apps[k] not in pk.get("Viterbi", {}):
                continue
            allem[k].append(np.abs(emv[:, k]))
            allrun[k].append(runs(vit[:, k] != y[:, k]))
    for k in range(len(apps)):
        if not allem[k]:
            continue
        me = float(np.concatenate(allem[k]).mean())
        rr = np.concatenate([r for r in allrun[k] if len(r)]) if any(len(r) for r in allrun[k]) else np.zeros(0)
        be = (2.0 * abs(float(sb[k])) / me * dt_s) if me > 1e-9 else np.inf
        print("   %-18s %10.2f %12.4f %12.0f %12.0f"
              % (apps[k], sb[k], me, be, (rr.max() * dt_s) if len(rr) else 0))
    print("   손익분기 = 전환 2회 값 / 단계당 방출. 이 길이보다 짧은 오류는 **원리적으로** 못 고친다")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        os.makedirs("results/plots", exist_ok=True)
        for stem, (pk, y, win, vit, mar, emv, onv, offv, t) in per_file.items():
            if "minipc" not in pk.get("Viterbi", {}):
                continue          # 그 파일에 미니PC 가 없다 (test_5)
            fig, ax = plt.subplots(2, 1, figsize=(13, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.3]})
            tm = t / 60.0
            for i, (lbl, v) in enumerate((("truth", y[:, km]), ("window", win[:, km]),
                                          ("viterbi", vit[:, km]), ("marginal", mar[:, km]))):
                ax[0].fill_between(tm, 3 - i, 3 - i + 0.8, where=v.astype(bool),
                                   step="mid", lw=0)
                ax[0].text(tm[0], 3 - i + 0.25, " " + lbl, fontsize=8, va="center")
            ax[0].set_yticks([]); ax[0].set_ylim(-0.2, 4.0)
            ax[0].set_title("%s  minipc  —  win %.3f / vit %.3f / mar %.3f"
                            % (stem, pk["창별"]["minipc"], pk["Viterbi"]["minipc"],
                               pk["주변확률"]["minipc"]), fontsize=10)
            ax[1].plot(tm, emv[:, km], lw=0.8, label="emission")
            ax[1].plot(tm, onv[:, km], lw=0.8, label="switch-on")
            ax[1].plot(tm, offv[:, km], lw=0.8, label="switch-off")
            ax[1].axhline(0, color="k", lw=0.5)
            ax[1].legend(fontsize=7, ncol=3); ax[1].set_xlabel("minute")
            fig.tight_layout()
            fig.savefig("results/plots/chainprop_%s.png" % stem, dpi=110)
            plt.close(fig)
        print("\n그림: results/plots/chainprop_test_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
