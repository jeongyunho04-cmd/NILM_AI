"""미니PC 유령은 왜 **오븐이 켤 때만** 나는가 — 재학습 없이 (12.154)

12.149.4 가 좁혀 놓은 것: `test_4` 에서 미니PC 유령 창 462개가 **하나도 빠짐없이**
오븐 통전 중이다 (오븐 OFF 인 유령 창 0개, 핫플 40.5%, 프로젝터·충전기 0%).
게이트가 진짜로 켜지므로 스켈치 사정권 밖이다.

지렛대부터 계산하면 어디를 봐야 하는지 나온다 (자료 불필요):

    차수   오븐@1100W   미니PC@10W    비
     h1     5098.8 mA     47.2 mA   108.1     <- 여기다
     h3       16.8 mA     45.6 mA     0.4
     h5~h15   18~63 mA   8~41 mA   0.9~4.6

**3차 이상에서는 오븐 지문이 7% 틀려도 미니PC 0.92W 어치다** (관측된 유령은
평균 3.41W / ON 창 7.02W). 3차 이상만으로는 설명이 안 된다. 그런데 `inv_h2`
가중은 h1 에 무게를 몰아 주고, h1 지렛대는 108배다.

그리고 12.151.1 이 그 h1 의 정체를 밝혔다 — `Re(sig[k,1,0]) = 1/V녹화` 라
지문의 h1 은 **기기가 아니라 그 녹화의 전압**이다 (기기 간 11.8% 가짜 판별자).

⚠ 12.140 은 오븐 **h3 비 0.93** 만 보고했다. **오븐 h1 정확도는 안 쟀다.**

이 스크립트가 재는 것 — 반증 조건을 먼저 적는다 (규칙 25/28)
--------------------------------------------------------------
(a) `test_4` 에서 미니PC 만 0 으로 두었을 때 `L_harm` 이 5% 미만으로 변하면
    **고조파가 시킨 것이 아니다.** 지문 갈래를 접고 게이트/입력을 본다.
(b) 그 ΔL_harm 의 절반 이상이 h1 에서 오지 않으면 **h1 지렛대 가설이 틀렸다.**
(c) 저항만 있는 대조 파일(오븐이 SMPS 없이 도는 곳)에서 미니PC 모양 잔차가
    0 에 가까우면 **오븐 지문은 결백하다** — 원인은 test_4 의 다른 것이다.
(d) 미니PC 게이트가 오븐 전력과 상관이 없으면 (|r| < 0.3) 인과가 아니다.

    python -X utf8 -m src.run_minipc_ghost_probe
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.postproc import apply_postproc, resistive_match, squelch
from src.run_gate_check import _signatures, forward_file, load_model

CTRL = ("test_9", "test_11", "test_12")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=["results/adapt_zi_s0.pt",
                                                  "results/adapt_zi_s1.pt",
                                                  "results/adapt_zi_s2.pt"])
    ap.add_argument("--stems", nargs="+", default=["test_4", "test_9", "test_11", "test_12"])
    ap.add_argument("--squelch", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/minipc_ghost.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    model, apps, _ = load_model(a.ckpt[0], dev)
    sig, sb_sig, nz_sig = (np.asarray(x, np.float64) for x in _signatures(list(apps)))
    from src.model.net import harmonic_scales
    from src.synthesis.segment_pool import SegmentPool
    hsc = np.asarray(harmonic_scales(
        SegmentPool(npz_dir="processed_data/npz", time_split="train"), apps), np.float64)
    H = sig.shape[1]
    w_h = 1.0 / (np.arange(1, H + 1, dtype=np.float64) ** 2)      # inv_h2
    w_h = w_h / w_h.max()
    MP, OV = apps.index("minipc"), apps.index("oven")

    acc: Dict[str, list] = {}
    dl_ord: Dict[str, list] = {}
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        for stem in a.stems:
            if stem not in ev:
                continue
            d = forward_file(model, stem, dev, stride=a.stride)
            for k in ("gate", "p_raw", "standby", "idle", "p_noise", "p_observed", "obs_harm"):
                d[k] = np.asarray(d[k], np.float64)
            P = squelch(d["gate"] * d["p_raw"], d["gate"], a.squelch)
            P, g = apply_postproc(P, d["gate"], apps)
            P, g = resistive_match(P, g, apps, d["p_observed"], d["v_rms"], d["standby"],
                                   d["p_noise"], obs_harm=d["obs_harm"], tol=0.02, snap=True)

            def terms(X):
                pred = (np.einsum("nk,khc->nhc", X, sig)
                        + np.einsum("nk,khc->nhc", d["idle"], sb_sig) + nz_sig[None])
                err = np.abs(pred - d["obs_harm"]) / hsc[None, :, None]
                return ((err * w_h[None, :, None]).mean(2), X.sum(1) + d["standby"].sum(1)
                        + d["p_noise"] - d["p_observed"])

            e1, r1 = terms(P)
            P0 = P.copy(); P0[:, MP] = 0.0
            e0, r0 = terms(P0)

            on = g[:, MP] > 0.5
            if stem == "test_4":
                tru, _ = build_on_off_truth(stem, apps, int(ev[stem]["cycles"]), events=ev)
                ovon = tru[np.clip(d["targets"], 0, len(tru) - 1), OV]
                m = on & ovon
            else:
                m = on
            grp = "대조" if stem in CTRL else "겨냥"
            acc.setdefault(stem, []).append((
                float(on.mean()), float(P[on, MP].mean()) if on.any() else 0.0,
                float(e1.sum(1)[m].mean()) if m.any() else np.nan,
                float(e0.sum(1)[m].mean()) if m.any() else np.nan,
                float(np.abs(r1)[m].mean()) if m.any() else np.nan,
                float(np.abs(r0)[m].mean()) if m.any() else np.nan,
                float(np.corrcoef(P[:, OV], g[:, MP])[0, 1]) if P[:, OV].std() > 1e-6 else np.nan,
                grp))
            if m.any():
                dl_ord.setdefault(stem, []).append((e0 - e1)[m].mean(0))   # 차수별 ΔL_harm

    print("\n── (a)(d) 미니PC 만 0 으로 둔 반사실 ─────────────────────────────")
    print(f"  {'파일':9s}{'층':5s}{'게이트ON':>9s}{'유령W':>8s}{'L_harm':>9s}"
          f"{'->미니0':>9s}{'Δ%':>8s}{'|잔차|':>8s}{'->미니0':>9s}{'r(오븐P,미니게이트)':>20s}")
    for stem, v in acc.items():
        A = np.array([x[:7] for x in v], dtype=float)
        m = A.mean(0)
        print(f"  {stem:9s}{v[0][7]:5s}{m[0]:9.3f}{m[1]:8.3f}{m[2]:9.4f}{m[3]:9.4f}"
              f"{100 * (m[3] / m[2] - 1):+8.1f}{m[4]:8.2f}{m[5]:9.2f}{m[6]:20.3f}")
    print("  판정: ΔL_harm 이 +5% 미만이면 고조파가 시킨 것이 아니다 (a)")
    print("        |r| < 0.3 이면 오븐 전력과 인과가 아니다 (d)")

    print("\n── (b) 그 ΔL_harm 은 어느 차수에서 오나 (inv_h2 가중 포함) ──────────")
    for stem, v in dl_ord.items():
        D = np.array(v).mean(0)
        tot = D.sum()
        if abs(tot) < 1e-12:
            continue
        top = np.argsort(-np.abs(D))[:4]
        print(f"  {stem:9s} 합 {tot:+.5f}  | " + "  ".join(
            f"h{h + 1}: {100 * D[h] / tot:+5.1f}%" for h in top))
    print("  판정: h1 몫이 50% 미만이면 h1 지렛대 가설이 틀렸다 (b)")

    print("\n── (c) 저항만 있는 대조 파일에서 미니PC 모양 잔차 ────────────────")
    for stem, v in acc.items():
        if stem not in CTRL:
            continue
        A = np.array([x[:7] for x in v], dtype=float).mean(0)
        print(f"  {stem:9s} 게이트ON {A[0]:.3f}  유령 {A[1]:.3f}W")
    print("  판정: 대조 파일에서도 미니PC 가 켜지면 **오븐 지문이 원인**이다.")
    print("        안 켜지면 오븐 지문은 결백하고 test_4 의 다른 것이 원인이다.")

    Path(a.out).write_text(json.dumps(
        {"_config": {"argv": sys.argv},
         "per_stem": {k: {"gate_on": float(np.mean([x[0] for x in v])),
                          "ghost_w": float(np.mean([x[1] for x in v])),
                          "l_harm": float(np.mean([x[2] for x in v])),
                          "l_harm_mp0": float(np.mean([x[3] for x in v])),
                          "corr_oven_gate": float(np.mean([x[6] for x in v]))}
                      for k, v in acc.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
