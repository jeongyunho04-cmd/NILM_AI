"""유령 에어컨은 **무엇을 흡수하고 있는가** — 그리고 막으면 어디로 가는가 (12.149)

12.148.3 이 남긴 결함 하나: `복소 Z·I` 의 유령8 6.78W 중 **5.59W 가 에어컨**이다.
에어컨은 11파일 어디서도 켜진 적이 없어 제약이 0 인 배출구다.

**그런데 채점 산출물을 먼저 읽으면 정체가 다르다.** `on_rate` 가 0.000 이고
하드 게이트에서는 0.021W 다. 즉 게이트는 한 번도 안 켜지고, 새는 것은
`P_hat = sigma(on) * p_raw` 에서 **sigma≈0.007 x p_raw≈800W** 다. BCE 로 보면
0.007 은 이미 "꺼짐"이라 아무 벌점이 없는데 와트로 보면 5.6W 다.

    유령8   소프트 6.78  ->  하드 0.83      (기준선은 3.54 -> 1.17)
    잔차    소프트 4.52  ->  하드 8.65      **4.1W 가 실려 있다**

그래서 묻는 것은 "에어컨을 어떻게 지우나" 가 아니라 **"그 4~6W 는 무엇이고
어디로 보내야 하나"** 다.

반증 조건 — 먼저 적는다 (규칙 25/28)
------------------------------------
(a) 에어컨만 0 으로 둘 때 `L_cons` 도 `L_harm` 도 5% 미만으로 변하면
    -> 흡수원이 아니라 그냥 잔향이다. **흡수 처방은 무의미하다. 멈춘다.**
(b) 에어컨을 뺀 뒤 와트당 비용 1위가 **겨냥 기기**(프로젝터/충전기/미니PC)면
    -> 열을 막으면 오차가 겨냥 기기로 간다 (규칙 29). **흡수원이 필요하다.**
(c) 남는 잔차 방향이 SMPS 판별 방향과 |cos| > 0.5 면
    -> 그 방향을 용서하면 판별이 죽는다 (규칙 31). **부분공간 불감대는 불가.**

    python -X utf8 -m src.run_ghost_absorb_probe --ckpt results/adapt_zi_s0.pt
"""
from pathlib import Path
from typing import Dict, Optional
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.model.realdata import SMPS_APPLIANCES, harmonic_offset
from src.run_gate_check import _signatures, forward_file, load_model

CTRL = ("test_9", "test_11", "test_12")


def harm_weights(h: int, mode: str) -> np.ndarray:
    hh = np.arange(1, h + 1, dtype=np.float64)
    if mode == "off":
        w = np.ones(h)
    elif mode == "inv_h":
        w = 1.0 / hh
    elif mode == "inv_h2":
        w = 1.0 / (hh * hh)
    else:
        raise ValueError(mode)
    return w / w.max()


def loss_terms(P: np.ndarray, d: dict, sig, sb_sig, nz_sig, hsc, hmask,
               off: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """창별 `L_cons` / `L_harm` 을 `losses.unlabeled` 와 같은 식으로 낸다."""
    recon = P.sum(1) + d["standby"].sum(1) + d["p_noise"]
    gap = d["p_observed"] - recon                     # + 면 과소예측
    pred = (np.einsum("nk,khc->nhc", P, sig)
            + np.einsum("nk,khc->nhc", d["idle"], sb_sig) + nz_sig[None])
    if off is not None:
        pred = pred + off
    err = np.abs(pred - d["obs_harm"]) / hsc[None, :, None]
    lh = (err * hmask[None, :, None]).mean(axis=(1, 2)) / hmask.mean()
    return {"gap": gap, "l_cons": np.abs(gap), "l_harm": lh,
            "resid_h": d["obs_harm"] - pred}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/adapt_zi_s0.pt", "results/adapt_zi_s1.pt",
                             "results/adapt_zi_s2.pt"])
    ap.add_argument("--harm-offset", default="results/norton_coef.npz")
    ap.add_argument("--harm-weight", default="inv_h2")
    ap.add_argument("--w-cons", type=float, default=0.1)
    ap.add_argument("--w-harm", type=float, default=4.0)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--out", default="results/ghost_absorb.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in ev if not s.startswith("_") and not is_sealed(s)]
    print(f"  장치 {dev} | 파일 {len(stems)} | 체크포인트 {len(a.ckpt)}")

    model, apps, _ = load_model(a.ckpt[0], dev)
    sig, sb_sig, nz_sig = _signatures(apps)
    sig = np.asarray(sig, np.float64)
    sb_sig = np.asarray(sb_sig, np.float64)
    nz_sig = np.asarray(nz_sig, np.float64)
    from src.model.net import harmonic_scales
    from src.synthesis.segment_pool import SegmentPool
    hsc = np.asarray(harmonic_scales(
        SegmentPool(npz_dir="processed_data/npz", time_split="train"), apps), np.float64)
    H = sig.shape[1]
    hmask = harm_weights(H, a.harm_weight)
    K = len(apps)
    AC = apps.index("air_conditioner")

    acc: Dict[tuple, list] = {}
    cost_rows, cost_rows_noac = [], []
    resid_dirs_noac = []
    per_app_leak: Dict[str, list] = {}
    stand_w, obs_w, gapall = [], [], []

    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        for stem in stems:
            d = forward_file(model, stem, dev, stride=a.stride)
            for k in ("gate", "p_raw", "standby", "idle", "p_noise",
                      "p_observed", "obs_harm"):
                d[k] = np.asarray(d[k], np.float64)
            off = None
            if a.harm_offset:
                off = np.asarray(harmonic_offset(
                    [stem] * len(d["targets"]), d["targets"], a.harm_offset,
                    n_harm=H), np.float64)
            P = d["gate"] * d["p_raw"]
            base = loss_terms(P, d, sig, sb_sig, nz_sig, hsc, hmask, off)
            P0 = P.copy()
            P0[:, AC] = 0.0
            noac = loss_terms(P0, d, sig, sb_sig, nz_sig, hsc, hmask, off)

            present = set(ev[stem]["appliances_present"])
            grp = "CTRL" if stem in CTRL else "TGT"
            for tag, t in (("base", base), ("noac", noac)):
                acc.setdefault((grp, tag), []).append(
                    (t["l_cons"].mean(), t["l_harm"].mean(),
                     (a.w_cons * t["l_cons"] + a.w_harm * t["l_harm"]).mean()))
            acc.setdefault((grp, "gap"), []).append(
                (base["gap"].mean(), np.median(base["gap"]), noac["gap"].mean()))

            for j, app in enumerate(apps):
                if app in present:
                    continue
                per_app_leak.setdefault(app, []).append(
                    (P[:, j].mean(), d["gate"][:, j].mean(), d["p_raw"][:, j].mean(),
                     float((d["gate"][:, j] > 0.5).mean())))
            stand_w.append(d["standby"].sum(1).mean())
            obs_w.append(d["p_observed"].mean())
            gapall.append(base["gap"])

            # ── 와트당 비용 (수치 미분, +1W) ────────────────────────────────
            for tag, bref, rows, Pb in (("base", base, cost_rows, P),
                                        ("noac", noac, cost_rows_noac, P0)):
                row = np.zeros(K)
                b0 = a.w_cons * bref["l_cons"] + a.w_harm * bref["l_harm"]
                for j in range(K):
                    Pp = Pb.copy()
                    Pp[:, j] += 1.0
                    t = loss_terms(Pp, d, sig, sb_sig, nz_sig, hsc, hmask, off)
                    row[j] = ((a.w_cons * t["l_cons"] + a.w_harm * t["l_harm"]) - b0).mean()
                rows.append((grp, row))

            R = noac["resid_h"]
            Rc = (R[:, :, 0] + 1j * R[:, :, 1]) / hsc[None]
            n = np.linalg.norm(Rc, axis=1, keepdims=True)
            resid_dirs_noac.append((Rc / np.clip(n, 1e-9, None))[np.ravel(n) > 1e-9])

    # ══ 보고 ═══════════════════════════════════════════════════════════════
    print("\n── [A] 누설의 정체 — 없는 기기에 붙은 전력 ───────────────────────")
    print(f"  {'기기':18s} {'누설W':>8s} {'sigma평균':>10s} {'p_raw평균':>10s} {'게이트>0.5':>10s}")
    for app, vs in sorted(per_app_leak.items(), key=lambda kv: -np.mean([x[0] for x in kv[1]])):
        m = np.mean([x[0] for x in vs])
        if m < 0.02:
            continue
        print(f"  {app:18s} {m:8.3f} {np.mean([x[1] for x in vs]):10.4f} "
              f"{np.mean([x[2] for x in vs]):10.1f} {np.mean([x[3] for x in vs]):10.4f}")

    print("\n── [B] 에어컨만 0 으로 둔 반사실 ────────────────────────────────")
    print(f"  {'층':5s} {'판':6s} {'L_cons W':>10s} {'L_harm':>9s} {'가중합':>9s}")
    for grp in ("TGT", "CTRL"):
        for tag in ("base", "noac"):
            v = np.array(acc[(grp, tag)])
            print(f"  {grp:5s} {tag:6s} {v[:, 0].mean():10.3f} {v[:, 1].mean():9.4f} "
                  f"{v[:, 2].mean():9.4f}")
        b, n2 = np.array(acc[(grp, "base")]), np.array(acc[(grp, "noac")])
        print(f"  {grp:5s} {'delta':6s} {100 * (n2[:, 0].mean() / b[:, 0].mean() - 1):+9.1f}% "
              f"{100 * (n2[:, 1].mean() / b[:, 1].mean() - 1):+8.1f}% "
              f"{100 * (n2[:, 2].mean() / b[:, 2].mean() - 1):+8.1f}%")

    print("\n── [C] 부호 있는 부족분 gap = P관측 − 재구성 ─────────────────────")
    g = np.concatenate(gapall)
    print(f"  전체 창 {len(g)}   평균 {g.mean():+.3f}W  중앙 {np.median(g):+.3f}W  "
          f"p10 {np.percentile(g, 10):+.2f}  p90 {np.percentile(g, 90):+.2f}")
    print(f"  gap>0 (과소예측) 창 비율 {float((g > 0).mean()):.3f}")
    for grp in ("TGT", "CTRL"):
        v = np.array(acc[(grp, "gap")])
        print(f"  {grp:5s} gap평균 {v[:, 0].mean():+7.3f}W  중앙 {v[:, 1].mean():+7.3f}W  "
              f"-> 에어컨 0 이면 {v[:, 2].mean():+7.3f}W")
    print(f"  규칙 40 — 기존 흡수원 용량: 대기항 {np.mean(stand_w):.2f}W "
          f"({100 * np.mean(stand_w) / np.mean(obs_w):.2f}% of 관측 {np.mean(obs_w):.1f}W), "
          f"p_noise 상수 1.40W")

    print("\n── [D] 와트당 비용 (+1W 의 Δ손실. 낮을수록 싼 배출구) ──────────────")
    print(f"  {'기기':18s} {'현행':>10s} {'에어컨0':>10s}")
    cb = np.array([r for gp, r in cost_rows if gp == "TGT"]).mean(0)
    cn = np.array([r for gp, r in cost_rows_noac if gp == "TGT"]).mean(0)
    order = np.argsort(cb)
    for j in order:
        mark = "  <- SMPS" if apps[j] in SMPS_APPLIANCES else ""
        star = "  *" if j == AC else ""
        print(f"  {apps[j]:18s} {cb[j]:10.5f} {cn[j]:10.5f}{mark}{star}")
    print(f"  현행 1위 = {apps[order[0]]} | 에어컨 뺀 뒤 1위 = {apps[int(np.argsort(cn)[0])]}")

    print("\n── [E] 남는 잔차 방향 vs 판별 방향 (규칙 31) ──────────────────────")
    D = np.concatenate(resid_dirs_noac)
    mu = D.mean(0)
    res_len = float(np.linalg.norm(mu))
    mu = mu / (res_len + 1e-12)
    print(f"  창 {len(D)}  평균방향 결집도 |mean| = {res_len:.3f} (1=완전 정렬, 0=무작위)")
    sc = (sig[:, :, 0] + 1j * sig[:, :, 1]) / hsc[None]
    cs = {}
    for j, app in enumerate(apps):
        v = sc[j] / (np.linalg.norm(sc[j]) + 1e-12)
        cs[app] = float(abs(np.vdot(v, mu)))
    print(f"  {'기기':18s} {'|cos(잔차, 지문)|':>20s}")
    for app, c in sorted(cs.items(), key=lambda kv: -kv[1]):
        print(f"  {app:18s} {c:20.3f}")
    print("  --- SMPS 판별 대비 (차 방향) ---")
    for x, y in (("beam_projector", "laptop_charger"), ("beam_projector", "minipc"),
                 ("laptop_charger", "minipc")):
        v = (sc[apps.index(x)] / np.linalg.norm(sc[apps.index(x)])
             - sc[apps.index(y)] / np.linalg.norm(sc[apps.index(y)]))
        v = v / (np.linalg.norm(v) + 1e-12)
        print(f"  {x[:12]:12s} - {y[:12]:12s} {abs(np.vdot(v, mu)):20.3f}")
    M = np.concatenate([D.real, D.imag], axis=1)
    s = np.linalg.svd(M, compute_uv=False)
    print(f"  잔차 주성분 설명력: PC1 {s[0] ** 2 / (s ** 2).sum():.3f}  "
          f"PC1-2 {(s[:2] ** 2).sum() / (s ** 2).sum():.3f}  "
          f"PC1-3 {(s[:3] ** 2).sum() / (s ** 2).sum():.3f}")

    Path(a.out).write_text(json.dumps({
        "_config": {"argv": sys.argv},
        "leak": {k: float(np.mean([x[0] for x in v])) for k, v in per_app_leak.items()},
        "cost_per_w": {apps[j]: float(cb[j]) for j in range(K)},
        "cost_per_w_noac": {apps[j]: float(cn[j]) for j in range(K)},
        "cos_resid_sig": cs,
        "resid_concentration": res_len,
        "gap_mean_w": float(g.mean()), "gap_median_w": float(np.median(g)),
        "standby_w": float(np.mean(stand_w)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
