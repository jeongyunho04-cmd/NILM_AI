"""흡수 기제를 더 밀 수 있는가 — 세 갈래를 재학습 없이 잰다 (12.152)

`absorb_residual` 은 12.104 에서 만들었다. 그 뒤로 **전제가 셋 바뀌었다**:
계통 임피던스 보정(12.148), 게이트 정합 스켈치(12.149), 기기별 천장(12.149.4).
그래서 지금 안 맞는 자리가 있는지 본다.

    ① Norton 미사용   손실은 `pred += harm_offset` 인데 흡수의 `pred_h` 에는 없다.
                     방향 판정을 **보정 안 된 잔차**로 한다.
    ② 코사인 배분     `w_k = max(0, cos(잔차, sig_k))` 는 지문마다 **독립**으로 잰다.
                     "이 잔차를 만들려면 각 기기가 몇 W 인가"(NNLS)가 아니다.
                     SMPS 3종 지문이 11.9도 안에 몰려 있어(12.145) 셋이 비슷해진다.
    ③ 크기 미사용     총전력 잔차 W 는 **반드시** 배분되고 고조파는 방향만 정한다.
                     고조파가 "그만큼 아니다" 라고 해도 못 막는다.

반증 조건 — 먼저 적는다 (규칙 25/28)
------------------------------------
(a) cos(보정 전 잔차방향, 보정 후) > 0.95 이면 **①은 값이 없다.** 접는다.
(b) NNLS 배분이 코사인 배분과 프로젝터 몫에서 3W 안이면 **②는 값이 없다.**
    (참값 46.9W 를 아는 프로젝터로 판정한다 — 어느 쪽이 참값에 가까운가)
(c) 고조파가 요구하는 W(NNLS 합)와 총전력 잔차 W 의 비가 0.8~1.25 안이면
    **③은 값이 없다.** 둘이 이미 일치한다는 뜻이다.
(d) 대조 파일 잔차의 절반 이상이 **저항 조합으로 설명 가능**하면 저항 수신자를
    여는 것이 값진 갈래다. 아니면 접는다.

    python -X utf8 -m src.run_absorb_improve_probe
"""
from pathlib import Path
from typing import Dict, List, Optional
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
from src.model.postproc import (ABSORB_CAP_W, ABSORB_MIN_GATE, RESISTIVE_OHM,
                                SMPS_GROUP, apply_postproc, resistive_match, squelch)
from src.model.realdata import harmonic_offset
from src.run_gate_check import _signatures, forward_file, load_model

CTRL = ("test_9", "test_11", "test_12")


def nnls(A: np.ndarray, b: np.ndarray, ub: np.ndarray, iters: int = 200) -> np.ndarray:
    """상한이 있는 비음수 최소제곱 `min ‖A x − b‖, 0 <= x <= ub`.

    열이 3개뿐이라 사영 경사법으로 충분하다 (scipy 의존을 안 늘린다).
    """
    x = np.zeros(A.shape[1])
    L = float(np.linalg.norm(A, 2) ** 2) + 1e-9
    for _ in range(iters):
        x = np.clip(x - (A.T @ (A @ x - b)) / L, 0.0, ub)
    return x


def cos_weights(Rc: np.ndarray, S: np.ndarray) -> np.ndarray:
    """`absorb_residual` 이 쓰는 그 가중 (3차 이상 코사인, 음수는 0)."""
    rn = np.linalg.norm(Rc)
    if rn <= 1e-12:
        return np.zeros(S.shape[0])
    w = np.array([np.real(Rc @ np.conj(s)) / (np.linalg.norm(s) * rn + 1e-12) for s in S])
    return np.clip(w, 0.0, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/adapt_zi_s0.pt")
    ap.add_argument("--harm-offset", default="results/norton_coef.npz")
    ap.add_argument("--squelch", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/absorb_improve.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    model, apps, _ = load_model(a.ckpt, dev)
    sig, sb_sig, nz_sig = (np.asarray(x, np.float64) for x in _signatures(list(apps)))
    H = sig.shape[1]
    cols = [j for j, x in enumerate(apps) if x in SMPS_GROUP]
    ub = np.array([ABSORB_CAP_W.get(apps[j], np.inf) for j in cols])
    hh = list(range(2, H, 2))                      # 3차 이상 홀수차 (0-based)
    S = np.array([sig[j][hh, 0] + 1j * sig[j][hh, 1] for j in cols])
    PJ = cols.index(apps.index("beam_projector"))
    rows: List[dict] = []

    for stem in stems:
        d = forward_file(model, stem, dev, stride=a.stride)
        for k in ("gate", "p_raw", "standby", "idle", "p_noise", "p_observed", "obs_harm"):
            d[k] = np.asarray(d[k], np.float64)
        off = np.asarray(harmonic_offset([stem] * len(d["targets"]), d["targets"],
                                         a.harm_offset, n_harm=H), np.float64)
        # 운영 파이프라인을 흡수 **직전**까지 재현한다
        P = squelch(d["gate"] * d["p_raw"], d["gate"], a.squelch)
        P, g = apply_postproc(P, d["gate"], apps)
        P, g = resistive_match(P, g, apps, d["p_observed"], d["v_rms"], d["standby"],
                               d["p_noise"], obs_harm=d["obs_harm"], tol=0.02, snap=True)

        pred_h = (np.einsum("nk,khc->nhc", P, sig)
                  + np.einsum("nk,khc->nhc", d["idle"], sb_sig) + nz_sig[None])
        R0 = d["obs_harm"] - pred_h                     # 현행 흡수가 보는 잔차
        R1 = R0 - off                                    # Norton 보정을 넣은 잔차
        resid_p = d["p_observed"] - (P.sum(1) + d["standby"].sum(1) + d["p_noise"])
        live = g[:, cols] >= ABSORB_MIN_GATE

        for i in range(len(P)):
            r0 = R0[i][hh, 0] + 1j * R0[i][hh, 1]
            r1 = R1[i][hh, 0] + 1j * R1[i][hh, 1]
            n0, n1 = np.linalg.norm(r0), np.linalg.norm(r1)
            if n0 < 1e-9 or n1 < 1e-9 or not live[i].any():
                continue
            # ① 방향이 얼마나 바뀌나
            c01 = float(np.real(r0 @ np.conj(r1)) / (n0 * n1))
            # ② 코사인 배분 vs NNLS 배분 (같은 총량 resid_p 를 나눈다)
            w = cos_weights(r0, S) * live[i]
            share = w / w.sum() if w.sum() > 0 else np.zeros(len(cols))
            add_cos = share * resid_p[i]
            # NNLS 는 **보정된** 잔차를 와트로 푼다 (천장·게이트 제약 포함)
            A = np.concatenate([S.T.real, S.T.imag], axis=0)
            b = np.concatenate([r1.real, r1.imag])
            hi = np.where(live[i], np.minimum(ub, 1e6), 0.0)
            x = nnls(A, b, hi)
            rows.append({
                "stem": stem, "cos01": c01, "resid_p": float(resid_p[i]),
                "nnls_sum": float(x.sum()), "nnls_pj": float(x[PJ]),
                "cos_pj": float(add_cos[PJ]), "p_pj": float(P[i, apps.index("beam_projector")]),
                "on_pj": bool(g[i, apps.index("beam_projector")] > 0.5),
                "harm_n": float(n1),
            })

    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("stem", "on_pj")}
    st = np.array([r["stem"] for r in rows])
    on = np.array([r["on_pj"] for r in rows])
    tgt = ~np.isin(st, CTRL)

    print(f"\n  창 {len(rows)}개 (겨냥 {int(tgt.sum())} / 대조 {int((~tgt).sum())})\n")
    print("── [①] Norton 보정이 흡수의 잔차 **방향**을 얼마나 바꾸나 ────────────")
    for nm, m in (("겨냥 8", tgt), ("대조 3", ~tgt)):
        c = A["cos01"][m]
        print(f"  {nm}  cos(보정전, 보정후) 중앙 {np.median(c):.3f}  "
              f"p10 {np.percentile(c, 10):.3f}  p90 {np.percentile(c, 90):.3f}  "
              f"| >0.95 인 창 {float((c > 0.95).mean()):.3f}")
    print("  판정: 중앙 > 0.95 이고 대부분이 0.95 위면 ① 은 값이 없다")

    print("\n── [②] 코사인 배분 vs NNLS 배분 — 프로젝터 몫 ───────────────────")
    m = tgt & on
    print(f"  프로젝터 ON 인 겨냥 창 {int(m.sum())}개")
    print(f"  흡수 전 프로젝터 예측       중앙 {np.median(A['p_pj'][m]):7.2f}W")
    print(f"  + 코사인 배분 (현행)        중앙 {np.median((A['p_pj'] + A['cos_pj'])[m]):7.2f}W")
    print(f"  + NNLS 배분                중앙 {np.median((A['p_pj'] + A['nnls_pj'])[m]):7.2f}W")
    print(f"  참값 46.9W 대비 |오차| 중앙   코사인 "
          f"{np.median(np.abs((A['p_pj'] + A['cos_pj'])[m] - 46.9)):.2f}W  "
          f"NNLS {np.median(np.abs((A['p_pj'] + A['nnls_pj'])[m] - 46.9)):.2f}W  "
          f"(흡수 전 {np.median(np.abs(A['p_pj'][m] - 46.9)):.2f}W)")
    print("  판정: 두 배분의 |오차| 차가 3W 안이면 ② 는 값이 없다")

    print("\n── [③] 고조파가 요구하는 W 와 총전력 잔차 W 가 맞는가 ──────────────")
    for nm, mm in (("겨냥 8", tgt), ("대조 3", ~tgt)):
        rp, ns = A["resid_p"][mm], A["nnls_sum"][mm]
        ok = np.abs(rp) > 0.5
        ratio = ns[ok] / rp[ok]
        print(f"  {nm}  총전력잔차 중앙 {np.median(rp):+7.2f}W  "
              f"고조파가 요구 중앙 {np.median(ns):7.2f}W  "
              f"비 중앙 {np.median(ratio):6.2f}  | 0.8~1.25 안 {float(((ratio > 0.8) & (ratio < 1.25)).mean()):.3f}")
    print("  판정: 비가 대체로 0.8~1.25 안이면 ③ 은 값이 없다")

    print("\n── [④] 대조 파일 잔차를 **저항 조합**이 설명하는가 ──────────────────")
    # 남는 잔차를 등가저항 조합으로 설명할 수 있는가 (V^2/R 의 부분집합 합)
    for stem in CTRL:
        m2 = st == stem
        if not m2.any():
            continue
        rp = A["resid_p"][m2]
        best = []
        for v in rp:
            # 저항 4종의 V^2/R 중 하나로 설명되는 크기인가 (220V 기준)
            cands = [220.0 ** 2 / r for r in RESISTIVE_OHM.values()]
            best.append(min(abs(v - c) / max(c, 1.0) for c in cands))
        print(f"  {stem:8s} 잔차 중앙 {np.median(rp):+8.2f}W  "
              f"|잔차| 중앙 {np.median(np.abs(rp)):7.2f}W  "
              f"저항 한 대로 설명되는 창 {float(np.mean(np.array(best) < 0.1)):.3f}")
    print("  판정: '저항 한 대' 로 설명되는 창이 드물면 저항 수신자는 답이 아니다")

    Path(a.out).write_text(json.dumps(
        {"_config": {"argv": sys.argv}, "n": len(rows),
         "cos01_median_tgt": float(np.median(A["cos01"][tgt])),
         "cos01_median_ctrl": float(np.median(A["cos01"][~tgt])),
         "pj_err_cos": float(np.median(np.abs((A["p_pj"] + A["cos_pj"])[m] - 46.9))),
         "pj_err_nnls": float(np.median(np.abs((A["p_pj"] + A["nnls_pj"])[m] - 46.9))),
         "pj_err_pre": float(np.median(np.abs(A["p_pj"][m] - 46.9)))},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
