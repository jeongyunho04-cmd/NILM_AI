"""흡수가 6W 를 **맞게** 나누려면 무엇을 더 봐야 하는가 — `Q` 다 (12.153)

12.152 이 잰 것: 총전력 잔차 중앙 **+5.96W** 인데 고조파가 지지하는 것은 1.18W 다.
그래서 고조파로 크기를 제한하면 흡수가 사실상 꺼진다 (잔차 6.62W = 흡수 끈 것과
같음). **6W 는 실재하고 배분돼야 하는데, 고조파 하나로는 못 정한다.**

미지수 셋(프로젝터·충전기·미니PC)에 방정식이 하나(`Σx = 6W`)뿐이다. 12.133 이
**두 번째 판별자**를 이미 쟀는데 후처리가 안 쓰고 있다:

    Q/P 중앙   프로젝터 −1.464   충전기 −1.606   미니PC −1.976
    폭/|중앙|        0.114        0.156        0.289   <- 전력 자체(0.077/0.722/1.162)
                                                          보다 충전기·미니PC 는 훨씬 안정
    판별력 d′    프로젝터vs충전기 2.31 / vs미니PC 4.64 / 충전기vs미니PC 3.11
                (고조파는 0.91 / 1.85 / 1.41 — Q 가 2.2~2.5배 낫다)

그러면 방정식이 둘이 된다:

    Σ x_k            = 총전력 잔차          (지금 쓰는 것)
    Σ (Q/P)_k · x_k  = 무효 잔차            (**안 쓰는 것**)

셋 중 둘이 정해지고 나머지 한 자유도를 고조파가 정한다.

반증 조건 — 먼저 적는다 (규칙 25/28)
------------------------------------
(a) |무효 잔차| 가 |Q/P_SMPS x 총전력 잔차| 의 1/3 미만이거나 3배 초과면
    **Q 는 이 6W 에 대한 정보가 없다.** 접는다.
(b) Q 를 넣어 푼 배분의 프로젝터가 참값 46.9W 에서 현행보다 멀면 접는다.
(c) 무효 잔차의 부호가 창마다 뒤집히면(부호 일치 < 0.7) 상쇄된 가짜다 (규칙 42).

    python -X utf8 -m src.run_absorb_q_probe
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

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.model.postproc import (ABSORB_CAP_W, ABSORB_MIN_GATE, SMPS_GROUP,
                                apply_postproc, resistive_match, squelch)
from src.run_gate_check import _signatures, forward_file, load_model

CTRL = ("test_9", "test_11", "test_12")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/adapt_zi_s0.pt")
    ap.add_argument("--squelch", type=float, default=0.1)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/absorb_q.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    model, apps, _ = load_model(a.ckpt, dev)
    sig, sb_sig, nz_sig = (np.asarray(x, np.float64) for x in _signatures(list(apps)))

    from src.model.net import noise_reactive, reactive_signatures
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    qp, usable = reactive_signatures(pool, apps)
    nq = float(noise_reactive(pool))
    qp = np.asarray(qp, np.float64)
    print(f"  Q/P  " + "  ".join(f"{x}={qp[j]:+.3f}{'' if usable[j] else '(못씀)'}"
                                 for j, x in enumerate(apps) if x in SMPS_GROUP))
    print(f"  계측계 무효 {nq:+.3f} VAR\n")

    cols = [j for j, x in enumerate(apps) if x in SMPS_GROUP]
    PJ = apps.index("beam_projector")
    hh = list(range(2, sig.shape[1], 2))
    S = np.array([sig[j][hh, 0] + 1j * sig[j][hh, 1] for j in cols])
    lim = np.array([ABSORB_CAP_W.get(apps[j], np.inf) for j in cols])
    rows: List[dict] = []

    for stem in stems:
        d = forward_file(model, stem, dev, stride=a.stride)
        for k in ("gate", "p_raw", "standby", "idle", "p_noise", "p_observed",
                  "obs_harm", "q_observed"):
            d[k] = np.asarray(d[k], np.float64)
        P = squelch(d["gate"] * d["p_raw"], d["gate"], a.squelch)
        P, g = apply_postproc(P, d["gate"], apps)
        P, g = resistive_match(P, g, apps, d["p_observed"], d["v_rms"], d["standby"],
                               d["p_noise"], obs_harm=d["obs_harm"], tol=0.02, snap=True)

        resid_p = d["p_observed"] - (P.sum(1) + d["standby"].sum(1) + d["p_noise"])
        q_pred = (P * qp[None]).sum(1) + (d["standby"] * qp[None]).sum(1) + nq
        resid_q = d["q_observed"] - q_pred
        pred_h = (np.einsum("nk,khc->nhc", P, sig)
                  + np.einsum("nk,khc->nhc", d["idle"], sb_sig) + nz_sig[None])
        R = (d["obs_harm"] - pred_h)[:, hh, :]
        Rc = R[:, :, 0] + 1j * R[:, :, 1]
        live = g[:, cols] >= ABSORB_MIN_GATE

        for i in range(len(P)):
            if not live[i].any():
                continue
            room = np.clip(lim - P[i, cols], 0.0, None) * live[i]
            # ── Q 를 넣어 푼다 ────────────────────────────────────────────
            #   Σx = resid_p (하드),  Σ qp·x = resid_q (연성),  고조파는 나머지
            # 정규화: 고조파는 mA, Q 는 VAR — 각자의 잔차 크기로 나눈다
            Ah = np.concatenate([S.T.real, S.T.imag], axis=0)
            bh = np.concatenate([Rc[i].real, Rc[i].imag])
            sh = max(np.linalg.norm(bh), 1e-6)
            aq = qp[cols][None, :]
            bq = np.array([resid_q[i]])
            sq = max(abs(resid_q[i]), 1e-6)
            A = np.concatenate([Ah / sh, aq / sq * 3.0], axis=0)
            b = np.concatenate([bh / sh, bq / sq * 3.0])
            # 사영 경사 + 합 제약 (단순: 매 스텝 합을 resid_p 로 되맞춘다)
            x = np.full(len(cols), max(resid_p[i], 0.0) / max(live[i].sum(), 1)) * live[i]
            L = float(np.linalg.norm(A, 2) ** 2) + 1e-9
            for _ in range(150):
                x = x - (A.T @ (A @ x - b)) / L
                x = np.clip(x, 0.0, room)
                s_ = x.sum()
                if s_ > 1e-9 and resid_p[i] > 0:
                    x = x * (resid_p[i] / s_)
                    x = np.clip(x, 0.0, room)
            rows.append({
                "stem": stem, "resid_p": float(resid_p[i]), "resid_q": float(resid_q[i]),
                "q_expect": float(qp[cols].mean() * resid_p[i]),
                "p_pj": float(P[i, PJ]), "x_pj": float(x[cols.index(PJ)]),
                "on_pj": bool(g[i, PJ] > 0.5),
                # 현행 코사인 배분
                "cos_pj": float(_cos_share(Rc[i], S, live[i]) * max(resid_p[i], 0.0)),
            })

    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("stem", "on_pj")}
    st = np.array([r["stem"] for r in rows])
    on = np.array([r["on_pj"] for r in rows])
    tgt = ~np.isin(st, CTRL)

    print("── [a] 무효 잔차가 6W 와 자릿수가 맞는가 ─────────────────────────")
    for nm, m in (("겨냥 8", tgt), ("대조 3", ~tgt)):
        rp, rq, qe = A["resid_p"][m], A["resid_q"][m], A["q_expect"][m]
        k = np.abs(rp) > 0.5
        print(f"  {nm}  총전력잔차 중앙 {np.median(rp):+7.2f}W  "
              f"무효잔차 중앙 {np.median(rq):+8.2f}VAR  "
              f"기대(Q/P x 잔차) {np.median(qe):+8.2f}VAR  "
              f"비 중앙 {np.median(rq[k] / qe[k]):6.2f}")
    print("  판정: 비가 1/3~3 밖이면 Q 에는 이 6W 의 정보가 없다")

    print("\n── [c] 무효 잔차의 부호가 일관되나 (규칙 42) ────────────────────")
    for nm, m in (("겨냥 8", tgt), ("대조 3", ~tgt)):
        rq = A["resid_q"][m]
        s_ = float((np.sign(rq) == np.sign(np.median(rq))).mean())
        print(f"  {nm}  부호 일치 {s_:.3f}  (중앙 {np.median(rq):+.2f} VAR)")

    print("\n── [b] Q 를 넣어 푼 배분 — 프로젝터 ─────────────────────────────")
    m = tgt & on
    print(f"  프로젝터 ON 인 겨냥 창 {int(m.sum())}개, 참값 46.9W")
    for nm, v in (("흡수 전", A["p_pj"][m]),
                  ("+ 코사인 (현행)", (A["p_pj"] + A["cos_pj"])[m]),
                  ("+ P&Q&고조파 (신규)", (A["p_pj"] + A["x_pj"])[m])):
        print(f"  {nm:22s} 중앙 {np.median(v):7.2f}W   "
              f"|오차| 중앙 {np.median(np.abs(v - 46.9)):6.2f}W   "
              f"p90 {np.percentile(np.abs(v - 46.9), 90):6.2f}W")
    print("  판정: |오차| 가 현행보다 크면 접는다")

    Path(a.out).write_text(json.dumps(
        {"_config": {"argv": sys.argv}, "n": len(rows),
         "qp": {apps[j]: float(qp[j]) for j in cols}, "noise_q": nq},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {a.out}")
    return 0


def _cos_share(Rc, S, live) -> float:
    rn = np.linalg.norm(Rc)
    if rn <= 1e-12:
        return 0.0
    w = np.clip(np.array([np.real(Rc @ np.conj(s)) / (np.linalg.norm(s) * rn + 1e-12)
                          for s in S]), 0.0, None) * live
    return float(w[0] / w.sum()) if w.sum() > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
