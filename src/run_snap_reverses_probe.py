"""스냅이 프로젝터↔충전기 맞바꿈을 **되돌리는가** (12.129)

12.128 이 기제를 닫았다 — 정답상 프로젝터가 켜진 창에서 적응은 프로젝터에
**+17.00W** 를 얹고 충전기에서 **−17.01W** 를 뺀다 (합 −0.01W, 완벽한 제로섬).
총전력과 무관하고, 12.122.2 가 모델 없이 예측한 +20/−16 과 같다.

그러면 프로젝터를 참값 46.9W 에 못 박으면 그 17W 는 갈 곳이 충전기밖에 없다.
**스냅이 맞바꿈을 되돌리는 것인가, 아니면 오차를 옮기기만 하는 것인가?**

⚠ 동어반복을 피한다
------------------
`snap_power` 는 **설계상 SMPS 그룹 안에서 재배분한다.** 그러니 "충전기로 가는가"
를 묻는 것은 답이 정해진 질문이다. 물어야 할 것은 **옳은 값에 착지하는가** 다.

독립 기준을 셋 세운다 (규칙 14 — 셋 다 참값이 아니다. 참값은 없다):

    ① 1단계 모델 (`cnn_ovh`)   맞바꿈 **전**의 배분. 합성만 보고 배웠다
    ② 정답 support NNLS        모델 없이, 사람 라벨이 켰다고 한 기기로만 푼다
                              **앵커를 끈다** — 프로젝터를 46.9 로 고정하면 순환이다
    ③ 스냅 뒤 2단계            우리가 시험하는 것

    통과 = ③ 이 ①·② 쪽으로 **되돌아온다**. 실패 = 지나치거나 엉뚱한 데로 간다

    python -m src.run_snap_reverses_probe
"""
from pathlib import Path
from typing import Dict, List, Sequence
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch
from scipy.optimize import nnls

from src.evaluation.real_events import build_on_off_truth, load_events
from src.evaluation.sealing import is_sealed
from src.model.realdata import SMPS_APPLIANCES
from src.run_gate_check import _signatures, forward_file, gated, load_model
from src.run_summarize_gate import human_stems

SNAP_W = 46.9        # power_ref.REFERENCE_W["beam_projector"]
H = 15


def pipeline(d: dict, apps: List[str], *, snap: float, resmatch: float) -> np.ndarray:
    """운영점 후처리를 run_gate_check 와 **같은 순서로** 건다."""
    from src.model.postproc import apply_postproc, resistive_match, snap_power
    P = gated(d, hard=False)
    g = d["gate"]
    P, g = apply_postproc(P, g, apps, gate_sync=False)
    if snap > 0:
        P, g = snap_power(P, g, apps, targets={"beam_projector": float(snap)},
                          bidirectional=True, share="gate", min_gate=0.5,
                          redistribute=True)
    if resmatch > 0:
        P, g = resistive_match(P, g, apps, d["p_observed"], d["v_rms"],
                               d["standby"], d["p_noise"], obs_harm=d["obs_harm"],
                               tol=resmatch, snap=True)
    return P


def nnls_reference(apps: Sequence[str], stems: Sequence[str], sig: np.ndarray,
                   nz: np.ndarray, window: int = 3600) -> Dict[str, list]:
    """② 정답 support NNLS — 모델 없이. **앵커를 끈다** (프로젝터를 고정하면 순환)."""
    from src.preprocessing import load_nilm_npz
    from src.run_fit_insitu_sig import EVAL_DIR
    ev = load_events()
    NZ = nz[:H, 0] + 1j * nz[:H, 1]
    out: Dict[str, list] = {a: [] for a in SMPS_APPLIANCES}
    pj = apps.index("beam_projector")
    for stem in stems:
        f = Path(EVAL_DIR) / f"{stem}.npz"
        if stem not in ev or not f.exists():
            continue
        z = load_nilm_npz(str(f))
        hc = np.asarray(z["harmonics_complex"])
        ok = np.asarray(z["is_valid"]).astype(bool)
        on, _ = build_on_off_truth(stem, apps, len(hc), ev)
        i = np.flatnonzero(ok)
        for k in range(0, len(i) - window, window):
            sl = i[k:k + window]
            sup = np.flatnonzero(on[sl].mean(0) > 0.5)
            if not len(sup):
                continue
            y = hc[sl].mean(0)[:H] - NZ
            A = np.array([np.concatenate([sig[j, :H, 0], sig[j, :H, 1]])
                          for j in sup]).T
            x, _ = nnls(A, np.concatenate([y.real, y.imag]))
            for j, v in zip(sup, x):
                if apps[j] in out:
                    out[apps[j]].append(float(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", default="results/cnn_ovh.pt")
    ap.add_argument("--stage2", nargs="+",
                    default=["results/adapt_ovh.pt", "results/adapt_ovh_s1.pt",
                             "results/adapt_ovh_s2.pt"])
    ap.add_argument("--snap", type=float, default=SNAP_W)
    ap.add_argument("--resmatch", type=float, default=0.02)
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--out", default="results/snap_reverses.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m1, apps, _ = load_model(a.stage1, dev)
    ev = load_events()
    stems = [s for s in human_stems(ev) if not is_sealed(s)]
    SM = [x for x in SMPS_APPLIANCES if x in apps]
    idx = {x: apps.index(x) for x in SM}
    pj = apps.index("beam_projector")

    print("=" * 92)
    print("스냅이 프로젝터↔충전기 맞바꿈을 되돌리는가 (12.128 의 후속)")
    print(f"층: **정답상 프로젝터 ON** 창 | 사람기록 {len(stems)}파일 | "
          f"2단계 시드 {len(a.stage2)}개")
    print("=" * 92)

    # 판 이름 -> 기기별 전력 표본
    arms: Dict[str, Dict[str, list]] = {}

    def add(arm: str, app: str, vals):
        arms.setdefault(arm, {a2: [] for a2 in SM})[app].extend(vals)

    for stem in stems:
        d1 = forward_file(m1, stem, dev, stride=a.stride)
        on, _ = build_on_off_truth(stem, apps, int(ev[stem]["cycles"]), ev)
        t1 = on[np.clip(d1["targets"], 0, len(on) - 1)]
        # **기기마다 그 기기가 정답상 켜진 창에서 잰다** (규칙 24).
        # 프로젝터 ON 창으로 뭉치면 꺼진 미니PC 의 0W 가 섞여 중앙값이 뜻을 잃는다.
        m_on = {x: (t1[:, idx[x]] > 0) for x in SM}
        if not any(m_on[x].any() for x in SM):
            continue
        P1 = gated(d1, hard=False)
        for x in SM:
            if m_on[x].any():
                add("① 1단계 (맞바꿈 전)", x, P1[m_on[x], idx[x]])
        for ck in a.stage2:
            m2, _, _ = load_model(ck, dev)
            d2 = forward_file(m2, stem, dev, stride=a.stride)
            n = min(len(d2["gate"]), len(t1))
            raw = gated(d2, hard=False)[:n]
            op = pipeline(d2, apps, snap=0.0, resmatch=a.resmatch)[:n]
            sp = pipeline(d2, apps, snap=a.snap, resmatch=a.resmatch)[:n]
            for x in SM:
                mm = m_on[x][:n]
                if not mm.any():
                    continue
                add("2단계 원시 (후처리 끔)", x, raw[mm, idx[x]])
                add("2단계 운영점 (스냅 없음)", x, op[mm, idx[x]])
                add("③ 2단계 운영점 + 스냅", x, sp[mm, idx[x]])
            del m2

    sig, _, nz = _signatures(apps)
    ref = nnls_reference(apps, stems, sig, nz)
    arms["② 정답 support NNLS"] = ref

    ORDER = ["① 1단계 (맞바꿈 전)", "② 정답 support NNLS", "2단계 원시 (후처리 끔)",
             "2단계 운영점 (스냅 없음)", "③ 2단계 운영점 + 스냅"]
    print(f"\n  {'판':<26s}" + "".join(f"{x:>17s}" for x in SM) + f"{'창':>8s}")
    print("  " + "-" * 92)
    med: Dict[str, Dict[str, float]] = {}
    for arm in ORDER:
        if arm not in arms:
            continue
        med[arm] = {x: float(np.median(arms[arm][x])) if arms[arm][x] else float("nan")
                    for x in SM}
        print(f"  {arm:<26s}" + "".join(
            f"{med[arm][x]:>13.1f} ({len(arms[arm][x]):>5d})" for x in SM))

    print("\n  [판정] 충전기가 맞바꿈 전으로 되돌아오는가 — 중앙 전력W")
    base1, base2 = med.get("① 1단계 (맞바꿈 전)"), med.get("② 정답 support NNLS")
    raw2 = med.get("2단계 원시 (후처리 끔)")
    snap3 = med.get("③ 2단계 운영점 + 스냅")
    if base1 and raw2 and snap3:
        print(f"  {'기기':<18s}{'①1단계':>10s}{'②NNLS':>10s}{'2단계원시':>11s}"
              f"{'③스냅':>10s}{'③−①':>9s}{'③−②':>9s}   판정")
        print("  " + "-" * 88)
        for x in SM:
            d1_ = snap3[x] - base1[x]
            d2_ = snap3[x] - base2[x] if base2 else float("nan")
            gap0 = raw2[x] - base1[x]           # 맞바꿈이 만든 편차
            if abs(gap0) < 1e-6:
                v = "-"
            elif abs(d1_) < abs(gap0) * 0.5:
                v = "**되돌아왔다** (편차 절반 이하)"
            elif abs(d1_) < abs(gap0):
                v = "부분 회복"
            elif d1_ * gap0 < 0:
                v = "**지나쳤다** (부호가 뒤집혔다)"
            else:
                v = "안 돌아왔다"
            print(f"  {x:<18s}{base1[x]:>10.1f}{base2[x] if base2 else float('nan'):>10.1f}"
                  f"{raw2[x]:>11.1f}{snap3[x]:>10.1f}{d1_:>+9.1f}{d2_:>+9.1f}   {v}")

    print("\n  ⚠ ①·② 어느 것도 참값이 아니다 (규칙 14). 충전기 참값은 없다 —")
    print("    격리 통전이 16~71W 라 `power_ref` 에서 제외됐다. 둘은 **독립 기준**일 뿐이다.")
    print("    ① 은 프로젝터를 +15.4W 과대예측하고, ② 는 12.122.2 의 순방향 오차를 안는다.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"median_w": med, "n_stage2": len(a.stage2), "stems": stems,
         "_config": {"argv": sys.argv, "args": vars(a)}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
