"""프로젝터는 총전력 부족분의 **배출구**인가 — 재학습 없이 정량화 (12.128)

가설 (12.126 이 지목, 12.125 가 귀결을 쟀다)
-----------------------------------------
`L_cons = |Σ P̂ + Σ대기 + p_noise − P관측|` 에 **슬랙이 없다** (`p_noise` 는
실측에서 상수 1.4W). 그래서 총전력 오차는 **반드시 기기 어딘가로 배분된다.**
그리고 12.120.3 이 프로젝터가 와트당 `L_harm` 이 가장 싸다고 쟀다
(0.1219 vs 충전기 0.1479 vs 미니PC 0.1697). 그러면 적응은 총합을 맞추면서
부족분을 프로젝터에 얹는다 — 12.125 가 그 귀결을 쟀다:

    프로젝터 평균오차   1단계(합성만) +15.4W  ->  2단계(적응) +31.6W
    오븐 중앙|오차|             59.2W   ->   16.5W

이 스크립트는 **그 사이의 인과 고리를 창 단위로** 잰다.

무엇을 재는가
-----------
같은 창에서 1단계와 2단계를 둘 다 돌려서

    부족분   gap_1  = P관측 − recon_1        (1단계가 총합을 얼마나 못 맞췄나)
    이동량   dP_i   = P̂_2,i − P̂_1,i         (적응이 기기별로 얼마를 옮겼나)

를 낸다. 배출 가설이 맞으면 셋이 성립해야 한다:

    (1) Σ_i dP_i ≈ gap_1              적응이 부족분을 채운다
    (2) dP_projector 가 gap_1 과 강하게 상관       프로젝터가 그것을 받는다
    (3) 프로젝터 몫이 **와트당 비용 순위**를 따른다   가장 싼 곳이 가장 많이 받는다

⚠ **반증 조건을 먼저 적는다.** (2) 의 상관이 약하거나 프로젝터 몫이 다른
기기와 비슷하면 **배출 가설이 틀린 것**이고, 슬랙은 처방이 아니다.
그러면 학습을 안 돌리고 여기서 멈춘다.

δ 크기
------
슬랙을 준다면 폭을 얼마로? `|gap_1|` 의 분포가 그것을 정한다. 저항이 낀 층과
아닌 층을 갈라 낸다 (규칙 24).

    python -m src.run_dump_probe
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

from src.evaluation.real_events import build_on_off_truth, load_events
from src.evaluation.sealing import is_sealed
from src.model.realdata import SMPS_APPLIANCES
from src.run_gate_check import forward_file, gated, load_model
from src.run_summarize_gate import human_stems

REF_PROJECTOR_W = 46.9      # power_ref.REFERENCE_W


def per_window(model, stem: str, dev: str, stride: int) -> dict:
    d = forward_file(model, stem, dev, stride=stride)
    P = gated(d, hard=False)                       # (n, K) 소프트 게이트 전력
    recon = P.sum(1) + d["standby"].sum(1) + d["p_noise"]
    return {"P": P, "recon": recon, "pobs": d["p_observed"],
            "targets": d["targets"], "gate": d["gate"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", default="results/cnn_ovh.pt")
    ap.add_argument("--stage2", nargs="+",
                    default=["results/adapt_ovh.pt", "results/adapt_ovh_s1.pt",
                             "results/adapt_ovh_s2.pt"])
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--out", default="results/dump_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m1, apps, _ = load_model(a.stage1, dev)
    ev = load_events()
    stems = [s for s in human_stems(ev) if not is_sealed(s)]
    smps = [apps.index(x) for x in SMPS_APPLIANCES if x in apps]
    pj = apps.index("beam_projector")
    res_idx = [i for i, x in enumerate(apps)
               if x in ("oven", "hotplate", "electiric_kettle")]

    print("=" * 88)
    print("배출 가설 — 적응이 채운 부족분은 어느 기기로 갔는가")
    print(f"1단계 {a.stage1} | 2단계 {len(a.stage2)}개 | 사람기록 {len(stems)}파일")
    print("=" * 88)

    rows: List[dict] = []
    for stem in stems:
        w1 = per_window(m1, stem, dev, a.stride)
        on, _ = build_on_off_truth(stem, apps, int(ev[stem]["cycles"]), ev)
        truth = on[np.clip(w1["targets"], 0, len(on) - 1)]     # (n, K) 정답 on/off
        for ck in a.stage2:
            m2, _, _ = load_model(ck, dev)
            w2 = per_window(m2, stem, dev, a.stride)
            n = min(len(w1["recon"]), len(w2["recon"]))
            gap1 = w1["pobs"][:n] - w1["recon"][:n]
            gap2 = w2["pobs"][:n] - w2["recon"][:n]
            dP = w2["P"][:n] - w1["P"][:n]
            for k in range(n):
                rows.append({
                    "stem": stem, "ck": Path(ck).stem,
                    "gap1": float(gap1[k]), "gap2": float(gap2[k]),
                    "dP": dP[k].astype(float),
                    "P2_pj": float(w2["P"][k, pj]), "P1_pj": float(w1["P"][k, pj]),
                    "pj_on": bool(truth[k, pj]),
                    "n_res_on": int(truth[k, res_idx].sum()),
                })
            del m2

    G1 = np.array([r["gap1"] for r in rows])
    DP = np.stack([r["dP"] for r in rows])
    print(f"\n창 {len(rows)}개 (stride {a.stride}, 2단계 시드 {len(a.stage2)}개 합산)\n")

    # ── (1) 적응이 부족분을 채우는가 ──────────────────────────────────────
    tot = DP.sum(1)
    print("[1] 적응이 1단계의 부족분을 채우는가")
    print(f"  gap1 중앙 {np.median(G1):+.1f}W   |gap1| 중앙 {np.median(np.abs(G1)):.1f}W")
    print(f"  Σ dP 중앙 {np.median(tot):+.1f}W   상관(gap1, Σ dP) = "
          f"{np.corrcoef(G1, tot)[0, 1]:+.3f}")

    # ── (2) 누가 받는가 ──────────────────────────────────────────────────
    print("\n[2] 부족분을 **누가** 받는가 — 기기별 dP 와 gap1 의 상관/기울기")
    print(f"  {'기기':<18s}{'dP 중앙W':>10s}{'상관':>8s}{'기울기 dP/gap1':>16s}"
          f"{'몫(회귀)':>10s}")
    print("  " + "-" * 64)
    order = []
    for i, app in enumerate(apps):
        v = DP[:, i]
        if np.allclose(v, 0):
            continue
        rho = float(np.corrcoef(G1, v)[0, 1]) if v.std() > 1e-9 else float("nan")
        sl = float(np.polyfit(G1, v, 1)[0])
        order.append((abs(sl), app, np.median(v), rho, sl))
    denom = sum(o[0] for o in order) or 1.0
    for _, app, med, rho, sl in sorted(order, reverse=True):
        mark = "  <- SMPS" if app in SMPS_APPLIANCES else ""
        print(f"  {app:<18s}{med:>+10.2f}{rho:>8.3f}{sl:>16.3f}"
              f"{abs(sl) / denom:>9.1%}{mark}")

    # ── (3) 저항 층에서 더 심한가 (규칙 24) ──────────────────────────────
    print("\n[3] 층별 — 저항이 켜진 창에서 배출이 더 큰가")
    lc = apps.index("laptop_charger") if "laptop_charger" in apps else None
    print(f"  {'층':<20s}{'창':>7s}{'|gap1| 중앙':>12s}"
          f"{'dP 프로젝터':>13s}{'dP 충전기':>12s}{'합':>8s}{'프로젝터 초과W':>15s}")
    print("  " + "-" * 84)
    for lab, sel in (("저항 0종", lambda r: r["n_res_on"] == 0),
                     ("저항 1종+", lambda r: r["n_res_on"] >= 1),
                     ("프로젝터 ON", lambda r: r["pj_on"]),
                     ("프로젝터 ON + 저항", lambda r: r["pj_on"] and r["n_res_on"] >= 1),
                     ("전체", lambda r: True)):
        v = [r for r in rows if sel(r)]
        if not v:
            continue
        ex = [r["P2_pj"] - REF_PROJECTOR_W for r in v if r["pj_on"]]
        dpj = float(np.median([r["dP"][pj] for r in v]))
        dlc = float(np.median([r["dP"][lc] for r in v])) if lc is not None else 0.0
        print(f"  {lab:<20s}{len(v):>7d}"
              f"{np.median([abs(r['gap1']) for r in v]):>12.1f}"
              f"{dpj:>+13.2f}{dlc:>+12.2f}{dpj + dlc:>+8.2f}"
              f"{(np.median(ex) if ex else float('nan')):>+15.1f}")

    # ── δ 후보 ───────────────────────────────────────────────────────────
    print("\n[δ 후보] 슬랙 폭은 |gap1| 의 분포가 정한다")
    for lab, sel in (("저항 0종", lambda r: r["n_res_on"] == 0),
                     ("저항 1종+", lambda r: r["n_res_on"] >= 1),
                     ("전체", lambda r: True)):
        g = np.abs([r["gap1"] for r in rows if sel(r)])
        if not len(g):
            continue
        p = np.percentile(g, [50, 75, 90])
        print(f"  {lab:<12s} p50 {p[0]:>7.1f}W   p75 {p[1]:>7.1f}W   p90 {p[2]:>7.1f}W")

    print("\n⚠ 반증 조건 — [2] 에서 프로젝터의 상관/기울기가 다른 SMPS 와 비슷하면")
    print("  배출 가설은 틀린 것이고 슬랙은 처방이 아니다. 학습을 안 돌린다.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "n_windows": len(rows), "stems": stems,
        "gap1_abs_p50": float(np.median(np.abs(G1))),
        "per_app": {apps[i]: {"dp_median": float(np.median(DP[:, i])),
                              "corr_gap1": (float(np.corrcoef(G1, DP[:, i])[0, 1])
                                            if DP[:, i].std() > 1e-9 else None),
                              "slope": float(np.polyfit(G1, DP[:, i], 1)[0])}
                    for i in range(len(apps))},
        "_config": {"argv": sys.argv, "args": vars(a)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
