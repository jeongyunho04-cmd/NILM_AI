"""물리 전력 상한 후처리를 잰다 (12.100절)

프로젝터는 격리에서 폭이 ±0.5W(48.5~49.3W)인데 복합 실측에서 100W 넘게 예측된다.
그 초과분을 잘라내면(그리고 다른 SMPS 로 넘기면) 무엇이 좋아지고 무엇이 나빠지는지
**학습 없이** 잰다.

    python -m src.run_cap_probe --ckpt results/adapt_smpsf.pt
    python -m src.run_cap_probe --ckpt results/adapt_ze1.pt --ckpt-smps results/cnn_ze1.pt
"""
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
from src.model.postproc import PHYSICAL_CAP_W, cap_power
from src.run_gate_check import forward_file, gated, load_model, merge_smps, score_one

SMPS = ("beam_projector", "laptop_charger", "minipc")
KO = {"beam_projector": "프로젝터", "laptop_charger": "충전기", "minipc": "미니PC"}


def main() -> int:
    ap = argparse.ArgumentParser(description="물리 전력 상한 후처리")
    ap.add_argument("--ckpt", default="results/adapt_smpsf.pt")
    ap.add_argument("--ckpt-smps", default="", metavar="PT")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--out", default="results/cap_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    model, apps, _ = load_model(a.ckpt, dev)
    m_smps = load_model(a.ckpt_smps, dev)[0] if a.ckpt_smps else None

    D = {}
    for stem in stems:
        d = forward_file(model, stem, dev, stride=a.stride)
        if m_smps is not None:
            d = merge_smps(d, forward_file(m_smps, stem, dev, stride=a.stride), apps)
        D[stem] = d

    # ── 1. 프로젝터가 실제로 얼마를 받고 있나 ────────────────────────────────
    print("=" * 96)
    print(f"[예측 전력 분포] {a.ckpt}" + (f" + {a.ckpt_smps}" if a.ckpt_smps else ""))
    print("=" * 96)
    print(f"  {'기기':8s}{'격리 최대':>10s}{'상한':>7s}"
          f"{'예측 p50':>10s}{'p90':>8s}{'p99':>8s}{'최대':>8s}{'상한 초과 창':>13s}")
    iso = {"beam_projector": 49.6, "laptop_charger": 70.3, "minipc": 26.9}
    for app in SMPS:
        j = apps.index(app)
        v = np.concatenate([gated(D[s], False)[:, j] for s in stems])
        on = v > 1.0
        q = v[on]
        cap = PHYSICAL_CAP_W[app]
        print(f"  {KO[app]:8s}{iso[app]:>10.1f}{cap:>7.0f}{np.percentile(q,50):>10.1f}"
              f"{np.percentile(q,90):>8.1f}{np.percentile(q,99):>8.1f}{q.max():>8.1f}"
              f"{100*(q > cap).mean():>12.1f}%")

    # ── 2. 후처리 변형별 채점 ────────────────────────────────────────────────
    VAR = [("없음 (기준)", False, False), ("상한만 (초과분 버림)", True, False),
           ("상한 + 재배분", True, True)]
    print()
    print("=" * 96)
    print("[후처리 변형] 유령W = 없는 기기에 붙은 전력, 잔차W = |관측 − 예측 합계|")
    print("=" * 96)
    print(f"  {'변형':24s}{'유령W':>8s}{'잔차W':>8s}{'프로젝터 평균W':>15s}"
          f"{'충전기':>9s}{'미니PC':>9s}")
    payload: Dict[str, dict] = {}
    for name, do_cap, redist in VAR:
        rows, means = [], {app: [] for app in SMPS}
        for stem in stems:
            d = D[stem]
            P = gated(d, False)
            if do_cap:
                P = cap_power(P, apps, gate=d["gate"], redistribute=redist)
            rows.append(score_one(d, P, stem, apps, ev))
            for app in SMPS:
                means[app].append(P[:, apps.index(app)].mean())
        gh = float(np.mean([r["absent_sum_w"] for r in rows]))
        rs = float(np.mean([r["residual_abs_w"] for r in rows]))
        mu = {app: float(np.mean(v)) for app, v in means.items()}
        print(f"  {name:24s}{gh:>8.2f}{rs:>8.2f}{mu['beam_projector']:>15.1f}"
              f"{mu['laptop_charger']:>9.1f}{mu['minipc']:>9.1f}")
        payload[name] = {"ghost_w": gh, "residual_w": rs, "mean_w": mu,
                         "per_file": {s: r for s, r in zip(stems, rows)}}

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_file"}
                   for k, v in payload.items()}, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
