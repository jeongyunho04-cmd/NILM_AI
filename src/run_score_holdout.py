"""
저장된 체크포인트를 임의의 합성 홀드아웃에 채점한다
=====================================================
`run_train_cnn` 은 학습 끝에 한 번 채점하고 만다. **같은 모델을 다른 홀드아웃에
대 보는** 도구가 없었다 — 12.63 의 반사실 평가 셋(기착 절제)처럼 데이터 쪽을
조작한 셋에 기존 모델을 그대로 물려 보려면 필요하다.

    # 기준 셋과 반사실 셋을 나란히
    python -m src.run_score_holdout --ckpt results/cnn_ov1.pt \
        --holdout processed_data/holdout60 processed_data/holdout60_noped

기본 셋과 다른 `target_index` 를 가진 홀드아웃은 거부한다 (12.45.3 의 가드).
"""
from pathlib import Path
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.holdout import load_holdout
from src.run_gate_check import load_model
from src.run_train_cnn import evaluate, prepare_holdout_inputs, report


def main() -> int:
    ap = argparse.ArgumentParser(description="체크포인트를 합성 홀드아웃에 채점")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--holdout", nargs="+", default=["processed_data/holdout60"])
    ap.add_argument("--apps", nargs="*", default=None,
                    help="이 가전만 표에 찍는다 (기본: 전부)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = {}
    for hd in a.holdout:
        hs = load_holdout(hd)
        prep = prepare_holdout_inputs(hs)
        abl = hs.meta.get("ablate_pedestal_apps") or []
        print(f"[{Path(hd).name}] {len(hs):,}창 | sha {hs.meta['content_sha256']}"
              + (f" | 기착 절제 {', '.join(abl)}" if abl else ""))
        for ck in a.ckpt:
            model, apps, _ = load_model(ck, dev)
            pred, on_prob = evaluate(model, prep, dev)
            sc, summ, _, resid = report(pred, on_prob, hs)
            rows[(Path(ck).stem, Path(hd).name)] = (
                {r.appliance: r for r in sc}, summ, resid)

    shown = a.apps or sorted({app for (d, _, _) in rows.values() for app in d})
    hds = [Path(h).name for h in a.holdout]
    cks = [Path(c).stem for c in a.ckpt]
    for ck in cks:
        print("\n" + "=" * (20 + 13 * len(hds)))
        print(f"[{ck}]  F1")
        print("=" * (20 + 13 * len(hds)))
        print(f"  {'기기':18s}" + "".join(f"{h[:12]:>13s}" for h in hds)
              + (f"{'Δ':>10s}" if len(hds) == 2 else ""))
        for app in shown:
            vals = [rows[(ck, h)][0][app].f1 for h in hds]
            line = f"  {app:18s}" + "".join(f"{v:>13.4f}" for v in vals)
            if len(hds) == 2:
                line += f"{vals[1] - vals[0]:>+10.4f}"
            print(line)
        for key, lab in (("f1_mean", "F1 평균"), ("mae_w_mean", "MAE 평균 W")):
            vals = [rows[(ck, h)][1][key] for h in hds]
            line = f"  {lab:18s}" + "".join(f"{v:>13.4f}" for v in vals)
            if len(hds) == 2:
                line += f"{vals[1] - vals[0]:>+10.4f}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
