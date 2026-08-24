"""
입력 절제로 "그 쌍의 판별이 무엇에 걸려 있는가" 를 묻는다 (설계 문서 12.65절)
================================================================================
12.62~12.64 가 후보 셋을 떨어뜨렸다 — 고조파 지문(k>=2), 기착, 절대 준위.
전부 **데이터를 조작해** 물었는데, 조작은 입력을 학습 분포 밖으로 밀어 분포 밖
충격과 교락된다 (12.64.3 의 한계). 절제는 그 교락이 모든 채널군에 똑같이 걸리므로
**채널군 사이의 상대 비교**가 성립한다.

    python -m src.run_pair_ablation --ckpt results/cnn_ov1.pt

[F1 이 아니라 쌍 정확도를 쓴다]
F1 은 검출(있나 없나)과 판별(어느 쪽인가)을 섞는다. 12.64 에서 프로젝터 F1 이
떨어졌는데 충전기는 안 떨어진 것이 그 탓이다 — 검출이 무너진 것이지 쌍이 섞인 것이
아니었다. 여기서는 **둘 중 정확히 하나만 켜진 창**만 골라 모델이 어느 쪽을 더
크게 예측하는지 본다. 검출 성능이 빠지고 판별만 남는다.

> ⚠ **0 으로 죽이는 것은 학습 분포 밖이다** (`run_ablation_probe` 서두와 같은
> 단서다). 이 시험이 힘을 갖는 방향은 **안 변할 때**다 — 분포 밖 입력을 줘도
> 판별이 그대로면 그 입력을 안 보고 있는 것이 확실하다. 크게 흔들렸을 때는
> "이 채널군이 그 판별을 옮긴다" 까지만 읽고, 채널군끼리 견준다.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.holdout import load_holdout
from src.model.inputs import FINE_CHANNELS
from src.run_gate_check import load_model
from src.run_train_cnn import evaluate, prepare_holdout_inputs

PAIR = ("beam_projector", "laptop_charger")

#: 채널군. `build_fine` 의 배치를 따른다 (inputs.py 주석 참조).
#: (이름, 세밀 채널 목록, 광역을 죽이는가)
GROUPS: List[Tuple[str, List[int], bool]] = [
    ("기준(절제 없음)",       [],                              False),
    ("기본파 Re/Im",          [0, 15],                          False),
    ("P",                     [30],                             False),
    ("Q",                     [31],                             False),
    ("P·Q",                   [30, 31],                         False),
    ("역률 ch46",             [46],                             False),
    ("기본파+P·Q+역률",       [0, 15, 30, 31, 46],              False),
    ("고차 Re/Im (2~15차)",   list(range(1, 15)) + list(range(16, 30)), False),
    ("ch35 |I2|/|I1| (짝수)",  [35],                             False),
    ("크기비 홀수 33,34,47",   [33, 34, 47],                     False),
    ("크기비 전부 33~35,47",   [33, 34, 35, 47],                 False),
    ("짝수 Re/Im (2,4..14차)", [1, 3, 5, 7, 9, 11, 13] + [16, 18, 20, 22, 24, 26, 28], False),
    ("홀수 Re/Im (3,5..15차)", [2, 4, 6, 8, 10, 12, 14] + [17, 19, 21, 23, 25, 27, 29], False),
    ("위상불변 φ3~φ9",        list(range(38, 46)),              False),
    ("고조파 전부",           list(range(0, 30)) + [33, 34, 35, 47] + list(range(38, 46)), False),
    ("추세제거 ch36~37",      [36, 37],                         False),
    ("전압 ch32",             [32],                             False),
    ("광역 갈래",             [],                               True),
]


def pair_accuracy(pred: np.ndarray, on_true: np.ndarray, apps: List[str]) -> dict:
    """둘 중 정확히 하나만 켜진 창에서 어느 쪽을 더 크게 예측하는가."""
    a, b = (apps.index(x) for x in PAIR)
    only_a = on_true[:, a].astype(bool) & ~on_true[:, b].astype(bool)
    only_b = on_true[:, b].astype(bool) & ~on_true[:, a].astype(bool)
    m = only_a | only_b
    if m.sum() == 0:
        return {"n": 0, "acc": float("nan")}
    picks_a = pred[m, a] > pred[m, b]
    truth_a = only_a[m]
    return {"n": int(m.sum()), "acc": float((picks_a == truth_a).mean()),
            "n_proj": int(only_a.sum()), "n_chg": int(only_b.sum()),
            "acc_proj": float(picks_a[truth_a].mean()) if truth_a.any() else float("nan"),
            "acc_chg": float((~picks_a[~truth_a]).mean()) if (~truth_a).any() else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description="쌍 판별의 입력 의존을 절제로 잰다")
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_ov1.pt"])
    ap.add_argument("--holdout", default="processed_data/holdout60")
    ap.add_argument("--out", default="results/pair_ablation.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hs = load_holdout(a.holdout)
    fine, wide = prepare_holdout_inputs(hs)
    print(f"[{Path(a.holdout).name}] {len(hs):,}창 | 세밀 {fine.shape[1]}채널")

    payload: Dict[str, dict] = {}
    for ckpt in a.ckpt:
        model, apps, ck = load_model(ckpt, dev)
        nch = int(ck.get("fine_channels", FINE_CHANNELS))
        print("\n" + "=" * 78)
        print(f"[{Path(ckpt).stem}] 세밀 {nch}채널 사용")
        print("=" * 78)
        print(f"  {'절제한 것':24s}{'n':>7s}{'쌍 정확도':>11s}{'Δ':>9s}"
              f"{'프로젝터':>10s}{'충전기':>10s}")
        base_acc = None
        rows = {}
        for name, chans, kill_wide in GROUPS:
            use = [c for c in chans if c < nch]
            if chans and not use:
                continue
            saved = fine[:, use].copy() if use else None
            if use:
                fine[:, use] = 0.0
            w = np.zeros_like(wide) if kill_wide else wide
            try:
                pred, _ = evaluate(model, (fine, w), dev)
            finally:
                if use:
                    fine[:, use] = saved
            r = pair_accuracy(pred, hs.y_on, apps)
            if base_acc is None:
                base_acc = r["acc"]
            d = r["acc"] - base_acc
            rows[name] = r
            print(f"  {name:24s}{r['n']:>7,d}{r['acc']:>11.4f}{d:>+9.4f}"
                  f"{r['acc_proj']:>10.4f}{r['acc_chg']:>10.4f}")
        payload[Path(ckpt).stem] = rows

    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
