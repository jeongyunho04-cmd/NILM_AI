"""
SMPS 층화 채점 — 12.121.7 과 같은 자 (2026-08-31)
====================================================
12.121.7 이 *"충전기 재현 0.641"* 을 잰 자리는 **파일 전체가 아니라 창 층**이다:

    SMPS 1대 + 저항 있음     프로젝터 정밀 0.491   충전기 재현 0.641
    SMPS 1대 + 저항 없음     프로젝터 정밀 0.472   충전기 재현 1.000

파일 전체로 재면 충전기 재현이 0.96 이라 **어려운 경우가 희석된다.** 그래서
`SMPS_PLAN` 이 주 판정으로 삼은 그 숫자를 재현하려면 같은 층으로 잘라야 한다.

    python -m src.run_stratified_smps --ckpt results/ronl_w0p3_s0_test_5.pt --stems test_5

[층을 어떻게 나누는가]
사이클마다 **정답** 기준으로 센다 (예측 기준이 아니다 — 예측으로 층을 나누면
층 자체가 처방에 따라 움직여 비교가 깨진다).

    n_smps  = 그 시점에 켜져 있는 SMPS 개수 (사람 라벨)
    저항여부 = 저항 5종 중 하나라도 켜져 있는가

`uncertain` 은 `build_on_off_truth` 규칙대로 양쪽 다 안 센다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence
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
from src.model.realdata import SMPS_APPLIANCES, upsample_to_cycles
from src.run_gate_check import forward_file

#: 저항 부하 — 이것 중 하나라도 켜져 있으면 "저항 있음" 층이다.
RESISTIVE = ("electiric_kettle", "oven", "hotplate", "hair_dryer", "air_conditioner")


def strata(on: np.ndarray, apps: Sequence[str]) -> Dict[str, np.ndarray]:
    """사이클별 층 마스크. **정답 기준**이다."""
    si = [apps.index(a) for a in SMPS_APPLIANCES if a in apps]
    ri = [apps.index(a) for a in RESISTIVE if a in apps]
    n_smps = on[:, si].sum(1)
    has_res = on[:, ri].any(1) if ri else np.zeros(len(on), bool)
    return {
        "SMPS 1대 + 저항": (n_smps == 1) & has_res,
        "SMPS 1대, 저항 없음": (n_smps == 1) & ~has_res,
        "SMPS 2대+ + 저항": (n_smps >= 2) & has_res,
        "SMPS 2대+, 저항 없음": (n_smps >= 2) & ~has_res,
    }


def score(pred_on: np.ndarray, truth: np.ndarray, scorable: np.ndarray,
          m: np.ndarray, apps: Sequence[str]) -> Dict[str, dict]:
    out = {}
    for a in SMPS_APPLIANCES:
        if a not in apps:
            continue
        j = apps.index(a)
        k = m & scorable[:, j]
        if not k.any():
            continue
        p, t = pred_on[k, j], truth[k, j]
        tp, fp, fn = float((p & t).sum()), float((p & ~t).sum()), float((~p & t).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        out[a] = {"precision": prec, "recall": rec,
                  "f1": (2 * prec * rec / (prec + rec)) if prec and rec and prec + rec else 0.0,
                  "n": int(k.sum()), "n_true_on": int(t.sum())}
    return out


def run(ckpt: str, stems: Sequence[str], dev: str, stride: int = 30) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    apps = ck["appliances"]
    from src.model.net import NILMNet, appliance_state_counts
    from src.model.inputs import LEGACY_FINE_CHANNELS
    model = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                    wide_summary=ck.get("wide_summary", False),
                    periodicity=ck.get("periodicity", False),
                    fine_dropout=ck.get("fine_dropout", 0.0),
                    prior_kappa=ck.get("prior_kappa", 0.0),
                    prior_beta=ck.get("prior_beta", 0.5),
                    fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ev = load_events()
    acc: Dict[str, Dict[str, List[dict]]] = {}
    for stem in stems:
        if is_sealed(stem) or stem not in ev:
            continue
        d = forward_file(model, stem, dev, stride=stride)
        n_cycles = int(ev[stem]["cycles"])
        on_c = upsample_to_cycles(d["gate"] > 0.5, d["targets"], n_cycles)
        truth, scorable = build_on_off_truth(stem, apps, n_cycles, ev)
        for lbl, m in strata(truth, apps).items():
            if not m.any():
                continue
            for a, s in score(on_c, truth, scorable, m, apps).items():
                acc.setdefault(lbl, {}).setdefault(a, []).append(s)
    # 층별로 tp/fp/fn 을 합쳐 다시 계산하지 않고, 파일별 값을 창 수로 가중 평균한다
    out: Dict[str, dict] = {}
    for lbl, per in acc.items():
        out[lbl] = {}
        for a, rows in per.items():
            w = np.array([r["n"] for r in rows], float)
            if w.sum() == 0:
                continue
            out[lbl][a] = {k: float(np.nansum(np.array([r[k] for r in rows]) * w) / w.sum())
                           for k in ("precision", "recall", "f1")}
            out[lbl][a]["n"] = int(w.sum())
            out[lbl][a]["n_true_on"] = int(sum(r["n_true_on"] for r in rows))
    return {"ckpt": ckpt, "stems": list(stems), "strata": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--stems", nargs="+", default=["test_5", "test_6", "test_7", "test_8", "test_13"])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = []
    for c in a.ckpt:
        r = run(c, a.stems, dev, a.stride)
        res.append(r)
        print("=" * 88)
        print(f"[{Path(c).stem}]  {', '.join(a.stems)}")
        for lbl, per in r["strata"].items():
            print(f"  {lbl}")
            for app, s in per.items():
                print(f"    {app:16s} F1 {s['f1']:.3f}  정밀 {s['precision']:.3f}  "
                      f"재현 {s['recall']:.3f}   (n={s['n']:,}, ON {s['n_true_on']:,})")
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
