"""
창 크기 × 타깃 위치 스윕
=========================
CNN 을 짜기 전에 두 가지를 실측으로 정한다.

  1. **타깃을 창 끝으로 옮기면 성능이 얼마나 떨어지는가** (인과성의 비용)
     같은 창 크기에서 타깃만 중앙 -> 끝-1초 로 옮겨 비교한다.
  2. **10초 창만으로 되는가** (긴 창의 가치)
     같은 타깃 위치에서 창을 10 / 60 / 120초로 늘려 비교한다.

시연에서 60초 지연은 치명적이므로, 이 두 축의 답이 창 설계를 결정한다.

    10초로 충분      -> 창 10초. 중앙 타깃(5초 지연)도 선택지가 된다
    120초가 필요     -> 반드시 끝 타깃(1초 지연). 중앙이면 60초 지연이라 못 쓴다

Phase 1 의 GBM 을 그대로 쓴다. **CNN 의 대리 지표**라는 한계는 있지만
(순서는 옮겨가도 크기는 다를 수 있다), 창 설계 결정에는 이것으로 충분하다.
특징 추출기는 창 길이에 맞춰 블록 수가 늘어나므로 긴 창을 쓸 길이 열려 있다.

# 기본 스윕 (5개 구성, 약 25분)
python -m src.run_window_sweep

# 빠르게
python -m src.run_window_sweep --windows 40000 --eval 4000
"""
from pathlib import Path
from typing import List, Tuple
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.baseline.train import BaselineModel, build_training_set, train
from src.evaluation.metrics import resistive_confusion, score_appliances, summarize
from src.run_baseline import LOW_LOAD, S_I

# (창 사이클, 타깃 lookahead, 이름). lookahead=None 이면 중앙 타깃.
CONFIGS: List[Tuple[int, int, str]] = [
    (600, 60, "10초 / 끝-1초"),
    (600, None, "10초 / 중앙"),
    (3600, 60, "60초 / 끝-1초"),
    (3600, None, "60초 / 중앙"),
    (7200, 60, "120초 / 끝-1초"),
]


def _lookahead(window: int, la) -> int:
    return (window // 2) if la is None else la


def run_one(window: int, la, n_train: int, n_eval: int, workers: int,
            max_iter: int, seed: int) -> dict:
    lookahead = _lookahead(window, la)
    tgt = window - 1 - lookahead
    print(f"\n  창 {window} ({window/60:.0f}초) | 타깃 {tgt} | 지연 {lookahead/60:.2f}초")

    Ftr, ytr, otr, apps = build_training_set(
        n_windows=n_train, window_cycles=window, time_split="train",
        seed=seed, n_workers=workers, target_lookahead_cycles=lookahead)
    Fev, yev, oev, _ = build_training_set(
        n_windows=n_eval, window_cycles=window, time_split="holdout",
        seed=987_654, n_workers=workers, target_lookahead_cycles=lookahead)

    t0 = time.time()
    model = train(Ftr, ytr, otr, apps, max_iter=max_iter, random_state=seed, verbose=False)
    pred, prob = model.predict(Fev)
    sc = score_appliances(yev, pred, apps, S_I,
                          on_true=oev.astype(bool), on_pred=prob > 0.5)
    summ = summarize(sc, low_load=LOW_LOAD)
    cm = resistive_confusion(yev, pred, apps)
    by = {x.appliance: x for x in sc}

    res = {
        "name": f"{window}/{lookahead}", "window_cycles": window,
        "target_index": tgt, "lookahead_cycles": lookahead,
        "latency_s": round(lookahead / 60.0, 3),
        "n_train": int(len(Ftr)), "n_eval": int(len(Fev)),
        "mae_mean": summ["mae_w_mean"], "f1_mean": summ["f1_mean"],
        "worst_f1": summ["worst_f1"][0], "worst_f1_app": summ["worst_f1"][1],
        "resistive_acc": cm["accuracy"] if cm else float("nan"),
        "oven_to_kettle": cm["matrix"][1][0] if cm else -1,
        "oven_solo": sum(cm["matrix"][1]) if cm else 0,
        "per_app_f1": {a: by[a].f1 for a in apps},
        "per_app_nmae": {a: by[a].nmae_on for a in apps},
        "per_app_bias": {a: by[a].bias_on_w for a in apps},
        "fa_pass": summ["fa_target_pass"],
        "train_s": round(time.time() - t0, 1),
    }
    print(f"    MAE {res['mae_mean']:.2f}W | F1 {res['f1_mean']:.3f} | "
          f"저항3종 {res['resistive_acc']:.3f} | 오븐→포트 {res['oven_to_kettle']}/{res['oven_solo']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="창 크기 x 타깃 위치 스윕")
    ap.add_argument("--windows", type=int, default=100_000, help="구성당 학습 창 수")
    ap.add_argument("--eval", type=int, default=8000, help="구성당 평가 창 수")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/window_sweep.json")
    a = ap.parse_args()

    print("=" * 84)
    print("[창 설계 스윕] 타깃 위치의 비용과 긴 창의 가치를 실측한다")
    print("=" * 84)

    rows = [run_one(w, la, a.windows, a.eval, a.workers, a.max_iter, a.seed)
            for w, la, _ in CONFIGS]
    names = {f"{w}/{_lookahead(w, la)}": nm for w, la, nm in CONFIGS}

    print("\n" + "=" * 84)
    print(f"{'구성':16s}{'지연':>8s}{'MAE':>9s}{'F1평균':>9s}{'최악F1':>9s}"
          f"{'저항3종':>9s}{'오븐→포트':>12s}")
    print("-" * 84)
    for r in rows:
        print(f"{names[r['name']]:16s}{r['latency_s']:>7.2f}s{r['mae_mean']:>9.2f}"
              f"{r['f1_mean']:>9.3f}{r['worst_f1']:>9.3f}{r['resistive_acc']:>9.3f}"
              f"{r['oven_to_kettle']:>7d}/{r['oven_solo']:<4d}")

    def find(w, la):
        return next(r for r in rows if r["window_cycles"] == w
                    and r["lookahead_cycles"] == _lookahead(w, la))

    print("\n[질문 1] 타깃을 중앙에서 끝-1초로 옮기는 비용")
    for w in (600, 3600):
        c, e = find(w, None), find(w, 60)
        print(f"  창 {w//60:>3d}초: MAE {c['mae_mean']:.2f} -> {e['mae_mean']:.2f}W "
              f"({100*(e['mae_mean']-c['mae_mean'])/max(c['mae_mean'],1e-9):+.1f}%) | "
              f"F1 {c['f1_mean']:.3f} -> {e['f1_mean']:.3f} | "
              f"저항3종 {c['resistive_acc']:.3f} -> {e['resistive_acc']:.3f} | "
              f"지연 {c['latency_s']:.0f}초 -> {e['latency_s']:.0f}초")

    print("\n[질문 2] 창을 늘리는 가치 (타깃은 끝-1초 고정)")
    base = find(600, 60)
    for w in (600, 3600, 7200):
        r = find(w, 60)
        print(f"  창 {w//60:>3d}초: MAE {r['mae_mean']:.2f}W "
              f"({100*(r['mae_mean']-base['mae_mean'])/max(base['mae_mean'],1e-9):+.1f}%) | "
              f"F1 {r['f1_mean']:.3f} | 저항3종 {r['resistive_acc']:.3f} | "
              f"오븐→포트 {r['oven_to_kettle']}/{r['oven_solo']} "
              f"({100*r['oven_to_kettle']/max(r['oven_solo'],1):.1f}%)")

    print("\n[오븐 상세] 0.4절대로 오븐 주기(최대 65초)를 담아야 포트와 갈리는가")
    for w in (600, 3600, 7200):
        r = find(w, 60)
        print(f"  창 {w//60:>3d}초: 오븐 F1 {r['per_app_f1']['oven']:.3f} | "
              f"nMAE {r['per_app_nmae']['oven']:.3f} | 편향 {r['per_app_bias']['oven']:+.1f}W | "
              f"포트 F1 {r['per_app_f1']['electiric_kettle']:.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"configs": rows, "names": names},
                                      ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"\n결과: {Path(a.out).resolve()}")
    print("=" * 84 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
