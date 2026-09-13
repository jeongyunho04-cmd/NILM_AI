"""
Phase 1 — 특징 기반 baseline 학습 및 평가
==========================================
설계 문서 4.1절의 권장 순서 1단계. **이후 모든 모델이 이 숫자와 비교된다.**

# 기본 (150,000창 학습 -> 고정 홀드아웃 평가)
python -m src.run_baseline

# 빠르게 확인
python -m src.run_baseline --windows 30000

# 특징 중요도까지 (느리다)
python -m src.run_baseline --importance

* 산출: results/baseline_gbm.json (지표), results/baseline_gbm.pkl (모델)
"""
from pathlib import Path
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 를 안 써도 BLAS 스레드는 묶는다

import numpy as np

from src.baseline.train import BaselineModel, build_training_set, permutation_importance_fast, train
from src.baseline.features import extract
from src.evaluation import (
    format_table,
    load_holdout,
    resistive_confusion,
    score_appliances,
    summarize,
    total_power_residual,
)

# 3.1절의 기기별 정격 스케일 (양성 창 p90). 손실 정규화와 FA_rel 계산에 쓴다.
#
# **데이터가 바뀌면 다시 재야 한다.** 2026-08-21 minipc_3/oven_2/beam_projector_2 를
# 추가한 뒤 1,500창으로 재측정한 결과, 미니PC 만 17.6 -> 21.7 (+23%) 로 움직였고
# 나머지는 전부 ±5% 안이라 측정 잡음과 구분되지 않았다. 그래서 미니PC 만 갱신한다.
# 미니PC 가 움직인 것은 minipc_3 이 24~34W 대역을 0.02% -> 5.65% 로 채웠기 때문이다
# (0.6절이 지적한 실측 28.4W vs 합성 17.6W 격차). 21.7 은 실측의 76% 로, 62% 에서 올랐다.
S_I = {
    "electiric_kettle": 1533.0, "oven": 1209.0, "hair_dryer": 967.0,
    "air_conditioner": 775.0, "hotplate": 548.0, "laptop_charger": 66.1,
    "beam_projector": 50.6, "fan": 38.3, "minipc": 21.7,
}
LOW_LOAD = ("beam_projector", "laptop_charger", "fan", "minipc")

# Phase 3·시각화가 "이것을 이겼는가" 를 찍을 때 쓰는 기준선.
# **상수로 박아 두면 안 된다.** 12.7절대로 `--seed` 를 줘도 워커 시드에 PID 가 섞여
# 재학습마다 MAE 가 0.02W 움직이고, 무엇보다 12.9.1절의 지표 수정 전에 박아 둔
# 저항3종 0.968 은 **결함 지표 값**이라 CNN 의 수정 지표 값과 나란히 놓으면 비교가
# 통째로 틀린다. 그래서 파일에서 읽고, 못 읽을 때만 아래로 떨어진다.
BASELINE_FALLBACK = {"mae": 1.45, "f1": 0.951, "resistive_acc": 0.990, "resid_abs": 13.25}


def baseline_reference(path="results/baseline_gbm.json") -> dict:
    """Phase 1 GBM 기준선을 읽는다. 결함 지표로 잰 파일이면 쓰지 않는다."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cm = d.get("resistive_confusion") or {}
        if float(cm.get("min_true_w", 0)) <= 0:      # 12.9.1절 이전 파일
            return dict(BASELINE_FALLBACK)
        return {"mae": float(d["summary"]["mae_w_mean"]),
                "f1": float(d["summary"]["f1_mean"]),
                "resistive_acc": float(cm["accuracy"]),
                "resid_abs": float(d["total_power_residual"]["mean_abs_w"])}
    except (OSError, KeyError, TypeError, ValueError):
        return dict(BASELINE_FALLBACK)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 1 특징 기반 baseline")
    ap.add_argument("--windows", type=int, default=150_000, help="학습 창 수")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--out", default="results")
    ap.add_argument("--importance", action="store_true", help="순열 중요도 계산 (느림)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    print("=" * 78)
    print("[Phase 1] 특징 기반 baseline — HistGradientBoosting (분해 과제)")
    print("=" * 78)

    hs = load_holdout()
    print(f"평가 셋: 홀드아웃 {len(hs):,}창 (원본 뒤 {hs.meta['holdout_frac']:.0%})"
          f" | sha {hs.meta['content_sha256']} | 타깃 {hs.meta['target_index']}")

    print(f"\n[1/4] 학습 창 생성 — 앞 80% 구간, 워커 {a.workers}개")
    F, yp, yo, apps = build_training_set(
        n_windows=a.windows, seed=a.seed, n_workers=a.workers,
        window_cycles=hs.meta["window_cycles"])
    if apps != hs.appliances:
        raise RuntimeError(f"가전 순서가 다릅니다: {apps} vs {hs.appliances}")

    print(f"\n[2/4] 학습 — 회귀 9개 + 분류 9개, 특징 {F.shape[1]}개")
    model = train(F, yp, yo, apps, max_iter=a.max_iter, random_state=a.seed)

    print(f"\n[3/4] 홀드아웃 평가")
    Fh = extract(hs.X, target_index=hs.meta["target_index"])
    p_pred, on_prob = model.predict(Fh)
    scores = score_appliances(hs.y_power, p_pred, apps, S_I,
                              on_true=hs.y_on.astype(bool), on_pred=on_prob > 0.5)
    print(format_table(scores))

    summ = summarize(scores, low_load=LOW_LOAD)
    cm = resistive_confusion(hs.y_power, p_pred, apps)
    resid = total_power_residual(p_pred, hs.p_observed, p_noise=hs.p_noise)

    print(f"\n  기기 평균 MAE {summ['mae_w_mean']:.2f}W | F1 평균 {summ['f1_mean']:.3f}"
          f" | 최악 F1 {summ['worst_f1'][0]:.3f} ({summ['worst_f1'][1]})")
    if cm:
        print(f"  저항3종 혼동 정확도 {cm['accuracy']:.3f} (단독 창 {cm['n_windows']:,}개)  "
              f"{cm['labels']}")
        for name, row in zip(cm["labels"], cm["matrix"]):
            print(f"    참={name:18s} 예측 {row}")
    print(f"\n  [4.3절 목표] 저부하 FA_rel(고부하 동시) < 0.15 : {summ['fa_target_pass']} 통과")
    for k, v in summ["low_load_fa_high_rel"].items():
        mark = "OK" if v == v and v < 0.15 else "초과"
        print(f"    {k:18s} {v:6.3f}  {mark}")
    print(f"  [0.6절 대응] 저부하 켜졌을 때 편향 (음수=과소예측)")
    for k, v in summ["low_load_bias_on_w"].items():
        print(f"    {k:18s} {v:+7.2f}W   nMAE {summ['low_load_nmae_on'][k]:.3f}   RE {summ['low_load_re_on'][k]:.3f}")
    print(f"\n  총전력 잔차: 평균 {resid['mean_w']:+.2f}W | 절대 평균 {resid['mean_abs_w']:.2f}W"
          f" | p95 {resid['p95_abs_w']:.2f}W")

    imp = {}
    if a.importance:
        print(f"\n[4/4] 특징 중요도 (순열, MAE 증가량)")
        imp = permutation_importance_fast(model, Fh, hs.y_power)
        for app, rows in imp.items():
            print(f"  {app:18s} " + "  ".join(f"{n}(+{g:.2f})" for n, g in rows[:4]))
    else:
        print(f"\n[4/4] 특징 중요도 생략 (--importance 로 켤 수 있다)")

    payload = {
        "phase": "1-baseline-gbm",
        "config": {**model.config, "n_features": int(F.shape[1]), "seed": a.seed},
        "holdout": {k: hs.meta[k] for k in
                    ("n_windows", "content_sha256", "target_index", "holdout_frac", "seed")},
        "per_appliance": [s.as_row() for s in scores],
        "summary": summ, "resistive_confusion": cm, "total_power_residual": resid,
        "feature_importance": {k: [[n, float(g)] for n, g in v] for k, v in imp.items()},
        "elapsed_s": round(time.time() - t_all, 1),
    }
    (out / "baseline_gbm.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    model.save(out / "baseline_gbm.pkl")

    print(f"\n{'=' * 78}")
    print(f"완료 {time.time() - t_all:.1f}s | 결과 {(out / 'baseline_gbm.json').resolve()}")
    print(f"{'=' * 78}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
