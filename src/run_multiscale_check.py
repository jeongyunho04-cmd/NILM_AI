"""
2갈래(multi-scale) 검증 + 잡음 바닥 측정
=========================================
12.8절 스윕에서 "60초 창이 10초보다 낫다" 는 **측정**했지만,
"그 이득을 다운샘플된 광역 갈래로도 얻는다" 는 **특징 목록을 보고 추론**한 것이었다.
여기서 직접 잰다.

[구성]
  A. 60초 전체 60Hz            스윕에서 이긴 구성
  B. 60초, 뒤 10초만 60Hz       ← 설계 문서 1.1절의 2갈래가 실제로 보는 정보
     + 앞 50초는 2Hz            (세밀 10초 @60Hz + 광역 60초 @2Hz)
  C. 10초 전체 60Hz            광역 갈래가 아예 없을 때

B 가 A 에 가까우면 2갈래 설계가 검증된다. B 가 C 에 가까우면 광역 갈래를
다운샘플하면 안 된다는 뜻이고, 60초를 통째로 60Hz 로 넣어야 한다.

[잡음 바닥]
스윕은 구성마다 평가창을 새로 만들었으므로 평가셋이 서로 다르다.
같은 구성을 평가 시드만 바꿔 여러 번 재서, 관측된 차이가 잡음보다 큰지 확인한다.

python -m src.run_multiscale_check
"""
from pathlib import Path
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.baseline.train import build_training_set, train
from src.evaluation.metrics import resistive_confusion, score_appliances, summarize
from src.run_baseline import LOW_LOAD, S_I

# (이름, 창 사이클, 다운샘플 경계). degrade=0 이면 전 구간 60Hz.
CONFIGS = [
    ("A. 60초 전체 60Hz",            3600, 0),
    ("B. 60초 (뒤10초 60Hz+2Hz)",    3600, 600),
    ("C. 10초 전체 60Hz",             600, 0),
]
EVAL_SEEDS = (987_654, 111_222, 333_444)


def evaluate(model, F, yp, yo, apps) -> dict:
    pred, prob = model.predict(F)
    sc = score_appliances(yp, pred, apps, S_I, on_true=yo.astype(bool), on_pred=prob > 0.5)
    summ = summarize(sc, low_load=LOW_LOAD)
    cm = resistive_confusion(yp, pred, apps)
    by = {x.appliance: x for x in sc}
    return {
        "mae": summ["mae_w_mean"], "f1": summ["f1_mean"],
        "resistive_acc": cm["accuracy"] if cm else float("nan"),
        "oven_err_pct": 100 * cm["matrix"][1][0] / max(sum(cm["matrix"][1]), 1) if cm else float("nan"),
        "oven_f1": by["oven"].f1, "oven_bias": by["oven"].bias_on_w,
        "minipc_f1": by["minipc"].f1, "hotplate_f1": by["hotplate"].f1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="2갈래 검증 + 잡음 바닥")
    ap.add_argument("--windows", type=int, default=80_000)
    ap.add_argument("--eval", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--out", default="results/multiscale_check.json")
    a = ap.parse_args()

    print("=" * 86)
    print("[2갈래 검증] 60초의 이득이 다운샘플된 광역 갈래로도 얻어지는가")
    print("=" * 86)

    rows = []
    for name, win, deg in CONFIGS:
        print(f"\n  {name}  (창 {win}, 다운샘플 경계 {deg})")
        t0 = time.time()
        Ftr, ytr, otr, apps = build_training_set(
            n_windows=a.windows, window_cycles=win, time_split="train",
            seed=0, n_workers=a.workers, target_lookahead_cycles=60,
            degrade_fine_cycles=deg)
        model = train(Ftr, ytr, otr, apps, max_iter=a.max_iter, random_state=0, verbose=False)

        reps = []
        for es in EVAL_SEEDS:
            Fe, ype, yoe, _ = build_training_set(
                n_windows=a.eval, window_cycles=win, time_split="holdout",
                seed=es, n_workers=a.workers, target_lookahead_cycles=60,
                degrade_fine_cycles=deg)
            reps.append(evaluate(model, Fe, ype, yoe, apps))
        agg = {k: (float(np.mean([r[k] for r in reps])), float(np.std([r[k] for r in reps])))
               for k in reps[0]}
        rows.append({"name": name, "window": win, "degrade": deg,
                     "agg": agg, "reps": reps, "elapsed_s": round(time.time() - t0, 1)})
        print(f"    MAE {agg['mae'][0]:.3f}±{agg['mae'][1]:.3f} | "
              f"F1 {agg['f1'][0]:.4f}±{agg['f1'][1]:.4f} | "
              f"오븐F1 {agg['oven_f1'][0]:.3f} | 오븐→포트 {agg['oven_err_pct'][0]:.1f}%")

    print("\n" + "=" * 86)
    print(f"{'구성':30s}{'MAE':>16s}{'F1평균':>16s}{'오븐 F1':>14s}{'오븐→포트':>12s}")
    print("-" * 86)
    for r in rows:
        g = r["agg"]
        print(f"{r['name']:30s}{g['mae'][0]:>10.3f}±{g['mae'][1]:<5.3f}"
              f"{g['f1'][0]:>10.4f}±{g['f1'][1]:<5.4f}"
              f"{g['oven_f1'][0]:>9.3f}±{g['oven_f1'][1]:<4.3f}{g['oven_err_pct'][0]:>11.1f}%")

    A, B, C = rows[0]["agg"], rows[1]["agg"], rows[2]["agg"]
    noise = float(np.mean([r["agg"]["mae"][1] for r in rows]))
    print(f"\n[잡음 바닥] 평가 시드 3개의 MAE 표준편차 평균 = {noise:.3f}W")
    print(f"  -> 이보다 작은 차이는 해석하지 말 것")

    print(f"\n[핵심 비교]")
    dAB = B["mae"][0] - A["mae"][0]
    dCB = C["mae"][0] - B["mae"][0]
    print(f"  A(60초 전체) vs B(2갈래) : MAE {A['mae'][0]:.3f} -> {B['mae'][0]:.3f}  "
          f"차이 {dAB:+.3f}W  ({'잡음 이내' if abs(dAB) < 2*noise else '유의미'})")
    print(f"  B(2갈래) vs C(10초만)    : MAE {B['mae'][0]:.3f} -> {C['mae'][0]:.3f}  "
          f"차이 {dCB:+.3f}W  ({'잡음 이내' if abs(dCB) < 2*noise else '유의미'})")
    print(f"\n  오븐 F1     A {A['oven_f1'][0]:.3f} / B {B['oven_f1'][0]:.3f} / C {C['oven_f1'][0]:.3f}")
    print(f"  오븐→포트   A {A['oven_err_pct'][0]:.1f}% / B {B['oven_err_pct'][0]:.1f}% / C {C['oven_err_pct'][0]:.1f}%")
    print(f"  미니PC F1   A {A['minipc_f1'][0]:.3f} / B {B['minipc_f1'][0]:.3f} / C {C['minipc_f1'][0]:.3f}")

    # 판정은 MAE 로 하지 않는다. MAE 는 고전력 기기(포트 1533W / 오븐 1209W)의
    # 절대 오차에 지배되어 평가셋 표본 잡음이 크다 - 이 실험에서 sigma 0.075W 로
    # 구성 간 차이(0.1W)와 같은 크기다. F1 과 오븐 혼동은 sigma 대비 5~11배라
    # 신호가 확실하다.
    f1_sd = float(np.mean([r["agg"]["f1"][1] for r in rows]))
    gain_total = A["f1"][0] - C["f1"][0]          # 60초가 10초보다 얻는 것
    gain_kept = B["f1"][0] - C["f1"][0]           # 2갈래가 그중 지켜낸 것
    kept = gain_kept / gain_total if abs(gain_total) > 1e-9 else float("nan")
    verdict = (
        f"2갈래 검증됨 - 60초 이득의 {100*kept:.0f}% 를 다운샘플로 지켜낸다"
        if gain_total > 3 * f1_sd and kept > 0.7 else
        "2갈래로는 60초 이득이 유지되지 않는다 - 광역 갈래 해상도를 올릴 것"
        if gain_total > 3 * f1_sd else
        "60초 자체의 이득이 잡음 이내다 - 10초로 충분하다")
    print(f"  F1 기준: 60초 이득 {gain_total:+.4f} (sigma {f1_sd:.4f}), "
          f"2갈래가 지켜낸 몫 {100*kept:.0f}%")
    print(f"\n[판정] {verdict}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"configs": rows, "noise_floor_mae": noise, "verdict": verdict,
         "eval_seeds": list(EVAL_SEEDS)},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n결과: {Path(a.out).resolve()}")
    print("=" * 86 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
