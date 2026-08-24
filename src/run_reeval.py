"""
재평가 — 12.9.1절 지표 수정을 저장된 모델에 소급 적용
======================================================
`results/*.json` 은 **지표를 고치기 전에** 생성됐다. 그 안의 저항3종 정확도는
오븐의 팬/조명(16W) 구간까지 세고 있어 난수에 가깝다 (12.9.1절).

여기서는 **재학습하지 않는다.** 저장된 가중치(`.pt` / `.pkl`)를 그대로 불러
같은 얼린 홀드아웃에 다시 채점만 한다. 그래야 지표 수정의 효과와 모델 차이가
섞이지 않는다.

    python -m src.run_reeval                    # 전부
    python -m src.run_reeval --models cnn_v2    # 하나만
    python -m src.run_reeval --dry-run          # 파일을 쓰지 않고 출력만

[홀드아웃이 모델마다 다르다 — 섞으면 안 된다]
GBM 은 10초 창(`processed_data/holdout`, sha afdf50...)에서 학습·평가됐고
CNN 은 60초 창(`processed_data/holdout60`, sha 721647...)이다.
`features.extract` 가 창 전체(`slice(0, w)`) 통계를 쓰므로 600사이클로 학습한
GBM 을 3600사이클 창에 그대로 먹이면 특징 분포가 어긋난다. **각자 자기 셋에서
채점한다.** 두 셋은 같은 시드·같은 레시피 혼합이지만 동일하지 않다.

[추가로 얻는 것]
`state_breakdown` (12.9.4절) 을 함께 계산해 JSON 에 남긴다. 기기 단위 집계가
묻어 버리는 저전력 부속 상태(오븐 팬/조명, 에어컨 송풍)의 실패가 여기서만 보인다.
"""
from pathlib import Path
from typing import Optional
import argparse
import datetime
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np

from src.evaluation import (
    format_state_table, format_table, load_holdout, resistive_confusion,
    score_appliances, state_breakdown, summarize, total_power_residual,
)
from src.run_baseline import LOW_LOAD, S_I

METRIC_VERSION = "12.9.1"          # resistive_confusion(min_true_w=300) + state_breakdown
GBM_HOLDOUT = "processed_data/holdout"
CNN_HOLDOUT = "processed_data/holdout60"


def _score(hs, pred: np.ndarray, on_pred: np.ndarray) -> dict:
    """수정된 지표 일습. 모델 종류와 무관하게 같은 함수를 쓴다."""
    sc = score_appliances(hs.y_power, pred, hs.appliances, S_I,
                          on_true=hs.y_on.astype(bool), on_pred=on_pred)
    summ = summarize(sc, low_load=LOW_LOAD)
    cm = resistive_confusion(hs.y_power, pred, hs.appliances)
    resid = total_power_residual(pred, hs.p_observed, p_noise=hs.p_noise)
    states = state_breakdown(hs.y_power, pred, hs.y_on, hs.y_state, hs.appliances)
    return {"per_appliance": [s.as_row() for s in sc], "summary": summ,
            "resistive_confusion": cm, "total_power_residual": resid,
            "state_breakdown": states, "_scores": sc}


def _f1_of(rows, app: str) -> Optional[float]:
    for r in rows:
        if r.get("appliance") == app:
            return r.get("f1")
    return None


def _old_snapshot(doc: dict) -> dict:
    """덮어쓰기 전의 값을 기록해 둔다. 무엇이 얼마나 바뀌었는지 남아야 한다."""
    cm = doc.get("resistive_confusion") or {}
    return {
        "mae_w_mean": doc.get("summary", {}).get("mae_w_mean"),
        "f1_mean": doc.get("summary", {}).get("f1_mean"),
        "resistive_acc": cm.get("accuracy"),
        "resistive_n_windows": cm.get("n_windows"),
        "resid_abs_w": doc.get("total_power_residual", {}).get("mean_abs_w"),
        "hotplate_f1": _f1_of(doc.get("per_appliance", []), "hotplate"),
    }


def predict_gbm(hs) -> tuple:
    from src.baseline.features import extract
    from src.baseline.train import BaselineModel
    model = BaselineModel.load("results/baseline_gbm.pkl")
    F = extract(np.asarray(hs.X), target_index=hs.meta["target_index"])
    p, on_prob = model.predict(F)
    return p, on_prob > 0.5


def predict_cnn(tag: str, hs, dev: str, prep) -> tuple:
    """`prep` 은 홀드아웃 입력 변환 결과다. 모델마다 다시 만들면 3.8GB memmap 을
    매번 훑어 24초씩 낭비한다 — 한 번 만들어 모든 체크포인트가 공유한다."""
    import torch

    from src.model.inputs import LEGACY_FINE_CHANNELS
    from src.run_gate_check import assert_target_config
    from src.model.net import NILMNet, appliance_state_counts
    from src.run_train_cnn import evaluate

    ck = torch.load(f"results/{tag}.pt", map_location="cpu", weights_only=False)
    assert_target_config(ck, f"results/{tag}.pt")   # 12.45.3
    apps = ck.get("appliances", hs.appliances)
    if apps != hs.appliances:
        raise RuntimeError(f"가전 순서가 다릅니다: {apps} vs {hs.appliances}")
    # 프라이어 설정을 체크포인트에서 그대로 되살린다. 빠뜨리면 평가가 학습과
    # 다른 모델이 된다 (kappa 기본값이 0 이라 프라이어가 조용히 꺼진다).
    model = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                    wide_summary=ck.get("wide_summary", False),
                    periodicity=ck.get("periodicity", False),
                    fine_dropout=ck.get("fine_dropout", 0.0),
                    prior_kappa=ck.get("prior_kappa", 0.0),
                    prior_beta=ck.get("prior_beta", 0.5),
                    fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS)).to(dev)
    model.load_state_dict(ck["model"])       # 구조가 다르면 여기서 걸린다
    p, on_prob = evaluate(model, prep, dev)
    return p, on_prob > 0.5


def run_one(tag: str, hs, dev: str, out_dir: Path, dry: bool, prep=None) -> Optional[dict]:
    jp = out_dir / f"{tag}.json"
    doc = json.loads(jp.read_text(encoding="utf-8"))
    t0 = time.time()

    if tag == "baseline_gbm":
        pred, on_pred = predict_gbm(hs)
    else:
        pred, on_pred = predict_cnn(tag, hs, dev, prep)

    new = _score(hs, pred, on_pred)
    old = _old_snapshot(doc)
    cm = new["resistive_confusion"]

    print(f"\n{'=' * 84}\n[{tag}]  홀드아웃 {len(hs):,}창 | sha {hs.meta['content_sha256']}"
          f" | 창 {hs.meta['window_cycles']} | {time.time() - t0:.1f}s\n{'=' * 84}")
    print(format_table(new["_scores"]))

    summ, resid = new["summary"], new["total_power_residual"]
    print(f"\n  {'지표':22s}{'이전':>12s}{'재평가':>12s}")
    for lab, o, n in [("기기 평균 MAE (W)", old["mae_w_mean"], summ["mae_w_mean"]),
                      ("F1 평균", old["f1_mean"], summ["f1_mean"]),
                      ("저항3종 정확도", old["resistive_acc"], cm["accuracy"] if cm else float("nan")),
                      ("총전력 잔차 (W)", old["resid_abs_w"], resid["mean_abs_w"]),
                      ("핫플레이트 F1", old["hotplate_f1"], _f1_of(new["per_appliance"], "hotplate"))]:
        o = float("nan") if o is None else o
        n = float("nan") if n is None else n
        print(f"  {lab:22s}{o:>12.4f}{n:>12.4f}")
    if cm:
        print(f"\n  저항3종 (참전력 {cm['min_true_w']:.0f}W 이상 단독 창 {cm['n_windows']:,}개)"
              f"  {cm['labels']}")
        for name, row in zip(cm["labels"], cm["matrix"]):
            print(f"    참={name:18s} 예측 {row}")
        n_oven = sum(cm["matrix"][1])
        if n_oven:
            print(f"    오븐→포트 {cm['matrix'][1][0]}/{n_oven} "
                  f"({100 * cm['matrix'][1][0] / n_oven:.1f}%)")

    print(f"\n  [12.9.4] 동작 상태별 — 상대오차 50% 초과에 <<< 표시")
    print(format_state_table(new["state_breakdown"]))
    bad = [r for r in new["state_breakdown"] if r["median_rel_err"] > 0.5]
    print(f"  50% 초과 상태 {len(bad)}개" + (": " + ", ".join(
        f"{r['appliance']}/{r['state']}" for r in bad) if bad else ""))

    if not dry:
        doc.update({k: new[k] for k in
                    ("per_appliance", "summary", "resistive_confusion",
                     "total_power_residual", "state_breakdown")})
        doc["reeval"] = {
            "metric_version": METRIC_VERSION,
            "ran_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "holdout_dir": str(hs.meta.get("_dir", "")),
            "holdout_sha256": hs.meta["content_sha256"],
            "note": ("재학습 없음 - 저장된 가중치를 같은 홀드아웃에 다시 채점했다. "
                     "CNN 은 .pt 가 최고 F1 epoch 이라, 마지막 epoch 을 적던 이전 "
                     "summary 와 저항3종 외의 값도 조금 다르다"),
            "superseded": old,
        }
        jp.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=float),
                      encoding="utf-8")
        print(f"\n  갱신: {jp}")
    return {"tag": tag, "old": old, "new": {
        "mae_w_mean": summ["mae_w_mean"], "f1_mean": summ["f1_mean"],
        "resistive_acc": cm["accuracy"] if cm else None,
        "resid_abs_w": resid["mean_abs_w"],
        "hotplate_f1": _f1_of(new["per_appliance"], "hotplate"),
        "n_bad_states": len(bad)}}


def main() -> int:
    ap = argparse.ArgumentParser(description="저장된 모델을 수정된 지표로 재채점")
    ap.add_argument("--models", nargs="*", default=None,
                    help="기본: results/ 에 있는 것 전부 (baseline_gbm, cnn_*)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--dry-run", action="store_true", help="JSON 을 쓰지 않는다")
    a = ap.parse_args()

    out = Path(a.out)
    tags = a.models
    if not tags:
        tags = []
        if (out / "baseline_gbm.pkl").exists():
            tags.append("baseline_gbm")
        tags += sorted(p.stem for p in out.glob("cnn*.pt"))

    print("=" * 84)
    print(f"[재평가] 지표 {METRIC_VERSION} — resistive_confusion(min_true_w=300)"
          f" + state_breakdown")
    print(f"대상 {len(tags)}개: {', '.join(tags)}" + ("  (dry-run)" if a.dry_run else ""))
    print("=" * 84)

    import torch
    env_guard.verify_numerics()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    rows, skipped = [], []
    hs_cache, prep = {}, None
    for tag in tags:
        d = GBM_HOLDOUT if tag == "baseline_gbm" else CNN_HOLDOUT
        if d not in hs_cache:
            hs_cache[d] = load_holdout(d)
            hs_cache[d].meta["_dir"] = d
        if tag != "baseline_gbm" and prep is None:
            from src.run_train_cnn import prepare_holdout_inputs
            t0 = time.time()
            prep = prepare_holdout_inputs(hs_cache[d])
            print(f"\n홀드아웃 입력 변환 {prep[0].shape} + {prep[1].shape} "
                  f"({time.time() - t0:.0f}s) — 모든 체크포인트가 공유한다")
        try:
            r = run_one(tag, hs_cache[d], dev, out, a.dry_run, prep)
        except Exception as e:                       # 구조가 바뀐 옛 체크포인트
            print(f"\n[{tag}] 건너뜀 — {type(e).__name__}: {str(e)[:200]}")
            skipped.append((tag, f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        if r:
            rows.append(r)

    print("\n" + "=" * 84)
    print("요약 — 이전 -> 재평가")
    print("=" * 84)
    h = (f"{'모델':14s}{'MAE(W)':>16s}{'F1 평균':>16s}{'저항3종':>16s}"
         f"{'잔차(W)':>16s}{'핫플F1':>14s}")
    print(h)
    print("-" * len(h))
    for r in rows:
        o, n = r["old"], r["new"]

        def pair(k, fmt="{:.3f}"):
            ov = "—" if o.get(k) is None else fmt.format(o[k])
            nv = "—" if n.get(k) is None else fmt.format(n[k])
            return f"{ov}->{nv}"

        print(f"{r['tag']:14s}{pair('mae_w_mean'):>16s}{pair('f1_mean'):>16s}"
              f"{pair('resistive_acc'):>16s}{pair('resid_abs_w'):>16s}"
              f"{pair('hotplate_f1'):>14s}")
    for tag, why in skipped:
        print(f"{tag:14s}  건너뜀 — {why}")
    print("=" * 84 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
