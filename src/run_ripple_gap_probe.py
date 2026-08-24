"""
고부하 위에서 핫플 리플 대비가 남는가 — 실측 vs 합성 (설계 문서 12.49절)
============================================================================
12.48 이 확정했다. **핫플 검출은 릴레이 리플 지문으로 한다** (오븐 잔여의 뺄셈이
아니다). 그러면 실측 `>=1300W` 에서 핫플 재현율이 무너지는 것(12.41.6)의 후보는
"오븐 추정" 이 아니라 **그 구간에서 리플 대비가 모자란 것** 이다.

12.9.13 ② 가 이미 그 방향을 쟀다:

    474W 핫플 리플이 ch30(asinh(P/100))에서 만드는 변화
      바닥에서 2.260,  오븐(1140W) 위 0.347   <- 1/6.5

여기서는 **실측과 합성을 나란히** 잰다. 모델을 안 돌린다 — 리플은 P 신호의 성질이라
원신호에서 바로 나온다. GPU 도 필요 없다.

    python -m src.run_ripple_gap_probe

[무엇을 비교하나 — 세 갈래]
```
A 실측 <1300W   핫플 ON, 오븐 히터 꺼짐      모델이 잘 맞히는 곳 (F1 0.96~0.99)
B 실측 >=1300W  핫플 ON + 오븐 히터 통전     무너지는 곳 (재현율 0.19~0.71)
C 합성 >=1300W  같은 구성                    학습에서 보는 것
```

[예측 — 돌리기 전에 적는다]
가설이 맞으면 리플 진폭이 **A > C > B** 여야 한다. 곧 합성 고부하 창은 실측보다
리플 대비를 더 남기고, 모델은 그 대비를 보고 배운 뒤 실측에서 못 찾는다.

- **B ~ C 로 비슷하면 가설이 틀린 것이다** — 실측과 합성의 리플이 같은데 실측만
  무너진다면 원인은 리플 대비가 아니다.
- **A > B 인데 C 도 B 만큼 낮으면** 문제는 합성이 아니라 물리다 (고부하 위에서는
  원래 안 보인다). 그러면 처방은 채널·합성이 아니라 **입력 표현**이다.

[두 층을 갈라 잰다]
    ① 물리:   r = P − 이동평균(P, ±0.5초)   의 진폭 (W)      <- 신호에 있는 것
    ② 표현: ch36 = asinh(r / RIPPLE_SCALE)  의 진폭          <- 모델이 받는 것
①은 같은데 ②만 다르면 스케일(`RIPPLE_SCALE`) 문제이고, ①부터 다르면 신호 문제다.
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.evaluation.real_events import load_events
from src.model.inputs import RIPPLE_HALF_SHORT, RIPPLE_SCALE, target_index

from src.model.realdata import DEFAULT_DIR

NPZ_DIR = Path(DEFAULT_DIR)   # 복합 실측은 composite_eval 에 있다


def movavg(a: np.ndarray, half: int) -> np.ndarray:
    k = 2 * half + 1
    pad = np.pad(a.astype(np.float64), (half, half), mode="edge")
    c = np.concatenate([[0.0], np.cumsum(pad)])
    return ((c[k:] - c[:-k]) / k).astype(np.float32)


def stats(r: np.ndarray, label: str, n_ctx: str) -> dict:
    """리플 진폭 요약. 표준편차와 p90-p10 폭 둘 다 본다."""
    ch = np.arcsinh(r / RIPPLE_SCALE)
    return {"label": label, "n": int(len(r)), "ctx": n_ctx,
            "p_std_w": float(r.std()),
            "p_span_w": float(np.percentile(r, 90) - np.percentile(r, 10)),
            "ch_std": float(ch.std()),
            "ch_span": float(np.percentile(ch, 90) - np.percentile(ch, 10))}


def mask_from(pairs, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for s, e in pairs or []:
        m[int(float(s) * 60):int(float(e) * 60)] = True
    return m


def real_groups(ev: dict) -> List[dict]:
    out = []
    for stem in ("test_4", "test_5", "test_6"):
        f = NPZ_DIR / f"{stem}.npz"
        if not f.exists():
            continue
        raw = np.load(f)
        p = np.asarray(raw["power_features"], np.float32)[:, 0]
        n = len(p)
        r = p - movavg(p, RIPPLE_HALF_SHORT)
        iv = ev[stem]["intervals"]
        hp = mask_from(iv.get("hotplate", {}).get("on"), n)
        ov = mask_from(iv.get("oven", {}).get("_heater_pulses"), n)
        hi = p >= 1300.0
        out.append({"stem": stem,
                    "A": r[hp & ~ov & ~hi], "B": r[hp & ov & hi],
                    "p_A": p[hp & ~ov & ~hi], "p_B": p[hp & ov & hi]})
    return out


def synth_group(n_win: int, window_cycles: int, seed: int) -> tuple:
    """합성: 오븐·핫플이 타깃에 동시 통전하는 창의 타깃 근방 리플."""
    from src.run_subtraction_probe import make_windows
    samples = make_windows(n_win, window_cycles, seed)
    ti = target_index(window_cycles)
    half = 120                      # 타깃 ±2초 (릴레이 주기 1회)
    R, P = [], []
    for s in samples:
        p = np.asarray(s.power_features, np.float32)[:, 0]
        r = p - movavg(p, RIPPLE_HALF_SHORT)
        sl = slice(max(0, ti - half), min(len(p), ti + half))
        keep = p[sl] >= 1300.0
        R.append(r[sl][keep]); P.append(p[sl][keep])
    return np.concatenate(R), np.concatenate(P)


def main() -> int:
    ap = argparse.ArgumentParser(description="리플 대비 격차 (12.49절)")
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--window-cycles", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ripple_gap_probe.json")
    a = ap.parse_args()

    ev = load_events()
    print("=" * 96)
    print("[리플 대비 격차]  핫플 릴레이 리플이 고부하 위에서 얼마나 남는가")
    print(f"  r = P − 이동평균(P, ±{RIPPLE_HALF_SHORT / 60:.1f}초),  "
          f"ch36 = asinh(r / {RIPPLE_SCALE:.0f})")
    print("=" * 96)

    rg = real_groups(ev)
    rows: List[dict] = []
    A = np.concatenate([g["A"] for g in rg]); pA = np.concatenate([g["p_A"] for g in rg])
    B = np.concatenate([g["B"] for g in rg]); pB = np.concatenate([g["p_B"] for g in rg])
    print(f"  합성 창 {a.windows}개 만드는 중...")
    C, pC = synth_group(a.windows, a.window_cycles, a.seed)

    rows.append(stats(A, "A 실측 <1300W (핫플 ON, 오븐 꺼짐)", f"P 중앙 {np.median(pA):.0f}W"))
    rows.append(stats(B, "B 실측 >=1300W (핫플 ON + 오븐 통전)", f"P 중앙 {np.median(pB):.0f}W"))
    rows.append(stats(C, "C 합성 >=1300W (같은 구성)", f"P 중앙 {np.median(pC):.0f}W"))

    print()
    print(f"  {'구간':<34s}{'표본':>9s}{'바탕':>12s}"
          f"{'① P리플 폭':>13s}{'② ch36 폭':>12s}{'ch36 표준편차':>14s}")
    print("  " + "-" * 94)
    for r in rows:
        print(f"  {r['label']:<34s}{r['n']:>9,d}{r['ctx']:>12s}"
              f"{r['p_span_w']:>12.1f}W{r['ch_span']:>12.3f}{r['ch_std']:>14.3f}")

    a_, b_, c_ = rows
    print()
    print(f"  A/B = {a_['ch_span'] / max(b_['ch_span'], 1e-9):.2f}배   "
          f"C/B = {c_['ch_span'] / max(b_['ch_span'], 1e-9):.2f}배   "
          f"(ch36 폭 기준)")
    print(f"  물리(① P리플)로는  A/B = {a_['p_span_w'] / max(b_['p_span_w'], 1e-9):.2f}배   "
          f"C/B = {c_['p_span_w'] / max(b_['p_span_w'], 1e-9):.2f}배")
    print()
    if c_["ch_span"] >= 1.3 * b_["ch_span"]:
        print("  판정: **C > B — 합성이 실측보다 리플을 더 남긴다.** 가설이 선다.")
    elif c_["ch_span"] <= 1.15 * b_["ch_span"]:
        print("  판정: **C ~ B — 합성과 실측의 리플이 비슷하다.** 가설이 틀렸다. "
              "원인은 리플 대비가 아니다.")
    else:
        print("  판정: 애매하다 (C/B 가 1.15~1.30). 표본을 늘려야 한다.")

    per_file = {}
    for g in rg:
        if len(g["B"]) > 20:
            per_file[g["stem"]] = {
                "A": stats(g["A"], "A", "") if len(g["A"]) > 20 else None,
                "B": stats(g["B"], "B", "")}
    print()
    print(f"  파일별 (② ch36 폭)   {'A <1300':>10s}{'B >=1300':>10s}{'A/B':>8s}")
    for stem, v in per_file.items():
        if v["A"]:
            print(f"    {stem:<18s}{v['A']['ch_span']:>10.3f}{v['B']['ch_span']:>10.3f}"
                  f"{v['A']['ch_span'] / max(v['B']['ch_span'], 1e-9):>8.2f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"groups": rows, "per_file": per_file},
                                      ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print()
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
