"""
레시피 믹스 후보의 동시성과 2차 배경을 미리 잰다 (설계 문서 12.67절)
======================================================================
12.66 이 격차를 닫았다 — 합성 창이 실측보다 한산해서(평균 1.41 vs 2.33,
빈 창 28% vs 5%) 2차 배경이 2.3배 조용하고, 그래서 |I2| 판별 신호 6.1 mA 가
합성에서만 살아남는다. 처방은 동시성을 올리는 것이다.

**믹스를 바꾸면 유령 전력이 돌아올 위험이 있다** — `standby_only` 와
`unplugged_baseline` 은 저부하 오탐 방지용이다 (0.3절). 그래서 학습 45분을
쓰기 전에 후보 믹스가 실제로 목표 동시성에 닿는지 여기서 먼저 본다.

    python -m src.run_recipe_mix_probe --preset half
    python -m src.run_recipe_mix_probe --mix standby_only=0.08 high_low_mixed=0.21

목표 (12.66.2, 12.66.3):
    실측    평균 2.33   빈 창  5%   |ΔI2| p95 12.26 mA
    현재    평균 1.41   빈 창 28%   |ΔI2| p95  5.42 mA
    절반    평균 1.87   빈 창 16%   |ΔI2| p95  8.8 mA  <- 이번 목표
"""
from typing import Dict
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.synthesis.dataset import DEFAULT_RECIPE_MIX, NILMBatchGenerator
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

SMPS = ("beam_projector", "laptop_charger", "minipc")

#: 12.66.5 가 지목한 이동. 빈 창을 만드는 셋에서 덜어 동시 부하 쪽으로 옮긴다.
PRESETS: Dict[str, Dict[str, float]] = {
    "half": {                        # 실측까지의 절반
        "random_realistic": 0.23,
        "random_uniform": 0.14,
        "standby_only": 0.08,
        "low_load_among_standby": 0.16,
        "high_power_resistive": 0.10,
        "high_low_mixed": 0.21,
        "resistive_overlap": 0.05,
        "unplugged_baseline": 0.03,
    },
    "full": {                        # 실측 수준을 겨냥
        "random_realistic": 0.28,
        "random_uniform": 0.14,
        "standby_only": 0.04,
        "low_load_among_standby": 0.12,
        "high_power_resistive": 0.10,
        "high_low_mixed": 0.26,
        "resistive_overlap": 0.05,
        "unplugged_baseline": 0.01,
    },
}


def measure(mix, n_windows=400, window_cycles=3600, seed=0, half=5.0, guard=1.0):
    """동시성과 |ΔI2| 널을 `audit_synth` 와 같은 기하로 잰다."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", holdout_frac=0.2)
    syn = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    gen = NILMBatchGenerator(segment_pool=pool, window_size_cycles=window_cycles,
                            recipe_mix=mix, synthesizer=syn, compute_gt_harmonics=False)
    nulls, n_active = [], []
    for _ in range(n_windows):
        r, _ = gen._synthesize_window()
        I = np.asarray(r.harmonics_complex)
        n = len(I)
        t = np.arange(n) / 60.0
        n_active.append(len(r.active_appliances))
        tt = []
        for app in SMPS:
            on = np.asarray(r.gt_is_on.get(app, np.zeros(n, np.int8)))
            tt.extend(np.flatnonzero(np.diff(on) != 0) / 60.0)
        tt = np.array(tt)
        cand = [c for c in np.arange(half + 7, t[-1] - half - 7, 2.0)
                if len(tt) == 0 or np.min(np.abs(tt - c)) > 12]
        if not cand:
            continue
        for c in rng.choice(cand, min(2, len(cand)), replace=False):
            pre = (t >= c - half) & (t <= c - guard)
            post = (t >= c + guard) & (t <= c + half)
            if pre.sum() < 30 or post.sum() < 30:
                continue
            nulls.append(np.abs((np.median(I[post].real, 0) + 1j * np.median(I[post].imag, 0))
                                - (np.median(I[pre].real, 0) + 1j * np.median(I[pre].imag, 0))))
    na = np.array(n_active)
    nl = np.array(nulls)
    return {"mean_active": float(na.mean()), "median_active": float(np.median(na)),
            "empty_share": float((na == 0).mean()), "n_null": len(nl),
            "i2_med_ma": float(np.median(nl[:, 1]) * 1000),
            "i2_p95_ma": float(np.percentile(nl[:, 1], 95) * 1000)}


def main() -> int:
    ap = argparse.ArgumentParser(description="레시피 믹스 후보를 미리 잰다")
    ap.add_argument("--preset", nargs="*", default=["half"], choices=list(PRESETS) + ["none"])
    ap.add_argument("--mix", nargs="*", default=None, metavar="KEY=VAL")
    ap.add_argument("--windows", type=int, default=400)
    a = ap.parse_args()

    cands = {"현재 (DEFAULT_RECIPE_MIX)": DEFAULT_RECIPE_MIX}
    for name in a.preset:
        if name != "none":
            cands[f"프리셋 {name}"] = PRESETS[name]
    if a.mix:
        m = dict(DEFAULT_RECIPE_MIX)
        for kv in a.mix:
            k, v = kv.split("=")
            m[k] = float(v)
        cands["직접 지정"] = m

    print("=" * 92)
    print("[레시피 믹스 후보] 동시성과 2차 배경 — 학습 전에 본다 (12.67절)")
    print("=" * 92)
    print(f"  {'믹스':28s}{'평균 활성':>10s}{'빈 창':>9s}{'|ΔI2| 중앙':>13s}{'|ΔI2| p95':>12s}")
    for name, mix in cands.items():
        tot = sum(mix.values())
        if abs(tot - 1.0) > 1e-6:
            print(f"  {name:28s}  ⚠ 합이 {tot:.3f} 입니다 — 건너뜁니다")
            continue
        r = measure(mix, n_windows=a.windows)
        print(f"  {name:28s}{r['mean_active']:>10.2f}{100*r['empty_share']:>8.0f}%"
              f"{r['i2_med_ma']:>11.2f}mA{r['i2_p95_ma']:>10.2f}mA")
    print(f"  {'실측 test_5/6/7':28s}{2.33:>10.2f}{5:>8.0f}%{1.21:>11.2f}mA{12.26:>10.2f}mA")
    print("\n  판별 신호 6.1 mA 가 널 p95 아래로 내려가야 |I2| 단서가 죽는다 (12.66.4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
