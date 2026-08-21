"""
고정 합성 홀드아웃 평가 셋 생성
================================
각 원본 녹화의 뒤 20% 구간에서만 평가 창을 만든다. 학습(앞 80%)에서 본 적 없는
파형이라 "합성 테스트 성능" 이 의미를 갖는다.

# 기본 (8,000창 x 10초, 약 0.65GB)
python -m src.run_build_holdout

# 크기 조정
python -m src.run_build_holdout --windows 20000

# 기존 셋 확인만
python -m src.run_build_holdout --inspect
"""
from pathlib import Path
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from src.evaluation.holdout import DEFAULT_DIR, DEFAULT_N_WINDOWS, DEFAULT_SEED, build_holdout, load_holdout


def inspect(d: str) -> int:
    hs = load_holdout(d)
    m = hs.meta
    print("\n" + "=" * 74)
    print(f"[홀드아웃] {Path(d).resolve()}")
    print("=" * 74)
    print(f"  창 {len(hs):,}개 x {m['window_cycles']/60:.0f}초 | 타깃 시점 {m['target_index']}"
          f" | seed {m['seed']} | sha {m['content_sha256']}")
    print(f"  원본의 뒤 {m['holdout_frac']:.0%} 구간만 사용")
    print(f"\n  {'가전':18s}{'홀드아웃 분량':>14s}{'양성 라벨률':>14s}")
    for a in hs.appliances:
        print(f"  {a:18s}{m['pool_holdout_minutes'].get(a, 0):>12.1f}분"
              f"{100*m['positive_rate'][a]:>13.1f}%")
    print(f"\n  레시피 구성:")
    for r, c in sorted(m["recipe_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {r:26s}{c:>7,d}  ({100*c/len(hs):5.1f}%)")
    print(f"\n  형태: X {hs.X.shape} {hs.X.dtype} | y_power {hs.y_power.shape}"
          f" | {m['bytes']/1e9:.2f} GB")
    thin = [a for a in hs.appliances if m["positive_rate"][a] * len(hs) < 200]
    if thin:
        print(f"\n  ⚠ 양성 창이 200개 미만인 가전: {', '.join(thin)}")
        print(f"    이 기기들의 지표는 표본 오차가 크다. --windows 를 늘리십시오.")
    print("=" * 74 + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="고정 합성 홀드아웃 평가 셋 생성")
    ap.add_argument("--out", default=str(DEFAULT_DIR))
    ap.add_argument("--windows", type=int, default=DEFAULT_N_WINDOWS)
    ap.add_argument("--window-cycles", type=int, default=600)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    if a.inspect:
        if not (Path(a.out) / "meta.json").exists():
            print(f"홀드아웃 셋이 없습니다: {Path(a.out).resolve()}")
            return 1
        return inspect(a.out)

    print("\n" + "=" * 74)
    print("[NILM AI] 고정 합성 홀드아웃 평가 셋 생성")
    print("=" * 74)
    build_holdout(out_dir=a.out, n_windows=a.windows, window_cycles=a.window_cycles,
                  holdout_frac=a.holdout_frac, seed=a.seed)
    return inspect(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
