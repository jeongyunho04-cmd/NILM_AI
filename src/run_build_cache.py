"""
학습용 윈도우 캐시 생성
========================
합성 시나리오를 미리 대량 생성해 디스크에 둔다. 학습 중에는 잘라 쓰기만 하므로
CPU 가 데이터 생성에 묶이지 않는다.

# 기본 (4000개 x 60초 = 신호 66.7시간, 약 3.4GB)
python -m src.run_build_cache

# 크기 조정
python -m src.run_build_cache --scenarios 8000 --seconds 60

# 확인만 (기존 캐시의 균형 점검)
python -m src.run_build_cache --inspect

* 출력: cache/train/ 아래 memmap 배열들과 meta.json
"""
from pathlib import Path
import argparse
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from src.synthesis.cache import WindowCache, build_cache


def inspect(cache_dir: str) -> None:
    cache = WindowCache(cache_dir)
    m = cache.meta
    print("\n" + "=" * 78)
    print(f"[캐시] {Path(cache_dir).resolve()}")
    print("=" * 78)
    print(f"  시나리오 {m['n_scenarios']:,}개 x {m['scenario_cycles']/60:.0f}초 "
          f"= 신호 {m['signal_hours']}시간 | {m['bytes']/1e9:.2f} GB")
    print(f"  창 {m['window_cycles']/60:.0f}초, 스트라이드 {m['stride_cycles']/60:.2f}초 "
          f"-> 부창 {m['n_windows']:,}개")
    print(f"\n  레시피 구성:")
    for r, c in sorted(m["recipe_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {r:26s}{c:>7,d}  ({100*c/m['n_scenarios']:5.1f}%)")

    uni = cache.class_share(weighted=False)
    wei = cache.class_share(weighted=True)
    print(f"\n  가전별 양성 라벨 비율 (창 중앙 시점)")
    print(f"    {'가전':18s}{'보정 없음':>12s}{'가중 보정':>12s}")
    for a in cache.appliances:
        print(f"    {a:18s}{100*uni[a]:>11.1f}%{100*wei[a]:>11.1f}%")
    u = np.array(list(uni.values())); w = np.array(list(wei.values()))
    print(f"    {'불균형(최다/최소)':18s}{u.max()/max(u.min(),1e-9):>10.1f}:1"
          f"{w.max()/max(w.min(),1e-9):>10.1f}:1")

    # ── 가중 추출로 잃는 다양성 ────────────────────────────────────────────
    ess, ratio = cache.effective_sample_size()
    print(f"\n  유효 표본 수 (Kish ESS)")
    print(f"    {'전체 부창':22s}{m['n_windows']:>12,d}")
    print(f"    {'ESS (균등 환산)':22s}{ess:>12,.0f}   ({100*ratio:.1f}%)")
    if cache.weights is not None:
        p = cache.weights.astype(np.float64); p = p / p.sum()
        print(f"    {'추출 확률 최대/최소':22s}{p.max()/max(p.min(), 1e-15):>11.1f}배")
    verdict = "양호" if ratio >= 0.4 else ("주의" if ratio >= 0.25 else "낮음 - 가중치 조정 필요")
    print(f"    {'판정':22s}{verdict:>12s}")
    # 재사용 횟수는 과적합 판단의 실질 지표다
    for epochs, per_epoch in [(50, 100_000)]:
        draws = epochs * per_epoch
        print(f"    {epochs} epoch x {per_epoch // 1000}k 샘플 = {draws:,} 추출 "
              f"-> 평균 {draws / max(ess, 1):.0f}회 재사용")
    print("    (ESS 는 가중치 불균등만 잰다. 스트라이드가 좁아 인접 창이 겹치는 것은")
    print("     포함되지 않으므로 진짜 독립 정보량은 이보다 적다)")

    # 실제로 뽑아 보고 확인
    rng = np.random.default_rng(0)
    idx = cache.sample_indices(4000, rng)
    got = cache.on_center[idx].mean(axis=0)
    print(f"\n  실제 추출 4,000창 검증:")
    for a, v in zip(cache.appliances, got):
        print(f"    {a:18s}{100*v:>11.1f}%")
    print(f"    {'불균형':18s}{got.max()/max(got.min(),1e-9):>10.1f}:1")

    one = cache.get(int(idx[0]))
    print(f"\n  창 1개 형태: X {one['X'].shape} {one['X'].dtype} | "
          f"y_power {one['y_power'].shape} | y_state {one['y_state'].shape}")
    print("=" * 78 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="학습용 윈도우 캐시 생성")
    ap.add_argument("--out", default="cache/train", help="출력 디렉터리")
    ap.add_argument("--scenarios", type=int, default=4000, help="시나리오 개수 (기본 4000)")
    ap.add_argument("--seconds", type=float, default=60.0, help="시나리오 1개 길이(초)")
    ap.add_argument("--window", type=int, default=600, help="학습 창 길이(사이클, 600=10초)")
    ap.add_argument("--stride", type=int, default=60, help="부창 간격(사이클, 60=1초)")
    ap.add_argument("--seed", type=int, default=None, help="재현용 난수 시드")
    ap.add_argument("--inspect", action="store_true", help="생성하지 않고 기존 캐시만 점검")
    args = ap.parse_args()

    if args.inspect:
        if not (Path(args.out) / "meta.json").exists():
            print(f"캐시가 없습니다: {Path(args.out).resolve()}")
            return 1
        inspect(args.out)
        return 0

    print("\n" + "=" * 78)
    print("[NILM AI] 학습용 윈도우 캐시 생성")
    print("=" * 78)
    t0 = time.time()
    build_cache(
        cache_dir=args.out,
        n_scenarios=args.scenarios,
        scenario_seconds=args.seconds,
        window_cycles=args.window,
        stride_cycles=args.stride,
        seed=args.seed,
    )
    print(f"\n생성 완료: {time.time() - t0:.1f}s")
    inspect(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
