"""
학습용 독립 창 캐시 생성
=========================
60초 창 합성이 550 win/s 라 GPU 가 7~10% 만 일한다 (12.8.2절). 미리 만들어 두면
학습이 GPU 병목으로 바뀌어 2M 창 기준 61분 -> 3.5분이 된다.

python -m src.run_build_traincache                  # 30만창, 약 13.5GB, 9분
python -m src.run_build_traincache --windows 100000 # 작게
"""
import argparse, sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401
from src.model.traincache import build_cache


def main() -> int:
    ap = argparse.ArgumentParser(description="학습용 독립 창 캐시 생성")
    ap.add_argument("--out", default="cache/train60")
    ap.add_argument("--windows", type=int, default=300_000)
    ap.add_argument("--window-cycles", type=int, default=3600)
    ap.add_argument("--split", default="train", choices=["train", "holdout", "all"])
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("=" * 78); print("[NILM AI] 학습용 독립 창 캐시"); print("=" * 78)
    build_cache(out_dir=a.out, n_windows=a.windows, window_cycles=a.window_cycles,
                time_split=a.split, seed=a.seed, n_workers=a.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
