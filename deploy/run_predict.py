"""배포 묶음 CLI — 수신기 CSV 를 읽어 기기별 전력을 찍는다.

    # 실시간 (수신기가 쓰고 있는 CSV 를 따라간다)
    python run_predict.py --csv data/live.csv

    # 재생 검증 (기존 CSV 를 처음부터)
    python run_predict.py --replay data/test_8.csv --speed 0

`--jsonl` 을 주면 결과를 한 줄 JSON 으로 흘려보낸다. **UI 는 이것만 읽어도 된다**
(별도 프로세스로 붙일 때). 같은 프로세스 안에서 쓰려면 `nilm_runtime.NILMPredictor`
를 직접 부르는 편이 낫다 — README 의 "UI 붙이기" 참조.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nilm_runtime import APPLIANCE_KO, NILMPredictor  # noqa: E402


def rows(path: Path, replay: bool, speed: float, poll: float = 0.2):
    """CSV 를 따라 읽는다. `replay` 면 처음부터, 아니면 끝에 붙어 기다린다."""
    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
        yield header, None
        if not replay:
            f.seek(0, 2)
        t0 = time.time()
        first_t = None
        while True:
            line = f.readline()
            if not line:
                if replay:
                    return
                time.sleep(poll)
                continue
            row = next(csv.reader([line]))
            if replay and speed > 0:
                try:
                    ts = float(row[header.index("t_s")])
                except (ValueError, IndexError):
                    ts = None
                if ts is not None:
                    first_t = ts if first_t is None else first_t
                    wait = (ts - first_t) / speed - (time.time() - t0)
                    if wait > 0:
                        time.sleep(wait)
            yield None, row


def main() -> int:
    ap = argparse.ArgumentParser(description="NILM 실시간 예측 (배포 묶음)")
    ap.add_argument("--csv", default="data/live.csv", help="수신기가 쓰는 CSV")
    ap.add_argument("--replay", default=None, help="기존 CSV 를 재생한다")
    ap.add_argument("--speed", type=float, default=20.0, help="재생 배속 (0=최대)")
    ap.add_argument("--ckpt", default=str(Path(__file__).parent / "models/adapt_smpsf.pt"))
    ap.add_argument("--postproc", default="on", choices=("off", "on", "sync"))
    ap.add_argument("--every", type=int, default=30, help="추론 간격 (사이클). 30=0.5초")
    ap.add_argument("--jsonl", default="", help="결과를 이 파일에 한 줄 JSON 으로")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    path = Path(a.replay or a.csv)
    pred = NILMPredictor(a.ckpt, postproc=a.postproc)
    print(f"[NILM] {a.ckpt} | 장치 {pred.dev} | 후처리 {a.postproc} | {path}")
    out = open(a.jsonl, "w", encoding="utf-8") if a.jsonl else None
    n = 0
    for header, row in rows(path, a.replay is not None, a.speed):
        if header is not None:
            pred.set_header(header)
            continue
        pred.push_row(row)
        n += 1
        if n % a.every:
            continue
        r = pred.predict()
        if r is None:
            continue
        if out:
            out.write(json.dumps({
                "t_s": round(r.t_s, 3), "observed_w": round(r.observed_w, 2),
                "total_w": round(r.total_w, 2), "residual_w": round(r.residual_w, 2),
                "power_w": {k: round(v, 2) for k, v in r.power_w.items()},
                "gate": {k: round(v, 4) for k, v in r.gate.items()},
            }, ensure_ascii=False) + "\n")
        if not a.quiet:
            on = sorted(((r.power_w[k], k) for k in r.on()), reverse=True)[:4]
            body = "  ".join(f"{APPLIANCE_KO.get(k, k)} {w:.0f}W" for w, k in on) or "-"
            print(f"  t={r.t_s:7.1f}s  관측 {r.observed_w:7.1f}W | 예측 {r.total_w:7.1f}W | {body}")
    if out:
        out.close()
    st = pred.stats()
    print(f"\n  사이클 {st['n_pushed']:,}개 | 순서 뒤바뀜 {100*st['reorder_rate']:.2f}% "
          f"(최대 역전 {st['max_backfill_s']:.2f}초) | 세션 이어붙임 {st['n_seam']}회")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
