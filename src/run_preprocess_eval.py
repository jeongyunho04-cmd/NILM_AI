"""복합 부하 실측 파일만 골라 전처리한다 — **세그먼트 풀은 안 건드린다** (12.155)

왜 따로 만드나
-------------
`run_preprocess_and_label` 은 CLI 인자가 없어 `data/` 전체를 돌고, 그 과정에서
`processed_data/npz/`(세그먼트 풀)를 **전부 다시 쓴다.** 풀이 바뀌면 지문·대기·
계측잡음·`harm_scale`·`Q/P` 상수가 전부 바뀌고 기존 체크포인트의 근거가 흔들린다.
새 복합 파일 몇 개를 넣자고 감수할 위험이 아니다.

그리고 미등록 파일이 하나라도 있으면 `require_known` 이 즉시 멈춘다. 지금
`hotplate_4_new` / `electric_kettle_3_new` 가 그렇다 — **다른 장소의 단독 녹화라
풀에 넣으면 안 된다** (12.151.1: 지문의 h1 실수부가 그 녹화의 선전압이라 장소를
섞으면 함의 전압이 오염된다). 그 둘은 `Z` 측정도 전이 지문 추출도 원본 CSV 를
직접 읽으므로 전처리 자체가 필요 없다.

여기서는 **이름을 명시한 복합 파일만** `processed_data/composite_eval/` 로 낸다.
`run_preprocess_and_label` 의 eval 갈래와 같은 코드 경로다.

    python -X utf8 -m src.run_preprocess_eval --stems test_14 test_15 test_16 test_17 test_18
    python -X utf8 -m src.run_preprocess_eval --stems test_14 --force
"""
from pathlib import Path
import argparse
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.preprocessing.file_registry import FileRole, classify_file
from src.preprocessing.numpy_exporter import NumpyDatasetExporter
from src.preprocessing.pipeline import PreprocessingPipeline


def _one_session(pipeline, path, which: int):
    """녹화 여러 개가 이어 붙은 파일에서 하나만 골라 전처리한다.

    세션 판정은 `DataCleaner.assign_sessions` 를 그대로 쓴다 — 여기서 따로 규칙을
    만들면 전처리와 갈라져 조용히 어긋난다 (`REORDER_TOLERANCE_FRAMES` 32 프레임).
    **원본 CSV 는 안 건드린다.** 읽어서 메모리에서 자르고 넘긴다.
    """
    import pandas as pd
    df = pd.read_csv(path)
    tagged, n = pipeline.cleaner.assign_sessions(df)
    if which >= n:
        raise SystemExit(f"{path.name} 에는 세션이 {n}개뿐입니다 (0..{n-1}).")
    sel = tagged[tagged["session_id"] == which].drop(columns=["session_id"])
    print(f"[세션 {which}/{n} · {len(sel):,}/{len(df):,}행] ", end="", flush=True)
    out, st = pipeline.cleaner.clean_dataframe(sel)
    out = pipeline.extractor.extract_features(out)
    st["file_role"] = "composite_eval"
    st.setdefault("v_ref_v", float(out["vrms"].median()))
    st.setdefault("duration_s", len(out) / 60.0)
    st.setdefault("cleaned_rows", len(out))
    st.setdefault("p_mean", float(out["p_w"].mean()))
    st.setdefault("p_max", float(out["p_w"].max()))
    return out, st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="+", required=True, help="처리할 복합 파일 stem")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="processed_data/composite_eval")
    ap.add_argument("--force", action="store_true", help="이미 있어도 다시 만든다")
    ap.add_argument("--session", type=int, default=None, metavar="N",
                    help="한 파일에 녹화가 여러 개면 그중 **하나만** 쓴다 (0부터). "
                         "`test_14` 가 그렇다 — seq 가 925 에서 0 으로 되감기고 "
                         "host_time 이 6분 점프한다. 사람 타임라인은 세션 1 것이다. "
                         "원본 CSV 는 안 건드리고 메모리에서만 자른다")
    ap.add_argument("--as-stem", default="", metavar="NAME",
                    help="다른 이름으로 저장 (세션을 따로 낼 때)")
    a = ap.parse_args()
    if a.as_stem and len(a.stems) != 1:
        raise SystemExit("--as-stem 은 파일 하나일 때만 됩니다.")

    raw, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = PreprocessingPipeline(sampling_hz=60.0, noise_floor_w=1.4)
    exporter = NumpyDatasetExporter(harmonics_count=15)

    # 역할을 **먼저 전부** 확인한다. 하나라도 복합이 아니면 아무것도 안 쓴다 —
    # 단일 가전 파일이 실수로 여기 들어오면 풀에 갈 파일을 엉뚱한 곳에 만든다.
    jobs = []
    for stem in a.stems:
        f = raw / f"{stem}.csv"
        if not f.exists():
            raise SystemExit(f"원본이 없습니다: {f}")
        c = classify_file(f)
        if c.role is not FileRole.COMPOSITE_EVAL:
            raise SystemExit(
                f"{stem} 의 역할이 {c.role.value} 입니다. 이 도구는 복합 부하 실측만 받습니다.\n"
                f"  단일 가전이면 file_registry 에 등록하고 run_preprocess_and_label 을 쓰십시오.\n"
                f"  ⚠ 다른 장소의 단독 녹화는 세그먼트 풀에 넣지 마십시오 (12.151.1).")
        jobs.append((f, c.stem))

    print("=" * 84)
    print(f"복합 부하 실측 전처리 {len(jobs)}개 -> {out.resolve()}  (세그먼트 풀 불변)")
    print("=" * 84)
    t0 = time.time()
    for f, stem in jobs:
        name = a.as_stem or stem
        dst = out / f"{name}.npz"
        if dst.exists() and not a.force:
            print(f"  {name:<10} 이미 있음 (--force 로 덮어씀)")
            continue
        print(f"  {name:<10} ... ", end="", flush=True)
        if a.session is None:
            df, st = pipeline.process_file(f)
        else:
            df, st = _one_session(pipeline, f, a.session)
        exporter.export_to_npz(
            df, output_path=dst,
            metadata={
                "source_file": f.name,
                "file_role": st["file_role"],
                "sampling_hz": 60.0,
                "duration_s": st["duration_s"],
                "v_ref_v": st["v_ref_v"],
                "note": "복합 부하 실측. 단일 가전이 아니므로 상태 라벨과 세그먼트 풀 사용 금지.",
            },
            compress=True,
        )
        print(f"행 {st['cleaned_rows']:>7,}  {st['duration_s']:>7.1f}s  "
              f"V_ref {st['v_ref_v']:6.1f}V  P {st['p_mean']:6.1f}/{st['p_max']:7.1f}W")
    print(f"완료 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
