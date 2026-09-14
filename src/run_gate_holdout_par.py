# -*- coding: utf-8 -*-
"""홀드아웃 **병렬 굽기**의 항등 관문 (14.51).

재는 것 — `traincache` 가 14.13 에서 통과한 그 검정을 홀드아웃에 그대로 댄다:

  ① 워커 **수를 바꿔도 같은 바이트**가 나온다 (2개 대 5개). 청크 번호로 시드하고
     `imap`(순서 보장)으로 받으므로 그래야 한다. 아니면 `imap_unordered` 를 쓴 것이거나
     워커 번호로 시드한 것이다 (12.11 이 잡았던 바로 그 버그).
  ② 병렬과 직렬은 **다르다** — 그리고 그것이 정상이다. 난수를 자르는 방식이 다르므로.
     ⚠ 이 줄이 "같다" 로 나오면 병렬이 **안 돈 것**이다 (워커가 0으로 떨어졌다).
  ③ 청크 크기를 바꾸면 내용이 바뀐다 — 경계가 난수를 가르므로. meta 가 그것을 적는가.
  ④ 두 경로가 **같은 조립기**(`_build_generator`)를 탄다 — 설정이 갈릴 자리가 없다.

⚠ 창 수를 적게 준다. 이것은 **배선 관문**이지 성능 측정이 아니다.
"""
from pathlib import Path
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import holdout as H       # noqa: E402

FAIL = []
ARR = ("y_power", "y_standby", "y_on", "y_state", "y_plugged", "p_noise", "p_observed", "recipe")


def ck(name: str, ok: bool, note: str = "") -> None:
    print(("  ✅ " if ok else "  ❌ ") + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


N_WIN = 150             #: 배선 관문이라 적게 굽는다. 창 하나가 45x3600x4B = 648KB 다.


def _build(d: Path, **kw) -> dict:
    return H.build_holdout(out_dir=d, n_windows=N_WIN, window_cycles=3600,
                           seed=4242, progress_every=0, **kw)


def _same(a: Path, b: Path) -> tuple:
    x = np.array_equal(np.load(a / "X.npy", mmap_mode="r"), np.load(b / "X.npy", mmap_mode="r"))
    ys = all(np.array_equal(np.load(a / f"{n}.npy"), np.load(b / f"{n}.npy")) for n in ARR)
    return x, ys


def main() -> int:
    global N_WIN
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=N_WIN, help="판마다 구울 창 수")
    ap.add_argument("--tmp", default="", metavar="DIR",
                    help="임시 폴더. 비우면 TMPDIR/시스템 임시. "
                         "⚠ **쿼터가 있는 홈에 두지 마라** — 네 판 x n x 648KB 다.")
    a = ap.parse_args()
    N_WIN = int(a.n)
    print("  판마다 %d창 x 60초 = %.0f MB · 네 판" % (N_WIN, N_WIN * 45 * 3600 * 4 / 1e6))
    print("홀드아웃 병렬 굽기 항등 관문 (14.51)\n")
    root = Path(tempfile.mkdtemp(prefix="hgate_",
                                 dir=(a.tmp or os.environ.get("TMPDIR") or None)))
    try:
        t = {}
        for tag, kw in (("w2", dict(workers=2)), ("w5", dict(workers=5)),
                        ("ser", dict(workers=0)), ("w2c", dict(workers=2, chunk_windows=40))):
            t0 = time.time()
            m = _build(root / tag, **kw)
            t[tag] = (time.time() - t0, m)
            print(f"     [{tag}] {t[tag][0]:5.1f}초 · sha {m['content_sha256']}")
        print()

        x, ys = _same(root / "w2", root / "w5")
        ck("① ★ 워커 2개와 5개가 **비트 동일** (X·정답 전부)", x and ys,
           "X %s · 정답 %s" % (x, ys))
        ck("① 해시도 같다",
           t["w2"][1]["content_sha256"] == t["w5"][1]["content_sha256"],
           t["w2"][1]["content_sha256"])

        x2, _ = _same(root / "w2", root / "ser")
        ck("② 병렬과 직렬은 다르다 (같으면 병렬이 안 돈 것이다)", not x2)
        ck("② 직렬 meta 가 자기를 직렬로 적는다",
           t["ser"][1]["workers_chunked"] is False and t["ser"][1]["chunk_windows"] == 0)
        import inspect as _i
        cw_def = _i.signature(H.build_holdout).parameters["chunk_windows"].default
        ck("② 병렬 meta 가 청크 크기를 적는다 (기본값을 그대로)",
           t["w2"][1]["workers_chunked"] is True and t["w2"][1]["chunk_windows"] == cw_def,
           "기본 %d창" % cw_def)

        x3, _ = _same(root / "w2", root / "w2c")
        ck("③ 청크 크기를 바꾸면 내용이 바뀐다 (경계가 난수를 가른다)", not x3,
           "기본 %d창 대 40창" % cw_def)

        import inspect
        src = inspect.getsource(H.build_holdout)
        ck("④ 두 경로가 같은 조립기를 탄다",
           "_build_generator(opts)" in src and src.count("SegmentPool(") == 0,
           "build_holdout 안에 SegmentPool 직접 조립이 없다")
        ck("④ 워커도 그 조립기를 탄다",
           "_build_generator(o, quiet=True)" in inspect.getsource(H._w_init))

        # 창 수가 같은가 — 청크가 하나라도 빠지면 조용히 절반만 나온다
        n_ok = all(np.load(root / g / "y_on.npy").shape[0] == N_WIN
                   for g in ("w2", "w5", "ser", "w2c"))
        ck("④ 네 판 다 %d창을 냈다" % N_WIN, n_ok)

        sp = t["ser"][0] / max(t["w5"][0], 1e-9)
        print(f"\n  참고 — 직렬 {t['ser'][0]:.1f}초 / 워커5 {t['w5'][0]:.1f}초 = **{sp:.1f}배**")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
