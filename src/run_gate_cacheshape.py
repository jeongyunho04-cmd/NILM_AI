# -*- coding: utf-8 -*-
"""학습 캐시의 **광역 모양**을 못 박는 관문 (14.93).

왜 필요했나
-----------
세밀은 12.34(38 -> 44채널) 사고 뒤로 `meta["fine_shape"]` 검사가 걸려 있었다.
**광역에는 아무것도 없었다** — meta 에 모양 키 자체가 없고(`n_wide` 뿐) 검사도 없다.
`WIDE_CHANNELS` 를 바꾸면 `np.load(mmap_mode="r")` 이 **조용히 엉뚱한 축으로** 읽는다.
세밀 쪽 주석이 그 사고를 이미 적어 뒀는데 광역만 안 막혀 있었다.

무엇을 확인하나
---------------
```
  [1] 멀쩡한 캐시는 **통과한다** (오탐 없음)
  [2] 광역 채널 수가 다르면 **운다**          <- 고침 전에는 조용히 통과했다
  [3] 광역 길이가 meta 의 n_wide 와 다르면 운다
  [4] 세밀 채널 수가 다르면 운다 (옛 검사가 살아 있다)
  [5] ★ **고침 전 코드에 돌리면 [2] 가 통과한다** — 그래야 이 관문이 무언가를 막는다
        ([[the-gate-must-build-the-real-object]]: 통과는 증거가 아니다, **깨지는 것**이 증거다)
```

    python -X utf8 src/run_gate_cacheshape.py
"""
from pathlib import Path
import json
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.model.inputs import FINE_CHANNELS, FINE_CYCLES, WIDE_CHANNELS  # noqa: E402
from src.model.traincache import CachedWindows  # noqa: E402

N, NW, K = 4, 2, 9


def make(d: Path, wc=WIDE_CHANNELS, nw=NW, fc=FINE_CHANNELS, meta_nw=None):
    """최소 캐시 하나를 짓는다. 모양만 맞으면 값은 아무거나 좋다."""
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "fine.npy", np.zeros((N, fc, FINE_CYCLES), np.float16))
    np.save(d / "wide.npy", np.zeros((N, wc, nw), np.float16))
    for nm, shp in (("y_power", (N, K)), ("y_on", (N, K)), ("y_plugged", (N, K)),
                    ("y_standby", (N, K)), ("obs_harm", (N, 15, 2)),
                    ("p_noise", (N,)), ("p_observed", (N,))):
        np.save(d / f"{nm}.npy", np.zeros(shp, np.float32))
    np.save(d / "y_state.npy", np.zeros((N, K), np.int8))
    (d / "meta.json").write_text(json.dumps({
        "n_windows": N, "n_wide": int(meta_nw if meta_nw is not None else nw),
        "appliances": ["a"] * K,
        "fine_shape": [int(fc), int(FINE_CYCLES)],
        "wide_shape": [int(wc), int(nw)],
    }), encoding="utf-8")
    return d


def load(d):
    """(울었나, 메시지)"""
    try:
        CachedWindows(d)
        return False, ""
    except Exception as e:                       # noqa: BLE001 — 무엇으로 울든 잡는다
        return True, str(e)[:90]


def main() -> int:
    print("학습 캐시 광역 모양 관문 (14.93)")
    print("  현재 코드: 세밀 %d x %d · 광역 %d채널"
          % (FINE_CHANNELS, FINE_CYCLES, WIDE_CHANNELS))
    print("")
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="gate_cacheshape_"))
    try:
        cases = [
            ("[1] 멀쩡한 캐시", dict(), False),
            ("[2] 광역 채널 +1", dict(wc=WIDE_CHANNELS + 1), True),
            ("[2] 광역 채널 −1", dict(wc=WIDE_CHANNELS - 1), True),
            ("[3] 광역 길이가 meta 와 다름", dict(nw=NW + 1, meta_nw=NW), True),
            ("[4] 세밀 채널 +1", dict(fc=FINE_CHANNELS + 1), True),
        ]
        for i, (name, kw, want_raise) in enumerate(cases):
            d = make(tmp / ("c%d" % i), **kw)
            raised, msg = load(d)
            good = raised == want_raise
            ok &= good
            print("  %-26s -> %-8s %s   %s"
                  % (name, "운다" if raised else "통과",
                     "OK" if good else "** 틀리다 **", msg if raised else ""))

        # ── [5] 고침 **전** 동작을 흉내 낸다 — meta 만 보고 배열은 안 보는 검사 ──
        print("")
        print("  [5] ★ 고침 **전** 코드(meta 의 fine_shape 만 검사)에서는 어땠나")
        d = make(tmp / "cbefore", wc=WIDE_CHANNELS + 1)
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        old_pass = list(m["fine_shape"]) == [FINE_CHANNELS, FINE_CYCLES]
        print("      광역 채널이 +1 인 캐시인데 옛 검사(fine_shape)는 %s"
              % ("**통과시킨다**" if old_pass else "운다"))
        arr = np.load(d / "wide.npy", mmap_mode="r")
        print("      그리고 배열은 실제로 (%d, **%d**, %d) 로 들어온다 — 코드는 %d 를 기대한다"
              % (arr.shape[0], arr.shape[1], arr.shape[2], WIDE_CHANNELS))
        ok &= old_pass
        print("      -> 이 관문이 막는 것이 **실재한다**   %s"
              % ("OK" if old_pass else "** 음성 대조 실패 **"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    print("관문 전부 통과" if ok else "** 관문 실패 **")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
