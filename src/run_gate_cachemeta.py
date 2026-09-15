# -*- coding: utf-8 -*-
"""관문 — 구운 캐시의 `meta` 가 **처치한 것만** 기준과 다른가 (14.107).

왜 도구로 빼나
--------------
이 검사는 `cache_v32*_par.sbatch` 안에 `$PY -c "..."` 로 박혀 있었다. 그래서
① sbatch 마다 복붙되어 조용히 갈라지고 ② **굽기가 다 끝난 뒤에** 돌아서, 22분
구운 뒤 키 이름 하나로 토막 여덟이 전부 `FAILED` 로 끝났다 (`984540`). 자료는
멀쩡했다 — `y_on` 이 `train60_v32h` 와 비트 동일이고 `fine`·`wide`·`obs_harm` 만
달랐으니 텍스처 처치가 정확히 들어간 꼴이다.

무엇을 확인하나
---------------
```
  [1] 처치 키가 **적힌 값 그대로** 들어 있다 (`--expect k=v` 로 준다)
  [2] 그 밖의 키는 기준과 같다. 단 **'키 없음' 과 '0/False' 는 같은 뜻**이다 —
      옛 캐시는 그 키가 생기기 전에 구워서 아예 없다
  [3] 기준에만 있는 **슬라이스 장부**(`_slice_of`·`_slice_rows`·`_why`)는 뺀다.
      ⚠ 빼기 전에 **구운 쪽에는 없다는 것을 확인**한다 — 있으면 기준을 잘못 준 것이다
  [4] 구운 쪽에만 있는 **새 키**는 값을 **검사한 뒤** 뺀다 (`--new k=v`).
      그냥 무시하면 관문을 늘려서 통과시키는 것이다
      ([[dont-loosen-a-gate-to-make-it-pass]])
  [5] 토막 장부(`shard`·`shard_chunks`·`n_windows`)가 서로 안 겹치고 빈틈이 없다
      — 여러 토막을 한꺼번에 주면 검사한다
```

    python -X utf8 src/run_gate_cachemeta.py --ref cache/_v32_head512 \
        --expect harmonic_z=True vtex_step_s=1.25 vtex_seg_s=1.25 vtex_coarse_s=25 \
        --new wide_shape=47,120 -- cache/train60_v32hs3_s0 ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

#: 굽기마다 반드시 다른 장부. 자료의 성질이 아니다.
BOOK = {"shard", "shard_chunks", "n_windows", "build_seconds", "bytes",
        "merged_from", "merged_partial", "positive_rate", "recipe_counts",
        "built_at", "content_sha256"}
#: 기준이 **슬라이스**일 때 슬라이스 도구가 붙이는 장부. 구운 쪽엔 없어야 한다.
SLICE_BOOK = {"_slice_of", "_slice_rows", "_why"}
OK, NG = "OK", "** 실패 **"


def _parse_kv(items):
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out


def _eq(want: str, got) -> bool:
    """문자열로 준 기대값을 실제 값과 견준다 — 숫자·불리언·목록을 다 받는다."""
    if want in ("True", "False"):
        return bool(got) == (want == "True")
    if "," in want:
        try:
            return [float(x) for x in want.split(",")] == [float(x) for x in (got or ())]
        except (TypeError, ValueError):
            return False
    try:
        return float(want) == float(got)
    except (TypeError, ValueError):
        return str(want) == str(got)


def _nz(v):
    """'키 없음' 과 '0/False/빈 문자열' 을 같은 것으로 본다."""
    return None if v is None or v == 0 or v == "" or v is False else v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+", help="검사할 캐시 디렉터리 (토막 여럿 가능)")
    ap.add_argument("--ref", required=True, help="기준 캐시 (슬라이스여도 된다)")
    ap.add_argument("--expect", nargs="*", default=[], metavar="K=V",
                    help="처치 키. 이 값이어야 하고, 기준과 달라도 된다")
    ap.add_argument("--new", nargs="*", default=[], metavar="K=V",
                    help="구운 쪽에만 새로 생긴 키. **값을 검사한 뒤** 뺀다")
    a = ap.parse_args()

    ref = json.load(open(Path(a.ref) / "meta.json", encoding="utf-8"))
    exp, new = _parse_kv(a.expect), _parse_kv(a.new)
    print("캐시 meta 관문 (14.107) · 기준 %s" % a.ref)
    print("  처치 %s" % (" ".join("%s=%s" % kv for kv in exp.items()) or "(없음)"))
    fail, seen = [], []

    for c in a.caches:
        d = json.load(open(Path(c) / "meta.json", encoding="utf-8"))
        bad = []
        for k, v in exp.items():                                   # [1]
            if not _eq(v, d.get(k)):
                bad.append("처치 %s 가 %r (기대 %s)" % (k, d.get(k), v))
        for k in SLICE_BOOK:                                       # [3]
            if k in d:
                bad.append("구운 쪽에 슬라이스 장부 %s 가 있다 — 기준을 잘못 줬다" % k)
        for k, v in new.items():                                   # [4]
            if k in ref:
                bad.append("%s 는 기준에도 있다 — `--new` 가 아니다" % k)
            elif not _eq(v, d.get(k)):
                bad.append("새 키 %s 가 %r (기대 %s)" % (k, d.get(k), v))
        skip = BOOK | SLICE_BOOK | set(exp) | set(new)             # [2]
        diff = sorted(k for k in (set(ref) | set(d)) - skip
                      if json.dumps(_nz(ref.get(k)), sort_keys=True, default=str)
                      != json.dumps(_nz(d.get(k)), sort_keys=True, default=str))
        if diff:
            bad.append("처치 밖의 키가 다르다: %s" % diff)
        seen.append((c, d.get("shard"), d.get("shard_chunks"), d.get("n_windows")))
        print("  %s %-34s 창 %-7s 청크 %s%s"
              % (OK if not bad else NG, Path(c).name, d.get("n_windows"),
                 d.get("shard_chunks"), "" if not bad else "\n      " + "\n      ".join(bad)))
        fail += bad

    if len(seen) > 1:                                              # [5]
        rng = sorted((s[2][0], s[2][1], s[0]) for s in seen if s[2])
        holes = [(rng[i][1], rng[i + 1][0]) for i in range(len(rng) - 1)
                 if rng[i + 1][0] != rng[i][1] + 1]
        tot = sum(s[3] or 0 for s in seen)
        want = json.load(open(Path(a.caches[0]) / "meta.json",
                              encoding="utf-8")).get("n_windows_total")
        good = not holes and (want is None or tot == want)
        if not good:
            fail.append("토막 장부")
        print("  %s 토막 %d개가 청크 %d~%d 를 빈틈없이 덮는다 · 창 합 %d (전체 %s)"
              % (OK if good else NG, len(seen), rng[0][0], rng[-1][1], tot, want))
        if holes:
            print("      ** 빈 구간: %s **" % holes)

    print("")
    if fail:
        print("관문 실패 %d건" % len(fail))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
