# -*- coding: utf-8 -*-
"""노드 여러 대가 나눠 구운 캐시 토막을 **번호순으로** 이어붙인다 (14.13).

`build_cache` 는 청크 **번호**로 시드하고(`chunk_seed`) `imap` 으로 순서를 지킨다. 그래서
청크 목록을 토막내 다른 노드에서 구워도 **같은 창**이 나오고, 번호순으로 이어붙이면
단일 노드 결과와 **비트 동일**하다. 그 성질이 이 도구의 전제다.

    # 노드 A                                    # 노드 B
    --out cache/v32_s0 --shard 0/2 \\            --out cache/v32_s1 --shard 1/2 \\
    --windows 300000 --chunk 250 --seed 0       --windows 300000 --chunk 250 --seed 0
    #                              ^^^^^^^^ 전체 기준값을 **양쪽 다** 같게

    python -X utf8 src/run_merge_traincache.py --out cache/train60_v32 \\
        cache/v32_s0 cache/v32_s1

⚠ 토막을 지우지 않는다. 지울지는 사람이 정한다.

⚠⚠ **memmap 으로 복사하지 마라 — GPFS 에서 120배 느리다** (2026-09-14 실측).
   처음에 `open_memmap` + 슬라이스 대입으로 짰더니 30만창에 **30분**이 걸렸다.
   ```
   WCHAN = ZN10gpfsNode_t8mmapLockEyPKjy    <- 페이지마다 GPFS **mmap 락**
   STAT  = Dl (중단 불가 I/O 대기)  ·  13 MB/s
   ```
   `.npy` 는 **머리말 + C순서 연속 바이트**라 축 0 이어붙이기는 **바이트 이어붙이기**와 같다.
   머리말만 새로 쓰고 자료부를 `shutil.copyfileobj` 로 순차 복사하면 **15초**다 (~1.6GB/s).
   사용자가 *"이어붙이는 게 그렇게 오래 걸려?"* 라고 묻지 않았으면 그게 정상인 줄 알았을 것이다.
   ⇒ 로그인 노드에서 15초면 되므로 별도 작업으로 뺄 이유도 없다.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

#: 이어붙일 배열들. `traincache._SPEC` 과 같아야 한다.
ARRAYS = ("fine", "wide", "y_power", "y_on", "y_plugged", "y_standby", "y_state",
          "obs_harm", "p_noise", "p_observed", "z_grid")

#: 토막 사이에 **같아야** 하는 meta 키. 하나라도 다르면 다른 생성기다.
SAME = ("window_cycles", "appliances", "time_split", "seed", "n_wide", "chunk",
        "n_windows_total", "recipe_mix", "smps_focus_off_p", "power_scale_std_map",
        "state_mix", "carrier_apps", "couple_ext", "sp_curves", "sp_per_texture",
        "vtail", "float_fill", "steady_crop", "standby_jitter_cap", "sibling_rotate",
        "exclude_activation_files", "dither_amp", "dither_phase_deg", "level_scramble")


def _npy_info(path):
    """`.npy` 의 (창 수, 자료부 시작 오프셋, dtype, fortran_order). 머리말을 건너뛴다."""
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        # ⚠ `_read_array_header` 는 비공개라 numpy 판마다 이름이 다르다 — 공개 API 로 나눈다.
        if ver == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif ver == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise SystemExit("[merge] 모르는 .npy 판 %s: %s" % (ver, path))
        return shape[0], f.tell(), dtype, bool(fortran)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="토막 디렉터리들 (순서는 상관없다 — shard 번호로 정렬한다)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="토막이 다 모이지 않아도 있는 것만 붙인다 (⚠ 단일 노드와 달라진다)")
    ap.add_argument("--rm-shards", action="store_true",
                    help="⚠⚠ **복사가 끝난 토막 배열 파일을 즉시 지운다** — 쿼터가 빠듯할 때. "
                         "안 주면 최대 사용량이 `토막 + 통짜` 라 두 배가 된다 (v32t 실측: "
                         "23G + 23G = 107G 가 하드 한도 110G 에 3G 남기고 닿았다). 주면 "
                         "`fine`(전체 바이트의 87%%)이 토막마다 바로 빠져 **89G** 로 눌린다. "
                         "⚠ 대가: 중간에 죽으면 토막이 반쯤 지워져 **다시 구워야 한다**. "
                         "복사 바이트 수를 확인한 뒤에만 지운다.")
    a = ap.parse_args()

    metas = []
    for d in a.shards:
        p = Path(d) / "meta.json"
        if not p.exists():
            raise SystemExit("[merge] meta.json 이 없다: %s" % p)
        metas.append((Path(d), json.loads(p.read_text(encoding="utf-8"))))

    # ── 관문 ① 전부 토막인가 ─────────────────────────────────────────────
    for d, m in metas:
        if not m.get("shard"):
            raise SystemExit("[merge] %s 는 토막이 아니다 (meta.shard 가 없다) — "
                             "--shard 없이 구운 통짜 캐시다" % d)
    n_tot = metas[0][1]["shard"][1]

    # ── 관문 ② 같은 생성기인가 ───────────────────────────────────────────
    base = metas[0][1]
    for d, m in metas[1:]:
        diff = [k for k in SAME if json.dumps(m.get(k), sort_keys=True, default=str)
                != json.dumps(base.get(k), sort_keys=True, default=str)]
        if diff:
            raise SystemExit("[merge] %s 의 설정이 %s 와 다르다: %s\n"
                             "  같은 생성기로 구운 토막만 붙일 수 있다."
                             % (d, metas[0][0], diff))

    # ── 관문 ③ 번호가 빠짐없이 0..n-1 인가 ──────────────────────────────
    metas.sort(key=lambda t: t[1]["shard"][0])
    got = [m["shard"][0] for _, m in metas]
    if len(set(got)) != len(got):
        raise SystemExit("[merge] 같은 토막 번호가 둘 이상이다: %s" % got)
    if any(m["shard"][1] != n_tot for _, m in metas):
        raise SystemExit("[merge] 토막들의 전체 개수가 다르다: %s"
                         % [m["shard"] for _, m in metas])
    missing = sorted(set(range(n_tot)) - set(got))
    if missing and not a.allow_partial:
        raise SystemExit("[merge] 토막이 빠졌다: %s (전체 %d개 중 %s 만 있다)\n"
                         "  굽기가 끝났는지 확인하라. 정말 일부만 붙이려면 --allow-partial."
                         % (missing, n_tot, got))

    # ── 관문 ④ 청크 범위가 이어지는가 ───────────────────────────────────
    prev_hi = -1
    for d, m in metas:
        lo, hi = m.get("shard_chunks") or (None, None)
        if lo is None:
            continue
        if not missing and lo != prev_hi + 1:
            raise SystemExit("[merge] 청크 번호가 안 이어진다: %s 가 %d 부터인데 앞이 %d 까지다"
                             % (d, lo, prev_hi))
        prev_hi = hi

    ns = [int(m["n_windows"]) for _, m in metas]
    N = sum(ns)
    print("토막 %d개 · 창 %s = **%s개** (전체 기준 %s)"
          % (len(metas), " + ".join("{:,}".format(x) for x in ns), "{:,}".format(N),
             "{:,}".format(int(base.get("n_windows_total") or 0))))
    if not missing and base.get("n_windows_total") and N != int(base["n_windows_total"]):
        raise SystemExit("[merge] 창 수가 전체 기준과 다르다: %d != %d"
                         % (N, int(base["n_windows_total"])))

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        srcs = [Path(d) / ("%s.npy" % name) for d, _ in metas]
        miss = [str(p) for p in srcs if not p.exists()]
        if miss:
            raise SystemExit("[merge] 배열이 없다: %s" % miss)
        a0 = np.load(srcs[0], mmap_mode="r")
        shape = (N,) + a0.shape[1:]
        row_bytes = int(np.prod(a0.shape[1:])) * a0.dtype.itemsize
        for p in srcs:
            src = np.load(p, mmap_mode="r")
            if src.shape[1:] != a0.shape[1:] or src.dtype != a0.dtype:
                raise SystemExit("[merge] %s 의 모양/자료형이 다르다: %s %s 대 %s %s"
                                 % (p, src.shape, src.dtype, a0.shape, a0.dtype))
            del src
        del a0
        # ⚠ **memmap 으로 복사하지 마라** (2026-09-14 실측). `open_memmap` + 슬라이스 대입은
        #   페이지마다 GPFS mmap 락을 잡아 **13MB/s** 밖에 안 난다 (`WCHAN=gpfsNode::mmapLock`,
        #   `STAT=Dl`). 24GB 에 30분이다.
        #   `.npy` 는 **머리말 + C순서 연속 바이트**라 축 0 이어붙이기는 **바이트 이어붙이기**와
        #   같다. 머리말만 새로 쓰고 자료부는 `copyfileobj` 로 순차 복사한다.
        _n0, _off0, _d0, _f0 = _npy_info(srcs[0])
        with open(out / ("%s.npy" % name), "wb") as fo:
            np.lib.format.write_array_header_2_0(
                fo, {"descr": np.lib.format.dtype_to_descr(_d0),
                     "fortran_order": False, "shape": shape})
            for p in srcs:
                n_, off_, dt_, fo_ = _npy_info(p)
                if fo_:
                    raise SystemExit("[merge] fortran_order 인 토막은 못 붙인다: %s" % p)
                with open(p, "rb") as fi:
                    fi.seek(off_)
                    at = fo.tell()
                    shutil.copyfileobj(fi, fo, length=32 * 1024 * 1024)
                    wrote = fo.tell() - at
                # ⚠ **세고 나서 지운다.** 바이트가 안 맞으면 지우지 않고 죽는다 —
                #   반쯤 복사된 것을 지우면 되돌릴 길이 없다.
                want = n_ * row_bytes
                if wrote != want:
                    raise SystemExit("[merge] %s 를 %d 바이트 복사해야 하는데 %d 를 썼다"
                                     % (p, want, wrote))
                if a.rm_shards:
                    p.unlink()
        print("   %-12s %s" % (name, shape))

    meta = dict(base)
    meta["n_windows"] = int(N)
    meta["shard"] = None
    meta["shard_chunks"] = None
    meta["merged_from"] = [str(d) for d, _ in metas]
    meta["merged_partial"] = bool(missing)
    # 양성률은 토막마다 조금씩 다르므로 **다시 센다** — 옛 값을 물려주면 거짓말이 된다.
    try:
        yo = np.load(out / "y_on.npy", mmap_mode="r")
        apps = base["appliances"]
        meta["positive_rate"] = {apps[i]: float((np.asarray(yo[:, i]) == 1).mean())
                                 for i in range(len(apps))}
    except Exception as e:                                    # noqa: BLE001
        print("   ⚠ 양성률을 다시 못 셌다: %s" % e)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print("\n이어붙임: %s" % out)
    print("   양성률 %s" % json.dumps({k: round(v, 3) for k, v in
                                      (meta.get("positive_rate") or {}).items()},
                                     ensure_ascii=False))
    free = shutil.disk_usage(out).free / 1e9
    if a.rm_shards:
        for d, _ in metas:
            for q in sorted(d.glob("*")):
                q.unlink()
            d.rmdir()
        print("   남은 자리 %.1f GB · **토막을 지웠다** (--rm-shards)" % free)
    else:
        print("   남은 자리 %.1f GB · ⚠ 토막은 안 지웠다 (%s)"
              % (free, " ".join(str(d) for d, _ in metas)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
