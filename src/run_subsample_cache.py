"""합성 replay 캐시를 솎는다 — 현장 재적응 꾸러미를 위해 (12.151)

왜
--
2단계 적응은 실측 파일로 돌지만 `--lam 0.5` 로 **합성 replay** 를 같이 먹인다
(4.2절, 망각 방지). 그 캐시가 `cache/train60_ovh` 이고 **20.9 GB** 다. 현장에서
장소가 바뀔 때마다 재적응해야 한다면(12.150.2) 이 20.9 GB 를 들고 다녀야 한다.

물어야 할 것 둘:
  ① `--lam 0` 으로 replay 를 아예 빼면 실측 성능이 무너지나?
  ② 무너진다면 캐시를 얼마나 줄일 수 있나?

이 도구가 ②를 위해 캐시를 **균등 스트라이드**로 솎는다. 앞에서 n 개를 자르지
않는 이유는 캐시가 250창 청크마다 레시피 믹스를 새로 뽑기 때문이다 — 앞을 자르면
믹스가 치우칠 수 있다.

    python -X utf8 -m src.run_subsample_cache --src cache/train60_ovh \
        --dst cache/train60_ovh_30k --n 30000
"""
from pathlib import Path
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cache/train60_ovh")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n", type=int, default=30_000)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    n0 = int(meta["n_windows"])
    if a.n >= n0:
        raise SystemExit(f"원본이 {n0:,}개라 {a.n:,}개로 솎을 수 없습니다.")
    idx = np.linspace(0, n0 - 1, a.n).round().astype(np.int64)
    idx = np.unique(idx)
    dst.mkdir(parents=True, exist_ok=True)

    names = [p.stem for p in src.glob("*.npy")]
    total = 0
    t0 = time.time()
    for name in names:
        s = np.load(src / f"{name}.npy", mmap_mode="r")
        o = np.lib.format.open_memmap(dst / f"{name}.npy", mode="w+",
                                      dtype=s.dtype, shape=(len(idx),) + s.shape[1:])
        for k in range(0, len(idx), 2000):          # memmap 무작위 접근을 끊어 읽는다
            j = idx[k:k + 2000]
            o[k:k + len(j)] = s[j]
        o.flush()
        b = o.nbytes; total += b
        print(f"  {name:<12} {str(o.shape):<24} {b/1e9:6.3f} GB")
        del o, s

    meta["n_windows"] = int(len(idx))
    meta["subsampled_from"] = str(src)
    meta["bytes"] = int(total)
    (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"[solk] {n0:,} -> {len(idx):,}창  {total/1e9:.2f} GB  "
          f"({time.time()-t0:.0f}s)  -> {dst.resolve()}")


if __name__ == "__main__":
    main()
