"""
두 학습 캐시가 **앞 N채널까지 같은 데이터인지** 확인한다 (설계 문서 12.40.1절)
================================================================================
새 채널은 반드시 뒤에 붙인다(12.34.3). 그래서 채널만 늘려 캐시를 다시 만들면
앞쪽은 바이트 단위로 같아야 한다 — 합성 난수가 청크 시드로 걸리고
(`traincache._chunk`), 채널 계산이 창별로 독립이기 때문이다.

**같으면 대조군을 새로 학습할 필요가 없다.** 옛 캐시로 학습한 체크포인트를 그대로
대조군으로 쓸 수 있다. 다르면 그 비교는 "채널 효과" 가 아니라 "다른 데이터" 다.

    python -m src.run_verify_cache_prefix cache/train60_ov cache/train60_tr2

앞 캐시가 기준(채널 적은 쪽), 뒤 캐시가 확장본이다.

[왜 표본인가]
`fine.npy` 가 17GB 다. 전부 읽으면 34GB I/O 인데, 불일치는 레시피나 시드가
어긋났을 때 생기고 그때는 **거의 모든 창이 다르다**. 무작위 2,000창이면
1창만 어긋나도 잡을 확률이 사실상 1 이다. 라벨·광역·관측은 작으므로 전부 본다.
`--full` 로 세밀 갈래까지 전수 비교한다.
"""
from pathlib import Path
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# 세밀 갈래를 제외한 배열. 전부 비교해도 1GB 남짓이다.
SMALL = ("wide", "y_power", "y_on", "y_plugged", "y_standby", "y_state",
         "obs_harm", "p_noise", "p_observed")
META_MUST_MATCH = ("n_windows", "window_cycles", "appliances", "time_split",
                   "seed", "n_wide", "exclude_activation_files")


def _meta(d: Path) -> dict:
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="캐시 앞 채널 일치 검증 (12.40.1절)")
    ap.add_argument("base", help="기준 캐시 (채널 적은 쪽)")
    ap.add_argument("ext", help="확장 캐시 (채널 많은 쪽)")
    ap.add_argument("--sample", type=int, default=2000, help="세밀 갈래 표본 창 수")
    ap.add_argument("--full", action="store_true", help="세밀 갈래도 전수 비교")
    ap.add_argument("--chunk", type=int, default=500, help="한 번에 읽을 창 수")
    a = ap.parse_args()

    A, B = Path(a.base), Path(a.ext)
    ma, mb = _meta(A), _meta(B)
    ok = True

    print("=" * 78)
    print(f"[캐시 앞 채널 검증]  기준 {A}  vs  확장 {B}")
    print("=" * 78)

    for k in META_MUST_MATCH:
        if ma.get(k) != mb.get(k):
            print(f"  ✗ meta '{k}' 불일치: {ma.get(k)!r} vs {mb.get(k)!r}")
            ok = False
    ca, cb = ma["fine_shape"][0], mb["fine_shape"][0]
    if ma["fine_shape"][1] != mb["fine_shape"][1]:
        print(f"  ✗ 세밀 길이 불일치: {ma['fine_shape']} vs {mb['fine_shape']}")
        return 1
    if cb < ca:
        print(f"  ✗ 확장본의 채널이 더 적습니다: {ca} -> {cb}")
        return 1
    print(f"  세밀 채널 {ca} -> {cb}  (뒤 {cb - ca}채널이 새로 붙었다)")
    if not ok:
        print("\n  메타가 다르면 앞 채널이 같아도 같은 실험이 아니다. 중단한다.")
        return 1

    n = ma["n_windows"]
    for name in SMALL:
        x = np.load(A / f"{name}.npy", mmap_mode="r")
        y = np.load(B / f"{name}.npy", mmap_mode="r")
        if x.shape != y.shape:
            print(f"  ✗ {name:11s} 모양 불일치 {x.shape} vs {y.shape}")
            ok = False
            continue
        # NaN 이 있으면 == 이 거짓이 된다. 바이트로 본다.
        same = all(np.array_equal(np.asarray(x[i:i + 20000]).view(np.uint8),
                                  np.asarray(y[i:i + 20000]).view(np.uint8))
                   for i in range(0, len(x), 20000))
        print(f"  {'✓' if same else '✗'} {name:11s} {'일치' if same else '**불일치**'}"
              f"  {x.shape}")
        ok &= same

    fa = np.load(A / "fine.npy", mmap_mode="r")
    fb = np.load(B / "fine.npy", mmap_mode="r")
    idx = (np.arange(n) if a.full
           else np.sort(np.random.default_rng(0).choice(n, min(a.sample, n), replace=False)))
    bad = 0
    for i in range(0, len(idx), a.chunk):
        sel = idx[i:i + a.chunk]
        u = np.asarray(fa[sel]).view(np.uint8)
        v = np.asarray(fb[sel][:, :ca]).view(np.uint8)
        if not np.array_equal(u, v):
            bad += int((u != v).any(axis=(1, 2)).sum())
    print(f"  {'✓' if bad == 0 else '✗'} {'fine[:, :%d]' % ca:11s} "
          f"{'일치' if bad == 0 else f'**{bad}창 불일치**'}  "
          f"({'전수 %d창' % n if a.full else '표본 %d창' % len(idx)})")
    ok &= bad == 0

    print()
    if ok:
        print("  판정: 앞 채널이 같다. **기준 캐시의 체크포인트를 대조군으로 쓸 수 있다.**")
        return 0
    print("  판정: 다르다. 확장 캐시에서 `--fine-channels %d` 로 대조군을 따로"
          " 학습해야 한다." % ca)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
