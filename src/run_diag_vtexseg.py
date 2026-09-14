# -*- coding: utf-8 -*-
"""창-안 전압 텍스처 변동의 **천장**을 잰다 (14.51 준비).

왜
--
14.50 이 쟀다: 합성 창 안의 `Re(V_h)` 변동이 실측의 **13~35%** 뿐이다. 기본파는 반대로
1.4~1.6배 **크다** — `v_open` 의 OU 표류가 모든 차수를 같은 비로 흔들기 때문이다.
즉 **텍스처가 소유한 축(파형의 모양)이 비어 있다**. 원인은 하나다:

    `_terminal_voltage_harmonics`:  V_h[t] = rel[h]·v_open[t] − z_h·I_h[t]
                                             ^^^^^^ 창 전체에서 **상수**

`rel` 은 창마다 하나 뽑히고 창 안에서 안 움직인다. `Texture` 는 녹화의 `step_s` 구간
중앙값이라 **그 녹화의 다음 텍스처**가 이미 라이브러리에 있는데 안 쓰고 있었다.

무엇을 재나
-----------
고침의 **천장**이다. 창 하나를 n 토막 내고 각 토막에 그 녹화의 **연속된** 텍스처를
얹으면 창-안 변동이 얼마나 살아나는가. 실측 녹화 자신에서 잰다 (합성기를 안 돈다):

    원시   : 60초 창 안 `rel_h = V_h/|V_1|` 의 사이클별 표준편차   <- 참값
    정적   : 창 전체 중앙값 하나 -> **0** (지금 이것이다)
    n토막  : 60/n 초 중앙값 n개의 계단함수의 표준편차              <- 고치면 이것

`rel` 로 재는 까닭: `v_open` 표류는 모든 차수를 같은 비로 흔들므로 `V_h` 절대값으로 재면
이미 있는 성분이 섞인다. `V_h/|V_1|` 이 **텍스처가 소유한 축**이다.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.inputs import VOLT_ORDERS      # noqa: E402
from src.synthesis.vtexture import H          # noqa: E402


def _cmed(x: np.ndarray) -> np.ndarray:
    """복소 중앙값 — 실수·허수 따로 (`vtexture` 와 같은 방식, 위상 상쇄를 피한다)."""
    return np.median(x.real, 0) + 1j * np.median(x.imag, 0)


def window_stats(rel: np.ndarray, ok: np.ndarray, w: int, nsegs, min_ok: int = 1800):
    """(창, 차수) 별 `Re/Im(rel_h)` 표준편차. 반환 {n: (창수, 표준편차 (K,2))}.

    `n = 0` 은 원시(사이클별)다. `n >= 1` 은 창을 n 토막 낸 계단함수다 (n=1 이 지금 판).
    """
    idx = [h - 1 for h in VOLT_ORDERS if h > 1]
    out = {n: [] for n in [0] + list(nsegs)}
    for a in range(0, len(rel) - w + 1, w):
        sl = ok[a:a + w]
        if int(sl.sum()) < min_ok:
            continue
        seg = rel[a:a + w][sl][:, idx]                      # (m, K) complex
        out[0].append(np.stack([seg.real.std(0), seg.imag.std(0)], 1))
        for n in nsegs:
            step = np.empty((w, len(idx)), dtype=np.complex128)
            bad = False
            for k in range(n):
                b0, b1 = k * w // n, (k + 1) * w // n
                s2 = ok[a + b0:a + b1]
                if int(s2.sum()) < 30:
                    bad = True
                    break
                step[b0:b1] = _cmed(rel[a + b0:a + b1][s2][:, idx])[None, :]
            if bad:
                out[n].append(np.full((len(idx), 2), np.nan))
                continue
            out[n].append(np.stack([step.real.std(0), step.imag.std(0)], 1))
    return {n: (len(v), np.asarray(v)) for n, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", default="processed_data/npz")
    ap.add_argument("--window-cycles", type=int, default=3600, help="창 길이 (60Hz 사이클)")
    ap.add_argument("--segs", default="1,2,3,6", help="토막 수 목록. 1 이 지금 판(정적)")
    ap.add_argument("--roles", default="device,noise")
    a = ap.parse_args()

    nsegs = [int(x) for x in a.segs.split(",") if x.strip()]
    roles = {r.strip() for r in a.roles.split(",") if r.strip()}
    w = int(a.window_cycles)
    ords_ = [h for h in VOLT_ORDERS if h > 1]

    acc = {n: [] for n in [0] + nsegs}
    files = 0
    for f in sorted(Path(a.npz_dir).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if "voltage_harmonics_complex" not in z.files:
            continue
        meta = json.loads(str(z["metadata_json"])) if "metadata_json" in z.files else {}
        if not meta.get("voltage_phase_available", False):
            continue
        if roles and meta.get("file_role") not in roles:
            continue
        V = np.asarray(z["voltage_harmonics_complex"], dtype=np.complex128)
        if V.ndim != 2 or V.shape[1] < H or len(V) < w:
            continue
        valid = (np.asarray(z["is_valid"]) == 1) if "is_valid" in z.files else np.ones(len(V), bool)
        v1 = np.abs(V[:, 0])
        ok = valid & (v1 > 150.0) & (v1 < 280.0) & np.isfinite(V[:, :H]).all(axis=1)
        rel = V[:, :H] / np.where(v1 > 0, v1, 1.0)[:, None]
        st = window_stats(rel, ok, w, nsegs)
        if st[0][0] == 0:
            continue
        files += 1
        for n, (_c, arr) in st.items():
            acc[n].append(arr)

    if not files:
        print("전압 위상이 있는 녹화가 없다")
        return 1
    for n in acc:
        acc[n] = np.concatenate(acc[n], 0)                  # (창, K, 2)
    nwin = acc[0].shape[0]

    print(f"녹화 {files}개 · 창 {nwin}개 ({w} 사이클 = {w / 60.0:.0f}초) · "
          f"창-안 `rel_h = V_h/|V_1|` 의 표준편차 (×1e-4, 창 중앙값)\n")
    hdr = "  차수  성분      원시  " + "".join(f"{('n=%d' % n):>9}" for n in nsegs)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    raw_med = np.nanmedian(acc[0], 0)                       # (K,2)
    for i, h in enumerate(ords_):
        for j, part in enumerate(("Re", "Im")):
            r0 = raw_med[i, j] * 1e4
            row = f"  h{h:<4d} {part}  {r0:9.2f}  "
            for n in nsegs:
                v = np.nanmedian(acc[n], 0)[i, j] * 1e4
                row += f"{v:6.2f}({100 * v / max(r0, 1e-12):3.0f}%)"
            print(row)
    print()
    # 한 줄 요약 — 차수·성분 전부의 회수율 중앙값
    print("  회수율(원시 대비, 전 차수·성분 중앙값):")
    for n in nsegs:
        rr = np.nanmedian(acc[n], 0) / np.maximum(raw_med, 1e-12)
        print(f"    n={n:<2d}  {100 * np.median(rr):5.1f}%   "
              f"(최소 {100 * rr.min():.1f}% · 최대 {100 * rr.max():.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
