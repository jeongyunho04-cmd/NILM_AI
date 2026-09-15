# -*- coding: utf-8 -*-
"""구운 캐시의 **창-안 전압 텍스처 변동**이 실측과 얼마나 닮았나 (14.110).

사용자: *"이번에 만든 캐시가 얼마나 실측과 비슷한지 검사해 보자."*

**우리가 바꾼 축에서 잰다.** `--vtex-seg-s/coarse-s` 가 사려던 것은 딱 하나다 —
세밀 갈래가 보는 10초 안에서 `V_h/|V_1|` 이 실측만큼 움직이는가. 14.50 이 잰 바로는
합성이 실측의 **13~35%** 뿐이었고, 14.65 가 천장을 이렇게 적었다 (세밀 창 기준):

```
  토막 n=1  0.0%   <- seg 끔. 세밀 창 안에 경계가 **구조적으로** 없다
        n=2 43.5% · n=3 61.8% · n=4 73.0% · n=6 83.7%
```
`train60_v32hs3` 는 세밀 창 안에 토막이 **9장**이니 90% 근처여야 한다. 실제로 그런가.

재는 법 — **두 쪽을 같은 자로**
```
  실측   processed_data/npz 의 `voltage_harmonics_complex` 로 사이클별
         `rel_h = V_h/|V_1|` 을 만들고, 600사이클(10초) 창마다 Re/Im 표준편차
  캐시   `fine.npy` 45~56 채널을 **눈금을 되돌려** 같은 `rel_h` 로 만든 뒤 같은 통계
         (h1: x*V_SPAN+V_CENTER · h>1: sinh(x)/VOLT_HARM_SCALE — `inputs.py` 의 역)
```
⚠ `V_h` 절대값이 아니라 `V_h/|V_1|` 로 재는 까닭은 14.51 에 있다 — `v_open` 표류가
  모든 차수를 같은 비로 흔들어서, 절대값으로 재면 텍스처가 아닌 성분이 섞인다.

    python -X utf8 src/run_diag_texreal.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.model.inputs import (FINE_VOLT0, V_CENTER, V_SPAN,  # noqa: E402
                              VOLT_HARM_SCALE, VOLT_ORDERS)

W = 600          #: 세밀 창 = 뒤 10초
NH = len(VOLT_ORDERS)


def rel_from_fine(f: np.ndarray) -> np.ndarray:
    """(57, 600) 세밀 -> (600, NH) 복소 `rel_h`. `inputs.py` 의 눈금을 되돌린다."""
    re = np.asarray(f[FINE_VOLT0:FINE_VOLT0 + NH], np.float64)          # (NH, 600)
    im = np.asarray(f[FINE_VOLT0 + NH:FINE_VOLT0 + 2 * NH], np.float64)
    vr, vi = np.empty_like(re), np.empty_like(im)
    for s, h in enumerate(VOLT_ORDERS):
        if h == 1:
            vr[s] = re[s] * V_SPAN + V_CENTER
            vi[s] = im[s] * V_SPAN
        else:
            vr[s] = np.sinh(re[s]) / VOLT_HARM_SCALE
            vi[s] = np.sinh(im[s]) / VOLT_HARM_SCALE
    v = (vr + 1j * vi).T                                                # (600, NH)
    v1 = np.abs(v[:, 0])
    return v / np.maximum(v1, 1e-9)[:, None]


def std_of(rel: np.ndarray) -> np.ndarray:
    """(600, NH) -> (NH, 3) 창 안 Re/Im 표준편차 **와 평균 |rel|**.

    평균을 같이 내는 까닭: 눈금 되돌리기가 틀렸으면 표준편차만 봐서는 모른다.
    평균 `|rel_h|` 는 실측과 **자릿수가 같아야** 한다 (h3/h1 은 0.5~3%).
    그게 맞은 뒤에야 표준편차 비교가 뜻을 갖는다 ([[check-conditioning-before-believing-a-fit]]).
    """
    return np.stack([rel.real.std(0), rel.imag.std(0), np.abs(rel).mean(0)], axis=1)


def from_cache(path: Path, n: int):
    arr = np.load(path / "fine.npy", mmap_mode="r")
    m = min(n, len(arr))
    return np.stack([std_of(rel_from_fine(arr[i])) for i in range(m)])   # (m, NH, 2)


def from_real(npz_dir: Path, n: int):
    from src.preprocessing import load_nilm_npz
    out = []
    files = sorted(npz_dir.glob("*.npz"))
    for p in files:
        try:
            r = load_nilm_npz(str(p))
        except Exception:                                   # noqa: BLE001
            continue
        vh = np.asarray(r.get("voltage_harmonics_complex"))
        if vh is None or vh.ndim != 2 or len(vh) < W:
            continue
        # ⚠ `voltage_harmonics_complex` 의 열은 **차수 1..N 순서**다 (0=h1, 1=h2, …).
        #   `vh[:, :NH]` 로 잘라 쓰면 h1~h6 을 뽑아놓고 h1·h3·h5·h7·h9·h11 이라 부르게
        #   된다 — 자 검정에서 "실측 h3 가 h5 보다 100배 작다" 는 뒤집힌 꼴로 잡혔다.
        col = [h - 1 for h in VOLT_ORDERS]
        v1 = np.abs(vh[:, 0])
        rel = vh[:, col] / np.maximum(v1, 1e-9)[:, None]
        ok = np.isfinite(rel).all(1) & (v1 > 100.0)
        for i in range(0, len(rel) - W, W):
            s = slice(i, i + W)
            if ok[s].mean() < 0.99:
                continue
            out.append(std_of(rel[s]))
            if len(out) >= n:
                return np.stack(out), len(files)
    return (np.stack(out) if out else np.zeros((0, NH, 2))), len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/train60_v32hs3")
    ap.add_argument("--ref", default="cache/train60_v32h", help="seg 끔 대조")
    ap.add_argument("--npz-dir", default="processed_data/npz")
    ap.add_argument("--n", type=int, default=400, help="양쪽에서 볼 창 수")
    a = ap.parse_args()

    real, nf = from_real(Path(a.npz_dir), a.n)
    if not len(real):
        raise SystemExit("실측 창을 못 모았다 — %s 를 봐라" % a.npz_dir)
    trt = from_cache(Path(a.cache), a.n)
    ref = from_cache(Path(a.ref), a.n) if (Path(a.ref) / "fine.npy").exists() else None

    print("창-안 전압 텍스처 변동 — 캐시 대 실측 (14.110)")
    print("  창 %d사이클(10초) · 실측 파일 %d개 · 창 실측 %d / 처치 %d%s\n"
          % (W, nf, len(real), len(trt), " / 대조 %d" % len(ref) if ref is not None else ""))
    print("  ★ 자 검정 — 평균 |rel_h| (자릿수가 같아야 한다)")
    print("    %-5s %11s %11s %11s" % ("차수", "실측", "처치", "대조"))
    for s, h in enumerate(VOLT_ORDERS):
        C = float(np.median(ref[:, s, 2])) if ref is not None else float("nan")
        print("    h%-4d %11.4e %11.4e %11.4e"
              % (h, float(np.median(real[:, s, 2])), float(np.median(trt[:, s, 2])), C))
    print("")
    print("  %-6s %2s | %11s %11s %11s | %9s %9s"
          % ("차수", "", "실측 sd", "처치 sd", "대조 sd", "처치 회수", "대조 회수"))
    rec_t, rec_r = [], []
    for s, h in enumerate(VOLT_ORDERS):
        for j, nm in enumerate(("Re", "Im")):
            if h == 1:
                continue                       # rel[0] = 1 (정의상 상수 — 잴 것이 없다)
            R = float(np.median(real[:, s, j]))
            T = float(np.median(trt[:, s, j]))
            C = float(np.median(ref[:, s, j])) if ref is not None else float("nan")
            rt, rc = 100.0 * T / max(R, 1e-12), 100.0 * C / max(R, 1e-12)
            rec_t.append(rt)
            if ref is not None:
                rec_r.append(rc)
            print("  h%-5d %2s | %11.3e %11.3e %11.3e | %8.0f%% %8.0f%%"
                  % (h, nm, R, T, C, rt, rc))
    print("")
    print("  **회수율 중앙 — 처치 %.0f%% · 대조 %.0f%%**"
          % (np.median(rec_t), np.median(rec_r) if rec_r else float("nan")))
    print("  (14.65 가 적은 천장: 토막 n=1 은 0% · n=2 43.5% · n=3 61.8% · n=6 83.7%)")
    print("  100% 를 넘으면 **과하다** — 텍스처가 실측보다 심하게 흔들린다는 뜻이고")
    print("  그것도 분포 차이다. 여기 sd 는 **단자 전압**이라 부하가 만드는 변동도 섞여 있다")
    print("  (그래서 seg 를 꺼도 0% 가 아니다 — 14.65 의 0% 는 텍스처 성분만 잰 값이다).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
