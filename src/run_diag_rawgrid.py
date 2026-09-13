# -*- coding: utf-8 -*-
"""원시 파형 스냅샷 재고 — **같은 기기, 다른 계통**이 생겼다 (13.84.65).

`circuit_model/README_v12.md` 의 "남은 일" 마지막 줄:
    *"저녁(216V, vh3<1%) 녹화는 R 을 어떻게 잡아도 15~30% — 원시 스냅샷이 전부 심야라서.
      **저녁 원시 스냅샷이 다음 자료**"*

그 자료가 들어와 있다. `data/raw_*.csv` 는 10,240 표본 @ 15.4kHz (= 40 사이클) 의 **파형**이고,
지금 두 계통이 갈린다:
    저녁  215~217V,  vh3 0.7~0.9%
    심야  229~232V,  vh3 2.8~4.1%
`circ12` 는 **심야만으로** 맞췄다 (raw_laptop_charger 5~8).

왜 이것이 큰가. 회로 모형의 가장 센 시험은 **같은 기기를 다른 전원에서** 예측하는 것이다 —
자유 모수가 0 이다. 그리고 13.84.56 이 잰 표류의 PC2 가 **ImV3 (14°)** 였다. 지금 두 계통의
vh3 가 0.8% 대 2.9~4.1% 니 **그 축을 4배로 벌려 놓은 짝 자료**다.

재는 것:
  ① 재고    파일마다 계통(Vrms·vh3)·동작점(P)·전류 크레스트
  ② 짝      같은 전력·다른 계통 짝에서 **전류가 얼마나 달라지나** (차수별)
  ③ 표류    그 차이가 13.84.52 의 **측정된 표류 부분공간** 안에 있나
  ④ 회로    회로가 그 차이를 예측하나 — `dI/dV3` 를 자유모수 없이 견준다

⚠ 계측 RC (τ=60µs) 는 두 파일에 **같게** 걸리므로 ②③ 의 비교는 그대로 선다.
  ④ 는 규약이 필요하다 (역RC(V) -> 시뮬 -> RC(i)) — `circuit_model/README_v12.md` 규약 2.

    python -X utf8 src/run_diag_rawgrid.py [--dev laptop_charger]
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

H = 15
ORD = [1, 3, 5, 7, 9, 11, 13, 15]


def load_raw(path):
    """(i, v, fs). 원시 스냅샷은 `i_a` · `v_v` 가 이미 스케일된 물리량이다."""
    a = np.genfromtxt(path, delimiter=",", names=True, usecols=("i_a", "v_v", "fs_hz"))
    return np.asarray(a["i_a"], float), np.asarray(a["v_v"], float), float(np.median(a["fs_hz"]))


def f0_of(v, fs, lo=58.0, hi=62.0):
    """전압에서 기본 주파수를 찾는다 — 표본/주기가 정수가 아니라(256.05) 빈으로 못 쓴다."""
    n = len(v)
    t = np.arange(n) / fs
    best, bf = -1.0, 60.0
    for f in np.linspace(lo, hi, 801):
        c = np.cos(2 * np.pi * f * t); s = np.sin(2 * np.pi * f * t)
        p = (v @ c) ** 2 + (v @ s) ** 2
        if p > best:
            best, bf = p, f
    return bf


def phasors(x, fs, f0, h=H):
    """h차까지 **최소제곱** 페이저 (복소, 진폭 규약). 창함수 없이 정확한 주파수로 맞춘다."""
    n = len(x); t = np.arange(n) / fs
    cols = [np.ones(n)]
    for k in range(1, h + 1):
        cols += [np.cos(2 * np.pi * k * f0 * t), np.sin(2 * np.pi * k * f0 * t)]
    A = np.stack(cols, 1)
    b = np.linalg.lstsq(A, x, rcond=None)[0]
    return np.array([b[1 + 2 * k] - 1j * b[2 + 2 * k] for k in range(h)])


def scan(dev, H=H):
    """⚠ **위상 기준을 반드시 옮긴다.** `phasors` 는 그 파일의 **첫 표본**을 시간 원점으로
    잡는데 캡처 시작 시각은 파일마다 제멋대로다 — 그대로 두면 파일끼리의 복소 차가 무의미하다
    (크기비는 멀쩡하다). 저장소 규약 `arg(X_h) − h·arg(V_1)` 로 옮기고 RMS 로 낸다.
    그러면 `voltage_harmonics_complex`·`sim_harmonics` 와 같은 영역이 된다.
    """
    out = []
    for p in sorted(glob.glob("data/raw_%s_*.csv" % dev),
                    key=lambda x: int(x.split("_")[-1].split(".")[0])):
        i, v, fs = load_raw(p)
        f0 = f0_of(v, fs)
        # ⚠ 원시는 15.4kHz(= 256 표본/주기) 라 **h128 까지** 담고 있다. h15 절단은
        #   2Hz 펌웨어의 한계지 이 파일의 한계가 아니다 — `H` 로 더 가져올 수 있다.
        vi = phasors(v, fs, f0, H); ii = phasors(i, fs, f0, H)
        h = np.arange(1, len(vi) + 1)
        rot = np.exp(-1j * h * np.angle(vi[0])) / np.sqrt(2.0)
        vi, ii = vi * rot, ii * rot
        out.append(dict(name=os.path.basename(p)[4:-4], P=float((v * i).mean()),
                        vrms=float(np.sqrt((v ** 2).mean())), f0=f0,
                        vh3=100 * abs(vi[2]) / abs(vi[0]), V=vi, I=ii,
                        crest=float(np.abs(i).max() / max(np.sqrt((i ** 2).mean()), 1e-12))))
    return out


def ang(u, v):
    a = np.concatenate([u.real, u.imag]); b = np.concatenate([v.real, v.imag])
    c = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-18)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+",
                    default=["laptop_charger", "minipc", "beam_projector"])
    ap.add_argument("--basis", default="results/drift_basis.npz")
    a = ap.parse_args()

    data = {d: scan(d) for d in a.dev}

    print("① **재고** — 원시 파형 스냅샷 (10,240 표본 @ 15.4kHz = 40 사이클)")
    print("   %-22s %8s %8s %8s %8s %8s  %s"
          % ("파일", "Vrms", "P W", "|I1| mA", "크레스트", "vh3 %", "계통"))
    for d in a.dev:
        for r in data[d]:
            grid = "저녁" if r["vh3"] < 1.5 else "심야"
            print("   %-22s %8.1f %8.1f %8.1f %8.2f %8.2f  %s"
                  % (r["name"], r["vrms"], r["P"], 1000 * abs(r["I"][0]),
                     r["crest"], r["vh3"], grid))

    print("\n② **짝** — 같은 전력·다른 계통. 전류가 차수별로 얼마나 다른가 (심야/저녁 크기비)")
    print("   %-34s %7s %7s  %s" % ("짝", "ΔP W", "ΔVrms", "  ".join("h%-4d" % o for o in ORD)))
    pairs = []
    for d in a.dev:
        ev = [r for r in data[d] if r["vh3"] < 1.5]
        nt = [r for r in data[d] if r["vh3"] >= 1.5]
        for e in ev:
            best = min(nt, key=lambda r: abs(r["P"] - e["P"]), default=None)
            if best is None or abs(best["P"] - e["P"]) > 0.15 * max(e["P"], 1.0):
                continue
            pairs.append((d, e, best))
            rr = np.abs(best["I"][[o - 1 for o in ORD]]) / np.maximum(
                np.abs(e["I"][[o - 1 for o in ORD]]), 1e-12)
            print("   %-34s %7.1f %7.1f  %s"
                  % ("%s -> %s" % (e["name"][-14:], best["name"][-10:]),
                     best["P"] - e["P"], best["vrms"] - e["vrms"],
                     "  ".join("%5.2f" % x for x in rr)))
    if not pairs:
        print("   짝이 없다")
        return 0

    print("\n③ **표류** — 그 차이가 13.84.52 의 측정된 표류 부분공간 안에 있나")
    try:
        B = np.load(a.basis, allow_pickle=True)
        V = np.asarray(B["V"], float)                       # (k, 2Ho)
        ords = [int(x) for x in B["orders"]]
    except Exception as e:
        print("   %s 를 못 읽었다 (%s) — `run_build_drift.py` 를 먼저 돌려라" % (a.basis, e))
        return 0
    idx = [o - 1 for o in ords]
    print("   기저 %d차원 · 차수 %s" % (V.shape[0], ords))
    print("   %-34s %10s %10s %s" % ("짝", "|ΔI| mA", "밖에 남음", "성분별 주각"))
    for d, e, n in pairs:
        # ⚠ **와트당**으로 견준다. 계통이 다르면 Vrms 도 전력도 조금씩 다른데, 정전력에서
        #   I ∝ 1/V 라 크기를 안 맞추면 h1 에 −7% 짜리 가짜 성분이 생긴다. 저장소의
        #   `sig` 규약(= I/P)과 같은 자를 쓰면 그 둘이 한꺼번에 빠진다.
        dv = (n["I"][idx] / n["P"] - e["I"][idx] / e["P"]) * e["P"]
        x = np.concatenate([dv.real, dv.imag])
        pr = V.T @ (V @ x)
        res = np.linalg.norm(x - pr) / max(np.linalg.norm(x), 1e-18)
        cs = [np.degrees(np.arccos(np.clip(abs(V[j] @ x) / max(np.linalg.norm(x), 1e-18), 0, 1)))
              for j in range(V.shape[0])]
        print("   %-34s %10.1f %9.1f%%  %s"
              % ("%s -> %s" % (e["name"][-14:], n["name"][-10:]),
                 1000 * np.linalg.norm(np.abs(dv)), 100 * res,
                 " ".join("PC%d %2.0f°" % (j + 1, c) for j, c in enumerate(cs))))
    null = 100 * np.sqrt(1.0 - V.shape[0] / float(2 * len(idx)))
    print("\n   무작위 벡터라면 밖에 **%.0f%%** 가 남는다 (%d차원 중 %d차원 기저)."
          % (null, 2 * len(idx), V.shape[0]))
    print("   읽는 법 — '밖에 남음' 이 그보다 훨씬 작으면 **두 계통의 차이가 곧 표류**다.")
    print("   ⚠ 기저는 %s 로 만들었다 — 그 둘이 안에 드는 것은 일부 자기참조다."
          % " · ".join(str(x) for x in B["sources"]))
    print("   **미니PC 는 기저에 안 들어갔다** — 독립 증거는 그쪽이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
