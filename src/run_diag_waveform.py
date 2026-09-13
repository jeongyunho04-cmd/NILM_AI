# -*- coding: utf-8 -*-
"""**파형에 더 안정한 것이 있는가** — 판별력의 바닥을 내릴 수 있는지 (13.84.47).

문제의 뿌리는 13.84.31·13.84.36 이 닫아 놓았다:
```
미니PC 12W 를 ±3W 로 읽으려면  형제 지문 오차 < 5.5%
그런데 충전기 h11~h13 은 같은 조건에서도 9~11% 흔들린다 (미니PC 자신은 0.3~4.1%)
```
13.84 의 모든 시도는 **같은 15개 고조파를 재배열**한 것이었다. 파이프라인은 사이클마다
고조파 15차(=900Hz)까지만 남기고 파형을 버리는데, 원시 CSV 에는 **15.36kHz**
(사이클당 256샘플) 가 40사이클씩 남아 있다.

묻는 것: **h11~h13 이 9~11% 흔들릴 때, 같이 안 흔들리는 파형 특징이 있는가.**
있으면 뺄셈을 거기 앵커해 바닥을 내릴 수 있다. 없으면 이 벽은 물리적이고,
정상상태 판별력을 올리려는 시도는 접는 것이 맞다.

⚠ 발췌마다 동작점이 다르면 비교가 무의미하다 — **와트당**으로 정규화하고 전력도 같이 찍는다.
⚠ 발췌가 40사이클(0.67초)뿐이라 13.84.36 의 61.7초 블록보다 **짧다.** 값이 아니라
  **특징 사이의 순위**를 읽어라.

    python -X utf8 src/run_diag_waveform.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

NH = 15


def load(path):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a["cyc"].astype(int), a["i_a"].astype(float), a["v_v"].astype(float)


def per_cycle(cyc, i, v):
    """사이클마다 (P, 고조파 (15,) complex, 파형 특징 dict)."""
    out = []
    for c in np.unique(cyc):
        m = cyc == c
        ii, vv = i[m], v[m]
        n = len(ii)
        if n < 200:
            continue
        P = float((ii * vv).mean())
        # 전압 기본파를 기준 위상으로 — 파일마다 시작 위상이 달라 그대로 비교하면 안 된다
        V = np.fft.rfft(vv) / n
        ph0 = np.angle(V[1])
        I = np.fft.rfft(ii) / n
        h = np.array([2 * I[k] * np.exp(-1j * k * ph0) for k in range(1, NH + 1)])
        rms = float(np.sqrt((ii ** 2).mean()))
        pk = float(np.abs(ii).max())
        # ── 파형 특징 (전압 위상 기준) ──────────────────────────────────────
        th = np.arange(n) * 2 * np.pi / n - ph0
        a_ = np.abs(ii)
        thr = 0.2 * pk
        cond = a_ > thr
        f = {
            "도통각(rad)": float(2 * np.pi * cond.mean()),
            "파고율": pk / rms if rms > 0 else np.nan,
            # 도통 구간의 **무게중심 위상** — 정류 구간이 전압 꼭대기 어디에 붙는가
            "도통중심(rad)": float(np.angle(np.sum(a_ * np.exp(1j * th)))),
            # 상승 기울기 — 다이오드가 켜지는 순간의 가파름
            "최대기울기": float(np.abs(np.diff(ii)).max() / (pk + 1e-12)),
            # 고차 에너지 몫 (h16 이상) — 15차 위에 무엇이 있나
            "h16+몫": float(np.sqrt((np.abs(I[NH + 1:]) ** 2).sum() * 2) / (rms + 1e-12)),
        }
        out.append((P, h, f))
    return out


def cv(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.std(x) / abs(np.mean(x))) if len(x) > 1 and abs(np.mean(x)) > 1e-12 else np.nan


def main():
    groups = {}
    for p in sorted(glob.glob("data/raw_*.csv")):
        stem = os.path.basename(p)[4:-4]
        app = stem.rsplit("_", 1)[0]
        groups.setdefault(app, []).append((stem, p))

    for app in ("laptop_charger", "minipc", "beam_projector"):
        if app not in groups:
            continue
        print("\n" + "=" * 92)
        print("## %s — 발췌 %d개 (각 40사이클 · 15.36kHz)" % (app, len(groups[app])))
        rows = []
        for stem, p in groups[app]:
            try:
                cyc, i, v = load(p)
            except Exception as e:
                print("   %-28s 못 읽음 (%s)" % (stem, e)); continue
            pc = per_cycle(cyc, i, v)
            if not pc:
                continue
            P = float(np.median([x[0] for x in pc]))
            if P < 3.0:
                print("   %-28s P %.1fW — 대기로 보여 뺀다" % (stem, P)); continue
            H = np.median(np.abs(np.stack([x[1] for x in pc])), axis=0) / P   # 와트당
            F = {k: float(np.median([x[2][k] for x in pc])) for k in pc[0][2]}
            rows.append((stem, P, H, F))
            print("   %-28s P %6.1fW · 사이클 %d" % (stem, P, len(pc)))
        if len(rows) < 3:
            print("   발췌가 3개 미만이라 산포를 못 잰다"); continue

        # 전력이 비슷한 것끼리만 — 충전기는 테이퍼라 동작점이 지문을 바꾼다 (13.84.32)
        Ps = np.array([r[1] for r in rows])
        med = np.median(Ps)
        sel = [r for r in rows if 0.6 * med <= r[1] <= 1.6 * med]
        print("   전력 중앙 %.1fW · 0.6~1.6배 안에 든 발췌 %d개" % (med, len(sel)))
        if len(sel) < 3:
            sel = rows
            print("   (너무 적어 전부 쓴다 — 동작점 교락 주의)")

        print("\n   발췌 간 변동계수 CV (작을수록 안정)")
        print("   %-16s %8s" % ("고조파(와트당)", "CV"))
        for o in (1, 5, 9, 11, 13, 15):
            print("   %-16s %7.1f%%" % ("h%d" % o, 100 * cv([r[2][o - 1] for r in sel])))
        print("   %-16s %8s" % ("파형 특징", "CV"))
        for k in sel[0][3]:
            print("   %-16s %7.1f%%" % (k, 100 * cv([r[3][k] for r in sel])))

    print("\n" + "=" * 92)
    print("읽는 법 — 충전기의 h11~h13 CV 보다 **뚜렷이 작은 파형 특징**이 있으면 앵커가 된다.")
    print("   요구치는 5.5% 다 (13.84.31 ④). 파형 특징도 10% 대면 벽은 물리적이다.")
    print("⚠ 발췌가 0.67초뿐이라 13.84.36 의 61.7초 블록보다 짧다 — **특징 사이 순위**만 읽어라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
