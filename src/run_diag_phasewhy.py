# -*- coding: utf-8 -*-
"""복합에서 위상이 왜 안 통하나 — 후보 셋을 가른다 (13.84.51).

13.84.50: 격리 녹화에서는 상대위상이 LORO AUC 1.000 인데 복합 전이 Δ 에서는 3° 안에 겹친다.
그 표의 차이가 **차수에 비례**한다는 것이 실마리다:
```
차이(복합−격리)  h5 +15  h9 +31  h11 +37  h13 +56  h15 +86  (도)
÷차수            3.0     3.4     3.4      4.3      5.7      -> 거의 상수 3~4°/h
```
상대위상은 `∠h_k − k·∠h1` 이라 **h1 위상 기준에 δ 오차가 있으면 정확히 −k·δ 로 나타난다.**
즉 사건마다 **스칼라 하나**짜리 방해변수일 수 있다. 그렇다면 빼면 되살아난다.

가르는 것 셋:
  ① 동작점   격리 녹화 안에서 상대위상이 **전력에 따라** 움직이는가 (움직이면 대역 차이가 원인)
  ② 회전     복합 Δ 를 사건마다 회전 c 하나로 격리 템플릿에 맞출 수 있는가. 맞춘 뒤 잔차는?
  ③ 판별     **회전을 허용한 뒤에도** 충전기와 미니PC 가 갈리는가 (갈리면 처방이 선다)

    python -X utf8 src/run_diag_phasewhy.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.preprocessing import load_nilm_npz
from src.run_diag_sepfloor import load_cycles

FS = 60
WIN, MARGIN = 8, 3
ORD = [3, 5, 7, 9, 11, 13, 15]
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
APPS = ["laptop_charger", "beam_projector", "minipc"]


def relph(d):
    p = np.angle(d) - np.arange(1, 16) * np.angle(d[0])
    return np.degrees(np.angle(np.exp(1j * p)))


def cmed(a):
    return float(np.degrees(np.angle(np.exp(1j * np.radians(np.asarray(a, float))).mean())))


def csd(a):
    R = np.abs(np.exp(1j * np.radians(np.asarray(a, float))).mean())
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))


def fit_rot(obs, tmpl, orders):
    """obs_k ≈ tmpl_k − k·c 가 되는 c 를 원형 최소제곱으로 (도/차수). 반환 (c, 잔차 RMS)."""
    k = np.array(orders, float)
    o = np.radians([obs[i - 1] for i in orders])
    t = np.radians([tmpl[i] for i in orders])
    best, bc = None, 0.0
    for c in np.arange(-15.0, 15.001, 0.05):          # ±15°/h 를 훑는다
        r = np.angle(np.exp(1j * (o + np.radians(c) * k - t)))
        v = float(np.sqrt((r ** 2).mean()))
        if best is None or v < best:
            best, bc = v, c
    return bc, float(np.degrees(best))


def main():
    # ── ① 동작점 — 격리 녹화 안에서 전력에 따라 움직이는가 ─────────────────
    print("① 동작점 — 격리 녹화의 상대위상이 전력을 타는가 (도)")
    print("   %-16s %-12s %5s | %s" % ("기기", "전력대(W)", "사이클", "".join("  %-7s" % ("h%d" % o) for o in ORD)))
    TMPL = {}
    for app, bands in (("laptop_charger", [(19, 28), (28, 40), (40, 55), (55, 70)]),
                       ("minipc", [(8, 14), (14, 19), (19, 24), (24, 30)])):
        for lo, hi in bands:
            G = load_cycles(app, lo, hi)
            if not G:
                continue
            hc = np.vstack([x[2] for x in G])
            ph = np.array([relph(h) for h in hc])
            med = [cmed(ph[:, o - 1]) for o in ORD]
            print("   %-16s %-12s %5d | %s"
                  % (app, "%d~%d" % (lo, hi), len(hc), "".join("  %7.0f" % v for v in med)))
            if (app == "laptop_charger" and (lo, hi) == (28, 40)) or \
               (app == "minipc" and (lo, hi) == (8, 14)):
                pass
            TMPL.setdefault(app, {})[(lo, hi)] = {o: m for o, m in zip(ORD, med)}

    # 복합 사건의 ΔP 중앙에 가장 가까운 대역을 템플릿으로 쓴다
    tmpl = {"laptop_charger": TMPL["laptop_charger"][(28, 40)],
            "minipc": TMPL["minipc"][(8, 14)]}
    print("   -> 템플릿: 충전기 28~40W · 미니PC 8~14W (복합 사건 ΔP 중앙 36W · 9W 에 맞춤)")

    # ── 복합 사건 모으기 ────────────────────────────────────────────────────
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    rows = {a: [] for a in APPS}
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(H)
        spec = ev[stem]
        # 그 사건 부근에서 **다른 기기**도 움직였나 (오염 판정)
        edges = []
        for x, d in spec["intervals"].items():
            for t0, t1 in d.get("on", []):
                edges += [(x, t0), (x, t1)]
        for app in APPS:
            for t0, t1 in spec["intervals"].get(app, {}).get("on", []):
                for t, sg in ((t0, 1.0), (t1, -1.0)):
                    i = int(round(t * FS))
                    a0, a1 = i - (MARGIN + WIN) * FS, i - MARGIN * FS
                    b0, b1 = i + MARGIN * FS, i + (MARGIN + WIN) * FS
                    if a0 < 0 or b1 > n:
                        continue
                    d = sg * (H[b0:b1].mean(0) - H[a0:a1].mean(0))
                    dp = sg * float(P[b0:b1].mean() - P[a0:a1].mean())
                    if abs(d[0]) < 1e-4 or dp < 3.0:
                        continue
                    dirty = any(x != app and abs(tt - t) <= 15.0 for x, tt in edges)
                    rows[app].append((stem, t, dp, relph(d), dirty))

    # ── ② 회전 하나로 맞춰지는가 ────────────────────────────────────────────
    print("\n② 사건마다 회전 c 하나로 격리 템플릿에 맞춘다 (h3~h15)")
    print("   %-16s %5s | %9s %9s | %9s" % ("기기", "사건", "회전 c 중앙", "c 표준편차", "맞춘 뒤 잔차"))
    fitted = {}
    for app in ("laptop_charger", "minipc"):
        R = rows[app]
        if not R:
            continue
        cs, res = [], []
        for x in R:
            c, v = fit_rot(x[3], tmpl[app], ORD)
            cs.append(c); res.append(v)
            fitted.setdefault(app, []).append((x, c, v))
        print("   %-16s %5d | %9.2f %9.2f | %8.0f°"
              % (app, len(R), np.median(cs), np.std(cs), np.median(res)))
    print("   견줌 — 맞추기 전 잔차:")
    for app in ("laptop_charger", "minipc"):
        R = rows[app]
        if not R:
            continue
        v0 = [fit_rot(x[3], tmpl[app], ORD)[1] for x in R]
        raw = []
        for x in R:
            k = np.array(ORD, float)
            o = np.radians([x[3][i - 1] for i in ORD])
            t = np.radians([tmpl[app][i] for i in ORD])
            raw.append(float(np.degrees(np.sqrt((np.angle(np.exp(1j * (o - t))) ** 2).mean()))))
        print("   %-16s %5s | %9s %9s | %8.0f° -> %.0f°"
              % (app, "", "", "", np.median(raw), np.median(v0)))

    # ── ③ 회전을 허용한 뒤에도 갈리는가 ─────────────────────────────────────
    print("\n③ **회전을 허용한 뒤** 두 템플릿 중 어느 쪽에 더 잘 맞는가 (최근접 잔차)")
    print("   %-16s %5s %9s %9s %8s" % ("참 기기", "사건", "자기 잔차", "남 잔차", "맞춘 몫"))
    tot_ok = tot_n = 0
    for app, other in (("laptop_charger", "minipc"), ("minipc", "laptop_charger")):
        R = rows[app]
        if not R:
            continue
        own, oth, ok = [], [], 0
        for x in R:
            _, v1 = fit_rot(x[3], tmpl[app], ORD)
            _, v2 = fit_rot(x[3], tmpl[other], ORD)
            own.append(v1); oth.append(v2); ok += int(v1 < v2)
        tot_ok += ok; tot_n += len(R)
        print("   %-16s %5d %9.0f° %9.0f° %7.2f" % (app, len(R), np.median(own), np.median(oth), ok / len(R)))
    print("   전체 맞춘 몫 %.2f  (우연 0.50)" % (tot_ok / max(tot_n, 1)))

    # ── 보너스: 오염 여부로 갈라 본다 ───────────────────────────────────────
    print("\n④ 오염(±15초 안에 다른 기기 전이) 여부로 갈라")
    for app in ("laptop_charger", "minipc"):
        for dirty in (False, True):
            R = [x for x in rows[app] if x[4] == dirty]
            if len(R) < 2:
                continue
            v = [fit_rot(x[3], tmpl[app], ORD)[1] for x in R]
            print("   %-16s %-6s %3d개 · 맞춘 뒤 잔차 중앙 %.0f°"
                  % (app, "오염" if dirty else "깨끗", len(R), np.median(v)))
    print("\n⚠ 사건 13~15개. 자릿수만 읽어라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
