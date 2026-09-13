# -*- coding: utf-8 -*-
"""13.84.48 의 위상 판별력이 **복합 실측에서도 서는가** (13.84.50).

13.84.48 은 **격리 녹화**에서 h1 기준 상대위상이 충전기/미니PC 를 LORO AUC 1.000 으로
가른다고 쟀다. 그러나 복합에서는 관측이 합이고 위상은 선형으로 안 더해진다 —
합에서 한 기기의 상대위상을 그냥 뽑을 수 없다.

복합에서 물을 수 있는 형태는 **Δ** 다. 전이 순간의 차분은 그 기기의 지문에 가깝다
(13.84.31 ⑥: 편차 277 -> 31mA, 9배). 그래서 셋을 본다.

  ① 복합 Δ 의 상대위상이 격리 녹화 값과 맞는가        (자리 맞춤)
  ② 기기끼리 갈리는가                                 (분리도 · AUC)
  ③ 형제가 켜져 있을 때도 갈리는가                     (실패 ②·③ 의 조건)

⚠ Δ 는 전후 8초 평균이라 그 사이 다른 기기가 움직이면 오염된다. 사건 수가 적다
  (미니PC 15개 안팎) — [[one-seed-cannot-rank-small-classes]].

    python -X utf8 src/run_diag_phasereal.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.preprocessing import load_nilm_npz

FS = 60
WIN, MARGIN = 8, 3
ORD = [5, 9, 11, 13, 15]
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
#: 13.84.48 ② 가 격리 녹화에서 잰 값 (도, h1 기준 상대위상 중앙)
ISO = {"laptop_charger": {5: -31.0, 9: -53.7, 11: -64.0, 13: -71.0, 15: -73.6},
       "minipc": {5: -91.0, 9: -149.6, 11: 146.8, 13: 127.2, 15: 28.9}}
WATCH = ["laptop_charger", "beam_projector", "minipc"]


def relph(d):
    """Δ 페이저 (15,) complex -> h1 기준 상대위상 (도)."""
    p = np.angle(d) - np.arange(1, 16) * np.angle(d[0])
    return np.degrees(np.angle(np.exp(1j * p)))


def circ_med(a):
    """각도 배열 (도) 의 원형 중앙 — 감김에 안 속는다."""
    z = np.exp(1j * np.radians(np.asarray(a, float)))
    return float(np.degrees(np.angle(z.mean())))


def circ_sd(a):
    z = np.exp(1j * np.radians(np.asarray(a, float)))
    R = np.abs(z.mean())
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(R, 1e-12)))))


def main():
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    rows = {a: [] for a in WATCH}
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(H)
        spec = ev[stem]
        # 형제 SMPS 가 켜져 있는 구간 표시 (③ 용)
        sib_on = np.zeros(n, bool)
        for x in ("laptop_charger", "beam_projector"):
            for t0, t1 in spec["intervals"].get(x, {}).get("on", []):
                sib_on[int(t0 * FS):int(min(t1 * FS, n))] = True
        for app in WATCH:
            for t0, t1 in spec["intervals"].get(app, {}).get("on", []):
                for t, sgn in ((t0, +1.0), (t1, -1.0)):
                    i = int(round(t * FS))
                    a0, a1 = i - (MARGIN + WIN) * FS, i - MARGIN * FS
                    b0, b1 = i + MARGIN * FS, i + (MARGIN + WIN) * FS
                    if a0 < 0 or b1 > n:
                        continue
                    d = sgn * (H[b0:b1].mean(0) - H[a0:a1].mean(0))
                    dp = sgn * float(P[b0:b1].mean() - P[a0:a1].mean())
                    if abs(d[0]) < 1e-4 or dp < 3.0:
                        continue           # 너무 작은 사건은 위상이 잡음이다
                    # 그 순간 다른 SMPS 형제가 켜져 있었나
                    j = max(0, i - MARGIN * FS)
                    on = bool(sib_on[j]) if app == "minipc" else False
                    rows[app].append((stem, t / 60.0, dp, relph(d), on))

    print("복합 실측의 **전이 Δ** 에서 잰 h1 기준 상대위상 (도)")
    print("   %-16s %5s %8s | %s" % ("기기", "사건", "ΔP중앙", "".join("  %-9s" % ("h%d" % o) for o in ORD)))
    for app in WATCH:
        R = rows[app]
        if not R:
            print("   %-16s %5d  (없음)" % (app, 0)); continue
        med = [circ_med([x[3][o - 1] for x in R]) for o in ORD]
        sd = [circ_sd([x[3][o - 1] for x in R]) for o in ORD]
        print("   %-16s %5d %8.1f | %s"
              % (app, len(R), np.median([x[2] for x in R]),
                 "".join("  %5.0f±%-3.0f" % (m, s) for m, s in zip(med, sd))))
        if app in ISO:
            print("   %-16s %5s %8s | %s"
                  % ("  (격리 녹화)", "", "", "".join("  %5.0f    " % ISO[app][o] for o in ORD)))

    print("\n② 충전기 대 미니PC — 복합 Δ 만으로 가를 수 있나 (사건 단위)")
    A = [x[3] for x in rows["laptop_charger"]]
    B = [x[3] for x in rows["minipc"]]
    if len(A) >= 3 and len(B) >= 3:
        for o in ORD:
            a = np.array([x[o - 1] for x in A]); b = np.array([x[o - 1] for x in B])
            # 원형 거리 기반 분리 — 각 군 중앙에서의 각거리로 최근접 분류
            ma, mb = circ_med(a), circ_med(b)
            def near(v):
                da = abs(np.degrees(np.angle(np.exp(1j * np.radians(v - ma)))))
                db = abs(np.degrees(np.angle(np.exp(1j * np.radians(v - mb)))))
                return da < db
            acc = (np.mean([near(v) for v in a]) + np.mean([not near(v) for v in b])) / 2
            sep = abs(np.degrees(np.angle(np.exp(1j * np.radians(ma - mb)))))
            print("   h%-3d 충전기 %5.0f° · 미니PC %5.0f° · 사이 %5.0f° · 최근접 정확도 %.2f"
                  % (o, ma, mb, sep, acc))
    else:
        print("   사건이 모자란다 (충전기 %d · 미니PC %d)" % (len(A), len(B)))

    print("\n③ 미니PC 사건을 **형제 ON/OFF** 로 갈라 — 실패 ②·③ 의 조건")
    for on in (False, True):
        R = [x for x in rows["minipc"] if x[4] == on]
        if len(R) < 2:
            print("   형제 %-4s 사건 %d개 — 못 잰다" % ("ON" if on else "OFF", len(R))); continue
        med = [circ_med([x[3][o - 1] for x in R]) for o in ORD]
        sd = [circ_sd([x[3][o - 1] for x in R]) for o in ORD]
        print("   형제 %-4s %3d개 | %s" % ("ON" if on else "OFF", len(R),
                                          "".join("  %5.0f±%-3.0f" % (m, s) for m, s in zip(med, sd))))
    print("\n⚠ 사건 수가 적다. 자릿수와 부호만 읽어라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
