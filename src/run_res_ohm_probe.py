"""실측에서 저항 부하의 등가저항을 장소별로 되짚는다 (12.164.13 의 후보 B).

`L_swap` 은 조합의 컨덕턴스가 `p_res/V²` 와 `swap_tol` 안에서 맞을 때만 감독한다.
12.164.12 에서 장소 B 의 드라이기 단독 창이 **참값조차 3.3~4.2% 벗어나** 항이
침묵하는 것을 봤다. 여기서는 라벨이 "그 기기만 켜져 있다" 고 말하는 구간의
`P_관측 / V²` 를 직접 재서, 표(`RESISTIVE_OHM`)가 그 장소에서 맞는지 본다.
모델을 안 쓴다 — 관측과 라벨만 쓴다.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import argparse
import json

import numpy as np

from src.model.postproc import RESISTIVE_OHM, HALFWAVE_OHM, HALFWAVE_ABS_MIN
from src.model.realdata import dense_targets

RES = ("electiric_kettle", "oven", "hair_dryer", "hotplate")
SITE = {"A": ("test_5", "test_6", "test_7", "test_8", "test_11", "test_12", "test_13"),
        "B": ("test_15", "test_16", "test_17", "test_18")}


def _mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs:
        m[int(s * 60):min(int(e * 60), n)] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description="장소별 등가저항 역산")
    ap.add_argument("--events", default="processed_data/real_events_refined.json")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--min-w", type=float, default=200.0)
    a = ap.parse_args()
    ev = json.load(open(a.events, encoding="utf-8"))["files"]

    print(f"{'':30s}{'n':>7s}{'P중앙':>9s}{'V중앙':>8s}{'R=V²/P':>10s}"
          f"{'표':>9s}{'차이':>8s}{'반파':>7s}{'바닥W':>8s}")
    for site, stems in SITE.items():
        print(f"\n=== 장소 {site} ===")
        for app in RES:
            tot = []
            for stem in stems:
                if stem not in ev or app not in ev[stem]["intervals"]:
                    continue
                n = int(ev[stem]["cycles"])
                on = _mask(ev[stem]["intervals"][app].get("on", []), n)
                other = np.zeros(n, bool)
                for b in RES:
                    if b != app and b in ev[stem]["intervals"]:
                        other |= _mask(ev[stem]["intervals"][b].get("on", []), n)
                try:
                    rw = dense_targets(stem, stride=a.stride)
                except Exception:
                    continue
                t = rw.target_cycle
                m = on[t] & ~other[t]
                if m.sum() < 5:
                    continue
                P = np.concatenate([rw.batch(np.arange(i, min(i + 512, len(rw))))[2]
                                    for i in range(0, len(rw), 512)])
                H = np.concatenate([rw.batch(np.arange(i, min(i + 512, len(rw))))[3]
                                    for i in range(0, len(rw), 512)])
                V = rw.v_observed.astype(np.float64)
                half = (np.hypot(H[:, 1, 0], H[:, 1, 1])
                        - np.hypot(H[:, 3, 0], H[:, 3, 1])) > HALFWAVE_ABS_MIN
                # 배경 보정: 저항 4종이 **하나도** 안 켜진 창의 관측 전력이
                # 그 파일의 SMPS·계측계 바닥이다. 안 빼면 큰 부하일수록 R 이
                # 작아 보인다 (600W 부하에 60W 바닥이면 R 이 9% 낮게 나온다).
                allres = np.zeros(n, bool)
                for b in RES:
                    if b in ev[stem]["intervals"]:
                        allres |= _mask(ev[stem]["intervals"][b].get("on", []), n)
                base = float(np.median(P[~allres[t]])) if (~allres[t]).sum() >= 5 else 0.0
                m &= P > a.min_w
                if m.sum() < 5:
                    continue
                tot.append((P[m] - base, V[m], half[m], np.full(int(m.sum()), base)))
            if not tot:
                continue
            P = np.concatenate([x[0] for x in tot])
            V = np.concatenate([x[1] for x in tot])
            hf = np.concatenate([x[2] for x in tot])
            bg = np.concatenate([x[3] for x in tot])
            for lab, sel, ref in (("", ~hf, RESISTIVE_OHM.get(app)),
                                  (" 반파", hf, HALFWAVE_OHM.get(app))):
                if sel.sum() < 5:
                    continue
                R = float(np.median(V[sel] ** 2 / P[sel]))
                d = (R / ref - 1) * 100 if ref else float("nan")
                print(f"  {app + lab:28s}{int(sel.sum()):7d}{np.median(P[sel]):9.0f}"
                      f"{np.median(V[sel]):8.1f}{R:10.2f}"
                      f"{(ref if ref else float('nan')):9.1f}{d:+7.1f}%{sel.mean():7.0%}"
                      f"{np.median(bg[sel]):8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
