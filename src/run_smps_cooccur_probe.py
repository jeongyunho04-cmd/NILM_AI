"""미니PC 창의 SMPS 동반 상태 — 합성 vs 실측 (12.177).

    python -m src.run_smps_cooccur_probe

실측의 SMPS 3종은 책상 세트라 **늘 같이 켜져 있다.** 합성이 그 조건을 얼마나
만드는지 센다. 9~13° 안에 몰린 세 기기가 뭉쳐 있는 것이 진짜 문제인데,
합성이 미니PC 를 혼자 켜는 창만 많이 만들면 모델은 쉬운 문제만 배운다.
"""
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from src.evaluation.holdout import load_holdout

NPZ = "processed_data/composite_eval/{}.npz"


def _mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs:
        m[int(s * 60):min(int(e * 60), n)] = True
    return m


def main(holdout="processed_data/holdout60_ac") -> int:
    hs = load_holdout(holdout)
    apps = hs.meta["appliances"]
    jm, jp, jc = (apps.index(a) for a in ("minipc", "beam_projector", "laptop_charger"))
    yo, yp = hs.y_on, hs.y_power
    sel = np.where((yo[:, jm] > 0.5) & (yp[:, jm] > 6) & (yp[:, jm] < 16))[0]
    p, c = yo[sel, jp] > 0.5, yo[sel, jc] > 0.5
    print(f"[합성] {holdout.split('/')[-1]}  미니PC ON & 6~16W 창 {len(sel)}개")
    print(f"   미니PC 만      {np.mean(~p & ~c):6.1%}")
    print(f"   + 프로젝터만   {np.mean(p & ~c):6.1%}")
    print(f"   + 충전기만     {np.mean(~p & c):6.1%}")
    print(f"   **3종 다**     {np.mean(p & c):6.1%}   ({int((p & c).sum())}창)")

    ev = json.load(open("processed_data/real_events_refined.json",
                        encoding="utf-8"))["files"]
    print()
    print("[실측] 미니PC ON 창에서 3종이 다 켜진 비율")
    for stem in ("test_15", "test_18"):
        if stem not in ev or "minipc" not in ev[stem]["intervals"]:
            continue
        n = int(ev[stem]["cycles"]); iv = ev[stem]["intervals"]
        on = _mask(iv["minipc"].get("on", []), n)
        z = np.load(NPZ.format(stem))
        nn = min(z["harmonics_complex"].shape[0], n)
        ok = z["is_valid"].astype(bool)[:nn] & on[:nn]
        both = np.ones(nn, bool)
        for a in ("beam_projector", "laptop_charger"):
            both &= _mask(iv.get(a, {}).get("on", []), n)[:nn]
        P = z["power_features"][:nn, 0].astype(float)
        print(f"   {stem}  ON창 {int(ok.sum()):6d}  3종 동시 {both[ok].mean():6.1%}"
              f"   총 관측전력 중앙 {np.median(P[ok]):5.0f}W"
              f"   미니PC 몫 ~{10 / max(np.median(P[ok]), 1):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
