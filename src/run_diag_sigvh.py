# -*- coding: utf-8 -*-
"""`sig` 의 **파형 몫**이 녹화 세션에 박제돼 있다 — 남은 수준 오차의 후보 (14.55).

순저항은 `I_h = V_h/R`, `P = V_1²/R` 이므로
```
    sig = median(I_h/P) = v_h_rel / V_1 ,    v_h_rel = |V_h|/|V_1|
```
14.49 의 앵커는 **`1/V_1` 쪽만** 고쳤다 (222V 고정 -> 기기별 적합 전압).
**`v_h_rel` 은 그 녹화 세션의 값 그대로 박제돼 있다.** 그런데 세션마다 다르다 —
자리 D 는 vh3/V1 0.6~0.85%, 자리 E 는 2.8~3.25% 로 **4배** 갈린다
(`grid_simulator.OBSERVED_VOLTAGE_CLUSTERS` 주석, 13.1).

그러면 실측 복합 창의 `v_h_rel` 이 그 기기 녹화의 `v_h_rel` 과 다른 만큼 `sig` 가 틀리고,
`L_harm` 이 `obs_harm ≈ Σ power·sig` 를 맞추면서 **전력을 그 비만큼 밀어 준다**:

    실측 v_h_rel < 녹화 v_h_rel  ->  같은 전력의 obs_harm 이 작다 -> 모델이 전력을 **줄인다**
                                    (= 과대예측으로 보이지 않고 **과소** 쪽) ... 부호는 아래 표로 읽어라

여기서는 **재기만** 한다: 기기 녹화의 통전 중 `v_h_rel` 대 실측 복합 고전력 창의 `v_h_rel`.

    python -X utf8 src/run_diag_sigvh.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

H = (3, 5, 7, 9, 11, 13, 15)
REC = {"oven": ("oven_1", "oven_2"), "hotplate": ("hotplate_1", "hotplate_2"),
       "electiric_kettle": ("electric_kettle_1",), "hair_dryer": ("hair_dryer_1",)}


def rel_of(path: Path, mask=None):
    z = np.load(path, allow_pickle=True)
    V = np.asarray(z["voltage_harmonics_complex"], dtype=np.complex128)
    ok = (np.asarray(z["is_valid"]) == 1) if "is_valid" in z.files else np.ones(len(V), bool)
    v1 = np.abs(V[:, 0])
    ok &= (v1 > 150) & (v1 < 280) & np.isfinite(V[:, :15]).all(1)
    if mask is not None:
        ok &= mask
    if ok.sum() < 30:
        return None, 0
    r = np.abs(V[ok][:, [h - 1 for h in H]]) / v1[ok][:, None]
    return np.median(r, 0), int(ok.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", default="processed_data/npz")
    ap.add_argument("--real-dir", default="processed_data/composite_eval")
    ap.add_argument("--p-hi", type=float, default=1500.0, help="실측 쪽 고전력 문턱 (W)")
    a = ap.parse_args()

    print("`v_h_rel = |V_h|/|V_1|` (%) — 기기 **녹화 통전 중** 대 **실측 복합 고전력**\n")
    print("  %-22s%7s" % ("", "사이클") + "".join("%8s" % ("h%d" % h) for h in H))

    rec = {}
    for app, stems in REC.items():
        for st in stems:
            p = Path(a.npz_dir) / ("%s.npz" % st)
            if not p.exists():
                continue
            z = np.load(p, allow_pickle=True)
            P = np.asarray(z["target_power_w"], dtype=np.float64)
            hi = P > 0.5 * np.percentile(P[P > 50], 80) if (P > 50).any() else None
            r, n = rel_of(p, hi)
            if r is None:
                continue
            rec.setdefault(app, []).append(r)
            print("  %-22s%7d" % (st, n) + "".join("%8.3f" % (100 * x) for x in r))
    print()
    real = []
    for p in sorted(Path(a.real_dir).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        P = np.asarray(z["power_features"])[:, 0].astype(np.float64)
        m = P > a.p_hi
        if m.sum() < 30:
            continue
        r, n = rel_of(p, m)
        if r is None:
            continue
        real.append(r)
        print("  %-22s%7d" % ("%s (P>%.0fW)" % (p.stem, a.p_hi), n)
              + "".join("%8.3f" % (100 * x) for x in r))
    if not real or not rec:
        print("표본이 모자라다")
        return 1
    R = np.median(np.stack(real), 0)
    print()
    print("  %-22s%7s" % ("실측 고전력 중앙", "") + "".join("%8.3f" % (100 * x) for x in R))
    print()
    print("■ 비 = 실측 / 녹화   — 1 에서 벗어난 만큼 `sig` 가 틀렸다")
    print("  %-22s%7s" % ("기기", "") + "".join("%8s" % ("h%d" % h) for h in H)
          + "%10s" % "h3~h9 중앙")
    for app, v in rec.items():
        m = np.median(np.stack(v), 0)
        q = R / np.maximum(m, 1e-12)
        print("  %-22s%7s" % (app, "") + "".join("%8.3f" % x for x in q)
              + "%10.3f" % float(np.median(q[:4])))
    print()
    print("  ⚠ 저항은 `sig = v_h_rel / V_1` 이다. 비가 0.99 면 실측에서 그 기기의 지문이")
    print("     1% 작다는 뜻이고, `L_harm` 이 `obs_harm ≈ Σ power·sig` 를 맞추려면")
    print("     **전력을 1% 더** 준다 (= 과대예측 쪽).")
    print("  ⚠ 14.49 의 앵커는 `1/V_1` 만 고쳤다. 이 축은 **아직 안 고쳤다.**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
