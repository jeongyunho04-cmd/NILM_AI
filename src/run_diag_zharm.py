# -*- coding: utf-8 -*-
"""**차수별 선로 임피던스** `Z_h` 를 계단 차분으로 잰다 (14.59).

왜 계단 차분인가
----------------
13.4 가 `Z_1` 을 믿을 만하게 잰 방법이다 — `Z_1 = −ΔV_1/Δ|I_1|`. **차분을 쓰면 상류
전압 `V0_h` 의 표류가 상쇄된다.** 창 전체를 회귀(`V_h = V0_h − Z_h·I_h`)하면 그 표류가
절편 가정을 깨고, 게다가 고차에서는 `I_h` 가 거의 안 움직여 **조건화가 안 된다** —
처음에 그렇게 쟀다가 h11·h13 에서 5.5~9.2Ω 이 나왔다 ([[check-conditioning-before-believing-a-fit]]).

    Z_h = −(V_h,뒤 − V_h,앞) / (I_h,뒤 − I_h,앞)      **복소**로 푼다

복소여야 R 과 X 가 갈린다. 크기만 쓰면 부호도 위상도 잃는다.

무엇을 겨냥하나
---------------
생성기는 `z_h = r_grid + j·h·x_grid` 다. `r 0.30~2.00 · x 0.02~0.15` 범위에서 이것은
**차수에 거의 평평하다** (r=1.15, x=0.085 면 `|z_3|/|z_1| = 1.02`). 그런데 실측은
전압 크기와 파형이 h3 에서 **r = +0.85** 로 같이 움직인다 (합성은 −0.10). 순저항의
`v_3_rel(ON)/v_3_rel(OFF) = (1+Z_1/R)/(1+Z_3/R)` 로 역산하면 `Z_3/Z_1 = 2~9` 가 나온다.
**그 역산을 직접 측정으로 확인하거나 반박하는 것이 이 자다.**

조건화와 음성 대조
------------------
· `|ΔI_h|` 가 그 구간 `I_h` 잡음의 `--snr` 배를 못 넘으면 **그 차수는 버린다.**
· ⚠ **음성 대조**: 계단이 아닌 조용한 구간 쌍으로도 같은 계산을 한다. 거기서도 비슷한
  값이 나오면 이 자는 계단을 재는 것이 아니다.

    python -X utf8 src/run_diag_zharm.py
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

H = (1, 3, 5, 7, 9, 11, 13, 15)
FS = 60.0


def _sites():
    try:
        from src.preprocessing.file_registry import site_of
        return site_of
    except Exception:
        return lambda s: ""


def steps_of(V, I, P, ok, guard, win, dp_min, snr, negative=False, rng=None):
    """[(h, Z_h 복소)] — 계단마다, 차수마다. `negative` 면 **계단이 아닌 곳**에서 잰다."""
    out = {h: [] for h in H}
    d = np.diff(P)
    if negative:
        # 조용한 구간 안에서 **같은 간격**으로 임의의 쌍을 잡는다 (계단이 없는 곳)
        cand = np.flatnonzero(np.abs(d) < 5.0)
        cand = cand[(cand > guard + win) & (cand < len(P) - guard - win - 1)]
        if len(cand) == 0:
            return out
        idx = rng.choice(cand, size=min(400, len(cand)), replace=False)
    else:
        idx = np.flatnonzero(np.abs(d) >= dp_min)
    for t in idx:
        a0, a1 = t - guard - win, t - guard
        b0, b1 = t + guard, t + guard + win
        if a0 < 0 or b1 >= len(P):
            continue
        if not (ok[a0:a1].all() and ok[b0:b1].all()):
            continue
        if P[a0:a1].std() > 40 or P[b0:b1].std() > 40:      # 양쪽이 조용할 때만 (13.4)
            continue
        for h in H:
            j = h - 1
            ia, ib = I[a0:a1, j], I[b0:b1, j]
            va, vb = V[a0:a1, j], V[b0:b1, j]
            dI = ib.mean() - ia.mean()
            # ── 조건화: ΔI 가 그 구간 잡음의 snr 배를 넘어야 한다 ──────────
            noise = 0.5 * (ia.std() + ib.std()) / np.sqrt(max(len(ia), 1))
            if abs(dI) < max(snr * noise, 1e-4):
                continue
            out[h].append(-(vb.mean() - va.mean()) / dI)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+",
                    default=["processed_data/npz", "processed_data/composite_eval"])
    ap.add_argument("--guard", type=int, default=45, help="계단 앞뒤 가드 (사이클, 0.75초)")
    ap.add_argument("--win", type=int, default=180, help="앞뒤 창 (사이클, 3초)")
    ap.add_argument("--dp-min", type=float, default=300.0, help="계단으로 볼 |ΔP| (W)")
    ap.add_argument("--snr", type=float, default=5.0,
                    help="`|ΔI_h|` 이 그 구간 잡음의 몇 배를 넘어야 쓰나 (조건화)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    site_of = _sites()
    rng = np.random.default_rng(a.seed)
    by_site = {}
    neg_site = {}
    for d in a.dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "voltage_harmonics_complex" not in z.files:
                continue
            meta = (json.loads(str(z["metadata_json"]))
                    if "metadata_json" in z.files else {})
            if d.endswith("npz") and not meta.get("voltage_phase_available", False):
                continue
            V = np.asarray(z["voltage_harmonics_complex"], dtype=np.complex128)
            I = np.asarray(z["harmonics_complex"], dtype=np.complex128)
            P = np.asarray(z["power_features"], dtype=np.float64)[:, 0]
            v1 = np.abs(V[:, 0])
            ok = ((np.asarray(z["is_valid"]) == 1) if "is_valid" in z.files
                  else np.ones(len(V), bool))
            ok &= (v1 > 150) & (v1 < 280) & np.isfinite(V[:, :15]).all(1)
            st = site_of(f.stem) or "?"
            for neg, store in ((False, by_site), (True, neg_site)):
                r = steps_of(V, I, P, ok, a.guard, a.win, a.dp_min, a.snr,
                             negative=neg, rng=rng)
                s = store.setdefault(st, {h: [] for h in H})
                for h in H:
                    s[h] += r[h]

    print("차수별 선로 임피던스 `Z_h = −ΔV_h/ΔI_h` (계단 차분, 복소)")
    print("  가드 %.2f초 · 창 %.1f초 · |ΔP| >= %.0fW · 조건화 |ΔI_h| > %.0f x 잡음\n"
          % (a.guard / FS, a.win / FS, a.dp_min, a.snr))
    for st in sorted(by_site):
        s = by_site[st]
        if len(s[1]) < 5:
            continue
        print("■ 자리 %s" % st)
        print("    %-5s%6s%9s%17s%8s  %s"
              % ("차수", "표본", "|Z_h|", "사분위", "Z/Z_1", "음성대조: 표본 중앙 [사분위]"))
        z1 = None
        for h in H:
            v = np.asarray(s[h])
            if len(v) < 5:
                print("    h%-4d%6d   (표본 부족)" % (h, len(v)))
                continue
            m = float(np.median(np.abs(v)))
            q1, q3 = np.percentile(np.abs(v), [25, 75])
            if h == 1:
                z1 = m
            ng = np.abs(np.asarray(neg_site.get(st, {}).get(h, [])))
            if len(ng) >= 5:
                n1, n3 = np.percentile(ng, [25, 75])
                ngs = "%4d  %7.2f [%6.2f~%7.2f]" % (len(ng), np.median(ng), n1, n3)
                # **가름**: 신호의 사분위와 음성대조의 사분위가 안 겹쳐야 잰 것이다
                sep = "  ✅" if q3 < n1 else ("  ⚠ 겹침" if q1 < n3 else "  ✅")
            else:
                ngs, sep = "   (없음)", ""
            wide = "  ⚠ 산포>2배" if (q3 - q1) > m else ""
            print("    h%-4d%6d%9.3f%9.2f~%-7.2f%8.2f  %s%s%s"
                  % (h, len(v), m, q1, q3, m / max(z1, 1e-9), ngs, sep, wide))
        print()
    r, x = 1.15, 0.085
    print("  생성기 모형 `r + j·h·x`  (r=%.2f, x=%.3f — 자리 D 중앙)" % (r, x))
    print("    %-5s" % "z_h/z_1"
          + "".join("%8.2f" % (abs(r + 1j * h * x) / abs(r + 1j * x)) for h in H))
    print("    %-5s" % "(차수)" + "".join("%8s" % ("h%d" % h) for h in H))
    print("\n  ⚠ **음성대조** 열이 |Z_h| 와 비슷하면 이 자는 계단을 재는 것이 아니다.")
    print("  ⚠ 사분위 폭이 중앙의 배를 넘으면 그 차수는 **읽지 마라.**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
