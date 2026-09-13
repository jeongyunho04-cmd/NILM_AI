# -*- coding: utf-8 -*-
"""실측 표류 증강 관문 — 항등·동작·폭·안전 (13.84.61).

`smps_drift` 는 형제 지문 변동을 **임의로 모수화**하던 것(회전+기울기+흔들기, 9자유도)을
버리고 13.84.52·56·57 이 잰 **실측 3차원 표류**를 그대로 더한다.

왜 생성기인가: 13.84.60 이 같은 부분공간을 `L_harm` 에서 **사영으로 지워** 봤고 졌다 —
표류 PC1 이 미니PC-형제 판별축과 24~25° 라 표류와 함께 판별력이 간다.
**지울 수 없으면 가르쳐야 한다.**

관문 넷:
  [1] 항등   목록에 없는 기기 · `drift_scale=0` 이면 **비트 단위로** 같다
  [2] 동작   뽑은 편차의 공분산이 **실측 표류**와 맞나 (부분공간 안에서)
  [3] 안전   차수별 3σ / 미니PC 신호 — 1 을 넘는 차수는 미니PC 를 묻는다
             ([[augmentation-can-erase-the-discriminant]], `smps_dev1` 이 빠진 함정)
  [4] 견줌   옛 프리셋들이 만드는 폭과 나란히

    python -X utf8 src/run_gate_driftaug.py
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.run_diag_driftcirc import cells
from src.run_diag_driftdim import ORD
from src.run_diag_driftlaw import minipc_sig
from src.synthesis.augmentor import SIBLING_ROTATE_PRESETS, DataAugmentor

NH = 15
T = 64
PRESETS = ("smps_rot10", "smps_dev1", "smps_dev2", "smps_drift", "smps_driftp", "smps_drift47")


def draw(name, n, seed=0):
    """프리셋이 만드는 **편차**를 n회 뽑는다. 기준 지문은 충전기 실측 중앙값."""
    from src.run_diag_driftdim import load
    for stem, P, hc, on, off in load("laptop_charger"):
        m = on & (P >= 28) & (P <= 40)
        if m.sum() > 600:
            base = (np.median(hc[m].real, 0) + 1j * np.median(hc[m].imag, 0))[:NH]
            break
    aug = DataAugmentor(sibling_rotate=SIBLING_ROTATE_PRESETS[name])
    np.random.seed(seed)
    x = np.repeat(base[None, :].astype(np.complex64), T, 0)
    out = []
    for _ in range(n):
        out.append(aug._apply_sibling_rotate("laptop_charger", x.copy())[0] - base)
    return np.asarray(out), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=4000)
    ap.add_argument("--check", default="",
                    help="이 프리셋에 **합격/불합격**을 낸다 (없으면 표만 찍고 0). "
                         "기준: 차수별 폭이 실측의 0.5~1.5배 · 표류 부분공간 밖이 35%% 아래")
    a = ap.parse_args()

    oi = [o - 1 for o in ORD]
    ms = minipc_sig()
    mp = np.array([np.hypot(ms[j], ms[j + len(oi)]) for j in range(len(oi))])
    D = np.vstack([Y for dev in ("laptop_charger", "beam_projector")
                   for _, Y, _, _, _ in cells(dev, 600)])
    Dm = np.array([np.sqrt((D[:, j] ** 2 + D[:, j + len(oi)] ** 2).mean()) for j in range(len(oi))])

    # ── [1] 항등 ────────────────────────────────────────────────────────────
    aug = DataAugmentor(sibling_rotate=SIBLING_ROTATE_PRESETS["smps_drift"])
    x = (np.random.RandomState(0).randn(T, NH) + 1j * np.random.RandomState(1).randn(T, NH)
         ).astype(np.complex64)
    np.random.seed(0)
    same = np.array_equal(aug._apply_sibling_rotate("oven", x.copy()), x)
    a0 = DataAugmentor(sibling_rotate={"laptop_charger": dict(
        SIBLING_ROTATE_PRESETS["smps_drift"]["laptop_charger"], drift_scale=0.0,
        tilt_lo=1.0, tilt_hi=1.0, dith_lo=1.0, dith_hi=1.0)})
    np.random.seed(0)
    same0 = np.array_equal(a0._apply_sibling_rotate("laptop_charger", x.copy()), x)
    print("[1] 항등  목록 밖 기기(oven) %s · drift_scale=0 %s"
          % ("**같다**" if same else "!! 다르다 !!", "**같다**" if same0 else "!! 다르다 !!"))

    # ── [2][3][4] ───────────────────────────────────────────────────────────
    print("\n[2] 동작 · [3] 안전 — 뽑은 편차의 차수별 RMS, 그리고 3σ / 미니PC 신호")
    print("   %-14s %8s | %s" % ("프리셋", "전체RMS", "  ".join("h%-4d" % o for o in ORD)))
    print("   %-14s %8.1f | %s   <- **실측 표류**"
          % ("(실측 표류)", 1000 * np.sqrt((D ** 2).sum(1).mean()),
             "  ".join("%5.1f" % (1000 * v) for v in Dm)))
    print("   %-14s %8s | %s   <- 미니PC 신호"
          % ("(미니PC)", "%.1f" % (1000 * np.linalg.norm(ms)),
             "  ".join("%5.1f" % (1000 * v) for v in mp)))
    for nm in PRESETS:
        E, base = draw(nm, a.n)
        Eo = E[:, oi]
        rms = np.sqrt((np.abs(Eo) ** 2).sum(1).mean())
        per = np.sqrt((np.abs(Eo) ** 2).mean(0))
        print("   %-14s %8.1f | %s" % (nm, 1000 * rms, "  ".join("%5.1f" % (1000 * v) for v in per)))
        r = 3 * per / np.maximum(mp, 1e-12)
        bad = [ORD[j] for j in range(len(oi)) if r[j] > 1.0]
        print("   %-14s %8s |   3σ/미니PC %s   %s"
              % ("", "", "  ".join("%5.2f" % v for v in r),
                 ("**h%s 가 묻힌다**" % ",".join(str(x) for x in bad)) if bad else "안전"))

    print("\n[4] 실측 대조 — 실측 표류 대비 폭 (1.00 이 실측 그대로)")
    for nm in PRESETS:
        E, _ = draw(nm, a.n)
        per = np.sqrt((np.abs(E[:, oi]) ** 2).mean(0))
        print("   %-14s %s" % (nm, "  ".join("h%d %4.2f" % (ORD[j], per[j] / max(Dm[j], 1e-12))
                                             for j in range(len(oi)))))
    print("\n   읽는 법 — [4] 가 1.00 근처면 증강 폭이 **실측 그대로**다.")
    print("   [3] 의 '묻힌다' 는 증강 결함이 **아니다** — 실측 표류 자체가 h9~h15 에서")
    print("   미니PC 기여를 넘는다 (표류 h15 8.3mA 대 미니PC 11.2mA, 3σ 면 2.2배). 정직한")
    print("   증강은 그 규칙을 만족할 수 없다. 합격 기준은 [3] 이 아니라 **[4] 와 모양**이다.")

    if not a.check:
        return 0
    # ── 합격/불합격 ────────────────────────────────────────────────────────
    # 기준을 [3](3σ/미니PC)이 아니라 [4](실측 대비 폭)와 **모양**으로 잡는 이유는 위와 같다.
    # smps_dev1 은 폭 6.6~12.9배 · 밖에 52.6% 로 떨어지고, smps_dev2(0.63~1.13 · 23.4%)와
    # smps_driftp(0.89~0.95 · 3.2%)는 통과한다.
    E, _ = draw(a.check, max(a.n, 2000))
    X = np.concatenate([E[:, oi].real, E[:, oi].imag], 1)
    per = np.sqrt((np.abs(E[:, oi]) ** 2).mean(0))
    w = per / np.maximum(Dm, 1e-12)
    Q = np.linalg.qr(np.linalg.svd(D - D.mean(0), full_matrices=False)[2][:3].T)[0]
    outf = float((np.linalg.norm(X - X @ Q @ Q.T, axis=1) ** 2).sum()
                 / max((np.linalg.norm(X, axis=1) ** 2).sum(), 1e-30))
    bad = [ORD[j] for j in range(1, len(oi)) if not (0.5 <= w[j] <= 1.5)]
    print("\n[판정] %s — 폭 %.2f~%.2f (h3~h15) · 표류 부분공간 밖 %.1f%%"
          % (a.check, w[1:].min(), w[1:].max(), 100 * outf))
    if bad:
        print("   !! 폭이 0.5~1.5 밖인 차수: h%s" % ",".join(str(x) for x in bad))
        return 1
    if outf > 0.35:
        print("   !! 부분공간 밖이 35%% 를 넘는다 — 실측이 아닌 방향을 흔든다")
        return 1
    print("   **합격**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
