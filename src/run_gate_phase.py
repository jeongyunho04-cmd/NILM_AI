# -*- coding: utf-8 -*-
"""형제 위상 증강의 폭이 **실측과 맞는가** (13.84.49 관문).

13.84.48 이 잰 실측 녹화 간 상대위상 산포 (표준편차, 도):
```
        h5    h9   h11   h13   h15
충전기  2.4   4.1   5.3   7.5  13.6
```
이것은 차수 비례 회전 `θ_h += c·h` 로 **c ≈ 0.5°/h** 다 (2.4/5 · 4.1/9 · 5.3/11 · 7.5/13 = 0.46~0.58).
13.84.8 이 고른 모형 모양은 맞았고 **폭만 20배 넓다** — 지금 `smps_dev1` 은 c_max=10°/h 라
h9 에서 ±90° 를 뿌리고, 거기에 h>=9 를 U(±180°) 로 **통째로 뒤섞기까지** 한다.

이 관문은 증강기가 실제로 만드는 산포를 재서 실측과 대조한다.

    python -X utf8 src/run_gate_phase.py [smps_dev1 smps_dev2 ...]
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.synthesis.augmentor import DataAugmentor, SIBLING_ROTATE_PRESETS

ORD = [5, 9, 11, 13, 15]
#: 13.84.48 ② 가 실측 격리 녹화에서 잰 **녹화 간** 상대위상 표준편차 (도).
REAL = {"laptop_charger": {5: 2.4, 9: 4.1, 11: 5.3, 13: 7.5, 15: 13.6}}
NTRY = 4000


def relph_std(cfg, app="laptop_charger", seed=0):
    """증강을 NTRY 번 뽑아 h1 기준 상대위상이 얼마나 흩어지는지 (도)."""
    aug = DataAugmentor(sibling_rotate={app: cfg} if cfg else None)
    np.random.seed(seed)
    base = np.ones((1, 15), np.complex64)          # 위상 0 에서 출발 — 증강분만 남는다
    out = []
    for _ in range(NTRY):
        c = aug._apply_sibling_rotate(app, base.copy())[0]
        ph = np.angle(c) - np.arange(1, 16) * np.angle(c[0])
        out.append(np.degrees(np.angle(np.exp(1j * ph))))
    A = np.asarray(out)
    # 원형 표준편차 — ±180° 감김에 안 속는다
    R = np.abs(np.exp(1j * np.radians(A)).mean(0))
    return np.degrees(np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0))))


def main():
    names = sys.argv[1:] or ["smps_dev1"]
    print("형제 위상 증강이 만드는 상대위상 산포 (원형 표준편차, 도) · %d회 추첨" % NTRY)
    print("   %-14s %s" % ("프리셋", "".join("  %-8s" % ("h%d" % o) for o in ORD)))
    print("   %-14s %s" % ("**실측(충전기)**",
                           "".join("  %8.1f" % REAL["laptop_charger"][o] for o in ORD)))
    rows = {}
    for nm in names:
        cfg = SIBLING_ROTATE_PRESETS.get(nm, {}).get("laptop_charger")
        if cfg is None:
            print("   %-14s (프리셋에 없다)" % nm); continue
        sd = relph_std(cfg)
        rows[nm] = sd
        print("   %-14s %s" % (nm, "".join("  %8.1f" % sd[o - 1] for o in ORD)))
    print()
    print("   %-14s %s" % ("실측 대비 배", ""))
    for nm, sd in rows.items():
        print("   %-14s %s" % (nm, "".join(
            "  %7.1f배" % (sd[o - 1] / REAL["laptop_charger"][o]) for o in ORD)))
    print()
    print("   통과 기준: 모든 차수에서 실측의 **0.5~2배** 안. 그 밖이면 폭이 틀렸다.")
    ok = all(all(0.5 <= sd[o - 1] / REAL["laptop_charger"][o] <= 2.0 for o in ORD)
             for sd in rows.values()) if rows else False
    for nm, sd in rows.items():
        r = [sd[o - 1] / REAL["laptop_charger"][o] for o in ORD]
        print("   %-14s %s" % (nm, "통과" if all(0.5 <= x <= 2.0 for x in r) else "**막힘**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
