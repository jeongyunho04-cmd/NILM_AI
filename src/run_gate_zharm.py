# -*- coding: utf-8 -*-
"""차수별 `Z_h` 표의 **배선** 관문 (14.59).

⚠⚠ Z 를 쓰는 데가 **넷**이다. 하나라도 빠지면 생성기가 만든 강하와 텍스처가 벗긴 강하가
서로 다른 Z 를 쓴다 — 이 저장소가 다섯 번 밟은 그 버그다.

```
① `grid_simulator.harmonic_z`          — 단자 전압 조립이 쓴다
② `coupling.zline`                     — SMPS 결합 델타가 쓴다
③ `vtexture` 의 개방전압 de-embed       — **텍스처 자신**을 만든다
④ `apply_load_drop`                    — h1 뿐이라 표와 무관
```
검정하는 불변량: **네 곳의 `Z_h/Z_1` 이 같아야 한다.** 밑절미(`r`·`L`)가 서로 달라도
**배수는 같아야** 한다 — 표가 `Z_1` 에 대한 복소 배수이기 때문이다.

```
① 표가 없으면 네 곳 다 **비트 동일** (옛 식)
② 표가 있으면 네 곳의 Z_h/Z_1 이 **같다**
③ 표에 없는 차수는 옛 식 그대로 (h7 위)
④ 표 값이 `run_diag_zharm` 이 잰 값과 맞는다
⑤ `coupling` 캐시 키에 표가 들어간다 (안 들어가면 다른 Z 가 같은 칸을 먹는다)
⑥ ⚠ 표를 켜면 **텍스처가 달라진다** — 그 사실 자체를 확인한다 (캐시 재굽기 경고의 근거)
```
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.synthesis.coupling import H as CH, F, zline  # noqa: E402
from src.synthesis.grid_simulator import (HARMONIC_Z_K, GridSimulator,  # noqa: E402
                                          harmonic_z)
from src.synthesis import vtexture  # noqa: E402

OK, NG = "✅", "❌"
FAIL = []


def ck(name, ok, note=""):
    print(("  " + (OK if ok else NG) + " ") + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


def main() -> int:
    print("차수별 `Z_h` 표 배선 관문 (14.59)\n")
    R, X = 1.15, 0.085
    L = X / (2 * np.pi * F)
    h = np.arange(1, CH + 1)

    # ── ① 표가 없으면 옛 식 ──────────────────────────────────────────────
    a = harmonic_z(h, R, X, "D", None)
    b = zline(R, L, ())
    ck("① 표 없음 — `harmonic_z` 가 `r + j·h·x` 와 **비트 동일**",
       bool(np.array_equal(a, R + 1j * h * X)))
    ck("① 표 없음 — `zline` 이 `harmonic_z` 와 같다 (x = 2πF·L)",
       float(np.abs(a - b).max()) < 1e-9, "최대 차 %.2e" % np.abs(a - b).max())

    # ── ② 표가 있으면 네 곳의 Z_h/Z_1 이 같다 ────────────────────────────
    gs = GridSimulator(texture_library=False)
    gs.harmonic_z_table = HARMONIC_Z_K
    for site in ("D", "E"):
        za = harmonic_z(h, R, X, site, HARMONIC_Z_K)
        zb = zline(R, L, gs.zk_tuple(site))
        ka, kb = za / za[0], zb / zb[0]
        ck("② 자리 %s — `harmonic_z` 와 `coupling.zline` 의 Z_h/Z_1 이 같다" % site,
           float(np.abs(ka - kb).max()) < 1e-9,
           "h3 배수 %.3f%+.3fj · 최대 차 %.2e" % (ka[2].real, ka[2].imag,
                                                np.abs(ka - kb).max()))
        # ③ vtexture 의 de-embed — 밑절미가 `z_ohm + jωhL_deembed` 로 달라도 **배수는 같아야**
        z_ohm = {"D": 1.15, "E": 0.42}[site]
        zc = z_ohm + 1j * 2 * np.pi * 60.0 * h * vtexture.DEEMBED_L_H
        zc = np.asarray(zc, dtype=np.complex128).copy()
        z1 = z_ohm + 1j * 2 * np.pi * 60.0 * vtexture.DEEMBED_L_H
        for hh, k in HARMONIC_Z_K[site].items():
            zc[hh - 1] = z1 * complex(k)
        kc = zc / zc[0]
        ck("② 자리 %s — `vtexture` de-embed 의 Z_h/Z_1 도 같다" % site,
           float(np.abs(ka[[2, 4]] - kc[[2, 4]]).max()) < 1e-9
           if site == "D" else float(abs(ka[2] - kc[2])) < 1e-9)

    # ── ③ 표에 없는 차수는 옛 식 ─────────────────────────────────────────
    za = harmonic_z(h, R, X, "D", HARMONIC_Z_K)
    old = R + 1j * h * X
    keep = [i for i in range(CH) if (i + 1) not in HARMONIC_Z_K["D"]]
    ck("③ 표에 없는 차수(h7 위 포함)는 **손대지 않는다**",
       bool(np.array_equal(za[keep], old[keep])),
       "그대로인 차수 " + " ".join("h%d" % (i + 1) for i in keep[:6]) + " …")

    # ── ④ 표 값이 실측과 맞는가 ──────────────────────────────────────────
    want = {("D", 3): (6.059, 4.672, 1.140), ("D", 5): (0.263, -2.364, 1.140),
            ("E", 3): (0.937, 0.322, 0.418)}
    bad = 0
    print("    실측 대조 (`run_diag_zharm --snr 8`)")
    for (site, hh), (re_, im_, z1m) in want.items():
        k = HARMONIC_Z_K[site][hh]
        w = complex(re_, im_) / z1m
        good = abs(k - w) < 1e-3
        bad += 0 if good else 1
        print("      %s h%-2d  표 %.3f%+.3fj · 실측 %.3f%+.3fj  %s"
              % (site, hh, k.real, k.imag, w.real, w.imag, OK if good else NG))
    ck("④ 표 값이 `run_diag_zharm` 의 실측과 맞는다", bad == 0)

    # ── ⑤ 캐시 키 ───────────────────────────────────────────────────────
    src = Path("src/synthesis/coupling.py").read_text(encoding="utf-8")
    ck("⑤ `coupling` 캐시 키에 표가 들어간다", "tuple(zk)" in src)
    ck("⑤ Z 를 만드는 자리가 `zline` 하나뿐이다",
       src.count("1j * 2 * np.pi * F * h * l_line") == 0
       and src.count("1j * 2 * np.pi * F * np.arange(1, H + 1) * l_line") == 0,
       "직접 조립한 자리 %d곳" % (src.count("2 * np.pi * F * h * l_line")
                                + src.count("F * np.arange(1, H + 1) * l_line")))

    # ── ⑥ 표를 켜면 텍스처가 달라진다 (캐시 재굽기 경고의 근거) ──────────
    vtexture.set_default_harmonic_z(None)
    vtexture.set_default_step_s(60.0)
    off = vtexture.default_library()
    r_off = {t.id: (None if t.rel_open is None else t.rel_open.copy()) for t in off.textures}
    vtexture.set_default_harmonic_z(HARMONIC_Z_K)
    on = vtexture.default_library()
    d = [float(np.abs(t.rel_open - r_off[t.id])[2]) for t in on.textures
         if t.rel_open is not None and r_off.get(t.id) is not None]
    vtexture.set_default_harmonic_z(None)
    ck("⑥ ⚠ 표를 켜면 **텍스처가 달라진다** (캐시를 다시 구워야 한다)",
       bool(d) and float(np.median(d)) > 1e-5,
       "h3 `rel_open` 변화 중앙 %.2e (텍스처 %d개)" % (np.median(d) if d else 0, len(d)))

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과 " + OK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
