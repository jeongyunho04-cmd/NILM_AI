# -*- coding: utf-8 -*-
"""회로 야코비를 **h15 까지** 넓혀 다시 잰다 (14.11).

사용자: *"PC3 가 계통전압 크기면 이미 15차 이상의 고차 전압 고조파까지 측정하고 있잖아"*

13.84.57·66 은 `VORD = (3, 5, 7)` 로 섭동했다 — 공변량 8개다. 그런데 `voltage_harmonics_complex`
에는 **h15 까지** 있고, 회귀로 재 보니 h9~h15 를 더하면 LORO R^2 가 0.260 -> 0.481 로 뛴다
(특히 PC2 가 −0.003 -> 0.501). **회로에 불완전한 구동을 주고 예측하라 한 것이다.**

여기서는 `jac()` 의 섭동 목록만 넓힌다. 회로 파라미터도 자유도도 그대로 0 이다.

⚠ numba/scipy 충돌 — `scipy.linalg.cython_blas` 를 먼저 import 해야 `RecursionError` 가 안 난다.

    python -X utf8 src/run_diag_driftcirc15.py
"""
import sys

import scipy.linalg.cython_blas          # noqa: F401  ⚠ 이 줄이 먼저여야 한다
import numba.np.arraymath                # noqa: F401

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NEW_VORD = (3, 5, 7, 9, 11, 13, 15)


def main():
    import runpy

    import src.run_diag_driftlaw as law
    law.VORD = NEW_VORD
    import src.run_diag_driftcirc as circ
    circ.VORD = NEW_VORD
    import src.run_diag_driftcirc12 as c12
    c12.VORD = NEW_VORD

    print("섭동 목록을 h15 까지 넓혔다 — VORD = %s · 공변량 %d개 (전: 8개)"
          % (NEW_VORD, 2 + 2 * len(NEW_VORD)))
    print("회로 파라미터·자유도는 그대로 0 이다.\n")
    sys.argv = ["run_diag_driftcirc12"]
    return c12.main()


if __name__ == "__main__":
    raise SystemExit(main())
