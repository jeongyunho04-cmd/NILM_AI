# -*- coding: utf-8 -*-
"""h15 **초과** 전압 고조파를 넣으면 값이 있나 — 민감도와 자료를 갈라 본다 (14.14).

사용자: *"15차 초과의 전압고조파를 추가했을 때의 효과도 분석해봐."*

14.11 이 h9~h15 를 공변량에 넣어 회로 야코비의 표류 설명력을 0.217 -> 0.654 로 올렸다.
그 다음 차수가 꼬리다. 그런데 질문이 **둘로 갈린다**:

  ① **민감도**  회로가 h17~h31 전압에 반응하나. `to_wave` 는 h31 까지 파형에 넣고
                정류기는 비선형이라 꼬리가 h1~h15 **전류**를 움직일 수 있다. 여기서 잰다.
  ② **구동**    그 전압이 실제로 얼마나 움직이나. `processed_data/vtail.npz` 는
                **세션당 값 하나** 뿐이다 (7세션) — 녹화 안 변동이 자료에 없다.
                ⇒ 민감도가 커도 **지금 자료로는 표류를 설명할 수 없다.**

⚠ 모델 입력 채널로 넣는 것은 또 다른 문제다. 세션당 상수를 채널로 주면 **세션 식별자**가
  되어 합성 전용 단서가 된다 ([[recording-session-leaks-into-the-signature]]).
  13.84.36 도 같은 이유로 꼬리를 세션 이동의 원인에서 뺐다 (7값 상수 대 연속 편차).

    python -X utf8 src/run_diag_vtail_sens.py
"""
import sys

import scipy.linalg.cython_blas          # noqa: F401  ⚠ numba 보다 먼저 (14.11 ⑥)
import numba.np.arraymath                # noqa: F401

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

V1 = 222.0
DV = 0.1                       # 섭동 크기 [V] — driftcirc12 와 같다
LOW = (3, 5, 7, 9, 11, 13, 15)
TAIL = (17, 19, 21, 23, 25, 27, 29, 31)
POWERS = (23.0, 47.0, 63.0)


def main():
    from src.synthesis import fcm
    from src.synthesis.circuit_sim import to_wave
    import circuit_model.circuit12 as c12

    mods = fcm.load_models()
    dev = "laptop_charger"
    par = None
    for name in ("laptop_charger",):
        m = mods.get(name)
        if m is not None:
            par = m
    if par is None:
        print("회로 파라미터가 없다"); return 1
    p = par.params if hasattr(par, "params") else par
    print("기기 %s · v12g 파라미터 %s" % (dev, np.round(np.asarray(p, float), 4)))

    def sim(P, V):
        H = c12.sim_harmonics(float(P), np.asarray(V, complex), tuple(p), npc=3072)
        return np.asarray(H)[:15]

    rows = {}
    for P in POWERS:
        V0 = np.zeros(31, complex); V0[0] = V1
        base = sim(P, V0)
        for h in LOW + TAIL:
            acc = []
            for ph in (0.0, np.pi / 2):
                Vp = V0.copy(); Vm = V0.copy()
                Vp[h - 1] += DV * np.exp(1j * ph)
                Vm[h - 1] -= DV * np.exp(1j * ph)
                acc.append((sim(P, Vp) - sim(P, Vm)) / (2 * DV))
            # 이 차수 전압이 h1~h15 전류를 움직이는 크기 (복소 2방향 합)
            rows.setdefault(h, []).append(
                float(np.sqrt(sum(float(np.abs(a) ** 2 @ np.ones(15)) for a in acc))))

    print("\n① **민감도** — 전압 V_h 를 1V 흔들면 h1~h15 전류가 얼마나 움직이나 (mA/V)")
    print("   %-6s %10s %10s %10s %10s" % ("차수", "23W", "47W", "63W", "평균"))
    ref = None
    for h in LOW + TAIL:
        v = rows[h]
        mean = float(np.mean(v))
        if h == 3:
            ref = mean
        mark = "  <- 꼬리" if h in TAIL else ""
        print("   h%-5d %8.2f %10.2f %10.2f %10.2f%s"
              % (h, 1e3 * v[0], 1e3 * v[1], 1e3 * v[2], 1e3 * mean, mark))
    lo = float(np.mean([np.mean(rows[h]) for h in LOW]))
    ta = float(np.mean([np.mean(rows[h]) for h in TAIL]))
    print("\n   h3~h15 평균 %.2f mA/V · h17~h31 평균 %.2f mA/V  =  **꼬리가 %.2f배**"
          % (1e3 * lo, 1e3 * ta, ta / max(lo, 1e-12)))

    # ── ② 구동 — 그 전압이 실제로 얼마나 움직이나 ─────────────────────────
    d = np.load("processed_data/vtail.npz", allow_pickle=True)
    t = np.asarray(d["tail"])
    print("\n② **구동** — 그 전압이 실제로 얼마나 움직이나")
    print("   꼬리(h17~31): `vtail.npz` 에 **세션당 값 하나**뿐 (%d세션). 녹화 안 변동이 **자료에 없다**"
          % len(d["keys"]))
    print("   세션 **사이** 표준편차: %s ppm"
          % " ".join("%.0f" % (1e6 * np.std(t[:, i])) for i in range(t.shape[1])))
    print("   견줌 h3~h15 의 **녹화 안** 산포: 535~1245 ppm (14.11 ①)")

    print("\n③ **기여 추정** = 민감도 x 구동")
    print("   %-14s %10s %12s %14s" % ("", "민감도 mA/V", "구동 ppm", "기여 mA"))
    drv_low = 800.0           # h3~h15 녹화 안 산포 대표값
    drv_tail = float(np.mean([np.std(t[:, i]) for i in range(t.shape[1])]))
    c_low = lo * drv_low * 1e-6 * V1 * len(LOW) ** 0.5
    c_tail = ta * drv_tail * V1 * len(TAIL) ** 0.5
    print("   %-14s %10.2f %12.0f %13.2f mA" % ("h3~h15 (녹화안)", 1e3 * lo, drv_low, 1e3 * c_low))
    print("   %-14s %10.2f %12.0f %13.2f mA  ⚠ 세션**사이** 값이다"
          % ("h17~h31", 1e3 * ta, 1e6 * drv_tail, 1e3 * c_tail))
    print("\n   실측 표류 (녹화 안) 28.9 mA 와 견주어라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
