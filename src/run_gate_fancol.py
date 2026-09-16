# -*- coding: utf-8 -*-
"""**선풍기 기둥**(`Budget(motor_cols=True)`)의 관문 — 14.359.

왜 — 선풍기 s3 은 THD 0.038 · ∠I1 +3.2° 로 **전기적으로 저항과 구분이 안 된다**.
기둥이 없으면 그 전류가 `G` 기둥으로 새고, §37 이 그것을 *"명목의 53~78%"* 로 쟀다.
37.6W 는 216V 에서 **0.8 mS** — 포트 식별 여유 **0.635 mS** 보다 크다.

⚠ Ĝ 는 조합 머리의 대들보다. **Ĝ 가 좋아지는 것이 채택 조건**이고, 나빠지면 접는다.

이 관문은 **참을 아는 혼합을 직접 지어** 잰다 (라벨 논쟁이 안 끼게):
    I_obs = G_true·V + Σ_상태 sig[fan,s]·P_fan   (+ SMPS 도 섞어 본다)

    python -X utf8 -m src.run_gate_fancol
"""
import numpy as np

from src import env_guard  # noqa: F401

from src.model import gbudget as GB  # noqa: E402
from src.model import inputs as _I  # noqa: E402
from src.model.net import harmonic_signatures_by_state  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-44s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def make_raw(g_ms, fan_w, fan_s, sig, v15, n=1):
    """(1, C, n) 원시. 채널 규약은 `physdecomp` 와 같다 — 0..14 Re I · 15..29 Im I ·
    33.. 전압 Re/Im (`VOLT_ORDERS`)."""
    nv = len(_I.VOLT_ORDERS)
    C = 33 + 2 * nv
    raw = np.zeros((1, C, n), np.float64)
    kf = APPS.index("fan")
    for s_, h in enumerate(_I.VOLT_ORDERS):
        raw[0, 33 + s_] = v15[h - 1].real
        raw[0, 33 + nv + s_] = v15[h - 1].imag
    I = np.zeros(15, complex)
    for s_, h in enumerate(_I.VOLT_ORDERS):          # 저항: I_h = G·V_h
        I[h - 1] += g_ms * 1e-3 * v15[h - 1]
    if fan_w > 0:                                     # 선풍기: sig(와트당)·P
        sg = sig[kf, fan_s]
        I += (sg[:, 0] + 1j * sg[:, 1]) * fan_w
    for h in range(15):
        raw[0, h] = I[h].real
        raw[0, 15 + h] = I[h].imag
    return raw


def main():
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig, us = harmonic_signatures_by_state(pool, APPS)
    v15 = np.zeros(15, complex)
    v15[0] = 216.0
    v15[2] = 216 * 0.006 * np.exp(-1j * np.deg2rad(110))
    v15[4] = 216 * 0.017 * np.exp(-1j * np.deg2rad(160))
    off = GB.Budget(APPS, v15, pool=pool, volt_re0=33, volt_orders=_I.VOLT_ORDERS)
    on = GB.Budget(APPS, v15, pool=pool, volt_re0=33, volt_orders=_I.VOLT_ORDERS,
                   motor_cols=True)
    print("선풍기 기둥 관문 (14.359) — 기둥 %d -> **%d**" % (off.k, on.k))

    # [1] 끄면 비트 동일
    r0 = make_raw(24.957, 0.0, 0, sig, v15)
    chk(1, "`motor_cols=False` 면 **옛 동작 그대로**",
        off.k == 9 and on.k == 12 and [c[1] for c in on.col_src[-3:]] == ["fan"] * 3,
        "끔 기둥 9 (FCM 3 + 에어컨 4) · 켬 **12** (+ 선풍기 3상태)")

    # [2] ★ 선풍기가 없으면 두 판이 같은 Ĝ 를 낸다
    d2 = abs(float(off.g_sum(r0)[0, 0]) - float(on.g_sum(r0)[0, 0]))
    chk(2, "★ 선풍기가 **꺼진** 창에서는 Ĝ 가 안 바뀌나", d2 < 0.02,
        "오븐 단독 24.957 mS — 끔 %.4f · 켬 %.4f · 차 **%.4f mS** (포트 여유 0.635 의 %.1f%%)"
        % (off.g_sum(r0)[0, 0], on.g_sum(r0)[0, 0], d2, 100 * d2 / 0.635))

    # [3] ★★ 선풍기가 켜진 창에서 **Ĝ 오염이 줄어드나**
    rows, worst_off, worst_on = [], 0.0, 0.0
    for gt, nm in ((0.0, "저항 없음"), (9.943, "핫플 단독"), (24.957, "오븐 단독")):
        for s_ in (1, 2, 3):
            if not us[APPS.index("fan"), s_]:
                continue
            pw = {1: 21.5, 2: 29.4, 3: 37.6}[s_]
            r = make_raw(gt, pw, s_, sig, v15)
            a = float(off.g_sum(r)[0, 0]) - gt
            b = float(on.g_sum(r)[0, 0]) - gt
            worst_off = max(worst_off, abs(a)); worst_on = max(worst_on, abs(b))
            rows.append("%s+fan s%d(%.1fW) 오차 **%+.3f -> %+.3f** mS" % (nm, s_, pw, a, b))
    chk(3, "★★ 선풍기 창에서 **Ĝ 오차가 줄어드나**", worst_on < worst_off * 0.5,
        " · ".join(rows) + "  |  최악 **%.3f -> %.3f mS** (포트 여유 0.635 의 %.0f%% -> %.0f%%)"
        % (worst_off, worst_on, 100 * worst_off / 0.635, 100 * worst_on / 0.635))

    # [4] 새 기둥이 **저항을 먹지 않는다** (반대 방향 사고)
    r4 = make_raw(28.122, 0.0, 0, sig, v15)          # 포트 단독
    e4 = abs(float(on.g_sum(r4)[0, 0]) - 28.122)
    chk(4, "⚠ 새 기둥이 **저항 전류를 훔치지 않나**", e4 < 0.05,
        "포트 단독 28.122 mS -> 켬 %.4f · 오차 **%.4f mS**"
        % (on.g_sum(r4)[0, 0], e4))

    # [5] 조건수
    import src.model.physdecomp as PD
    for nm, bud in (("끔", off), ("켬", on)):
        A = PD.build_design(make_raw(24.957, 29.4, 2, sig, v15), bud.T,
                            volt_orders=_I.VOLT_ORDERS, volt_re0=33)[0, 0]
        c_ = np.linalg.cond(A)
        if nm == "끔":
            c_off = c_
        else:
            c_on = c_
    chk(5, "설계행렬 **조건수가 안 터지나**", c_on < max(3.0 * c_off, 1e3),
        "끔 **%.1f** -> 켬 **%.1f** (%.2f배) — [[check-conditioning-before-believing-a-fit]]"
        % (c_off, c_on, c_on / c_off))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
