# -*- coding: utf-8 -*-
"""녹화 정렬이 **왜 틀렸는지**를 재는 진단 (14.396, §54). 관문이었다가 내려왔다.

⚠⚠ 이 파일은 `run_gate_sigalign` 이었다. 관문이 7/7 을 냈는데 **자가 틀렸다** —
`irr(정렬한 사이클, 정렬한 sig)` 를 쟀다. 학습 자료의 사이클은 **안 돌아간다**.
사이클을 그대로 두고 재면 부호가 뒤집힌다. 그 대조를 여기 남긴다.

    python -X utf8 -m src.run_diag_sigalign
"""
from typing import List

import numpy as np
import torch

from src import env_guard  # noqa: F401

from src.model.net import (harmonic_signatures,  # noqa: E402
                           harmonic_signatures_by_state, state_power_w)
from src.model.sigalign import (FIT_ORDERS, MIN_R2, derotate,  # noqa: E402
                                med_phasor, rec_of, recording_rotations,
                                rotation_report)
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

#: ⚠ `run_gate_powsig_instate` 와 **글자 그대로 같은 차례**다 — 다르면 인덱스가 어긋난다
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
SMPS = ("minipc", "laptop_charger", "beam_projector")
QUIET = ("oven", "electiric_kettle", "hair_dryer", "hotplate", "fan")
DISC = (9, 11, 13)                      # 미니PC↔충전기를 가르는 차수 (14.367)
OK: List[bool] = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-52s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def cells(pool, app, sid):
    """(기기, 상태) 칸의 사이클을 녹화별로 — `harmonic_signatures_by_state` 와 같은 규약."""
    by = {}
    for a in pool.appliance_activations.get(app, []):
        st = getattr(a, "state_id", None)
        if st is None:
            continue
        pw = state_power_w(a, False)
        m = (pw > 1.0) & (np.asarray(st) == sid)
        if m.any():
            by.setdefault(rec_of(a), []).append(
                np.asarray(a.net_harmonics_complex)[m]
                / np.maximum(pw[m], 1e-6)[:, None])
    return {r: np.concatenate(v) for r, v in by.items()}


def irr_of(per_w, sig, orders):
    out = []
    for h in orders:
        i = h - 1
        mag = np.median(np.abs(per_w[:, i]))
        out.append(np.median(np.abs(per_w[:, i] - sig[i])) / max(mag, 1e-12))
    return float(np.mean(out))


def main() -> int:
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    rots = recording_rotations(pool, APPS)
    print("녹화 위상 정렬 관문 (14.395) — 적합 차수 %s · h1 제외\n" % (FIT_ORDERS,))

    # [1] 안 켜면 **바이트 동일**
    b0 = harmonic_signatures(pool, APPS)
    b1 = harmonic_signatures(pool, APPS, rotations=None)
    s0, u0 = harmonic_signatures_by_state(pool, APPS)
    s1, u1 = harmonic_signatures_by_state(pool, APPS, rotations=None)
    chk(1, "안 켜면 **바이트 동일**인가 (두 입구 다)",
        b0.tobytes() == b1.tobytes() and s0.tobytes() == s1.tobytes()
        and np.array_equal(u0, u1),
        "sig %s · sig_state %s — `rotations=None` 이 옛 경로 그대로여야 지난 판과 비교가 선다"
        % (b0.shape, s0.shape))

    bA = harmonic_signatures(pool, APPS, rotations=rots)
    sA, uA = harmonic_signatures_by_state(pool, APPS, rotations=rots)

    # [2] **h1 은 한 비트도 안 바뀐다** (규약 ①)
    h1_same = (b0[:, 0].tobytes() == bA[:, 0].tobytes()
               and s0[:, :, 0].tobytes() == sA[:, :, 0].tobytes())
    _moved = np.abs(sA[:, :, 1:] - s0[:, :, 1:]).max()
    chk(2, "★ **h1 은 한 비트도 안 바뀌나** (규약 ①)", h1_same,
        "h1 최대차 **%.3e** · h3 이상 최대차 %.3e — 돌리면 충전기 irr 이 "
        "0.032 -> 0.060 으로 **두 배** 나빠진다"
        % (np.abs(sA[:, :, 0] - s0[:, :, 0]).max(), _moved))

    # [3] **전체 위상을 안 옮긴다** (규약 ②) — 회전의 중앙값이 0
    meds = {a: float(np.median(list(v.values()))) for a, v in rots.items() if v}
    chk(3, "★ **전체 위상을 안 옮기나** (회전 중앙값 = 0)",
        all(abs(x) < 1e-9 for x in meds.values()),
        "기기별 회전 중앙값 최대 |·| = **%.2e** · 기기 %d종 — 안 빼면 기기 간 "
        "상대 위상이 깨져 `Σ_k P_k·sig_k` 가 무너진다"
        % (max(abs(x) for x in meds.values()) if meds else 0.0, len(meds)))

    # [4] 채택이 갈리나 (이 부분은 여전히 맞다)
    rep = rotation_report(pool, APPS)
    smps_r2 = [rep[a][0] for a in SMPS if a in rep and np.isfinite(rep[a][0])]
    quiet_r2 = [rep[a][0] for a in QUIET if a in rep and np.isfinite(rep[a][0])]
    chk(4, "채택이 갈리나 (SMPS 만 · 문턱 %.2f)" % MIN_R2,
        bool(smps_r2) and bool(quiet_r2) and min(smps_r2) > MIN_R2 > max(quiet_r2),
        " · ".join("%s R² %s" % (a[:9], ("%.2f" % rep[a][0]) if np.isfinite(rep[a][0])
                                 else "녹화1") for a in SMPS + QUIET if a in rep))

    # [5] ⚠⚠⚠ **두 자를 나란히 놓는다** — 이것이 §54 의 반증이다
    s0, u0 = harmonic_signatures_by_state(pool, APPS)
    sA, _ = harmonic_signatures_by_state(pool, APPS, rotations=rots)
    rows, ok5 = [], True
    for app, sid in (("laptop_charger", 2), ("minipc", 2), ("beam_projector", 2),
                     ("beam_projector", 1)):
        j = APPS.index(app)
        if not u0[j, sid]:
            continue
        C = cells(pool, app, sid)
        if len(C) < 2:
            continue
        raw = np.concatenate([C[r] for r in sorted(C)])
        alg = np.concatenate([derotate(C[r], float(rots.get(app, {}).get(r, 0.0)))
                              for r in sorted(C)])
        #: 오라클 — 사이클도 같이 돌려 놓고 잰다 (관문이 하던 것)
        o0 = irr_of(raw, med_phasor(raw), DISC)
        o1 = irr_of(alg, med_phasor(alg), DISC)
        #: ★ 올바른 자 — 사이클은 **그대로**, 손실이 쓸 sig 만 바꾼다
        g0 = s0[j, sid, :, 0] + 1j * s0[j, sid, :, 1]
        gA = sA[j, sid, :, 0] + 1j * sA[j, sid, :, 1]
        t0, t1 = irr_of(raw, g0, DISC), irr_of(raw, gA, DISC)
        rows.append("%s s%d 오라클 %+.1f%% **대 올바른 자 %+.1f%%**"
                    % (app[:9], sid, 100 * (o1 - o0) / o0, 100 * (t1 - t0) / t0))
        ok5 &= (100 * (t1 - t0) / t0) > -5.0        # 올바른 자로는 **이득이 없어야** 한다
    chk(5, "⚠⚠ 오라클과 올바른 자가 **다른가** (정렬은 이득이 없다)", ok5,
        " · ".join(rows))

    # [5b] ★ 되푼 와트 — 줄어든 사전이 **편향이 없다**
    j = APPS.index("laptop_charger")
    z = np.concatenate([v for _, v in sorted(cells(pool, "laptop_charger", 2).items())])
    g0 = s0[j, 2, :, 0] + 1j * s0[j, 2, :, 1]
    gA = sA[j, 2, :, 0] + 1j * sA[j, 2, :, 1]
    def wat(g, h):
        i = h - 1
        return float(np.mean((z[:, i] * np.conj(g[i])).real) / max(abs(g[i]) ** 2, 1e-18))
    chk(51, "★ **줄어든 사전이 편향이 없나** (되푼 와트, 참값 1.000)",
        abs(wat(g0, 15) - 1.0) < abs(wat(gA, 15) - 1.0),
        "충전기 h15 — 지금 **%.3f** 대 정렬 %.3f · h11 %.3f 대 %.3f "
        "⇒ `E[P*] = P·E[cosθ]/ρ = P`. 길이를 되찾으면 **낮게** 치우친다"
        % (wat(g0, 15), wat(gA, 15), wat(g0, 11), wat(gA, 11)))

    # [6] 녹화가 하나뿐인 기기는 **항등**
    one = [a for a in APPS if a not in rots]
    idxs = [APPS.index(a) for a in one]
    chk(6, "녹화가 **하나뿐인 기기**는 항등인가", not idxs
        or s0[idxs].tobytes() == sA[idxs].tobytes(),
        "표에 없는 기기 %s — 맞출 것이 없으면 건드리면 안 된다"
        % (", ".join(one) if one else "없음"))

    # [7] ⚠⚠ **진짜 손실 객체**를 지어 본다 (함수가 아니라 배선을 잰다)
    from src.model.lossbuild import build_loss
    L0 = build_loss(APPS, "cpu", state_signatures=True, verbose=False)
    L1 = build_loss(APPS, "cpu", state_signatures=True, align_recordings=True, verbose=False)
    d = (L1.sig_state - L0.sig_state).abs()
    si = [APPS.index(a) for a in SMPS]
    qi = [APPS.index(a) for a in QUIET if a in APPS]
    rs = float(d[si].max()) / max(float(L0.sig_state[si].abs().max()), 1e-12)
    rq = float(d[qi].max()) / max(float(L0.sig_state[qi].abs().max()), 1e-12)
    B = 4
    out = {"power": torch.rand(B, len(APPS)) * 30,
           "power_raw": torch.rand(B, len(APPS)) * 30 + 1,
           "on_logit": torch.zeros(B, len(APPS)),
           "plugged_logit": torch.zeros(B, len(APPS))}
    pr = L1._harm_pred_active(out, out["power"])
    chk(7, "⚠⚠ **진짜 손실 객체**가 바뀌고 forward 가 유한한가",
        rs > 5 * rq and bool(torch.isfinite(pr).all()) and float(d.max()) > 0,
        "`build_loss(align_recordings=True)` 의 sig_state 상대 변화 — SMPS **%.4f** 대 "
        "대조군 %.4f (**%.0f배**) · `_harm_pred_active` 유한 %s · h1 불변 %s"
        % (rs, rq, rs / max(rq, 1e-12), bool(torch.isfinite(pr).all()),
           bool(torch.equal(L0.sig_state[:, :, 0], L1.sig_state[:, :, 0]))))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
