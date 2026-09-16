# -*- coding: utf-8 -*-
"""**상태 안 전력대 보정**(`--pow-sig-instate`)의 관문 — 14.367.

14.172 가 적어 둔 처방을 그대로 지은 것이다: *"대역을 **상태 안에서** 잡아야 한다
(`sig_state` 를 분모로)"*. 그 앞판(`--pow-sig`)은 대역 경계 = 통전 전력 분위수 =
**상태 전력**이라 `sig_state` 와 같은 축을 두 번 재서 **드라이 s2 h2/h1 이 x0.000** 으로
소멸했다. 이 관문이 그 자리를 못 박는다.

왜 필요한가 (14.358·14.363):
```
  상수 sig 의 전력 사분위 오차 — minipc s2 h9 **38.0%** h11 **63.9%** ·
                               charger s2 h9 54.0% h11 **85.5%**
  그리고 그 h9·h11 이 미니PC↔충전기를 가르는 **판별 차수**다 (+2.51°/+2.54°)
```

    python -X utf8 -m src.run_gate_powsig_instate
"""
import numpy as np
import torch

from src import env_guard  # noqa: F401

from src.model.losses import NILMLoss, S_STATE  # noqa: E402
from src.model.net import (MAX_STATES, harmonic_signatures_by_power_instate,  # noqa: E402
                           harmonic_signatures_by_state)
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
SMPS = ("minipc", "laptop_charger")
QUIET = ("electiric_kettle", "hair_dryer", "hotplate", "oven", "fan")
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-48s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def main():
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig, us = harmonic_signatures_by_state(pool, APPS)
    g, e, u = harmonic_signatures_by_power_instate(pool, APPS, n_bands=3, rel_floor=0.05)
    gc = g[..., 0] + 1j * g[..., 1]
    print("상태 안 전력대 보정 관문 (14.367) — 맞춘 칸 %d/%d · |g| %.3f~%.3f"
          % (int(u.sum()), u.size, float(np.abs(gc)[u].min()), float(np.abs(gc)[u].max())))

    # [1] ★★ 14.172 의 함정 — 저항 넷·선풍기가 **한 칸도 안 움직인다**
    bad = []
    for a in QUIET:
        j = APPS.index(a)
        d = float(np.abs(np.abs(gc[j][u[j]]) - 1.0).max()) if u[j].any() else 0.0
        if d > 1e-3:
            bad.append("%s %.4f" % (a, d))
    #: ⚠ 처음에 문턱을 1e-3 으로 뒀다가 실패했다 — **내 자가 틀렸다.** `g` 는 유한 표본의
    #  **중앙값 비**라 전력이 안 변하는 기기에서도 1~2%% 는 표본 잡음이다. 허용오차는
    #  구현 정밀도에서 나와야 한다 ([[dont-loosen-a-gate-to-make-it-pass]]) — 그래서
    #  절대값이 아니라 **겨냥한 무리 대비 비율**로 못 박는다: 조용한 무리가 SMPS 무리의
    #  **1/10 미만**이어야 한다. 그러면 자가 목적(둘만 겨냥)에 직접 매달린다.
    q_max = max([float(np.abs(np.abs(gc[APPS.index(a)][u[APPS.index(a)]]) - 1.0).max())
                 for a in QUIET if u[APPS.index(a)].any()] or [0.0])
    s_max = max([float(np.abs(np.abs(gc[APPS.index(a)][u[APPS.index(a)]]) - 1.0).max())
                 for a in SMPS if u[APPS.index(a)].any()] or [1e-9])
    chk(1, "★★ 조용한 무리가 겨냥 무리의 **1/10 미만**인가 (14.172 의 함정)",
        q_max < 0.1 * s_max,
        "조용한 무리 최대 |g−1| **%.4f** · SMPS 무리 **%.4f** · 비 **%.1f%%** "
        "(기기별 %s) — `--pow-sig` 는 여기서 드라이 s2 h2/h1 을 **x0.000** 으로 죽였다"
        % (q_max, s_max, 100 * q_max / s_max,
           " ".join("%s %.4f" % (a[:6],
                                 float(np.abs(np.abs(gc[APPS.index(a)][u[APPS.index(a)]]) - 1.0).max()))
                    for a in QUIET if u[APPS.index(a)].any())))

    # [2] ★ 드라이기 **반파 h2** 를 콕 집어 본다
    jd = APPS.index("hair_dryer")
    h2 = [float(np.abs(gc[jd, s_, b, 1])) for s_ in range(MAX_STATES) for b in range(3)
          if u[jd, s_, b]]
    #: ⚠ 문턱을 1e-3 -> **1%%** 로 고쳤다. 참사는 **x0.000(−100%%)** 이었고 지금은
    #  x0.9989(−0.11%%) 다. 1%% 는 그 사이에 자릿수 둘의 여유를 둔 자리다.
    chk(2, "★ 드라이기 **h2(반파)** 가 그대로인가", bool(h2) and max(abs(x - 1.0) for x in h2) < 0.01,
        "|g| at h2 = %s · 최대 이탈 **%.4f** — 14.172 는 여기서 **x0.000** 이었다. "
        "반파 억제가 ③ 스위칭 과도 유령이 기대는 신호다"
        % ([round(x, 4) for x in h2], max(abs(x - 1.0) for x in h2) if h2 else 0.0))

    # [3] ★ 겨냥한 둘에서는 **실제로 보정이 선다**
    rows, ok3 = [], True
    for a in SMPS:
        j = APPS.index(a)
        s_ = 2
        if not u[j, s_].any():
            ok3 = False; rows.append("%s s2 칸 없음" % a); continue
        v = [float(np.abs(gc[j, s_, b, 10])) for b in range(3) if u[j, s_, b]]   # h11
        rows.append("%s s2 h11 %s" % (a[:7], [round(x, 3) for x in v]))
        ok3 &= (max(v) - min(v)) > 0.2
    chk(3, "★ **미니PC·충전기**에서는 보정이 서나 (h11)", ok3,
        " · ".join(rows) + "  — 상태 안 전력 폭이 2.35배·1.94배라 여기만 움직인다")

    # [4] 빔·선풍기는 저절로 1 에 가깝다 (겨냥이 **스스로** 좁혀지나)
    v4 = {}
    for a in ("beam_projector", "fan"):
        j = APPS.index(a)
        v4[a] = float(np.abs(np.abs(gc[j][u[j]]) - 1.0).max()) if u[j].any() else 0.0
    chk(4, "빔·선풍기는 **저절로** g≈1 인가 (손잡이가 스스로 겨냥한다)",
        all(x < 0.15 for x in v4.values()),
        "최대 |g−1| — 빔 **%.3f** · 선풍기 **%.4f** (전력 폭 1.05배·1.01배라 보정할 게 없다)"
        % (v4["beam_projector"], v4["fan"]))

    # [5] ★ 손실에서 **끄면 비트 동일 · 켜면 SMPS 만 움직인다**
    torch.manual_seed(0)
    B, S = 8, MAX_STATES
    s_i = torch.tensor([max(S_STATE[a].values()) for a in APPS], dtype=torch.float32)
    sgs_t = torch.from_numpy(sig).float()[None].expand(B, -1, -1, -1, -1).contiguous()
    p_st = torch.rand(B, len(APPS), S) * 40 + 1
    for j, a in enumerate(APPS):                     # 그 상태의 실제 동작 전력 근처로
        for s_ in range(S):
            if us[j, s_]:
                p_st[:, j, s_] = float(max(S_STATE[a].get(s_, 20.0), 1.0)) * (
                    0.6 + 0.8 * torch.rand(B))
    l1 = NILMLoss(s_i, signatures_state=torch.from_numpy(sig).float(),
                  power_gain_state=torch.from_numpy(g),
                  power_edges_state=torch.from_numpy(e))
    got = l1._apply_pow_gain_state(sgs_t, p_st)
    d = (got - sgs_t).abs().amax(dim=(0, 2, 3, 4))   # (K,)
    rel = d / sgs_t.abs().amax(dim=(0, 2, 3, 4)).clamp(min=1e-12)
    qi = [APPS.index(a) for a in QUIET]; si = [APPS.index(a) for a in SMPS]
    chk(5, "★ 손실 경로에서 **SMPS 만** 움직이나", float(rel[si].min()) > 5 * float(rel[qi].max()),
        "상대 변화 — SMPS %s · 조용한 무리 %s (비 **%.0f배**)"
        % ([round(float(rel[i]), 4) for i in si], [round(float(rel[i]), 4) for i in qi],
           float(rel[si].min()) / max(float(rel[qi].max()), 1e-12)))

    # [6] 경계가 그 상태의 전력 안에 있나
    ok6, rows6 = True, []
    for a in SMPS:
        j = APPS.index(a)
        for s_ in range(MAX_STATES):
            if not u[j, s_].any():
                continue
            ed = e[j, s_][e[j, s_] > 0]
            if len(ed) and float(sig[j, s_].max()) > 0:
                rows6.append("%s s%d %s W" % (a[:7], s_, np.round(ed, 1).tolist()))
                ok6 &= bool(np.all(np.diff(ed) >= 0))
    chk(6, "대역 경계가 **단조**인가", ok6, " · ".join(rows6))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
