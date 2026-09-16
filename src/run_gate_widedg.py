# -*- coding: utf-8 -*-
"""`--wide-dg` 의 관문 — 끄면 항등, 켜면 **진짜 계단**을 잰다 (14.315, 제안 ③).

무엇을 거나 — 광역 60초에서 `G(tau) = P(tau)/V(tau)**2` 의 **가장 큰 계단**을 찾아
그 앞뒤 차이를 채널 넷으로 몸통 입력에 더한다. 창 안 전이는 **한 번에 한 대**라
([[nilm-onestep-one-switch]]) 그 차이가 기기 하나의 G 다.

왜 광역인가 (합성 300기록):
```
  기기가 켜진 창 안에 **그 기기 자신의** 전이가 있는 비율
    세밀 10초   오븐(가열) 47.9%  ·  포트 10.3%
    광역 60초   오븐(가열) **89.1%** ·  포트 57.8%
```

⚠ 14.94 가 광역 축을 닫았다 (풀링·수용영역·타깃탭 셋 다 실패, 넷 다 **씨앗 산포가
  기준선의 2~7배**). 그 셋은 광역에 **용량**을 준 것이고 이건 **정보**다. 그래서
  광역에는 파라미터를 한 톨도 안 준다 — 광역은 **숫자의 출처**로만 쓰고 파생 넷은
  세밀 갈래(`_conv_in`)에 넣는다. 그래도 산포가 늘면 그 자리에서 기각이다.

    python -X utf8 -m src.run_gate_widedg [--cache cache/seqraw_k]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model import inputs as _I
from src.model.inputs import (POWER_SCALE, V_CENTER, V_SPAN, WIDE_BLOCK,
                              build_inputs, wide_target_index)
from src.model.net import (NILMNet, P_CH_WIDE, V_CH_WIDE, WDG_CLIP, WDG_SCALE,
                           WDG_THR, appliance_state_counts)

EVEN_K = 5          #: 우리 처방(`--even-median 5`)에 맞춘다. 관문 안에서만 일관되면 된다
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-38s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def build(**kw):
    """진짜 객체를 짓는다 — 관문 전용 축소판을 재면 진짜 경로가 반쪽이어도 통과한다
    ([[the-gate-must-build-the-real-object]])."""
    apps = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
            "hotplate", "laptop_charger", "minipc", "oven"]
    torch.manual_seed(0)
    return NILMNet(apps, appliance_state_counts(apps), fine_channels=57,
                   fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4),
                   p_state_cap=3.0, prior_kappa=8.0, **kw), apps


def windows(cache, n=384):
    """원시 45채널 창과, 그 창의 **참값** 전력·상태·전압을 같이 돌려준다."""
    C = Path(cache)
    m = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    tgt, wc = int(m["target_offset"]), int(m["window_cycles"])
    grid = np.arange(tgt, int(m["record_s"] * 60) - 13 * 60 - 1, int(m["grid_s"] * 60))
    raw = np.load(C / "raw.npy", mmap_mode="r")
    yp = np.load(C / "y_power.npy", mmap_mode="r")
    ys = np.load(C / "y_state.npy", mmap_mode="r")
    W, P, S = [], [], []
    for i in range(len(raw)):
        r = np.asarray(raw[i])
        for t, c in enumerate(grid):
            seg = r[:, c - tgt:c - tgt + wc]
            if seg.shape[1] < wc:
                continue
            W.append(seg)
            P.append(np.asarray(yp[i, t]))
            S.append(np.asarray(ys[i, t]))
        if len(W) >= n:
            break
    return (np.stack(W[:n]).astype(np.float32), np.stack(P[:n]).astype(np.float64),
            np.stack(S[:n]), m, grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_k")
    a = ap.parse_args()
    _I.EVEN_MEDIAN = EVEN_K
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Wr, Ptrue, Strue, meta, _ = windows(a.cache)
    f, w = build_inputs(Wr)
    fa = torch.from_numpy(f).to(dev)
    wa = torch.from_numpy(w).to(dev)
    print("창 %d · 세밀 %s · 광역 %s · even_median=%d\n" % (len(Wr), f.shape[1:], w.shape[1:], EVEN_K))

    # ── [1] 끄면 **항등** ────────────────────────────────────────────────
    off, _ = build(wide_dg=False)
    off = off.to(dev).eval()
    same = off._conv_in(fa, wa) is fa
    chk(1, "끄면 `_conv_in` 이 항등", same and off.n_wdg == 0,
        "`_conv_in(fine, wide) is fine` = %s · n_wdg=%d -> 옛 경로와 **객체까지 같다**"
        % (same, off.n_wdg))

    on, apps = build(wide_dg=True)
    on = on.to(dev).eval()
    ko, kk = apps.index("oven"), apps.index("electiric_kettle")

    # ── [2] 채널 배치를 **재서** 확인한다 ────────────────────────────────
    #     이름표를 믿으면 안 된다 — 교차인 줄 알았는데 블록이라 결론 둘을 철회한 적이 있다
    #     ([[verify-channel-layout-by-measurement]]).
    #     ⚠ **원소별**로 견뎌야 한다. 첫 판은 "창 전체 중앙값의 평균" 대 "블록 중앙값의
    #     평균" 을 견줘 18.5% 가 났는데, 듀티 부하는 평균≠중앙값이라 **자가 틀린 것**이었다.
    nb = w.shape[-1]
    blk = lambda a: np.median(a[:, :nb * WIDE_BLOCK].reshape(len(a), nb, WIDE_BLOCK), 2)
    p_ref, v_ref = blk(Wr[:, 30]), blk(Wr[:, 32])
    p_ch = np.sinh(w[:, P_CH_WIDE].astype(np.float64)) * POWER_SCALE
    v_ch = w[:, V_CH_WIDE].astype(np.float64) * V_SPAN + V_CENTER
    e_p = float(np.median(np.abs(p_ch - p_ref) / np.maximum(np.abs(p_ref), 1.0)))
    e_v = float(np.median(np.abs(v_ch - v_ref) / v_ref))
    chk(2, "P_CH_WIDE=0 · V_CH_WIDE=2 가 맞나", e_p < 1e-3 and e_v < 1e-4,
        "블록 %d칸 원소별 상대오차 중앙 — 전력 **%.2e** · 전압 **%.2e** "
        "(asinh 왕복 + float32 저장의 정밀도)" % (nb, e_p, e_v))

    with torch.no_grad():
        D = on._wide_dg(wa).cpu().numpy()                      # (B,4)
    mag, sgn, pos, ok = D[:, 0], D[:, 1], D[:, 2], D[:, 3]

    # ── [3] 무효 창은 **정확히 0** ───────────────────────────────────────
    z = D[ok == 0]
    chk(3, "무효 창은 넷 다 정확히 0", z.size == 0 or float(np.abs(z).max()) == 0.0,
        "유효 %d / %d 창 (%.1f%%) · 무효 창의 |최대| = %s"
        % (int(ok.sum()), len(ok), 100 * ok.mean(),
           "창 없음" if z.size == 0 else "%.3e" % float(np.abs(z).max())))

    # ── [4] ★ ΔĜ 가 **진짜 계단**을 잡나 ────────────────────────────────
    #     참값으로 판정한다: 창 안에서 딱 한 대만 상태가 바뀌고 그 전력차가 300W 넘는 창.
    #     ⚠ `gt.max(1)` 과 견주면 느슨하다 — 전이의 주인이 그 창의 최대 기기라는 보장이
    #     없다. **어느 기기의 참 G 와도 안 맞으면** 계단이 아닌 것을 잡은 것이다.
    gt = 1e3 * Ptrue / v_ref.mean(1)[:, None] ** 2              # (B,K) mS — 참 컨덕턴스
    big = (gt.max(1) > 5.0) & (ok > 0)
    got = mag[big] * WDG_SCALE                                  # mS
    #: ⚠ 마스크는 **나눗셈 뒤**에 건다 — inf 를 넣고 나누면 inf/inf = nan 이 된다.
    gb = gt[big]
    rel = np.abs(gb - got[:, None]) / np.maximum(gb, 1e-6)      # (n,K)
    best = np.where(gb > 2.0, rel, np.inf).min(1)               # 켜져 있는 기기만 후보
    hit15 = float((best < 0.15).mean())
    chk(4, "ΔĜ 가 **어느 한 기기**의 G 인가", hit15 > 0.60,
        "유효·대형 창 %d개 · 가장 가까운 기기와의 상대거리 중앙 **%.1f%%** · "
        "15%% 안에 드는 창 **%.0f%%** (아무 수나 나오면 여기가 무너진다)"
        % (big.sum(), 100 * float(np.median(best)), 100 * hit15))

    # ── [5] 부호·위치·클립이 규약대로인가 ───────────────────────────────
    nt = w.shape[-1]
    lo, hi = (0 - wide_target_index(nt)) / nt, (nt - 2 - wide_target_index(nt)) / nt
    good = (set(np.unique(sgn[ok > 0])) <= {-1.0, 1.0}
            and float(mag.max()) <= WDG_CLIP + 1e-6
            and float(pos[ok > 0].min()) >= lo - 1e-6
            and float(pos[ok > 0].max()) <= hi + 1e-6)
    chk(5, "부호 ±1 · 크기 클립 · 위치 범위", good,
        "부호 %s · |ΔĜ|최대 %.3f (클립 %.1f) · 위치 [%.3f, %.3f] (허용 [%.3f, %.3f])"
        % (sorted(set(np.unique(sgn[ok > 0]))), float(mag.max()), WDG_CLIP,
           float(pos[ok > 0].min()), float(pos[ok > 0].max()), lo, hi))

    # ── [6] 문턱이 **일을 하나** ────────────────────────────────────────
    #     WDG_THR 를 1.0 으로 낮추면 유효 창이 늘어야 한다. 안 늘면 문턱이 죽은 코드다.
    import src.model.net as _N
    _keep = _N.WDG_THR
    try:
        _N.WDG_THR = 1.0
        with torch.no_grad():
            ok_lo = float(on._wide_dg(wa).cpu().numpy()[:, 3].mean())
    finally:
        _N.WDG_THR = _keep
    chk(6, "문턱 WDG_THR 가 실제로 자른다", ok_lo > ok.mean() + 0.02,
        "THR=%.1f -> 유효 %.1f%% · THR=1.0 -> **%.1f%%**" % (WDG_THR, 100 * ok.mean(), 100 * ok_lo))

    # ── [7] v2 와의 조합은 **막힌다** ───────────────────────────────────
    try:
        build(wide_dg=True, head_layout="v2")
        raised = False
    except ValueError as ex:
        raised = "_conv_in" in str(ex)
    chk(7, "head_layout=v2 조합을 막나", raised,
        "v2 는 `_feats_v2` 가 `_conv_in` 을 안 타서 **조용히 아무 일도 안 난다** -> "
        "%s" % ("ValueError 로 막힌다" if raised else "안 막힌다"))

    # ── [8] ★★ 파생 넷이 **머리까지 닿나** (공허한 통과 막이) ───────────
    #     `_wide_dg` 를 상수로 갈아 끼우고 출력이 움직이는지 본다. 안 움직이면 채널을
    #     만들기만 하고 아무도 안 읽는 것이다 ([[the-gate-must-build-the-real-object]]).
    with torch.no_grad():
        base = on(fa, wa)["power"].clone()
        _orig = on._wide_dg
        on._wide_dg = lambda _w: torch.zeros(_w.shape[0], 4, device=_w.device,
                                             dtype=_w.dtype)
        zero = on(fa, wa)["power"].clone()
        on._wide_dg = _orig
        back = on(fa, wa)["power"]
    moved = float((base - zero).abs().max())
    exact = float((base - back).abs().max())
    chk(8, "파생 넷이 출력을 실제로 움직이나", moved > 1e-3 and exact == 0.0,
        "넷을 0 으로 덮으면 전력 최대차 **%.4g W** (0 이면 아무도 안 읽는 것) · "
        "되돌리면 %.3e" % (moved, exact))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
