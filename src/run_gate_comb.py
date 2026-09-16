# -*- coding: utf-8 -*-
"""**조합 머리**(`--comb-tau`)의 관문 — 14.347~348.

저항 4종을 독립 게이트가 아니라 **24개 조합 위의 softmax** 로 낸다:
```
  z_{k,OFF} = log( (1−σ_k) + σ_k·Σ_{s∉표} mix_s )      <- 컨덕턴스 0 은 OFF 만이 아니다
  z_{k,s}   = log σ_k + log mix_k[s]                    (s ∈ PIN_MS[k])
  score_c   = Σ_k β_k·z_{k,c_k}  −  |Ĝ − ΣG(c)| / τ
  P_k       = Σ_c w_c·G(k,c_k)·|V₁|²  +  (표에 없는 켜짐 상태의 모델 자유 크기)
```
왜 이 모양인가 (14.343): Ĝ 는 **계량기**(총 전력 −0.39%)지 **분류기**가 아니다
(조합 84.5%). 모델은 94.9%. **합치면 99.69%** — 이득이 결합에서만 나온다.

⚠ **`comb_tau>0` 이면 저항 전력이 `G 표`에서 나온다.** τ→∞ 가 지금과 같아지는 것은
  **게이트·상태 주변확률**뿐이고 전력은 아니다. 끄고 켜는 스위치는 `comb_tau == 0` 이다.

    python -X utf8 -m src.run_gate_comb
"""
import itertools
import sys

import numpy as np
import torch

from src import env_guard  # noqa: F401

from src.model import gbudget as GB  # noqa: E402
from src.model import inputs as _I  # noqa: E402
from src.model.losses import S_STATE  # noqa: E402
from src.model.net import MAX_STATES, NILMNet, appliance_state_counts  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-44s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def build(**kw):
    """**진짜 객체**를 짓는다 ([[the-gate-must-build-the-real-object]]).

    `prior_kappa` 는 트레이너 값(8.0)을 쓴다 — 기본 0.0 으로 지으면 물리 프라이어 블록이
    아예 안 돌아 관문이 반쪽을 재게 된다 (14.138 이 그렇게 당했다).
    """
    torch.manual_seed(0)
    m = NILMNet(APPS, appliance_state_counts(APPS), fine_channels=_I.FINE_CHANNELS,
                fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4), p_state_cap=3.0,
                prior_kappa=8.0, aux_z=True, **kw)
    m.eval()
    return m


def main():
    torch.manual_seed(1)
    B = 16
    f = torch.randn(B, _I.FINE_CHANNELS, 600) * 0.3
    w = torch.randn(B, _I.WIDE_CHANNELS, 120) * 0.3
    g = torch.rand(B) * 45.0
    off, on, inf = build(), build(comb_tau=1.0), build(comb_tau=1e9)
    with torch.no_grad():
        a, b, c = off(f, w), on(f, w, g), inf(f, w, g)
    R = on.comb_cols.tolist()
    non = [j for j in range(len(APPS)) if j not in R]
    print("조합 머리 관문 (14.348) — 저항 %s · 후보 %d개"
          % ([APPS[j][:6] for j in R], on.comb_g.shape[0]))

    # [1] 끄면 비트 동일
    same = all(torch.equal(a[k], off(f, w)[k]) for k in ("power", "on_logit", "state"))
    with torch.no_grad():
        a2 = build()(f, w)
    chk(1, "`comb_tau=0` 이면 **비트 동일**",
        same and all(torch.equal(a[k], a2[k]) for k in ("power", "on_logit", "state")),
        "같은 씨앗 두 판이 power·on_logit·state 에서 한 비트도 안 다르다")

    # [2] ★ τ→∞ 주변확률 항등식
    st = inf.comb_state
    marg = torch.stack([c["comb_w"][:, st[:, r] > 0].sum(-1) for r in range(len(R))], 1)
    pin = torch.zeros(len(R), MAX_STATES)
    for i, j in enumerate(R):
        for s_ in GB.PIN_MS[APPS[j]]:
            pin[i, s_] = 1.0
    want = torch.sigmoid(c["on_logit"][:, R]) * (c["power_mix"][:, R] * pin[None]).sum(-1)
    d2 = float((marg - want).abs().max())
    chk(2, "★ τ→∞ 주변확률 == `σ(on)·Σ_{s∈표} mix_s`", d2 < 1e-5,
        "최대차 **%.3e** · 기기별 %s — 물리를 끄면 조합 softmax 가 **독립 시그모이드로 "
        "정확히 되돌아간다**는 뜻이다"
        % (d2, [round(float(x), 7) for x in (marg - want).abs().max(0).values.tolist()]))

    # [3] 파라미터
    n0, n1 = (sum(p.numel() for p in off.parameters()),
              sum(p.numel() for p in on.parameters()))
    chk(3, "파라미터가 **바닥 + %d** 뿐인가" % len(R), n1 - n0 == len(R),
        "끔 %d · 켬 %d · 차 **%d** (기기별 β %d개)" % (n0, n1, n1 - n0, len(R)))

    # [4] 후보표가 PIN_MS 와 맞고 반파가 갈려 있다
    want_c = list(itertools.product(*[[0.0] + [v for _, v in sorted(GB.PIN_MS[APPS[j]].items())]
                                      for j in R]))
    tbl_ok = (on.comb_g.shape[0] == len(want_c)
              and np.allclose(sorted(on.comb_gsum.tolist()),
                              sorted(float(sum(x)) for x in want_c)))
    hd = dict(GB.app_states("hair_dryer"))
    chk(4, "후보 %d개가 `PIN_MS` 와 맞고 **반파가 갈려 있나**" % len(want_c),
        tbl_ok and len(hd) == 2 and abs(min(hd.values()) - 9.45) < 0.5,
        "총합 최소 %.3f · 최대 %.3f mS · 드라이 %s — 뭉개면 14.327 처럼 전력오차가 "
        "146 -> 628W 로 터진다"
        % (float(on.comb_gsum.min()), float(on.comb_gsum.max()),
           " ".join("s%d %.2f" % (s, v) for s, v in sorted(hd.items()))))

    # [5] ★ Ĝ 를 주면 그 조합이 뽑히나 (물리만 남기고 본다)
    phys = build(comb_tau=0.3)
    with torch.no_grad():
        phys.comb_beta.zero_()                       # 모델 의견을 끈다 = 물리 단독
        rows, ok5 = [], True
        for nm in ("oven", "electiric_kettle", "hotplate"):
            gk = GB.max_ms(nm)
            o5 = phys(f, w, torch.full((B,), gk))
            j = int(o5["comb_w"].mean(0).argmax())
            picked = [APPS[R[r]][:6] for r in range(len(R)) if phys.comb_state[j, r] > 0]
            rows.append("Ĝ=%.2f -> %s" % (gk, "+".join(picked) or "(없음)"))
            ok5 &= (picked == [nm[:6]])
    chk(5, "★ `Ĝ` 를 주면 **그 조합**이 뽑히나 (β=0, 물리 단독)", ok5,
        " · ".join(rows))

    # [6] 저항 아닌 기기 격리
    chk(6, "저항 아닌 %d종이 **한 칸도 안 바뀐다**" % len(non),
        torch.equal(a["power"][:, non], b["power"][:, non])
        and torch.equal(a["on_logit"], b["on_logit"]) and torch.equal(a["state"], b["state"]),
        "power 최대차 %.3e · on_logit·state 비트 동일"
        % float((a["power"][:, non] - b["power"][:, non]).abs().max()))

    # [7] g_hat 없이 부르면 멈춘다
    try:
        on(f, w)
        ok7, msg7 = False, "**안 멈췄다** — 조용히 반쪽으로 돈다"
    except ValueError as e:
        ok7, msg7 = True, "`%s`" % str(e)[:70]
    chk(7, "★ `g_hat` 없이 부르면 **멈추나**", ok7, msg7)

    # [8] β 에 기울기가 흐르나
    on.train()
    d8 = on(f, w, g)
    d8["power"].sum().backward()
    gb = on.comb_beta.grad
    chk(8, "기울기가 **β 에 흐르나** (죽은 파라미터가 아니다)",
        gb is not None and float(gb.abs().sum()) > 0,
        "β 기울기 %s" % [round(float(x), 3) for x in (gb if gb is not None else []).tolist()])
    on.eval()

    # [9] ★ 표에 없는 켜짐 상태가 안 사라지나
    small = {APPS[R[i]]: [s for s in range(MAX_STATES) if on.comb_small[i, s] > 0]
             for i in range(len(R))}
    want_small = {APPS[j]: sorted(s for s in range(MAX_STATES)
                                  if on.power_mix_mask[j, s] > 0 and s not in GB.PIN_MS[APPS[j]])
                  for j in R}
    with torch.no_grad():
        zero = on(f, w, torch.zeros(B))              # 예산 0 -> 통전 조합은 안 뽑힌다
    # 오븐 FAN_LIGHT 의 몫이 남아 있어야 한다
    jo = APPS.index("oven")
    left = float(zero["power"][:, jo].max())
    chk(9, "★ 표에 없는 켜짐 상태(오븐 FAN_LIGHT)가 **안 사라지나**",
        small == want_small and left > 0.0,
        "`comb_small` %s (기대 %s) · Ĝ=0 에서도 오븐 전력 최대 **%.2fW** 남는다 — "
        "0 이면 FAN_LIGHT 14.6W 가 통째로 증발한 것이다" % (small, want_small, left))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
