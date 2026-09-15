# -*- coding: utf-8 -*-
"""관문 — `--on-detach-gate` 가 **경사만** 끊는가 (14.147).

사용자: *"게이트랑 전력이랑 결합되어 있으니까, 학습할 때도 그렇고 필연적으로
불확실성이 생길 수밖에 없는 전이 순간에 게이트가 전력 예측을 흔들면서 문제를
발생시킬 수도 있나?"*

**그렇다. 재서 확인했다 (14.146).** `power = σ(on_logit) · p_raw` 라 전력 손실의
하강 방향이 둘이다 — `p_raw` 를 낮추거나 **게이트를 닫거나**. 합성 홀드아웃
8000창·3시드에서 **참ON 창**의 게이트와 (참전력/슬롯 비) 상관이

```
  에어컨 **0.777** · **오븐 0.618** · 포트 0.333 · 선풍 0.261 · 미니PC 0.233
  (순수 검출기면 0 이어야 한다 · 참OFF 게이트는 0.0000~0.0020 으로 멀쩡하다)
  참ON 인데 게이트<0.9 인 창: 미니PC 46.2% · 에어컨 39.3% · 선풍 22.2%
```
그리고 저장소에 **이미 한 번 기록돼 있었다** (13.98) — *"작은 와트 슬롯이 없으면
'켜짐인데 0W' 를 표현할 수 없어 게이트가 0 으로 간다"*. 핫플 재현 0.41 -> 0.990 이
슬롯을 옮기기만 해서 났고 **전력 출력은 그대로였다.**

⇒ **참ON 창에서만** 게이트를 detach 한다. `--off-detach-praw`(13.11)의 **짝**이다:
```
  참OFF 창   `p_raw` 를 끊는다  -> 게이트가 0 으로 만드는 일을 맡는다   (기존)
  참ON  창   **게이트를 끊는다** -> `p_raw` 가 진폭을 맡는다            (이번)
```

무엇을 확인하나
---------------
```
  [1] 끄면 손실 값도 **경사도** 비트 동일
  [2] 켜도 **손실 값은 비트 동일** — detach 는 값이 아니라 경사만 바꾼다
  [3] ★ 참ON 창에서 `on_logit` 이 **전력 손실 경사를 못 받는다** (정확히 0)
  [4] ★ `p_raw` 경사는 **안 줄어든다** (진폭 학습이 살아 있다)
  [5] 참**OFF** 창에서는 게이트가 **여전히** 전력 손실 경사를 받는다 (거긴 안 끊는다)
  [6] 게이트 BCE 는 여전히 `on_logit` 에 닿는다 — 게이트가 안 죽는다
  [7] `--off-detach-praw` 와 같이 켜면 **둘 다** 걸린다
  [8] 사영(proj>0)과 같이 쓰면 **거부**
```

    python -X utf8 src/run_gate_ondetach.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.losses import (LossWeights, NILMLoss, S_STATE,  # noqa: E402
                              build_state_scales)

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
H = 15
OK, NG = "OK", "** 실패 **"
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % (OK if good else NG, name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


def _loss(**kw):
    s_i = torch.tensor([max(max(S_STATE.get(a, {1: 10.0}).values()), 10.0) for a in APPS])
    r = np.random.RandomState(0)
    return NILMLoss(
        s_i=s_i,
        signatures=torch.from_numpy((r.randn(len(APPS), H, 2) * 1e-3).astype(np.float32)),
        standby_sig=torch.zeros(len(APPS), H, 2),
        noise_sig=torch.zeros(H, 2), harm_scale=torch.ones(H),
        s_state=build_state_scales(APPS, s_i.tolist()),
        weights=LossWeights(power=1.0, state=0.0, on=0.0, plugged=0.0,
                            standby=0.0, harm=0.0, cons=0.0, over=0.0),
        **kw)


def _io(B=64, seed=0):
    """`on_logit`·`p_raw` 를 **잎 텐서**로 두고 출력을 짓는다 — 경사를 직접 읽으려고."""
    g = torch.Generator().manual_seed(seed)
    K, S = len(APPS), 5
    on_logit = (torch.randn(B, K, generator=g) * 1.5).requires_grad_(True)
    p_raw = (torch.rand(B, K, generator=g) * 800 + 50).requires_grad_(True)
    y_on = (torch.rand(B, K, generator=g) > 0.5).float()
    out = {"power": torch.sigmoid(on_logit) * p_raw, "power_raw": p_raw,
           "on_logit": on_logit, "plugged_logit": torch.zeros(B, K),
           "standby": torch.zeros(B, K),
           "state": torch.randn(B, K, S, generator=g),
           "power_mix": torch.softmax(torch.randn(B, K, S, generator=g), -1),
           "power_states": torch.rand(B, K, S, generator=g) * 800}
    tgt = {"y_power": (torch.rand(B, K, generator=g) * 800), "y_on": y_on,
           "y_plugged": y_on.clone(), "y_standby": torch.zeros(B, K),
           "p_noise": torch.zeros(B), "y_state": torch.zeros(B, K, dtype=torch.long)}
    return on_logit, p_raw, y_on, out, tgt


def _grads(**kw):
    on_logit, p_raw, y_on, out, tgt = _io()
    L = _loss(**kw)
    parts = L(out, tgt)
    tot = parts["power"] if isinstance(parts, dict) else parts[0]["power"]
    tot.backward()
    return float(tot), on_logit.grad.clone(), p_raw.grad.clone(), y_on.bool()


def main() -> int:
    print("`--on-detach-gate` — **켜진 창에서 게이트에 경사를 주지 않는다** (14.147)")
    print("  `--off-detach-praw`(13.11) 의 짝이다. 값은 안 바뀌고 **경사만** 바뀐다.\n")

    v0, g0, p0, on = _grads()
    v0b, g0b, p0b, _ = _grads(on_detach_gate=False)
    ck("[1] 끄면 손실 값도 **경사도** 비트 동일",
       v0 == v0b and torch.equal(g0, g0b) and torch.equal(p0, p0b),
       "손실 %.8f" % v0)

    v1, g1, p1, _ = _grads(on_detach_gate=True)
    ck("[2] 켜도 **손실 값은 비트 동일** (detach 는 경사만 끊는다)", v0 == v1,
       "%.8f == %.8f" % (v0, v1))

    ck("[3] ★ **참ON 창**에서 `on_logit` 이 전력 손실 경사를 **못 받는다**",
       float(g1[on].abs().max()) == 0.0 and float(g0[on].abs().max()) > 0,
       "끄면 최대 %.4g -> 켜면 **%.4g**" % (g0[on].abs().max(), g1[on].abs().max()))

    ck("[4] ★ `p_raw` 경사는 **안 줄어든다** (진폭 학습이 살아 있다)",
       bool(torch.equal(p0, p1)),
       "참ON 창 경사 합 %.6g == %.6g" % (p0[on].abs().sum(), p1[on].abs().sum()))

    ck("[5] 참**OFF** 창에서는 게이트가 **여전히** 경사를 받는다 (거긴 안 끊는다)",
       float(g1[~on].abs().max()) > 0 and torch.equal(g1[~on], g0[~on]),
       "최대 %.4g (끄면 %.4g)" % (g1[~on].abs().max(), g0[~on].abs().max()))

    # ── [6] BCE 는 여전히 닿는가 ──────────────────────────────────────────
    ol, pr, yon, out, tgt = _io()
    L = _loss(on_detach_gate=True)
    L.w.power, L.w.on = 0.0, 1.0
    parts = L(out, tgt)
    parts["on"].backward()
    gb = ol.grad
    ck("[6] 게이트 **BCE** 는 여전히 `on_logit` 에 닿는다 — 게이트가 안 죽는다",
       gb is not None and float(gb[yon.bool()].abs().max()) > 0,
       "참ON 창 BCE 경사 최대 %.4g" % (gb[yon.bool()].abs().max() if gb is not None else 0))

    # ── [7] 둘 다 켜면 둘 다 걸리나 ───────────────────────────────────────
    vb, gbo, pbo, _ = _grads(on_detach_gate=True, off_detach_praw=True)
    _, gof, pof, _ = _grads(off_detach_praw=True)
    ck("[7] `off_detach_praw` 와 같이 켜면 **둘 다** 걸린다",
       float(gbo[on].abs().max()) == 0.0 and float(pof[~on].abs().max()) == 0.0
       and float(pbo[~on].abs().max()) == 0.0 and vb == v0,
       "참ON 게이트 경사 0 · 참OFF p_raw 경사 0 · 손실 %.8f == %.8f" % (vb, v0))

    # ── [9~12] ★ B: 마스크 독립 손실 (`on_power_praw`) ───────────────────
    v2, g2, p2, _ = _grads(on_power_praw=True)
    ck("[9] ★ B 도 참ON 창에서 `on_logit` 에 경사를 **안 준다**",
       float(g2[on].abs().max()) == 0.0,
       "최대 %.4g" % g2[on].abs().max())
    #: A 는 p_raw 경사에 gate 가 곱해져 있다 -> p_raw 가 **y/gate 로 밀린다**(보상 왜곡).
    #: B 는 그 곱이 없으므로 참ON 창 p_raw 경사가 A 보다 **크다** (gate<1 이므로).
    ra = float(p1[on].abs().sum()); rb = float(p2[on].abs().sum())
    ck("[10] ★ B 는 `p_raw` 경사에 **게이트가 안 곱해진다** (A 보다 크다)", rb > ra * 1.05,
       "A %.6g -> B **%.6g** (%.2f배)" % (ra, rb, rb / max(ra, 1e-12)))
    ck("[11] B 는 손실 **값이 바뀐다** (A 는 안 바뀐다 — 다른 처치다)",
       v2 != v0 and v1 == v0, "A %.8f == 대조 · B **%.8f**" % (v1, v2))
    ck("[12] B 도 참**OFF** 창 게이트 경사는 **그대로**",
       torch.equal(g2[~on], g0[~on]), "최대 %.4g" % g2[~on].abs().max())
    bad2 = ""
    try:
        _grads(on_detach_gate=True, on_power_praw=True); g13 = False
    except ValueError as e:
        g13, bad2 = True, str(e)[:44]
    ck("[13] A 와 B 를 같이 쓰면 **거부**", g13, bad2)

    # ── [8] 사영과 같이 쓰면 거부 ─────────────────────────────────────────
    bad = ""
    try:
        ol2, pr2, y2, out2, tgt2 = _io()
        #: ⚠ `_proj_seen` 은 `forward` 안에서 `out` 에 `proj_r` 이 있는지로 **매번 다시
        #:   세워진다** (losses.py:848). 밖에서 세우면 덮인다 — 표식을 `out` 에 넣어야 한다.
        out2["proj_r"] = torch.zeros(1)
        L2 = _loss(on_detach_gate=True)
        L2(out2, tgt2)
        good = False
    except ValueError as e:
        good, bad = True, str(e)[:52]
    ck("[8] 사영(proj>0)과 같이 쓰면 **거부**", good, bad)

    print("")
    print("  ⚠ **예측을 먼저 적는다.** 게이트가 검출만 하게 되면 참ON 창의 게이트가 1 에")
    print("    더 붙고, `p_raw` 가 진폭을 통째로 맡는다. 통전 헛detect 가 줄어들지는")
    print("    **모른다** — 14.146 은 상관만 보였지 인과를 안 보였다.")
    print("    ⇒ 통과 조건은 **'저항 4종 신원·판정줄·잔차가 안 나빠진다'** 이고,")
    print("      통전 자(포트창·겹침창)는 **관찰**한다.")
    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
