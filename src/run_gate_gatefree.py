# -*- coding: utf-8 -*-
"""관문 — `--gate-free-power` 가 **게이트와 전력을 완전히 분해하는가** (14.150).

사용자: *"또 다른 게이트와의 연결점이 있는지 검토하고, 완전히 게이트랑 전력예측을
분해하고 학습을 돌려봐야 하지 않겠어?"*

전수조사에서 연결점이 **여섯**이었다.
```
  (1) 값     out["power"] = σ(on)·p_raw
  (2) 경사   ∂power/∂p_raw = **σ(on)**  — 게이트가 0 이면 전력 기울기도 죽는다
             (13.80 주석: *"잘못 포화한 게이트는 **흡수 상태**다"*)
  (3) 경사   ∂power/∂on_logit = σ'·p_raw
  (4) 프라이어 `on_logit += logsigmoid(κ·gap)` -> 게이트 -> 전력
  (5) 공유머리 `on_logit`·`state`·`p_states` 가 **같은 `hd(z)`** 에서 나온다
  (6) 손실   `L_harm`(w 0.1) 도 `out["power"]` 를 쓴다
```
14.147 의 A(`on_detach_gate`)·B(`on_power_praw`)는 (2)(3)을 **참ON 창에서만** 끊었고
(1)(4)는 그대로였다. 14.149 가 잰 결과 — 출력 결손의 **100%가 게이트 항**이었고
`p_raw` 는 기준선에서 이미 정직했다(-0.0005). **끊은 곳이 틀렸던 것이다.**

⚠ **곱을 그냥 뗄 수는 없다.** `power_mix_mask` 가 state 0 을 빼서
`p_raw = Σ_{s>=1} mix_s·p_states_s` 는 **구조적으로 "켜졌다면 얼마"** 이고 0 이 못 된다.
0W 를 낼 수 있는 **유일한 장치가 게이트**다. 그래서:

```
  state 0 을 혼합에 **넣고** 그 전력을 **0** 으로 둔다
  -> `p_raw` 가 0 을 표현할 수 있다 -> OFF 일을 **상태 머리**가 맡는다
  -> `power = p_raw` 로 곱을 뗀다 -> (1)(2)(3)(4)가 한꺼번에 사라진다
  -> 게이트는 `out["on_logit"]` 로만 남아 **검출·채점**에만 쓰인다
  (5)만 남는다 — 머리를 쪼개야 없앨 수 있다
```

무엇을 확인하나
---------------
```
  [1] 끄면 **비트 동일**
  [2] ★ `power == p_raw` — 곱이 떨어졌다
  [3] ★ state 0 이 혼합에 들어간다 (기준선은 정확히 0)
  [4] ★ state 0 의 전력이 **0**
  [5] `Σ mix·p_states == p_raw` 이고 혼합 합이 1 (`L_harm` 의 분해가 깨지지 않는다)
  [6] ★ `power` 의 경사가 **`on_logit` 에 안 닿는다**
  [7] ★ **프라이어도 전력에 못 닿는다** (κ 8 대 0 에서 power 가 동일)
  [8] 파라미터 수 불변   [9] autocast(bfloat16)
```

    python -X utf8 src/run_gate_gatefree.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.net import NILMNet, appliance_state_counts  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
#: ⚠ `prior_kappa` 를 **켠다** — 클래스 기본은 0.0 인데 트레이너 기본이 8.0 이다 (14.138).
BASE = dict(fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4),
            prior_kappa=8.0, p_state_cap=3.0)
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % ("OK" if good else "** 실패 **", name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


def _net(**kw):
    torch.manual_seed(0)
    return NILMNet(APPS, appliance_state_counts(APPS), **dict(BASE, **kw))


def main() -> int:
    print("`--gate-free-power` — 게이트와 전력을 **완전히** 분해한다 (14.150)")
    print("  연결점 여섯 중 (1)값·(2)흡수상태·(3)게이트경사·(4)프라이어를 끊는다.")
    print("  (5) 공유 머리만 남는다 — 머리를 쪼개야 없앨 수 있다.\n")
    f, w = torch.randn(4, 57, 600), torch.randn(4, 47, 120)
    a, b, c = _net(), _net(gate_free_power=True), _net(gate_free_power=False)
    for m in (a, b, c):
        m.eval()
    with torch.no_grad():
        o0, o1, oc = a(f, w), b(f, w), c(f, w)

    ck("[1] 끄면 **비트 동일**",
       all(torch.equal(o0[k], oc[k]) for k in o0 if torch.is_tensor(o0[k])))
    ck("[2] ★ `power == p_raw` — 곱이 떨어졌다", bool(torch.equal(o1["power"], o1["power_raw"])),
       "gate-free 최대차 0 · 기준선 %.1fW" % float((o0["power"] - o0["power_raw"]).abs().max()))
    ck("[3] ★ state 0 이 혼합에 **들어간다**",
       float(o1["power_mix"][..., 0].max()) > 0.01
       and float(o0["power_mix"][..., 0].max()) < 1e-9,
       "mix[0] 최대 %.3f (기준선 %.3g)"
       % (o1["power_mix"][..., 0].max(), o0["power_mix"][..., 0].max()))
    ck("[4] ★ state 0 의 **전력이 0**", float(o1["power_states"][..., 0].abs().max()) == 0.0)
    ck("[5] `Σ mix·p_states == p_raw` · 혼합 합 1 (`L_harm` 분해가 안 깨진다)",
       bool(torch.allclose((o1["power_mix"] * o1["power_states"]).sum(-1),
                           o1["power_raw"], atol=1e-4))
       and bool(torch.allclose(o1["power_mix"].sum(-1),
                               torch.ones_like(o1["power_mix"][..., 0]), atol=1e-5)))

    gv = {}
    for tag, kw in (("기준선", {}), ("gate-free", dict(gate_free_power=True))):
        m = _net(**kw)
        m.train()
        o = m(f, w)
        o["on_logit"].retain_grad()
        o["power"].sum().backward()
        gv[tag] = 0.0 if o["on_logit"].grad is None else float(o["on_logit"].grad.abs().max())
    ck("[6] ★ `power` 의 경사가 **`on_logit` 에 안 닿는다**",
       gv["gate-free"] == 0.0 and gv["기준선"] > 0,
       "기준선 %.4g -> gate-free **%.4g**" % (gv["기준선"], gv["gate-free"]))

    p0 = _net(gate_free_power=True, prior_kappa=0.0)
    p0.eval()
    with torch.no_grad():
        o8 = p0(f, w)
    ck("[7] ★ **프라이어도 전력에 못 닿는다** (κ 8 대 0 에서 power 동일)",
       bool(torch.equal(o8["power"], o1["power"]))
       and not bool(torch.equal(o8["on_logit"], o1["on_logit"])),
       "power 최대차 %.3g · on_logit 최대차 %.3g"
       % (float((o8["power"] - o1["power"]).abs().max()),
          float((o8["on_logit"] - o1["on_logit"]).abs().max())))
    ck("[8] 파라미터 수 불변",
       sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters()))
    err = ""
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
            v = float(b(f, w)["power"].float().abs().max())
        g9 = bool(v == v)
    except Exception as e:                                # noqa: BLE001
        g9, err = False, "%s: %s" % (type(e).__name__, e)
    ck("[9] autocast(bfloat16) 에서 돈다", g9, err or "OK")

    print("")
    print("  ⚠ **예측을 먼저 적는다.** OFF 일이 게이트에서 **상태 머리**로 넘어간다.")
    print("    13.11 이 반대 방향에서 본 병(드라이 s2 가 0W 로 죽음)이 여기서 재발할 수")
    print("    있다 — 이제 OFF 창이 `p_raw` 를 직접 0 으로 민다. **상태별 지문**과")
    print("    **혼자켜짐 전력오차**를 같이 봐야 한다.")
    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
