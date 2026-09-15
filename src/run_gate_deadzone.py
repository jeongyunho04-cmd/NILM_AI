# -*- coding: utf-8 -*-
"""관문 — **죽은구역 보존 손실** (14.162). 여섯 줄.

```
  [1] 끄면(δ<0) `parts["cons"]` 가 옛 절대 와트 식과 **비트 동일**
  [2] `--w-cons 0` 이면 δ 와 무관하게 항이 **안 걸린다** (옛 경로 비트 동일)
  [3] 죽은구역 안(|r| <= δ)에서는 손실이 **정확히 0**, 기울기도 **정확히 0**
  [4] 죽은구역 밖에서는 `relu(|r|−δ)/max(P관측,10)` 과 **값이 일치**
  [5] **양방향** — 과잉(+)과 과소(−)가 같은 크기면 손실도 같다
  [6] 실제 모델·실제 손실로 지어서 확인한다 (따로 지은 물건이 아니다)
```

    python -X utf8 src/run_gate_deadzone.py
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.losses import LossWeights, NILMLoss  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
K, H = len(APPS), 15


def build(w_cons, dz):
    torch.manual_seed(0)
    return NILMLoss(
        s_i=torch.full((K,), 1000.0),
        signatures=torch.zeros(K, H, 2),
        standby_sig=torch.zeros(K, H, 2),
        noise_sig=torch.zeros(H, 2),
        harm_scale=torch.ones(H),
        weights=LossWeights(harm=0.0, cons=w_cons, over=0.0, z=0.0),
        cons_deadzone=dz,
    )


def sample(b=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {"power": torch.rand(b, K, generator=g) * 400,
           "standby": torch.rand(b, K, generator=g) * 2,
           "on_logit": torch.randn(b, K, generator=g),
           "state": torch.randn(b, K, 5, generator=g),
           "plugged_logit": torch.randn(b, K, generator=g),
           "power_raw": torch.rand(b, K, generator=g) * 400,
           "power_states": torch.rand(b, K, 5, generator=g) * 400,
           "power_mix": torch.rand(b, K, 5, generator=g)}
    tgt = {"y_power": torch.rand(b, K, generator=g) * 400,
           "y_on": (torch.rand(b, K, generator=g) > 0.5).float(),
           "y_plugged": torch.ones(b, K),
           "y_standby": torch.zeros(b, K),
           "y_state": torch.zeros(b, K, dtype=torch.long),
           "p_noise": torch.full((b,), 5.0),
           "p_observed": torch.rand(b, generator=g) * 3000 + 100}
    return out, tgt


def main() -> int:
    ok = True
    o, t = sample()

    def cons(w_cons, dz):
        return float(build(w_cons, dz)(dict(o), dict(t))["cons"])

    # [1] 끄면 비트 동일
    recon = o["power"].sum(1) + o["standby"].sum(1) + t["p_noise"]
    r = recon - t["p_observed"]
    old = float(r.abs().mean())
    v = cons(0.4, -1.0)
    print("[1] δ<0 이면 옛 절대 와트 식과 비트 동일  %.10f 대 %.10f  %s"
          % (v, old, "OK" if v == old else "FAIL"))
    ok &= (v == old)

    # [2] w_cons 0 이면 항이 안 걸린다
    v0a, v0b = cons(0.0, -1.0), cons(0.0, 100.0)
    print("[2] w_cons=0 이면 δ 와 무관하게 0     %.10f · %.10f  %s"
          % (v0a, v0b, "OK" if v0a == 0.0 == v0b else "FAIL"))
    ok &= (v0a == 0.0 and v0b == 0.0)

    # [3] 죽은구역 안이면 손실·기울기가 정확히 0
    o2, t2 = sample(seed=1)
    o2["power"] = o2["power"] * 0.0
    o2["standby"] = o2["standby"] * 0.0
    t2["p_observed"] = torch.full((len(t2["p_noise"]),), 1000.0)
    t2["p_noise"] = torch.full((len(t2["p_noise"]),), 950.0)   # r = −50W
    o2["power"].requires_grad_(True)
    c = build(0.4, 100.0)(dict(o2), dict(t2))["cons"]
    gr = torch.autograd.grad(c, o2["power"], allow_unused=True)[0]
    gmax = 0.0 if gr is None else float(gr.abs().max())
    print("[3] |r|=50W < δ=100W 이면 손실 %.10f · 기울기 최대 %.3e  %s"
          % (float(c), gmax, "OK" if float(c) == 0.0 and gmax == 0.0 else "FAIL"))
    ok &= (float(c) == 0.0 and gmax == 0.0)

    # [4] 밖에서는 식이 일치
    o3, t3 = sample(seed=2)
    c4 = build(0.4, 100.0)(dict(o3), dict(t3))["cons"]
    r3 = (o3["power"].sum(1) + o3["standby"].sum(1) + t3["p_noise"]) - t3["p_observed"]
    want = float((torch.relu(r3.abs() - 100.0)
                  / t3["p_observed"].clamp(min=10.0)).mean())
    print("[4] 죽은구역 밖 식 일치            %.10f 대 %.10f  %s"
          % (float(c4), want, "OK" if abs(float(c4) - want) < 1e-12 else "FAIL"))
    ok &= abs(float(c4) - want) < 1e-12

    # [5] 양방향 — 부호만 다른 두 창의 손실이 같다
    vals = []
    for sgn in (+1.0, -1.0):
        o5, t5 = sample(seed=3)
        o5["power"] = o5["power"] * 0.0
        o5["standby"] = o5["standby"] * 0.0
        t5["p_observed"] = torch.full((len(t5["p_noise"]),), 2000.0)
        t5["p_noise"] = torch.full((len(t5["p_noise"]),), 2000.0 + sgn * 270.0)
        vals.append(float(build(0.4, 100.0)(dict(o5), dict(t5))["cons"]))
    print("[5] 양방향 (+270W 대 −270W)        %.10f 대 %.10f  %s"
          % (vals[0], vals[1], "OK" if abs(vals[0] - vals[1]) < 1e-12 else "FAIL"))
    ok &= abs(vals[0] - vals[1]) < 1e-12

    # [6] 실제 모델을 지어서 — `out` 을 따로 안 만든다
    from src.model.net import NILMNet, appliance_state_counts
    from src.model.inputs import FINE_CHANNELS, WIDE_CHANNELS
    net = NILMNet(APPS, appliance_state_counts(APPS), prior_kappa=8.0)
    net.eval()
    g = torch.Generator().manual_seed(7)
    f = torch.randn(8, FINE_CHANNELS, 600, generator=g)
    w = torch.randn(8, WIDE_CHANNELS, 120, generator=g)
    oo = net(f, w)
    _, tt = sample(b=8, seed=4)
    a = float(build(0.4, -1.0)(dict(oo), dict(tt))["cons"])
    b = float(build(0.4, 100.0)(dict(oo), dict(tt))["cons"])
    print("[6] 진짜 모델 출력으로  옛 식 %.4f · 죽은구역 %.6f  %s"
          % (a, b, "OK" if b <= a else "FAIL"))
    ok &= (b <= a)

    print("\n%s" % ("전부 통과" if ok else "**실패한 줄이 있다**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
