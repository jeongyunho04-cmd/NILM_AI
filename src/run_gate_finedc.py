# -*- coding: utf-8 -*-
"""관문 — **시간축 DC 분리** `--fine-dc split` (14.224). 여섯 줄.

14.220~221 이 잰 것: 첫 conv 의 응답이 같은 L2 섭동에서 **DC 가 AC 의 776배**다
(15판 중앙, 범위 63~938). 커널 시간합 |Σw|/|w| = 4.33 으로 무작위 기대(2.12)의 2배라
DC 를 막기는커녕 더 통과시킨다. 실측 실패 창의 변위는 창 전체에 고르고(8~15%/칸)
맞히는 창은 타깃에 78% 몰려 있어 **실패만 이 통로로 직통**한다.

```
  [1] `keep` 이면 **비트 동일** (출력·파라미터 수)
  [2] `split` 은 파라미터가 **정확히 n_dc x 256** 만 는다 (trunk.0 한 줄)
  [3] conv 스택이 받는 입력의 채널별 시간평균이 **정확히 0**
  [4] ★ 첫 conv 의 **DC/AC 이득비가 ~1 로 떨어진다** (지금 776배)
  [5] 정보를 **안 버린다** — 머리 특징 맨 뒤 n_dc 칸이 그 평균과 일치
  [6] 폭이 달라지므로 `load_state_dict` 가 **조용히 통과하면 안 된다**
```

    python -X utf8 -m src.run_gate_finedc
"""
import sys

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.losses import S_STATE  # noqa: E402
from src.model.net import NILMNet  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
NS = [1 + max(S_STATE.get(a, {1: 0}).keys()) for a in APPS]


def mk(dc, seed=0):
    torch.manual_seed(seed)
    return NILMNet(APPS, NS, fine_channels=57, fine_extra_dilations=(32, 64),
                   tap_layers=(0, 1, 4), p_state_cap=3.0, prior_kappa=8.0,
                   fine_dc=dc).eval()


def main() -> int:
    ok = True
    torch.manual_seed(7)
    f = torch.randn(4, 57, 600) * 0.3
    w = torch.randn(4, 47, 120) * 0.3

    a, b = mk("keep"), mk("keep")
    with torch.no_grad():
        d = max(float((a(f, w)[k] - b(f, w)[k]).abs().max())
                for k in ("on_logit", "power", "state"))
    n_a = sum(p.numel() for p in a.parameters())
    print("[1] `keep` 두 판이 **비트 동일** %.3e · 파라미터 %d  %s"
          % (d, n_a, "OK" if d == 0.0 else "FAIL"))
    ok &= (d == 0.0)

    m = mk("split")
    n_m = sum(p.numel() for p in m.parameters())
    want = m.n_dc * 256
    print("[2] `split` 파라미터 %d (+%d) · 기대 n_dc(%d) x 256 = %d  %s"
          % (n_m, n_m - n_a, m.n_dc, want, "OK" if n_m - n_a == want else "FAIL"))
    ok &= (n_m - n_a == want)

    cap = {}
    h = m.fine[0].register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("v", inp[0].detach()))
    with torch.no_grad():
        m(f, w)
    h.remove()
    mx = float(cap["v"].mean(-1).abs().max())
    print("[3] conv 입력의 채널별 시간평균 |max| **%.3e**  %s"
          % (mx, "OK" if mx < 1e-5 else "FAIL"))
    ok &= (mx < 1e-5)

    def gain(model):
        base = {}
        hh = model.fine[0].register_forward_hook(
            lambda mo, i_, o_: base.__setitem__("v", o_.detach()))

        def act(x):
            with torch.no_grad():
                model(x, w[:1])
            return base["v"]
        f0 = f[:1]
        z = act(f0).clone()
        rng = np.random.default_rng(0)
        rs = []
        for _ in range(5):
            v = torch.from_numpy(rng.normal(0, 1, (1, 57, 1)).astype(np.float32))
            dc = (v / v.norm()) * float(np.sqrt(600.0))
            g = torch.from_numpy(rng.normal(0, 1, (1, 57, 600)).astype(np.float32))
            g = g - g.mean(-1, keepdim=True)
            ac = g / g.norm()
            e = 0.3
            rs.append(((act(f0 + e * dc.expand(-1, -1, 600)) - z).norm().item(),
                       (act(f0 + e * ac) - z).norm().item()))
        hh.remove()
        return (float(np.median([r[0] for r in rs])),
                float(np.median([r[1] for r in rs])))

    d0, a0 = gain(a)
    d1, a1 = gain(m)
    r0, r1 = d0 / max(a0, 1e-12), d1 / max(a1, 1e-12)
    print("[4] ★ 첫 conv DC/AC 이득비  keep **%.0f배** -> split **%.2f배**  %s"
          % (r0, r1, "OK" if (r0 > 50 and r1 < 2.0) else "FAIL"))
    print("       (keep DC %.2f AC %.4f · split DC %.4f AC %.4f)" % (d0, a0, d1, a1))
    ok &= (r0 > 50 and r1 < 2.0)

    trunk_in = {}
    h2 = m.trunk.register_forward_pre_hook(
        lambda mod, inp: trunk_in.__setitem__("v", inp[0].detach()))
    with torch.no_grad():
        m(f, w)
    h2.remove()
    tail = trunk_in["v"][:, -m.n_dc:]
    with torch.no_grad():
        want_dc = m._conv_in(f).mean(-1)
    d5 = float((tail - want_dc).abs().max())
    print("[5] 머리 뒤 %d칸 = 떼어 낸 DC (정보 보존) |max차| **%.3e**  %s"
          % (m.n_dc, d5, "OK" if d5 < 1e-5 else "FAIL"))
    ok &= (d5 < 1e-5)

    try:
        m.load_state_dict(a.state_dict())
        bad = True
    except Exception:
        bad = False
    print("[6] `keep` 가중치를 `split` 판에 **못 싣는다** (폭이 다르다)  %s"
          % ("FAIL — 조용히 통과했다" if bad else "OK"))
    ok &= (not bad)

    print("\n%s" % ("전부 통과" if ok else "**실패한 줄이 있다**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
