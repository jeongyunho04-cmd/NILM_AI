# -*- coding: utf-8 -*-
"""`--on-detach-gate` 배선 관문 (14.76).

무엇을 막으려는가
-----------------
`power = sigmoid(on_logit) * p_raw` 라 `L_power` 의 경사가 **게이트로도** 흐른다.
그래서 합이 안 맞으면 **게이트를 눌러 맞추는 것**이 허용돼 있고, A 기기의 수준 오차가
B 기기의 게이트를 죽인다. 실측 (14.75, test_4):

```
  빔프 +7.4W 과대  ->  충전기가 꺼지면 남는 자리 2.5W
  미니PC 는 p_raw 9.96W 로 **맞게** 내는데 게이트가 0.004 로 눌려 10W 가 사라진다 (참 ON).
```

검정하는 불변량
---------------
```
(1) 끄면 **비트 동일** — 두 손잡이 다 끄면 옛 경로와 손실이 한 비트도 안 다르다
(2) 켜면 **켜진 창의 `on_logit` 경사가 정확히 0** (L_power 로부터)
(3) ⚠ 음성 대조 — **꺼진 창의 게이트에는 경사가 남아야** 한다.
    안 그러면 샌 전력을 게이트로 눌러 0 을 만들 수 없다 (`off_detach_praw` 주석의 설계).
(4) `p_raw` 는 켜진 창에서 계속 경사를 받는다 — '얼마인가' 는 그쪽 일이다
```

    python -X utf8 src/run_gate_ondetach.py
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.model.losses import NILMLoss  # noqa: E402

OK, NG = "OK", "NG"
FAIL = []


def ck(name, ok, note=""):
    print("  [" + (OK if ok else NG) + "] " + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


def _mk(K, S, B, seed=0):
    """손실이 forward 에서 만지는 것만 최소로 짓는다."""
    torch.manual_seed(seed)
    on_logit = (torch.randn(B, K) * 2).requires_grad_(True)
    p_raw = (torch.rand(B, K) * 400 + 20).requires_grad_(True)
    out = {
        "on_logit": on_logit,
        "power_raw": p_raw,
        "power": torch.sigmoid(on_logit) * p_raw,
        "plugged_logit": torch.randn(B, K),
        "standby": torch.rand(B, K) * 2,
        "state": torch.randn(B, K, S),
        "power_states": torch.rand(B, K, S) * 400 + 20,
    }
    y_on = (torch.rand(B, K) > 0.5).float()
    tgt = {
        "y_on": y_on,
        "y_power": torch.rand(B, K) * 400 + 20,
        "y_plugged": torch.ones(B, K),
        "y_standby": torch.rand(B, K) * 2,
        "y_state": torch.randint(0, S, (B, K)).float(),
        "p_noise": torch.rand(B) * 5,
        "p_observed": torch.rand(B) * 1000 + 100,
    }
    return out, tgt, on_logit, p_raw, y_on


def main() -> int:
    print("`--on-detach-gate` 배선 관문 (14.76)")
    print()
    K, S, B = 9, 4, 16
    s_i = torch.arange(K) % 3

    def run(on_detach, off_detach, seed=0):
        out, tgt, g, pr, y_on = _mk(K, S, B, seed)
        L = NILMLoss(s_i=s_i, off_detach_praw=off_detach, on_detach_gate=on_detach)
        parts = L(out, tgt)
        parts["power"].backward()
        return float(parts["power"]), g.grad.clone(), pr.grad.clone(), y_on > 0.5

    v0, g0, p0, on = run(False, False)
    v0b, g0b, p0b, _ = run(False, False)
    ck("(1) 끄면 되풀이해도 비트 동일", v0 == v0b and torch.equal(g0, g0b),
       "손실 %.10g" % v0)

    v1, g1, p1, on1 = run(True, False)
    ck("(1) 켜도 **손실 값**은 같다 (경사 경로만 바뀐다)", abs(v1 - v0) < 1e-12,
       "끔 %.10g · 켬 %.10g" % (v0, v1))

    ck("(2) 켜면 **켜진 창의 게이트 경사가 정확히 0**",
       bool((g1[on1].abs().max() == 0)),
       "켜진 칸 %d개 · 최대 |grad| %.3e" % (int(on1.sum()), float(g1[on1].abs().max())))
    ck("(2) 끄면 그 자리에 경사가 **있었다** (이 관문이 뭔가를 재고 있다는 증거)",
       bool(g0[on1].abs().max() > 0),
       "최대 |grad| %.3e" % float(g0[on1].abs().max()))

    ck("(3) 음성 대조 — **꺼진 창의 게이트 경사는 남는다**",
       bool(g1[~on1].abs().max() > 0),
       "꺼진 칸 %d개 · 최대 |grad| %.3e" % (int((~on1).sum()), float(g1[~on1].abs().max())))

    ck("(4) `p_raw` 는 켜진 창에서 경사를 계속 받는다",
       bool(p1[on1].abs().max() > 0),
       "최대 |grad| %.3e" % float(p1[on1].abs().max()))

    v2, g2, p2, on2 = run(True, True)
    ck("(5) `off_detach_praw` 와 같이 켜도 선다 (둘은 서로 다른 축이다)",
       bool(g2[on2].abs().max() == 0) and bool(g2[~on2].abs().max() > 0))

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
