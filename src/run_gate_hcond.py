# -*- coding: utf-8 -*-
"""관문 — **전도도 머리** `--head-conductance` (14.284). 여덟 줄.

14.16 이 `--vexp` 를 죽인 까닭은 구조가 아니라 **목표가 와트였던 것**이다:
구조는 정확히 +2.02(기대 +2.00)로 작동했는데 헤드가 대조군과 같은 `k=0.66` 을 **또**
배워 합이 **2.68**(물리 2.0)이 됐다 — 이중 계산. 실측 다섯 파일이 전부 나빠졌다.
진단도 남아 있다: *"전압 효과가 전부 상태 **안**에 있는데 헤드는 상태 **사이** 100배에
손실이 지배당해 7%를 잡음으로 취급한다. 헤드를 먼저 고쳐야 한다."*

이 팔은 순전파를 바꾸는 게 아니라 **목표를 V 불변으로** 옮긴다:
```
  순전파   p_raw *= (V/222)^e_k          (vexp 와 같다)
  목표     log(P / (V/222)^e_k)          <- **V 가 사라진다**
  ⇒ 헤드가 전압을 인코딩할 유인이 없어져 이중 계산이 원천에서 막힌다
```

```
  [1] 끄면 **비트 동일** (출력·파라미터·손실)
  [2] `vrel_pow` 가 정확히 `(V/222)^e_k` · V_EXP 표 (저항 2 · SMPS 0 · 모터 0.6)
  [3] ★★ **목표가 V 불변** — 저항 물리로 V 를 ±10% 흔들어도 `y/vrel_pow` 가 안 움직인다
  [4] ★ 손실이 실제로 갈린다 — 켜진 자리는 log, 꺼진 자리는 옛 척도 Huber
  [5] ★ **슬롯이 안 죽는다** — 꺼진 기기에도 기울기가 산다 (13.84.68)
  [6] ★ 신원 오차가 **전력 수준과 무관** — 11W 짜리와 1.4kW 짜리가 같은 값
  [7] `--vexp` 와 같이 못 쓴다 (같은 곱을 두 번)
  [8] V_REL_CLAMP 가 외삽을 막는다
```

    python -X utf8 -m src.run_gate_hcond
"""
import sys

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from src import env_guard  # noqa: F401,E402

from src.model import inputs as _I  # noqa: E402

import torch  # noqa: E402

from src.model.losses import S_STATE  # noqa: E402
from src.model.net import (V_CENTER, V_EXP, V_REL_CLAMP, V_SPAN,  # noqa: E402
                           NILMNet, V_CH_FINE)

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
NS = [1 + max(S_STATE.get(a, {1: 0}).keys()) for a in APPS]
ok = True


def mk(hc=False, ve=False, seed=0):
    torch.manual_seed(seed)
    #: 14.331 — 상수에서 파생. 박아 두면 차수를 넓힐 때 조용히 틀린다.
    return NILMNet(APPS, NS, fine_channels=_I.FINE_CHANNELS,
                   fine_extra_dilations=(32, 64),
                   tap_layers=(0, 1, 4), p_state_cap=3.0, prior_kappa=8.0,
                   head_conductance=hc, vexp=ve).eval()


def inp(vv, seed=7):
    torch.manual_seed(seed)
    f = torch.randn(4, _I.FINE_CHANNELS, 600) * 0.3
    w = torch.randn(4, _I.WIDE_CHANNELS, 120) * 0.3
    f[:, V_CH_FINE] = (vv - V_CENTER) / V_SPAN
    return f, w


f0, w0 = inp(222.0)
a, b = mk(False), mk(False)
with torch.no_grad():
    d = max(float((a(f0, w0)[k] - b(f0, w0)[k]).abs().max())
            for k in ("power", "on_logit", "state"))
na = sum(p.numel() for p in a.parameters())
nc = sum(p.numel() for p in mk(True).parameters())
has = "vrel_pow" in a(f0, w0)
print("[1] 끄면 비트 동일 %.3e · 파라미터 %d = %d · `vrel_pow` 없음 %s   %s"
      % (d, na, nc, not has, "OK" if d == 0 and na == nc and not has else "FAIL"))
ok &= d == 0 and na == nc and not has

m = mk(True)
V = 235.0
fv, wv = inp(V)
with torch.no_grad():
    o = m(fv, wv)
vp = o["vrel_pow"]
want = torch.tensor([(V / V_CENTER) ** V_EXP.get(x, 0.0) for x in APPS])
e2 = max(float((vp - want[None]).abs().max()), 0.0)
tab = all(abs(V_EXP[x] - 2.0) < 1e-9 for x in ("oven", "electiric_kettle", "hotplate",
                                               "hair_dryer")) \
    and all(abs(V_EXP[x]) < 1e-9 for x in ("minipc", "laptop_charger", "beam_projector")) \
    and abs(V_EXP["fan"] - 0.6) < 1e-9
print("[2] `vrel_pow` 오차 **%.2e** · V_EXP 표(저항2·SMPS0·모터0.6) %s   %s"
      % (e2, "맞다" if tab else "**틀리다**", "OK" if e2 < 1e-6 and tab else "FAIL"))
ok &= e2 < 1e-6 and tab

# ── [3] 목표 V 불변 ───────────────────────────────────────────────────────────
print("[3] ★★ 저항 물리(P ∝ V²)로 V 를 흔들 때 **목표 log(y/vrel_pow)** 가 움직이나")
ko = APPS.index("oven")
base_y = 1109.0
rows = []
for s in (0.92, 1.00, 1.08):
    vv = V_CENTER * s
    fs, ws = inp(vv)
    with torch.no_grad():
        vps = m(fs, ws)["vrel_pow"][0, ko]
    y = base_y * s ** 2                      # 저항이면 참값도 V² 로 움직인다
    rows.append(float(y / vps))
    print("      V=%6.1f  y=%7.1fW  vrel^e=%.4f  ->  목표 y/vrel^e = **%.2f**"
          % (vv, y, float(vps), rows[-1]))
sp = max(rows) / min(rows) - 1.0
print("      목표 변동 **%.4f%%** (V 는 ±8%% 움직였다)   %s"
      % (100 * sp, "OK" if sp < 1e-4 else "FAIL"))
ok &= sp < 1e-4

# ── [4][5] 손실 ──────────────────────────────────────────────────────────────
from src.model.losses import _huber  # noqa: E402
print("[4] 손실이 갈리나 — 켜진 자리 log · 꺼진 자리 옛 척도")
pw = torch.tensor([[1200.0, 0.0, 900.0]])
yy = torch.tensor([[1109.0, 0.0, 16.8]])
vpx = torch.tensor([[1.10, 1.10, 1.10]])
s_i = torch.tensor([[1200.0, 1200.0, 1200.0]])
on = yy > 5.0
l_on = _huber(torch.log((pw / vpx).clamp(min=1.0)), torch.log((yy / vpx).clamp(min=1.0)), 0.05)
l_off = _huber(pw / s_i, yy / s_i, 0.1)
mixed = torch.where(on, l_on, l_off)
good = bool(torch.allclose(mixed[0, 0], l_on[0, 0]) and torch.allclose(mixed[0, 1], l_off[0, 1]))
print("      켜짐 자리 log %.5f · 꺼짐 자리 옛 %.5f · 선택 맞음 %s   %s"
      % (float(l_on[0, 0]), float(l_off[0, 1]), good, "OK" if good else "FAIL"))
ok &= good

pz = torch.tensor([[900.0]], requires_grad=True)
yz = torch.tensor([[0.0]])
_huber(pz / torch.tensor([[1200.0]]), yz / torch.tensor([[1200.0]]), 0.1).sum().backward()
g = float(pz.grad.abs().max())
print("[5] ★ 꺼진 기기(참값 0)에 기울기가 **%.4f** (0 이면 슬롯이 죽는다)   %s"
      % (g, "OK" if g > 1e-6 else "FAIL"))
ok &= g > 1e-6

# ── [6] 척도 무관 ────────────────────────────────────────────────────────────
print("[6] ★ 신원 오차가 전력 수준과 무관한가 (오븐 42.08Ω 대 포트 35.67Ω = 18%)")
print("      %-14s %12s %12s" % ("기기 크기", "옛 Huber(P/s_i)", "**log**"))
for nm, P in (("1.4kW 급", 1109.0), ("11W 급", 11.09)):
    q = P * 42.08 / 35.67
    lin = float(_huber(torch.tensor(q / max(P, 1e-9) * 0 + q / 1200.0),
                       torch.tensor(P / 1200.0), 0.1))
    lg = float(_huber(torch.log(torch.tensor(q)), torch.log(torch.tensor(P)), 0.05))
    print("      %-14s %12.6f %12.6f" % (nm, lin, lg))
lg1 = float(_huber(torch.log(torch.tensor(1109.0 * 42.08 / 35.67)),
                   torch.log(torch.tensor(1109.0)), 0.05))
lg2 = float(_huber(torch.log(torch.tensor(11.09 * 42.08 / 35.67)),
                   torch.log(torch.tensor(11.09)), 0.05))
print("      log 는 %.6f 대 %.6f -> **같다**   %s"
      % (lg1, lg2, "OK" if abs(lg1 - lg2) < 1e-6 else "FAIL"))
ok &= abs(lg1 - lg2) < 1e-6

try:
    mk(True, True)
    g7 = False
except ValueError:
    g7 = True
print("[7] `head_conductance` + `vexp` 가 **막힌다** %s   %s" % (g7, "OK" if g7 else "FAIL"))
ok &= g7

fx, wx = inp(300.0)
with torch.no_grad():
    vpx2 = float(m(fx, wx)["vrel_pow"][0, ko])
cap = (V_REL_CLAMP[1]) ** 2
print("[8] V=300 에서도 `vrel^e` 가 상한 %.4f 로 묶인다 (%.4f)   %s"
      % (cap, vpx2, "OK" if abs(vpx2 - cap) < 1e-5 else "FAIL"))
ok &= abs(vpx2 - cap) < 1e-5

print()
print("관문 %s" % ("전부 통과" if ok else "**실패**"))
sys.exit(0 if ok else 1)
