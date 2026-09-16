# -*- coding: utf-8 -*-
"""**선로 임피던스 주입**(`--z-input`)의 관문 — 14.354.

```
  zf    = [ (log r − Z_LOG_MEAN)/Z_LOG_STD , known ]     <- 모르면 [0, 0]
  z_pre = trunk(...)                       <- ★ 보조머리 `log_z` 는 **여기**를 읽는다
  z     = z_pre + z_proj(zf)               <- 전력·게이트 머리만 Z 를 본다
```
왜 갈라 놓나 (14.354): 가린 창에서만 `log_z` 를 감독하면 몸통 세금이 절반이 되는데,
그 세금의 이득 기전이 **미확인**이다 (§7.6 — log_z 는 Z 가 아니라 '창에 계단이 있나'
에 +0.622 로 반응한다). 줄이면 충전기·미니PC AUC 이득이 같이 날아갈 수 있다.

    python -X utf8 -m src.run_gate_zin
"""
import numpy as np
import torch

from src import env_guard  # noqa: F401

from src.model import inputs as _I  # noqa: E402
from src.model import net as _NET  # noqa: E402
from src.model.net import NILMNet, appliance_state_counts  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-46s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def build(**kw):
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
    r = torch.tensor([1.15, 0.42, 1.19, 0.30, 2.00, 0.90] + [float("nan")] * (B - 6))
    off, on = build(), build(z_input=True)
    on.load_state_dict(off.state_dict(), strict=False)
    with torch.no_grad():
        a, b = off(f, w), on(f, w, None, r)
    print("선로 임피던스 주입 관문 (14.354) — 눈금 평균 %.4f · 표준편차 %.4f"
          % (_NET.Z_LOG_MEAN, _NET.Z_LOG_STD))

    # [1] ★ 0 초기화라 **출발이 비트 동일**
    chk(1, "★ `z_proj` 0 초기화 -> **출발이 바닥과 비트 동일**",
        all(torch.equal(a[k], b[k]) for k in ("power", "on_logit", "state", "log_z")),
        "power·on_logit·state·log_z 가 한 비트도 안 다르다 — 이득이 나오면 그건 "
        "**배운 것**이지 초기화가 준 것이 아니다")

    # [2] 파라미터가 딱 Linear(2,h) 뿐
    d = (sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters()))
    h = on.z_proj.weight.shape[0]
    chk(2, "파라미터가 **Linear(2,%d) 뿐인가**" % h, d == 2 * h + h,
        "차 **%d** = 가중치 %d + 편향 %d (바닥의 %.3f%%)"
        % (d, 2 * h, h, 100.0 * d / sum(p.numel() for p in off.parameters())))

    # [3] ★ `z_in` 없이 부르면 멈춘다
    try:
        on(f, w)
        ok3, m3 = False, "**안 멈췄다** — 조용히 '모름' 으로 돌면 배치 조건을 못 잰다"
    except ValueError as e:
        ok3, m3 = True, "`%s`" % str(e)[:70]
    chk(3, "★ `z_in` 없이 부르면 **멈추나**", ok3, m3)

    # [4] ★★ 보조머리는 **주입 전**을 읽는다 — Z 를 바꿔도 log_z 가 안 움직인다
    with torch.no_grad():
        on.z_proj.weight.normal_(0, 0.5); on.z_proj.bias.normal_(0, 0.5)
        o1 = on(f, w, None, torch.full((B,), 0.30))
        o2 = on(f, w, None, torch.full((B,), 2.00))
    dz = float((o1["log_z"] - o2["log_z"]).abs().max())
    dp = float((o1["power"] - o2["power"]).abs().max())
    chk(4, "★★ `log_z` 는 **주입 전**을 읽나 (세금이 안 시시해지나)", dz == 0.0 and dp > 0,
        "Z 를 0.30 -> 2.00 으로 바꿔도 log_z 최대차 **%.3e** 인데 전력은 **%.2fW** 움직인다 "
        "— 몸통은 여전히 스스로 Z 를 담아야 하고, 머리만 준 값을 쓴다" % (dz, dp))

    # [5] ★ "모름" 이 "평균값" 과 다르다
    with torch.no_grad():
        o_un = on(f, w, None, torch.full((B,), float("nan")))
        o_av = on(f, w, None, torch.full((B,), float(np.exp(_NET.Z_LOG_MEAN))))
    d5 = float((o_un["power"] - o_av["power"]).abs().max())
    chk(5, "★ **모름**과 **평균 Z**가 다른 값인가", d5 > 1e-4,
        "모름(NaN) 대 r=%.3fΩ(정규화 0) 전력 최대차 **%.2fW** — `known` 칸이 없으면 "
        "둘이 뭉개져 '모른다' 를 표현할 방법이 사라진다" % (float(np.exp(_NET.Z_LOG_MEAN)), d5))

    # [6] 정규화가 실측 자리를 분포 한가운데 놓나
    vals = {"E1 0.42": 0.42, "D1 1.15": 1.15, "D2 1.19": 1.19}
    zs = {k: (np.log(v) - _NET.Z_LOG_MEAN) / _NET.Z_LOG_STD for k, v in vals.items()}
    chk(6, "실측 세 자리가 **분포 안**인가 (|ẑ| < 3)", all(abs(v) < 3.0 for v in zs.values()),
        " · ".join("%s -> **%+.3f**" % (k, v) for k, v in zs.items())
        + "  (학습 분포 −3.33~+1.80)")

    # [7] 기울기가 z_proj 에 흐른다
    on.train()
    on(f, w, None, r)["power"].sum().backward()
    g7 = on.z_proj.weight.grad
    chk(7, "기울기가 **`z_proj` 에 흐르나** (죽은 파라미터가 아니다)",
        g7 is not None and float(g7.abs().sum()) > 0,
        "‖grad‖ **%.4f**" % float(g7.abs().sum() if g7 is not None else 0.0))
    on.eval()

    # [8] ★ NaN 이 전력으로 새지 않는다
    with torch.no_grad():
        o8 = on(f, w, None, torch.tensor([float("nan")] * B))
    chk(8, "★ 모르는 창의 출력에 **NaN 이 없다**",
        bool(torch.isfinite(o8["power"]).all() and torch.isfinite(o8["on_logit"]).all()),
        "power·on_logit 전부 유한 — `torch.where` 로 log(NaN) 을 먼저 막는다")

    # [9] x_grid 를 안 쓴다 (배치에 없는 입력을 학습에 넣지 않는다)
    chk(9, "⚠ 입력이 **r 뿐**인가 (x_grid 미사용)", on.z_proj.weight.shape[1] == 2,
        "z_proj 입력 %d칸 = [ẑ, known] — 실측 `SITE_SESSIONS` 에 R 뿐이고 X 는 "
        "미측정이라(§13.20) 넣으면 배치에 없는 입력이 된다" % on.z_proj.weight.shape[1])

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
