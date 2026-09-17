# -*- coding: utf-8 -*-
"""**`--w-harm-margin` 관문** (14.412).

주장: *"`L_harm` 은 절대적합이라 잔차의 95%를 차지하는 순방향 오차를 배분이 흡수한다.
SMPS 쌍 사이로 δ 와트를 옮겼을 때 오차가 **눈에 띄게** 커지도록 요구하면, 평평한
골짜기(§62: 충전기/빔 **22.6°**)에 앉는 것을 벌할 수 있다."*

    python -X utf8 -m src.run_gate_hmargin
"""
from typing import List
import inspect
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.lossbuild import build_loss  # noqa: E402

APPS = ("air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven")
SMPS = ("minipc", "laptop_charger", "beam_projector")
OK: List[bool] = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-52s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def mk(**kw):
    return build_loss(list(APPS), "cpu", harm_even_magnitude=True,
                      state_signatures=True, standby_operating="session", **kw)


def main() -> int:
    l0, l1 = mk(), mk(harm_margin=1.0)
    K, H, B = len(APPS), l1.harm_scale.numel(), 16

    chk(1, "★ 가중 0 이면 쌍 표가 **비어 있다** (비트 동일)",
        l0.mg_i.numel() == 0 and l1.mg_i.numel() > 0,
        "w=0 쌍 %d개 · w=1 쌍 **%d개** — 0 은 곱이 아니라 **분기**라 옛 판과 계속 비교된다"
        % (l0.mg_i.numel(), l1.mg_i.numel()))

    # [2] ★ **SMPS 쌍만** 걸렸나 (저항이 섞이면 14.35 의 재판이다)
    idx = set(l1.mg_i.tolist()) | set(l1.mg_j.tolist())
    want = {APPS.index(x) for x in SMPS}
    chk(2, "★ **SMPS 셋만** 들었나 (저항이 섞이면 안 된다)", idx == want,
        "든 기기 %s · 기대 %s · 방향 포함 %d쌍 (3종이면 3x2=6)"
        % (sorted(APPS[i] for i in idx), sorted(SMPS), l1.mg_i.numel()))

    # [3] ★★ **최적점에서 자동 충족**인가 — 안 그러면 최적해를 옮긴다
    torch.manual_seed(0)
    obs = torch.randn(B, H, 2) * l1.harm_scale[None, :, None] * 0.3
    m_ = l1.harm_mask[None, :, None]
    d_ = l1.harm_mask.mean().clamp(min=1e-6)
    e0 = (l1._harm_err(obs, obs) * m_).mean((1, 2)) / d_          # pred == obs
    gaps, mins = [], []
    for t in range(l1.mg_i.numel()):
        dv = l1.harm_margin_delta * (l1.sig[l1.mg_j[t]] - l1.sig[l1.mg_i[t]])
        e1 = (l1._harm_err(obs + dv[None], obs) * m_).mean((1, 2)) / d_
        gaps.append(float((e1 - e0).mean()))
        mins.append(float(l1.mg_m[t]))
    ok3 = all(g > m for g, m in zip(gaps, mins))
    chk(3, "★★ `pred=obs` 에서 여유가 **자동 충족**되나", ok3,
        "쌍별 달성 간격 %s · 요구 %s — frac<1 이라 최적해를 **안 옮긴다**"
        % ([round(g, 4) for g in gaps], [round(m, 4) for m in mins]))

    # [4] ★ 제일 좁은 쌍이 **충전기/빔** 인가 (§62 가 22.6° 로 잰 그 쌍)
    pair = {}
    for t in range(l1.mg_i.numel()):
        a, b = sorted((APPS[l1.mg_i[t]], APPS[l1.mg_j[t]]))
        pair[(a, b)] = gaps[t]
    narrow = min(pair, key=pair.get)
    chk(4, "★ 제일 좁은 쌍이 **충전기/빔** 인가 (§62 의 22.6°)",
        set(narrow) == {"laptop_charger", "beam_projector"},
        "쌍별 간격 " + " · ".join("%s/%s %.4f" % (a[:5], b[:5], v)
                                 for (a, b), v in sorted(pair.items(), key=lambda x: x[1])))

    # [5] ★★ 기울기가 **망까지** 가나 (사전은 버퍼라 경로가 끊길 수 있다)
    out = {"power_raw": torch.rand(B, K) * 50 + 1, "on_logit": torch.randn(B, K),
           "plugged_logit": torch.randn(B, K), "standby": torch.rand(B, K),
           "power_mix": torch.softmax(torch.randn(B, K, 5), -1),
           "power_states": torch.rand(B, K, 5) * 50 + 1, "state": torch.randn(B, K, 5)}
    out["power"] = torch.sigmoid(out["on_logit"]) * out["power_raw"]
    for v in out.values():
        v.requires_grad_(True)
    tgt = {"obs_harm": torch.randn(B, H, 2) * 0.05, "y_power": torch.rand(B, K) * 50,
           "y_on": (torch.rand(B, K) > 0.5).float(),
           "y_plugged": (torch.rand(B, K) > 0.5).float(),
           "y_standby": torch.rand(B, K), "y_state": torch.randint(0, 5, (B, K)),
           "p_observed": torch.rand(B) * 500, "p_noise": torch.full((B,), 1.4),
           "v_rms": torch.full((B,), 220.0)}
    parts = l1(out, tgt)
    parts["hmargin"].backward()
    gp = float(out["power_raw"].grad.abs().sum())
    gg = float(out["on_logit"].grad.abs().sum())
    chk(5, "★★ 기울기가 `power_raw`·`on_logit` 까지 닿나", gp > 0 and gg > 0,
        "hmargin %.5f · power_raw |g|합 **%.3e** · on_logit |g|합 **%.3e** — "
        "`sig` 는 학습 안 되는 버퍼라 여기가 0 이면 항이 **아무 일도 안 한다**"
        % (float(parts["hmargin"]), gp, gg))

    # [6] 배선이 양쪽 학습기에 닿나
    import src.run_adapt as A
    import src.run_train_cnn as T
    sg = inspect.signature(build_loss).parameters
    s1 = "harm_margin=a.w_harm_margin" in inspect.getsource(T)
    s2 = "harm_margin=a.w_harm_margin" in inspect.getsource(A)
    ck = '"w_harm_margin": float(a.w_harm_margin)' in inspect.getsource(T)
    chk(6, "⚠ 1단계·2단계·체크포인트 셋 다 닿나",
        "harm_margin" in sg and s1 and s2 and ck,
        "`build_loss` %s · 1단계 %s · 2단계 %s · 체크포인트 %s"
        % ("harm_margin" in sg, s1, s2, ck))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
