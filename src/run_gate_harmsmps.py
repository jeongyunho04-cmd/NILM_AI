# -*- coding: utf-8 -*-
"""**SMPS 전용 고조파 항**(`--w-harm-smps`)의 관문 — 14.375.

사용자 지적에서 나왔다: *"L_harm 에서 저항성 부하 분리되어 있어? 이미 총량을 알면
거기서 저항성 부하 몫의 고조파 빼는 건 쉬운 일이잖아"*. 맞았다 —
```
  지금  pred = Σ(기기 전부) + 대기 + 잡음 ;  err = |pred − obs|    <- **잔차가 하나**
  ⇒ SMPS 목표를 바꾸면 기울기가 **저항 넷에도** 간다 (14.373 의 C포트오탐 2 -> 12)
  ⇒ 그리고 그 잔차를 **저항이 지배한다** (14.374):
      test_5 관측 |I1| 7,915mA 중 저항 몫 **7,896mA (100%)** · SMPS 77mA (1%)
  새 항  err_smps = | Σ_smps + 대기 + 잡음 − ( obs − **Ĝ·V_h** ) |
  `Ĝ·V_h` 는 **입력만의 함수**라 결합이 원리상 없다. ⚠ `detach` 로는 안 끊긴다 —
  결합은 기울기 식이 아니라 **잔차 값**에 있다.
```

    python -X utf8 -m src.run_gate_harmsmps
"""
import numpy as np
import torch

from src import env_guard  # noqa: F401

from src.model.losses import LossWeights, NILMLoss, S_STATE  # noqa: E402
from src.model.net import MAX_STATES, harmonic_signatures_by_state  # noqa: E402
from src.model.postproc import SMPS_GROUP  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
RES = ("electiric_kettle", "hair_dryer", "hotplate", "oven")
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-46s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def make(w_smps, sig):
    torch.manual_seed(0)
    s_i = torch.tensor([max(S_STATE[a].values()) for a in APPS], dtype=torch.float32)
    return NILMLoss(
        s_i, signatures=torch.from_numpy(sig[:, 1]).float(),
        signatures_state=torch.from_numpy(sig).float(),
        weights=LossWeights(harm=0.1, harm_smps=w_smps),
        harm_smps=w_smps, harm_smps_min_order=3,
        smps_sel=(torch.tensor([1.0 if a in SMPS_GROUP else 0.0 for a in APPS])
                  if w_smps > 0 else None))


def fake(B=6, H=15, g=None):
    """★ **진짜 `NILMNet` 을 지어 돌린다** ([[the-gate-must-build-the-real-object]]).

    가짜 `out` 을 손으로 채우면 손실이 요구하는 키를 계속 놓친다 — 그리고 그렇게 지은
    물건을 재면 진짜 경로가 반쪽이어도 관문이 통과한다.
    """
    from src.model import inputs as _I
    from src.model.net import NILMNet, appliance_state_counts
    torch.manual_seed(1)
    m = NILMNet(APPS, appliance_state_counts(APPS), fine_channels=_I.FINE_CHANNELS,
                fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4), p_state_cap=3.0,
                prior_kappa=8.0, aux_z=True)
    m.train()
    f = torch.randn(B, _I.FINE_CHANNELS, 600) * 0.3
    w = torch.randn(B, _I.WIDE_CHANNELS, 120) * 0.3
    out = m(f, w)
    S = out["power_states"].shape[-1]
    v = torch.zeros(B, H, 2); v[:, 0, 0] = 220.0
    for h in (3, 5, 7, 9, 11, 13, 15):
        v[:, h - 1, 0] = 220.0 * 0.01 / h
    tgt = {"obs_harm": torch.randn(B, H, 2) * 0.05,
           "g_hat": (torch.full((B,), 24.96) if g is None else g),
           "volt_harm": v, "harm_offset": None,
           "y_power": torch.rand(B, len(APPS)) * 60,
           "y_on": (torch.rand(B, len(APPS)) > 0.5).float(),
           "y_plugged": (torch.rand(B, len(APPS)) > 0.3).float(),
           "y_standby": torch.rand(B, len(APPS)) * 3,
           "y_state": torch.randint(0, S, (B, len(APPS))),
           "p_noise": torch.rand(B) * 2, "p_observed": out["power"].sum(1).detach(),
           "vrel": torch.ones(B), "v_rms": torch.full((B,), 220.0),
           "log_z": torch.randn(B), "r_grid": torch.rand(B) + 0.5}
    return m, out, tgt


def main():
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig, _us = harmonic_signatures_by_state(pool, APPS)
    l0, l1 = make(0.0, sig), make(0.2, sig)
    print("SMPS 전용 고조파 항 관문 (14.375) — SMPS %s · 최소차수 h%d"
          % (list(SMPS_GROUP), l1._smps_min_order))

    # [1] 끄면 비트 동일
    _m, o, t = fake()
    a = l0(o, t); b = l1(o, t)
    chk(1, "★ `w_harm_smps=0` 이면 `harm` 항이 **비트 동일**",
        torch.equal(a["harm"], b["harm"]) and float(a["harm_smps"]) == 0.0,
        "harm 최대차 **%.3e** · 끔 판의 harm_smps = %.1f"
        % (float((a["harm"] - b["harm"]).abs().max()), float(a["harm_smps"])))

    # [2] ★★ 새 항에 **저항 파라미터 기울기가 안 간다**
    _m, o, t = fake()
    o["on_logit"].retain_grad()
    b = l1(o, t)
    b["harm_smps"].backward(retain_graph=True)
    gr = o["on_logit"].grad.abs().sum(0)
    ri = [APPS.index(x) for x in RES]
    si = [APPS.index(x) for x in SMPS_GROUP]
    chk(2, "★★ 새 항의 기울기가 **저항 넷에 안 가나**",
        float(gr[ri].max()) < 1e-9 < float(gr[si].min()),
        "저항 넷 |grad| 최대 **%.3e** · SMPS 셋 최소 **%.3e** — `Ĝ·V_h` 에 모델 "
        "파라미터가 없어 저항 예측과 결합이 **원리상 없다**"
        % (float(gr[ri].max()), float(gr[si].min())))

    # [3] ★ `Ĝ` 를 바꾸면 항이 움직인다 (뺄셈이 실제로 걸리나)
    _m, o, t1 = fake(g=torch.full((6,), 0.0))
    _m2, _o2, t2 = fake(g=torch.full((6,), 40.0))
    v1 = float(l1(o, t1)["harm_smps"]); v2 = float(l1(o, t2)["harm_smps"])
    chk(3, "★ `Ĝ` 를 0 -> 40 mS 로 바꾸면 항이 **움직이나**", abs(v2 - v1) > 1e-6,
        "harm_smps **%.5f -> %.5f** (차 %.5f) — 안 움직이면 뺄셈이 안 걸린 것이다"
        % (v1, v2, v2 - v1))

    # [4] ★ h1 이 빠져 있나
    chk(4, "★ **h1 이 빠져 있나** (저항이 지배하는 차수)",
        float(l1.smps_ord[0]) == 0.0 and float(l1.smps_ord[1]) == 0.0
        and float(l1.smps_ord[2]) == 1.0,
        "가면 h1 %.0f · h2 %.0f · h3 %.0f — 저항 h1 이 7,900mA 라 0.3%% 오차가 24mA 고 "
        "신호:오차가 **1.8:1** 이다 (h3 이상은 3.5~5.6:1)"
        % (l1.smps_ord[0], l1.smps_ord[1], l1.smps_ord[2]))

    # [5] `Ĝ` 가 NaN 이면 뺄셈을 건너뛴다 (옛 캐시·모르는 창)
    _m, o, tn = fake(g=torch.full((6,), float("nan")))
    vn = l1(o, tn)["harm_smps"]
    chk(5, "`Ĝ` 가 NaN 이면 **NaN 이 안 새나**", bool(torch.isfinite(vn)),
        "harm_smps = **%.5f** (유한) — 옛 캐시엔 `g_hat` 이 없다" % float(vn))

    # [6] 가중 필드가 손실 합에 들어가나
    _m, o, t = fake()
    r = l1(o, t)
    man = sum(getattr(l1.w, n) * float(v) for n, v in r.items() if n != "total")
    chk(6, "`harm_smps` 가 **손실 합에 들어가나**", abs(man - float(r["total"])) < 1e-4,
        "수동합 %.6f · total %.6f · w.harm_smps = **%.2f**"
        % (man, float(r["total"]), l1.w.harm_smps))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
