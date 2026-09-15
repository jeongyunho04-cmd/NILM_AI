# -*- coding: utf-8 -*-
"""관문 — `L_gcond` 가 **게이트를 통전에 묶는가** (14.106).

무엇을 고치려는 것인가
----------------------
학습 자료(`train60_v32h`, 20만 창)의 상태 라벨을 세면 **오븐만** 다르다:
```
  오븐   ON 62897   state 1 **35.3%** 중앙 **15.0W**  |  state 2 64.7% 중앙 1125W
  핫플   ON 43417   state 2 100.0% 중앙 455W          <- state 1 이 아예 없다
  포트   ON 45439   state 1 100.0% 중앙 1272W
  드라이  ON 46228   state 1 43.6% 442W / state 2 56.4% 863W   <- 둘 다 크다
```
`power = σ(on_logit)·p_raw` 이므로 **게이트가 서면 전력 통로가 열린다.** 오븐은
ON 라벨의 3분의 1이 전력 15W 라, `L_on` 이 "증거 없이 게이트를 세라" 고 가르친다.
실측에서 오븐 헛게이트가 **10.1%** 로 저항 4종 중 압도적이고(포트 0.2 · 핫플 0.3 ·
드라이 1.1), 그 **75.7%가 전력 없는 팬·조명**이다 (14.105).

`L_gcond` 는 `res_cond_state > 0` 인 기기의 `on_logit` 을 **통전 라벨**
`1{y_state == 통전상태}` 에 맞추는 BCE 다. `L_on` 과 팬·조명 창에서 반대 방향이라
그 창의 최적 게이트가 `w_on / (w_on + w_gcond)` 로 내려간다.

⚠ 이것은 **12.162.4 가 미뤄 둔 고침**이다. 그때 결론이 *"NILM 의 목적이 전력
   분해이므로 통전이 채점 기준이어야 하지만, 그러면 라벨 정의를 바꾸는 것이고
   `build_on_off_truth` 에 손대야 한다. **안 했다**"* 였다. 여기서는 **라벨이 아니라
   손실에서** 한다 — 채점기를 안 건드리니 옛 판과 비교가 선다.

무엇을 확인하나
---------------
```
  [1] 0 이면 **키조차 안 생긴다** — `parts` 가 옛 경로와 같고 `total` 이 비트 동일
  [2] 켜면 항이 실제로 걸린다 (`gate_cond` 가 0 이 아니고 `total` 이 달라진다)
  [3] 목표가 정확히 `1{y_state == res_cond_state}` — 손으로 계산한 BCE 와 일치
  [4] `res_cond_state == 0` 인 기기에는 기울기가 **정확히 0** (가면이 실재한다)
  [5] ★ 해석적 최적 — 팬·조명 창의 최적 게이트가 `w_on/(w_on+W)` 다.
        0.3/0.9/2.7 세 값을 실제로 최적화해서 확인한다 (눈금이 문서와 맞는가)
  [6] ★ **핫플에는 아무 일도 안 일어난다** — `y_state` 가 늘 통전이면 목표가
        `y_on` 과 같아 `L_on` 과 안 싸운다. 손잡이가 구조적으로 오븐 전용인 근거다
  [7] **autocast(bfloat16)** 에서 돈다 — 982877 이 여기서 죽었다. 관문이 fp32 로만
        돌면 못 잡는다 ([[verify-the-gate-runs-that-path]])
  [8] 음성 대조 — 가면을 일부러 없앤 본을 잡아낸다 (통과가 증거가 되려면)
  [9] **배선** — `run_train_cnn` 이 `--w-gate-cond` 를 `LossWeights.gate_cond` 로
        넘기는가. 이 관문은 손실을 따로 짓기 때문에 배선은 따로 봐야 한다
        ([[the-gate-must-build-the-real-object]])
```

    python -X utf8 src/run_gate_gatecond.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.model.losses import (LossWeights, NILMLoss, S_STATE,  # noqa: E402
                              build_state_scales)
from src.model.postproc import HALFWAVE_OHM, RESISTIVE_OHM  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
KO = {"air_conditioner": "에어컨", "beam_projector": "빔프", "electiric_kettle": "포트",
      "fan": "선풍", "hair_dryer": "드라이", "hotplate": "핫플",
      "laptop_charger": "충전", "minipc": "미니PC", "oven": "오븐"}
RES = ("electiric_kettle", "oven", "hotplate", "hair_dryer")
COND = {"oven": 2, "hotplate": 2}
H, S = 15, 5
OK, NG = "OK", "** 실패 **"
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % (OK if good else NG, name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


def _crit(cond=COND, **wkw):
    s_i = torch.tensor([max(max(S_STATE.get(a, {1: 10.0}).values()), 10.0) for a in APPS])
    r = np.random.RandomState(0)
    return NILMLoss(
        s_i=s_i,
        signatures=torch.from_numpy((r.randn(len(APPS), H, 2) * 1e-3).astype(np.float32)),
        standby_sig=torch.zeros(len(APPS), H, 2),
        noise_sig=torch.zeros(H, 2), harm_scale=torch.ones(H),
        s_state=build_state_scales(APPS, s_i.tolist()),
        weights=LossWeights(**wkw),
        res_ohm=torch.tensor([RESISTIVE_OHM[a] if a in RES else 0.0 for a in APPS]),
        res_ohm_half=torch.tensor([HALFWAVE_OHM[a] if (a in RES and a in HALFWAVE_OHM)
                                   else 0.0 for a in APPS]),
        res_cond_state=torch.tensor([(cond or {}).get(a, 0) for a in APPS], dtype=torch.long))


def _batch(B=96, seed=0, oven_state=None, hot_state=None):
    """오븐의 state 를 골라 넣을 수 있게 한 판. 기본은 무작위."""
    g = torch.Generator().manual_seed(seed)
    lg = (torch.randn(B, len(APPS), generator=g) * 2).requires_grad_(True)
    pr = (torch.rand(B, len(APPS), generator=g) * 800 + 10).requires_grad_(True)
    sb = (torch.rand(B, len(APPS), generator=g) * 3).requires_grad_(True)
    st = torch.randn(B, len(APPS), S, generator=g)
    out = dict(on_logit=lg, power_raw=pr, power=torch.sigmoid(lg) * pr, standby=sb,
               state=st, power_mix=st.softmax(-1),
               power_states=pr[..., None].expand(-1, -1, S),
               plugged_logit=torch.randn(B, len(APPS), generator=g))
    ys = torch.randint(0, 3, (B, len(APPS)), generator=g)
    if oven_state is not None:
        ys[:, APPS.index("oven")] = int(oven_state)
    if hot_state is not None:
        ys[:, APPS.index("hotplate")] = int(hot_state)
    tgt = dict(p_observed=torch.rand(B, generator=g) * 2000 + 200,
               p_noise=torch.full((B,), 1.9), v_rms=torch.full((B,), 222.0),
               obs_harm=torch.randn(B, H, 2, generator=g) * 0.05,
               y_power=torch.rand(B, len(APPS), generator=g) * 100,
               y_on=(ys > 0).float(),
               y_plugged=torch.ones(B, len(APPS)),
               y_standby=torch.rand(B, len(APPS), generator=g),
               y_state=ys)
    return out, tgt, (lg, pr, sb)


def main() -> int:  # noqa: C901
    print("`L_gcond` — 게이트를 통전에 묶는다 (14.106)")
    print("  `res_cond_state` = %s\n" % COND)

    # ── [1] 0 이면 키조차 안 생긴다 ──────────────────────────────────────
    out, tgt, _ = _batch()
    p0 = _crit().forward(out, tgt)
    p0b = _crit(gate_cond=0.0).forward(out, tgt)
    same = ("gate_cond" not in p0 and "gate_cond" not in p0b
            and torch.equal(p0["total"], p0b["total"]))
    ck("[1] `--w-gate-cond 0` 이면 `parts` 에 키가 없고 `total` 이 비트 동일", same,
       "총 %.9f" % float(p0["total"]))

    # ── [2] 켜면 걸린다 ──────────────────────────────────────────────────
    p1 = _crit(gate_cond=1.0).forward(out, tgt)
    on_ = ("gate_cond" in p1 and float(p1["gate_cond"]) > 0
           and not torch.equal(p1["total"], p0["total"]))
    ck("[2] 켜면 `gate_cond` 가 생기고 `total` 이 달라진다", on_,
       "gate_cond %.5f · total %.5f -> %.5f"
       % (float(p1["gate_cond"]), float(p0["total"]), float(p1["total"])))

    # ── [3] 목표가 정확히 1{y_state == 통전상태} ─────────────────────────
    cs = torch.tensor([COND.get(a, 0) for a in APPS], dtype=torch.long)
    gm = (cs > 0).float()[None]
    t_want = (tgt["y_state"].long() == cs[None]).float()
    bce = F.binary_cross_entropy_with_logits(out["on_logit"], t_want, reduction="none")
    want = float((bce * gm).sum() / gm.expand_as(bce).sum())
    d = abs(want - float(p1["gate_cond"]))
    ck("[3] 손으로 계산한 BCE 와 일치", d < 1e-6,
       "손계산 %.9f · 손실 %.9f · 차 %.3e" % (want, float(p1["gate_cond"]), d))

    # ── [4] 가면 — cond 0 인 기기에는 기울기가 정확히 0 ──────────────────
    out2, tgt2, (lg2, _p, _s) = _batch(seed=1)
    L = _crit(gate_cond=1.0).forward(out2, tgt2)["gate_cond"]
    g4 = torch.autograd.grad(L, lg2)[0]
    leak = max(float(g4[:, j].abs().max()) for j, a in enumerate(APPS) if a not in COND)
    hit = min(float(g4[:, APPS.index(a)].abs().max()) for a in COND)
    ck("[4] `res_cond_state=0` 인 일곱에 기울기 **정확히 0**", leak == 0.0,
       "누수 %.3e · 겨냥한 둘은 %.3e" % (leak, hit))

    # ── [5] ★ 해석적 최적 = w_on / (w_on + W) ────────────────────────────
    print("  [5] ★ 팬·조명 창의 최적 게이트 — 문서가 적은 `w_on/(w_on+W)` 와 맞나")
    w_on = LossWeights().on
    ok5 = True
    for W in (0.3, 0.9, 2.7):
        z = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([z], lr=0.5, max_iter=250)

        def closure():
            opt.zero_grad()
            # 팬·조명 창 하나: y_on = 1, 통전 라벨 = 0
            loss = (w_on * F.binary_cross_entropy_with_logits(z, torch.ones(1))
                    + W * F.binary_cross_entropy_with_logits(z, torch.zeros(1)))
            loss.backward()
            return loss
        opt.step(closure)
        got = float(torch.sigmoid(z.detach()))
        want5 = w_on / (w_on + W)
        good = abs(got - want5) < 5e-4
        ok5 &= good
        print("      W=%.1f  최적 게이트 %.4f · 식 %.4f   %s"
              % (W, got, want5, OK if good else NG))
    ck("[5] 눈금이 문서와 맞는다 (w_on = %.1f)" % w_on, ok5)

    # ── [6] ★ 핫플에는 아무 일도 안 일어난다 ─────────────────────────────
    #   학습 자료에 핫플 state 1 이 **하나도 없다** (100% state 2). 그런 기기는
    #   `1{y_state==2}` 가 `y_on` 과 같아 `L_on` 과 싸우지 않는다.
    o6, t6, (lg6, _, _) = _batch(seed=2, hot_state=2)
    ih = APPS.index("hotplate")
    t6["y_state"][:, ih] = torch.where(t6["y_on"][:, ih] > 0.5,
                                       torch.full_like(t6["y_state"][:, ih], 2),
                                       torch.zeros_like(t6["y_state"][:, ih]))
    tc6 = (t6["y_state"].long() == cs[None]).float()
    agree = bool(torch.equal(tc6[:, ih], t6["y_on"][:, ih]))
    L6 = _crit(gate_cond=1.0).forward(o6, t6)["gate_cond"]
    g6 = torch.autograd.grad(L6, lg6)[0]
    io = APPS.index("oven")
    ck("[6] ★ 통전 상태만 있는 기기(핫플)는 목표가 `y_on` 과 같다", agree,
       "핫플 ∂on 평균 %.3e · 오븐 %.3e (기울기는 있으나 `L_on` 과 **같은 방향**)"
       % (float(g6[:, ih].abs().mean()), float(g6[:, io].abs().mean())))

    # ── [7] autocast (bfloat16) ─────────────────────────────────────────
    got7, err = None, ""
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            got7 = float(_crit(gate_cond=1.0).forward(*_batch(seed=3)[:2])["gate_cond"])
    except Exception as e:                                    # noqa: BLE001
        err = "%s: %s" % (type(e).__name__, e)
    ck("[7] autocast(bfloat16) 에서 돈다 (982877 이 여기서 죽었다)",
       got7 is not None and np.isfinite(got7), err or "gate_cond %.5f" % (got7 or 0.0))

    # ── [8] 음성 대조 — 가면을 없앤 본을 잡아내나 ────────────────────────
    o8, t8, _ = _batch(seed=4)
    cs_all = torch.full((len(APPS),), 2, dtype=torch.long)     # 일부러 전 기기에 건다
    t_all = (t8["y_state"].long() == cs_all[None]).float()
    bad = float(F.binary_cross_entropy_with_logits(o8["on_logit"], t_all).mean())
    good_ = float(_crit(gate_cond=1.0).forward(o8, t8)["gate_cond"])
    ck("[8] 음성 대조 — 가면 없는 본은 **다른 값**을 낸다", abs(bad - good_) > 1e-3,
       "가면 있음 %.5f · 없음 %.5f" % (good_, bad))

    # ── [9] 배선 — run_train_cnn 이 정말 넘기는가 ────────────────────────
    import ast
    src = Path("src/run_train_cnn.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    wired = False
    for nd in ast.walk(tree):
        if (isinstance(nd, ast.Call) and getattr(nd.func, "id", "") == "LossWeights"):
            for kw in nd.keywords:
                if kw.arg == "gate_cond":
                    wired = (isinstance(kw.value, ast.Attribute)
                             and kw.value.attr == "w_gate_cond")
    has_cli = '"--w-gate-cond"' in src
    ck("[9] `run_train_cnn` 이 `--w-gate-cond` -> `LossWeights.gate_cond` 로 넘긴다",
       wired and has_cli, "CLI %s · 배선 %s" % (has_cli, wired))

    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
