# -*- coding: utf-8 -*-
"""관문 — **DC 를 크기 경로에만 되돌린다** `--fine-dc mag` (14.251). 여덟 줄.

까닭 (14.243~14.250 이 잰 것). `ch23 = asinh(P/100)` 의 창 평균 = **절대 전력 수준**은
**두 일을 한꺼번에** 한다:

```
  숏컷이다 — 오븐/큰저항부하 신원을 합성에서 AUC 0.943 으로 가르는데 실측에서 **0.521**
  신호다   — `split` 로 몸통에서 통째로 떼니 실측 포트 게이트는 6/6 으로 고쳐졌지만
             홀드아웃 MAE 4.65 -> **7.75** · F1 0.933 -> 0.904 (여섯 대 셋, 겹침 0)
             게다가 예측이 흔들렸다 — 정지창 |ΔP| 3.06 -> **7.40** W/0.5초 (계단근처 4.74배)
```
RevIN(ICLR 2022)의 표준은 **대칭**이다 — 입력에서 창 통계를 빼고 **출력에서 도로 넣는다**.
`split` 은 앞 절반만 했다. `mag` 가 뒤 절반이다: DC 는 몸통에 **안 들어가고**
`p_states`(크기)에만 더해진다. `on_logit`(신원)·`state`(상태혼합)에는 **구조적으로 못 닿는다.**

```
  [1] `keep`·`split` 이 **비트 동일** — 옛 판과 안 겹친다
  [2] 파라미터가 **정확히 n_dc x (K x S) + (K x S)** 만 는다 (mag 는 몸통을 안 늘린다)
  [3] ★★ `dc_pow` 를 **크게 무작위로** 채워도 `on_logit`·`state`·`plugged`·`standby` 가
      **정확히 0** 만큼 움직이고 `power_states` 만 움직인다
  [4] ★★ 기울기 차단 — `∂on_logit/∂dc_pow` 가 **정확히 0** (autograd 가 말한다)
  [5] conv 가 받는 입력의 채널별 시간평균이 **정확히 0** (DC 가 진짜로 빠졌다)
  [6] 0 초기화라 **출발점의 되돌림 항이 정확히 0**
  [7] `split` 과 **다른 물건** — 몸통 폭이 keep 과 같고 교차 로드가 조용히 통과하면 안 된다
  [8] `head_layout=v2` 와 못 쓴다
```

    python -X utf8 -m src.run_gate_finedcmag
"""
import sys

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.losses import S_STATE  # noqa: E402
from src.model.net import MAX_STATES, NILMNet  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
NS = [1 + max(S_STATE.get(a, {1: 0}).keys()) for a in APPS]
GATEK = ("on_logit", "state", "plugged_logit", "standby", "power_mix")


def mk(dc, seed=0, **kw):
    torch.manual_seed(seed)
    return NILMNet(APPS, NS, fine_channels=57, fine_extra_dilations=(32, 64),
                   tap_layers=(0, 1, 4), p_state_cap=3.0, prior_kappa=8.0,
                   fine_dc=dc, **kw).eval()


def main() -> int:  # noqa: C901
    ok = True
    torch.manual_seed(7)
    f = torch.randn(4, 57, 600) * 0.3
    w = torch.randn(4, 47, 120) * 0.3
    N = {d: sum(p.numel() for p in mk(d).parameters()) for d in ("keep", "split", "mag")}

    # ── [1] 옛 갈래 불변 ──────────────────────────────────────────────────────
    d1 = 0.0
    for dc in ("keep", "split"):
        a, b = mk(dc), mk(dc)
        with torch.no_grad():
            d1 = max(d1, max(float((a(f, w)[k] - b(f, w)[k]).abs().max())
                             for k in ("on_logit", "power", "state")))
    print("[1] `keep`·`split` 재현 %.3e · 파라미터 keep %d · split %d  %s"
          % (d1, N["keep"], N["split"], "OK" if d1 == 0.0 else "FAIL"))
    ok &= d1 == 0.0

    # ── [2] 파라미터 ──────────────────────────────────────────────────────────
    m = mk("mag")
    want = m.n_dc * (len(APPS) * MAX_STATES) + len(APPS) * MAX_STATES
    got = N["mag"] - N["keep"]
    good = got == want and N["split"] - N["keep"] == m.n_dc * 256
    print("[2] mag 증가 %d = n_dc(%d) x (K x S = %d) + %d = %d · split 증가 %d = n_dc x 256  %s"
          % (got, m.n_dc, len(APPS) * MAX_STATES, len(APPS) * MAX_STATES, want,
             N["split"] - N["keep"], "OK" if good else "FAIL"))
    ok &= good

    # ── [3] ★★ 구조적 차단 ───────────────────────────────────────────────────
    with torch.no_grad():
        o0 = {k: v.clone() for k, v in m(f, w).items()}
        torch.manual_seed(3)
        m.dc_pow.weight.normal_(0, 5.0)
        m.dc_pow.bias.normal_(0, 5.0)
        o1 = m(f, w)
    gd = {k: float((o1[k] - o0[k]).abs().max()) for k in GATEK}
    ps = float((o1["power_states"] - o0["power_states"]).abs().max())
    good = all(v == 0.0 for v in gd.values()) and ps > 1.0
    print("[3] dc_pow 를 크게 흔듦 -> %s | power_states **%.3f 움직임**  %s"
          % (" · ".join("%s %.1e" % (k, v) for k, v in gd.items()), ps,
             "OK" if good else "FAIL"))
    ok &= good

    # ── [4] ★★ 기울기 차단 ───────────────────────────────────────────────────
    m.zero_grad(set_to_none=True)
    m(f, w)["on_logit"].abs().sum().backward()
    gw = m.dc_pow.weight.grad
    gb = m.dc_pow.bias.grad
    g_on = float(gw.abs().max()) if gw is not None else 0.0
    g_on = max(g_on, float(gb.abs().max()) if gb is not None else 0.0)
    m.zero_grad(set_to_none=True)
    m(f, w)["power_raw"].abs().sum().backward()
    g_pw = float(m.dc_pow.weight.grad.abs().max())
    good = g_on == 0.0 and g_pw > 0.0
    print("[4] ∂|on_logit|/∂dc_pow = **%.3e** (0 이어야) · ∂|power_raw|/∂dc_pow = %.3e  %s"
          % (g_on, g_pw, "OK" if good else "FAIL"))
    ok &= good

    # ── [5] conv 입력의 시간평균 ─────────────────────────────────────────────
    with torch.no_grad():
        ci, dcv = m._split_dc(m._conv_in(f))
    mu = float(ci.mean(-1).abs().max())
    good = mu < 1e-6 and dcv is not None and dcv.shape[1] == m.n_dc
    print("[5] conv 입력의 채널별 시간평균 |max| %.3e · 떼어 낸 DC 폭 %d  %s"
          % (mu, 0 if dcv is None else dcv.shape[1], "OK" if good else "FAIL"))
    ok &= good

    # ── [6] 0 초기화 ─────────────────────────────────────────────────────────
    m2 = mk("mag")
    z = max(float(m2.dc_pow.weight.abs().max()), float(m2.dc_pow.bias.abs().max()))
    print("[6] 초기 dc_pow |max| = **%.3e** (되돌림 항이 처음엔 0)  %s"
          % (z, "OK" if z == 0.0 else "FAIL"))
    ok &= z == 0.0

    # ── [7] split 과 다른 물건 ───────────────────────────────────────────────
    ms = mk("split")
    tw_keep = mk("keep").trunk[0].weight.shape[1]
    good = m2.trunk[0].weight.shape[1] == tw_keep and ms.trunk[0].weight.shape[1] == tw_keep + m.n_dc
    cross = "통과했다(나쁘다)"
    try:
        ms.load_state_dict(m2.state_dict())
    except Exception:                                 # noqa: BLE001
        cross = "막혔다"
    good &= cross == "막혔다"
    print("[7] 몸통 입력폭 keep %d · mag %d · split %d | mag->split 로드 **%s**  %s"
          % (tw_keep, m2.trunk[0].weight.shape[1], ms.trunk[0].weight.shape[1], cross,
             "OK" if good else "FAIL"))
    ok &= good

    # ── [8] v2 금지 ──────────────────────────────────────────────────────────
    try:
        mk("mag", head_layout="v2")
        good = False
    except ValueError:
        good = True
    print("[8] `mag` + head_layout=v2 가 **막힌다**  %s" % ("OK" if good else "FAIL"))
    ok &= good

    # ── [9] `_conv_in` 누수 차단 ─────────────────────────────────────────────
    lk = {}
    for deriv in ("window",):
        mm = mk("mag", fine_derive=deriv)
        g = f.clone()
        g[:, 23] += 0.30                       # 순수 DC 이동 (AC 는 그대로)
        with torch.no_grad():
            a0 = mm._split_dc(mm._conv_in(f))[0]
            a1 = mm._split_dc(mm._conv_in(g))[0]
        lk[deriv] = float((a1 - a0).abs().max()) / float(a0.abs().max())
    blocked = False
    try:
        mk("mag", fine_derive="both")
    except ValueError:
        blocked = True
    good = lk["window"] < 1e-5 and blocked
    print("[9] ch23 DC 만 밀어도 `_conv_in` 뒤 AC 상대변화 **%.2e** (window) · "
          "`both` 조합은 **%s**  %s"
          % (lk["window"], "막힌다" if blocked else "안 막힌다", "OK" if good else "FAIL"))
    ok &= good

    print()
    print("관문 %s" % ("전부 통과" if ok else "**실패**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
