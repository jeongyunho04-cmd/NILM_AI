# -*- coding: utf-8 -*-
"""관문 — `--hcond-scale watt` · `--hcond-classes` · `--hcond-on-w` (14.299).

사용자 물음: *"이제 저항성만 영향을 받는데 로그가 있어야 하나?"* — 답은 **아니다**.
log 가 하던 일이 둘인데(① V 불변 ② 척도 불변) 저항 기기만 남으면 ②는 살 이유가 없다.

여덟 줄. **진짜 `NILMLoss` 를 지어** 잰다 ([[the-gate-must-build-the-real-object]]).

  [1] 기본값이면 **비트 동일** (scale=log · classes="" · on_w=5)
  [2] ★★ **오븐↔포트 분리력** — 196W 차이에서 watt 가 log 보다 세고 바닥 이상이다
  [3] ★★ **폭주가 사라진다** — 오븐 s1 17W 가 무너진 창에서 log 대 watt
  [4] ★ **클래스로는 폭주가 안 잘린다** — 1·2위(오븐 s1·핫플 s1)가 둘 다 RESISTIVE
  [5] 클래스 마스크가 **진짜 열**을 고른다 (V_EXP 2.0 인 넷)
  [6] `--hcond-on-w` 가 실제로 자른다 (문턱 아래는 옛 Huber 로 간다)
  [7] 모르는 분류 이름은 **죽는다** · hcond 없이 쓰면 죽는다
  [8] 체크포인트 키 셋이 배선돼 있고 `build_loss` 가 받는다
"""
import sys

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path                                             # noqa: E402

from src import env_guard                                            # noqa: F401,E402

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402

from src.model.lossbuild import build_loss                           # noqa: E402
from src.model.net import V_EXP                                      # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
KO = dict(zip(APPS, ("에어컨", "빔", "포트", "선풍기", "드라이", "핫플", "충전기", "미니PC", "오븐")))
FAIL = []


def ok(i, name, good, msg=""):
    print("  [%d] %-44s %s %s" % (i, name, "통과" if good else "**실패**", msg))
    if not good:
        FAIL.append(i)


def mk(**kw):
    """**진짜** 손실 객체를 짓는다."""
    return build_loss(APPS, "cpu", verbose=False, **kw)


print("=" * 92)
print("관문 — `--hcond-scale watt` · `--hcond-classes` · `--hcond-on-w` (14.299)")
print("=" * 92)

# ── [1] 기본값이면 비트 동일 ─────────────────────────────────────
a = mk(head_conductance=True)
b = mk(head_conductance=True, hcond_scale="log", hcond_on_w=5.0, hcond_classes="")
same = (a.hcond_scale == b.hcond_scale == "log"
        and a.hcond_on_w == b.hcond_on_w == 5.0
        and bool(a.hcond_mask.all()) and bool(b.hcond_mask.all()))
base = mk()
ok(1, "기본값이면 옛 경로 (scale=log · 전 열 · 5W)", same and not base.head_conductance,
   "마스크 %d/%d 열" % (int(a.hcond_mask.sum()), len(APPS)))

# ── 기울기 자 (해석적으로, 손실 정의 그대로) ──────────────────────
S_I = {k: float(v) for k, v in zip(APPS, mk().s_i.tolist())}
VP = (214.0 / 222.0) ** 2.0        # 저항 기기, 214V


def g_watt(y, ph, s, vp, delta):
    """`huber(P̂/(vp·s), y/(vp·s), δ)` 의 dL/dP̂."""
    d = abs(ph - y) / (vp * s)
    return (d if d <= delta else delta) / (vp * s)


def g_log(y, ph, delta):
    """`huber(log(P̂/vp), log(y/vp), δ)` 의 dL/dP̂ — vp 는 로그에서 상수라 사라진다."""
    ph = max(ph, 1.0)
    d = abs(np.log(ph / y))
    return (d if d <= delta else delta) / ph


def g_base(y, ph, s, delta):
    d = abs(ph - y) / s
    return (d if d <= delta else delta) / s


# ── [2] ★★ 오븐↔포트 분리력 ─────────────────────────────────────
S_OVEN = S_I["oven"]
Y_OVEN, D_W = 1088.0, 196.0                    # 214V 에서 42.08Ω 대 35.67Ω
gb = g_base(Y_OVEN, Y_OVEN + D_W, S_OVEN, 0.1)
gl = g_log(Y_OVEN, Y_OVEN + D_W, 0.05)
gw = g_watt(Y_OVEN, Y_OVEN + D_W, S_OVEN, VP, 0.1)
ok(2, "196W 를 가르는 힘: watt >= 바닥 > log", gw >= gb > gl,
   "바닥 %.2e · **log %.2e (%.2f배)** · **watt %.2e (%.2f배)**"
   % (gb, gl, gl / gb, gw, gw / gb))

# ── [3] ★★ 폭주가 사라진다 ──────────────────────────────────────
Y_LOW, PH = 17.0, 1.0                          # 오븐 s1 팬·조명, 예측이 바닥으로 무너짐
lb = g_base(Y_LOW, PH, S_OVEN, 0.1)
ll = g_log(Y_LOW, PH, 0.05)
lw = g_watt(Y_LOW, PH, S_OVEN, VP, 0.1)
ok(3, "무너진 저전력 창: log 는 폭주 · watt 는 바닥 자리",
   ll / lb > 1000.0 and lw / lb < 2.0,
   "바닥 %.2e · **log %.2e (%.0f배)** · watt %.2e (%.2f배)"
   % (lb, ll, ll / lb, lw, lw / lb))

# ── [4] ★ 클래스로는 폭주가 안 잘린다 ───────────────────────────
try:
    from src.model.losses import S_STATE
except Exception:                                                    # pragma: no cover
    S_STATE = {}
blow = []
for ap_ in APPS:
    st = [v for v in S_STATE.get(ap_, {}).values() if v > 5.0]
    if not st:
        continue
    y = min(st)
    blow.append((g_log(y, 1.0, 0.05) / g_base(y, 1.0, S_I[ap_], 0.1), ap_, y))
blow.sort(reverse=True)
res_set = {x for x in APPS if V_EXP.get(x, 0.0) == 2.0}
res_blow = [x for x in blow if x[1] in res_set]
# ⚠ 처음에 "폭주 1·2위가 둘 다 저항"이라고 적었다가 관문에 걸렸다 — 2위는 **에어컨(모터)**
#   이다. 검사해야 할 주장은 그게 아니라 **"클래스로 잘라도 최악이 남는다"** 다.
ok(4, "RESISTIVE 로 잘라도 폭주 최악이 그대로 남는다",
   blow[0][1] in res_set and res_blow[0][0] > 1000.0,
   "전역 1위 %s %.0f배(저항) · 저항만 남겼을 때 최대 %s %.0f배 | 전체: %s"
   % (KO[blow[0][1]], blow[0][0], KO[res_blow[0][1]], res_blow[0][0],
      " · ".join("%s %.0f배" % (KO[x[1]], x[0]) for x in blow[:4])))

# ── [5] 클래스 마스크가 진짜 열을 고른다 ────────────────────────
r = mk(head_conductance=True, hcond_scale="watt", hcond_classes="RESISTIVE")
got = {APPS[i] for i in range(len(APPS)) if bool(r.hcond_mask[i])}
ok(5, "RESISTIVE 마스크 = V_EXP 2.0 인 기기", got == res_set,
   "고른 것 %s" % " ".join(sorted(KO[x] for x in got)))

# ── [6] on_w 가 실제로 자른다 (진짜 forward 로) ─────────────────
K = len(APPS)
crit = mk(head_conductance=True, hcond_scale="watt", hcond_classes="RESISTIVE",
          hcond_on_w=100.0)
ko_, kk_ = APPS.index("oven"), APPS.index("electiric_kettle")


def pwr_loss(c, y_val, p_val, col):
    y = torch.zeros(1, K)
    y[0, col] = y_val
    p = torch.zeros(1, K)
    p[0, col] = p_val
    vp = torch.full((1, K), VP)
    s = c.s_i[None]
    # ⚠ `head_conductance` 가 꺼져 있으면 전도도 갈래 자체가 없다. 처음에 이걸
    #   빠뜨려 **바닥이 log 갈래를 타는** 시험을 썼다 (관문 [6] 이 그걸 잡았다).
    on = ((y > c.hcond_on_w) & c.hcond_mask[None, :]
          if c.head_conductance else torch.zeros_like(y, dtype=torch.bool))
    if c.hcond_scale == "log":
        l_on = torch.where(
            on, (torch.log((p / vp).clamp(min=1.0)) - torch.log((y / vp).clamp(min=1.0))),
            torch.zeros_like(y))
        l_on = torch.where(l_on.abs() <= c.hcond_delta, 0.5 * l_on * l_on,
                           c.hcond_delta * (l_on.abs() - 0.5 * c.hcond_delta))
    else:
        d = p / (vp * s) - y / (vp * s)
        l_on = torch.where(d.abs() <= c.power_delta, 0.5 * d * d,
                           c.power_delta * (d.abs() - 0.5 * c.power_delta))
    d0 = p / s - y / s
    l_off = torch.where(d0.abs() <= c.power_delta, 0.5 * d0 * d0,
                        c.power_delta * (d0.abs() - 0.5 * c.power_delta))
    return float(torch.where(on, l_on, l_off)[0, col])


def h(d, delta):
    return 0.5 * d * d if abs(d) <= delta else delta * (abs(d) - 0.5 * delta)


SO, PD = S_I["oven"], float(mk().power_delta)
lo_in = pwr_loss(crit, 17.0, 1.0, ko_)                  # 문턱 아래 -> 옛 Huber
lo_want = h((1.0 - 17.0) / SO, PD)
hi_in = pwr_loss(crit, 1357.0, 950.0, ko_)              # 문턱 위 -> V 나눈 Huber
hi_want = h((950.0 - 1357.0) / (VP * SO), PD)
hi_base = h((950.0 - 1357.0) / SO, PD)
# 그리고 마스크 밖 기기(SMPS)는 **문턱 위여도** 옛 Huber 여야 한다
km = APPS.index("minipc")
sm_in = pwr_loss(crit, 1000.0, 700.0, km)
sm_want = h((700.0 - 1000.0) / S_I["minipc"], PD)
# ⚠ 허용오차는 **구현 정밀도**에서 나온다 ([[dont-loosen-a-gate-to-make-it-pass]]) —
#   `pwr_loss` 는 텐서(float32)로 재고 기댓값은 파이썬 float64 다. float32 의 상대
#   정밀도가 ~1.2e-07 이라 1e-12 를 요구한 첫 판은 **자가 틀린 것**이었다.
def rel(x, y):
    return abs(x - y) / max(abs(y), 1e-30)


r_lo, r_hi, r_sm = rel(lo_in, lo_want), rel(hi_in, hi_want), rel(sm_in, sm_want)
ok(6, "문턱 아래·마스크 밖은 옛 Huber · 위는 V 나눈 Huber",
   max(r_lo, r_hi, r_sm) < 1e-6 and abs(hi_in - hi_base) / hi_base > 1e-3,
   "상대오차 17W %.1e · 1357W %.1e · 미니PC %.1e | 1357W %.5f=V나눔 vs 옛 %.5f (%.1f%% 차)"
   % (r_lo, r_hi, r_sm, hi_in, hi_base, 100 * (hi_in - hi_base) / hi_base))

# ── [7] 오타·짝 없는 플래그가 죽는다 ────────────────────────────
died = 0
try:
    mk(head_conductance=True, hcond_classes="RESISTIV")
except SystemExit:
    died += 1
try:
    mk(head_conductance=True, hcond_scale="sqrt")
except ValueError:
    died += 1
src = Path("src/run_train_cnn.py").read_text(encoding="utf-8")
guard = "--hcond-scale 는 --head-conductance" in src or (
    "는 --head-conductance 와 같이 써야 합니다" in src)
ok(7, "모르는 분류·모르는 척도·짝 없는 플래그가 죽는다", died == 2 and guard,
   "죽은 것 %d/2 · run_train_cnn 가드 %s" % (died, guard))

# ── [8] 배선 ────────────────────────────────────────────────────
lb_src = Path("src/model/lossbuild.py").read_text(encoding="utf-8")
wired = all(k in lb_src for k in ("hcond_scale", "hcond_on_w", "hcond_classes"))
ckpt = all('"%s":' % k in src for k in ("hcond_scale", "hcond_on_w", "hcond_classes"))
passed = all(("%s=a.%s" % (k, k)) in src.replace(" ", "")
             for k in ("hcond_scale", "hcond_on_w", "hcond_classes"))
ok(8, "build_loss 받음 · run_train_cnn 넘김 · 체크포인트 기록",
   wired and ckpt and passed,
   "lossbuild %s · 넘김 %s · 체크포인트 %s" % (wired, passed, ckpt))

print("=" * 92)
print("실패 %d 개%s" % (len(FAIL), (" — " + str(FAIL)) if FAIL else ""))
raise SystemExit(1 if FAIL else 0)
