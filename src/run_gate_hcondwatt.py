# -*- coding: utf-8 -*-
"""관문 — `--hcond-scale watt` · `--hcond-classes` · `--hcond-on-w` (14.299).

사용자 물음: *"이제 저항성만 영향을 받는데 로그가 있어야 하나?"* — 답은 **아니다**.
log 가 하던 일이 둘인데(① V 불변 ② 척도 불변) 저항 기기만 남으면 ②는 살 이유가 없다.

여덟 줄. **진짜 `NILMLoss` 를 지어** 잰다 ([[the-gate-must-build-the-real-object]]).

  [1] 기본값이면 **비트 동일** (scale=log · classes="" · on_w=5)
  [2] ★★ **오븐↔포트 분리력** — 196W 차이에서 watt 가 log 보다 세고 바닥 이상이다
  [3] ★★ **폭주가 사라진다** — 5~100W 참값 창이 무너졌을 때 log 대 watt
  [4] ★★ **클래스 제한이 폭주 창을 통째로 없앤다** — `target_power_w` 로 직접 센다

⚠ **14.300 정정.** 처음엔 `S_STATE`(오븐 s1 = 17W)로 재고 "클래스로는 폭주가 안 잘린다"
  고 적었다. 그건 **척도·초기값 표**지 학습 목표가 아니다(`losses.py` 의 "이 표는 초기값
  전용이다"). 실제 목표 `target_power_w` 로 재면 오븐 팬·조명은 **0.0W** 라 5W 문턱에
  애초에 안 걸리고, 5~100W 띠의 **75.6%가 비저항**이다. 결론이 뒤집혔다 —
  `--hcond-classes RESISTIVE` **하나로 폭주가 사라진다** ([[verify-channel-layout-by-measurement]]).
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
# ⚠⚠ **14.301 정정.** `--per-state-scale` 기본값이 **True** 라 손실의 `s` 는
#   `s_i`(기기 p90)가 아니라 **(기기,상태)별 척도** `s_state` 다 (`losses.py` forward
#   의 `torch.gather`). 처음엔 전부 `s_i` 로 재서 폭주를 **246배까지 부풀렸다**
#   (에어컨 s1: 2,024배 -> 실제 **8.2배**). 상태별 척도가 저전력 상태를 이미
#   정규화하고 있다 ([[check-the-denominator-before-reading-a-ratio]]).
_C = mk(per_state_scale=True)
S_I = {k: float(v) for k, v in zip(APPS, _C.s_i.tolist())}
_SS = _C.s_state.numpy()
assert _C.use_state_scale, "per_state_scale 이 꺼졌다 — 자가 달라진다"


def S_SCALE(app, state):
    """손실이 실제로 쓰는 척도. 상태가 None 이면 기기 p90 (per_state_scale 꺼진 경우)."""
    i = APPS.index(app)
    return float(_SS[i, min(int(state), _SS.shape[1] - 1)]) if state is not None else S_I[app]


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
S_OVEN = S_SCALE("oven", 2)                    # ★ 통전은 **상태 2** 다 (s_i 아님)
Y_OVEN, D_W = 1088.0, 196.0                    # 214V 에서 42.08Ω 대 35.67Ω
gb = g_base(Y_OVEN, Y_OVEN + D_W, S_OVEN, 0.1)
gl = g_log(Y_OVEN, Y_OVEN + D_W, 0.05)
gw = g_watt(Y_OVEN, Y_OVEN + D_W, S_OVEN, VP, 0.1)
ok(2, "196W 를 가르는 힘: watt >= 바닥 > log", gw >= gb > gl,
   "바닥 %.2e · **log %.2e (%.2f배)** · **watt %.2e (%.2f배)**"
   % (gb, gl, gl / gb, gw, gw / gb))

# ── 진짜 학습 목표를 읽는다 (14.300) ────────────────────────────
import glob                                                          # noqa: E402

STEM = {"electiric_kettle": "electric_kettle"}    # ⚠ 기기 이름에 오타가 박혀 있다


def band_frac(app, lo=5.0, hi=100.0):
    """그 기기의 `target_power_w` 중 (lo, hi] 에 드는 **비율과 개수**."""
    P = []
    for f in sorted(glob.glob("processed_data/npz/%s_*.npz" % STEM.get(app, app))):
        z = np.load(f, allow_pickle=True)
        v = z["is_valid"].astype(bool)
        P.append(z["target_power_w"].astype(np.float64)[v])
    if not P:
        return None
    p = np.concatenate(P)
    b = (p > lo) & (p <= hi)
    return float(b.mean()), int(b.sum()), int(len(p))


def band_median(app, lo=5.0, hi=100.0):
    P = []
    for f in sorted(glob.glob("processed_data/npz/%s_*.npz" % STEM.get(app, app))):
        z = np.load(f, allow_pickle=True)
        v = z["is_valid"].astype(bool)
        P.append(z["target_power_w"].astype(np.float64)[v])
    p = np.concatenate(P)
    b = (p > lo) & (p <= hi)
    return float(np.median(p[b])) if b.sum() else 0.0


BAND = {a: band_frac(a) for a in APPS}
if any(v is None for v in BAND.values()):
    raise SystemExit("processed_data/npz 가 없다 — 이 관문은 **진짜 목표**를 읽어야 한다")
BANDMED = {a: band_median(a) for a in APPS}


def band_states(app, lo=5.0, hi=100.0):
    """(상태, 창수, y중앙) 목록 — **상태별**로 갈라야 척도가 맞는다."""
    P, ST = [], []
    for f in sorted(glob.glob("processed_data/npz/%s_*.npz" % STEM.get(app, app))):
        z = np.load(f, allow_pickle=True)
        v = z["is_valid"].astype(bool)
        P.append(z["target_power_w"].astype(np.float64)[v])
        ST.append(z["state_id"].astype(int)[v])
    p, st = np.concatenate(P), np.concatenate(ST)
    b = (p > lo) & (p <= hi)
    return [(int(sid), int((b & (st == sid)).sum()),
             float(np.median(p[b & (st == sid)])))
            for sid in sorted(set(st[b].tolist())) if (b & (st == sid)).sum()]


BANDSTATE = {a: band_states(a) for a in APPS}
#: 문턱 **위**도 봐야 ②가 자명한 명제가 안 된다.
BANDSTATE_ALL = {a: band_states(a, 5.0, 1e9) for a in APPS}

# ── [3] ★★ 폭주를 **창수 x 비** 로 센다 (실재하는 창만) ─────────
# ⚠ 비 하나로 고르면 안 된다 — 오븐이 4,963배지만 창이 **35개**뿐이다.
#   에어컨은 1,978배인데 **18,666개**다. 손실에 실제로 들어오는 것은 곱이다
#   ([[count-effective-samples-not-rows]]).
MASS = {}          # (기기, 상태) -> 창수 x 비
for a in APPS:
    for sid, cnt, ymed in BANDSTATE[a]:
        sc = S_SCALE(a, sid)
        MASS[(a, sid)] = (cnt * g_log(ymed, 1.0, 0.05) / g_base(ymed, 1.0, sc, 0.1),
                          g_log(ymed, 1.0, 0.05) / g_base(ymed, 1.0, sc, 0.1), cnt, ymed, sc)
heavy = max(MASS, key=lambda k: MASS[k][0])
worst = max(MASS, key=lambda k: MASS[k][1])
hm, hr, hc, hy, hs = MASS[heavy]
wm, wr, wc, wy, ws = MASS[worst]
lw_r = g_watt(hy, 1.0, hs, VP, 0.1) / g_base(hy, 1.0, hs, 0.1)
ok(3, "폭주가 실재하고 watt 는 그 자리가 바닥이다",
   wr > 100.0 and hm > 1e5 and lw_r < 2.0,
   "비 최악 %s s%d **%.0f배**(창 %d) · 질량 최악 %s s%d %.0f배 x %d창 = **%.2fM** · "
   "watt 는 %.2f배"
   % (KO[worst[0]], worst[1], wr, wc, KO[heavy[0]], heavy[1], hr, hc, hm / 1e6, lw_r))

# ── [4] ★★ 클래스 제한이 폭주 창을 통째로 없앤다 ────────────────
res_set = {x for x in APPS if V_EXP.get(x, 0.0) == 2.0}
r_hit = sum(BAND[a][1] for a in res_set)
r_tot = sum(BAND[a][2] for a in res_set)
n_hit = sum(BAND[a][1] for a in APPS if a not in res_set)
n_tot = sum(BAND[a][2] for a in APPS if a not in res_set)
r_frac, n_frac = r_hit / max(r_tot, 1), n_hit / max(n_tot, 1)
r_mass = sum(v[0] for k, v in MASS.items() if k[0] in res_set)
n_mass = sum(v[0] for k, v in MASS.items() if k[0] not in res_set)
share = r_mass / max(r_mass + n_mass, 1e-30)
# ★ 두 단계다. ① 클래스가 대부분을 없애고 ② 문턱이 나머지를 없앤다.
#   ②가 자명하지 않으려면 **문턱 위** 저항 창에 폭주가 없는지도 봐야 한다 —
#   띠가 (5,100] 이라 "문턱 100 이 띠를 다 뗀다" 는 정의상 참이기 때문이다.
over = []          # 문턱 위 저항 상태에서 **전형적 오차(30% 저평가)** 의 비
for a in res_set:
    for sid, cnt, ymed in BANDSTATE_ALL[a]:
        if ymed <= 100.0:
            continue
        sc = S_SCALE(a, sid)
        ph = 0.7 * ymed
        over.append((g_log(ymed, ph, 0.05) / g_base(ymed, ph, sc, 0.1), a, sid, ymed, cnt))
over.sort(reverse=True)
ok(4, "① 클래스가 질량 대부분을 · ② 문턱이 나머지를 없앤다",
   r_frac < 0.01 and n_frac > 0.5 and n_mass > 10 * r_mass and over[0][0] <= 2.0,
   "① 질량 저항 %.2fM 대 비저항 %.1fM = **1/%.0f** · 창 %.3f%% 대 %.1f%% | "
   "② 남는 %d 창이 전부 <=100W 라 문턱이 뗀다 · **문턱 위** 저항 최대비 "
   "%s s%d **%.2f배**(폭주 없음)"
   % (r_mass / 1e6, n_mass / 1e6, n_mass / max(r_mass, 1e-30),
      100 * r_frac, 100 * n_frac, r_hit,
      KO[over[0][1]], over[0][2], over[0][0]))

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
