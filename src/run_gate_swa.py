# -*- coding: utf-8 -*-
"""관문 — `--swa-start` 궤적 평균 (14.295).

여덟 줄. **진짜 객체를 짓고** 실제 학습 경로를 돌려 잰다
([[the-gate-must-build-the-real-object]]).

  [1] 끄면 **비트 동일** — 같은 씨앗 두 판이 가중치까지 같다
  [2] ★★ **무동작 감지** — 코사인이 0 으로 떨어진 구간을 평균하면 w̄ ≈ w_T 다.
      LR 을 고정하면 퍼짐이 생긴다. 두 경우의 ||w̄−w_T||/||w_T|| 를 나란히 찍는다
  [3] ★ **BN 재추정 불필요** — 실제 모델에 running stat 을 가진 정규화가 없다
  [4] 누적 평균 = 산술평균 (부동소수 오차 안)
  [5] 정수·불리언 버퍼가 **안 깨진다** (dtype·값 보존)
  [6] LR 이 swa_start 전엔 코사인, 후엔 **상수**다 (실제 스케줄러로)
  [7] 평균판이 **저장·재적재되고 유한한 출력**을 낸다 (진짜 save/load 경로)
  [8] 체크포인트에 swa_start·swa_lr·swa_every·swa_n 이 적히고 배선돼 있다
"""
import sys

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math                                                          # noqa: E402
import tempfile                                                      # noqa: E402
from pathlib import Path                                             # noqa: E402

from src import env_guard                                            # noqa: F401,E402

import torch                                                         # noqa: E402

from src.model import inputs as _I  # noqa: E402
from src.model.net import NILMNet                                    # noqa: E402
from src.model.losses import S_STATE                                 # noqa: E402
from src.run_train_cnn import WeightAverager                         # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
NS = [1 + max(S_STATE.get(a, {1: 0}).keys()) for a in APPS]
FAIL = []


def ok(i, name, good, msg=""):
    print("  [%d] %-42s %s %s" % (i, name, "통과" if good else "**실패**", msg))
    if not good:
        FAIL.append(i)


def build(seed=0):
    """**학습이 짓는 것과 같은 모델**을 짓는다 — 관문이 딴 물건을 재면 안 된다
    ([[the-gate-must-build-the-real-object]])."""
    torch.manual_seed(seed)
    #: ⚠ 14.368 — 채널 수를 **박아 두면 배치가 바뀔 때 조용히 만료된다**
    #  (57/47 은 14.331 이전 값이다. `run_gate_pstatecap` 이 같은 꼴로 986157 을 죽였다).
    return NILMNet(APPS, NS, fine_channels=_I.FINE_CHANNELS, fine_extra_dilations=(32, 64),
                   tap_layers=(0, 1, 4), p_state_cap=3.0, prior_kappa=8.0)


def run(model, steps, lr_fn, avg=None, avg_every=0, seed=1):
    """진짜 순전파·역전파로 궤적을 만든다. `lr_fn(step)` 이 학습률."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0, weight_decay=0.01)
    fc = model.fine_channels
    for t in range(steps):
        f = torch.randn(4, fc, 600, generator=g) * 0.3
        w = torch.randn(4, _I.WIDE_CHANNELS, 120, generator=g) * 0.3
        out = model(f, w)
        loss = out["power"].pow(2).mean() + out["on_logit"].pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for gr in opt.param_groups:
            gr["lr"] = lr_fn(t)
        opt.step()
        if avg is not None and avg_every and (t + 1) % avg_every == 0:
            avg.add(model)
    return model


print("=" * 84)
print("관문 — `--swa-start` 궤적 평균 (14.295)")
print("=" * 84)

STEPS, LR0 = 12, 3e-4


def cos_lr(t):
    return LR0 * (1 + math.cos(math.pi * t / STEPS)) / 2


# ── [1] 끄면 비트 동일 ────────────────────────────────────────────
m_a = run(build(0), STEPS, cos_lr)
m_b = run(build(0), STEPS, cos_lr)
same = all(torch.equal(x, y) for x, y in zip(m_a.state_dict().values(),
                                             m_b.state_dict().values()))
ok(1, "끄면 비트 동일 (같은 씨앗 두 판)", same)

# ── [2] ★★ 무동작 감지 ───────────────────────────────────────────
HALF = STEPS // 2
CONST = cos_lr(HALF)
m0 = run(build(0), STEPS, cos_lr)                       # 공통 출발점

def clone_of(src_model):
    """`deepcopy` 는 순전파가 붙여 둔 비잎 텐서 때문에 못 쓴다 — 같은 구조를 새로 짓고
    가중치만 싣는다."""
    c = build(0)
    c.load_state_dict(src_model.state_dict())
    return c


m_cos = clone_of(m0)
a_cos = WeightAverager(m_cos)
run(m_cos, STEPS, lambda t: cos_lr(HALF + t) * 1e-3, a_cos, 1, seed=7)
s_cos = a_cos.rel_shift(m_cos)

m_con = clone_of(m0)
a_con = WeightAverager(m_con)
run(m_con, STEPS, lambda t: CONST, a_con, 1, seed=7)
s_con = a_con.rel_shift(m_con)
ok(2, "LR 0 근처면 무동작 · 고정하면 퍼짐",
   s_con > 10 * max(s_cos, 1e-30) and s_con > 1e-5,
   "코사인꼬리 %.2e · **고정 %.2e** (%.0f배)" % (s_cos, s_con, s_con / max(s_cos, 1e-30)))

# ── [3] ★ BN 재추정 불필요 ───────────────────────────────────────
real = build(0)
bad3 = [n for n, mo in real.named_modules()
        if isinstance(mo, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                           torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm))
        or getattr(mo, "running_mean", None) is not None]
ngn = sum(1 for _, mo in real.named_modules() if isinstance(mo, torch.nn.GroupNorm))
ok(3, "running stat 가진 정규화 없음 (BN 재추정 불필요)",
   not bad3, "GroupNorm %d 개 · running stat %d 개" % (ngn, len(bad3)))

# ── [4] 누적 평균 = 산술평균 ──────────────────────────────────────
m4 = build(0)
av = WeightAverager(m4)
pts = []
for _k in range(5):
    with torch.no_grad():
        for prm in m4.parameters():
            prm.add_(torch.randn_like(prm) * 0.01)
    av.add(m4)
    pts.append({kk: v.detach().clone().float() for kk, v in m4.state_dict().items()})
key = next(k for k in av.float_keys if pts[0][k].numel() > 100)
truth = torch.stack([p[key] for p in pts]).mean(0)
err = float((av.avg[key] - truth).abs().max())
ok(4, "누적 평균이 산술평균과 같다", err < 1e-5 and av.n == 5,
   "n=%d · 최대오차 %.2e" % (av.n, err))

# ── [5] 정수·불리언 버퍼 보존 ────────────────────────────────────
# ⚠ 기본 구성에는 비부동소수 텐서가 **하나도 없다** — 그대로 재면 관문이 공허하게
#   통과한다 ([[the-gate-must-build-the-real-object]]). `--w-swap` 을 켜면 생기는
#   `_swap_combos` 같은 정수·불리언 버퍼를 **직접 달아** 그 경로를 태운다.
m5 = build(0)
m5.register_buffer("_gate_i64", torch.arange(16, dtype=torch.int64), persistent=True)
m5.register_buffer("_gate_bool", torch.tensor([True, False, True]), persistent=True)
m5.register_buffer("_gate_i8", torch.tensor([3, -7], dtype=torch.int8), persistent=True)
sd0 = {k: v.clone() for k, v in m5.state_dict().items()}
nonf = [k for k, v in sd0.items() if not v.is_floating_point()]
av5 = WeightAverager(m5)
for _k in range(4):                       # 부동소수만 움직인다. 정수는 그대로 둔다
    with torch.no_grad():
        for prm in m5.parameters():
            prm.add_(torch.randn_like(prm) * 0.01)
    av5.add(m5)
back = av5.state_dict(m5)
bad5 = [k for k in nonf
        if not (back[k].dtype == sd0[k].dtype and torch.equal(back[k], sd0[k]))]
dtb = [k for k in back if back[k].dtype != sd0[k].dtype]
# 그리고 **부동소수는 실제로 평균이 됐는지** — 안 그러면 위 셋이 통과해도 무의미하다
fkey = next(k for k in av5.float_keys if sd0[k].numel() > 100)
moved = float((back[fkey] - sd0[fkey]).abs().max()) > 1e-6
ok(5, "정수·불리언은 그대로 · 부동소수만 평균", not bad5 and not dtb and moved,
   "비부동소수 %d 개(i64·bool·i8 포함) · 값 어긋남 %d · dtype 어긋남 %d · 부동소수 움직임 %s"
   % (len(nonf), len(bad5), len(dtb), moved))

# ── [6] 실제 스케줄러로 LR 궤적 ──────────────────────────────────
EP, SPE, SWA_START = 10, 3, 7
mm = build(0)
opt6 = torch.optim.AdamW(mm.parameters(), lr=LR0)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt6, T_max=EP * SPE)
swa_lr = LR0 * (1 + math.cos(math.pi * ((SWA_START - 1) * SPE) / (EP * SPE))) / 2
seen = []
for ep in range(1, EP + 1):
    for _ in range(SPE):
        seen.append(opt6.param_groups[0]["lr"])
        if ep >= SWA_START:
            for gr in opt6.param_groups:
                gr["lr"] = swa_lr
        else:
            sch.step()
pre = seen[:(SWA_START - 1) * SPE]
post = seen[(SWA_START - 1) * SPE + 1:]
ok(6, "swa_start 전 코사인 · 후 상수",
   all(pre[i] >= pre[i + 1] for i in range(len(pre) - 1))
   and max(post) - min(post) < 1e-12 and abs(post[0] - swa_lr) < 1e-12,
   "코사인 %.2e -> %.2e · 고정 %.2e" % (pre[0], pre[-1], post[0]))

# ── [7] 저장·재적재·유한 출력 (진짜 경로) ─────────────────────────
m7 = build(0)
a7 = WeightAverager(m7)
run(m7, 8, lambda t: CONST, a7, 2, seed=3)
m7.load_state_dict(a7.state_dict(m7))
with tempfile.TemporaryDirectory() as d:
    pth = Path(d) / "swa_gate.pt"
    torch.save({"model": m7.state_dict(), "appliances": APPS, "width": 48,
                "epoch": EP, "swa_start": SWA_START, "swa_lr": float(swa_lr),
                "swa_every": 2, "swa_n": int(a7.n)}, pth)
    ck = torch.load(pth, map_location="cpu", weights_only=False)
    m8 = build(1)
    m8.load_state_dict(ck["model"])
    m8.eval()
    with torch.no_grad():
        o = m8(torch.randn(2, m8.fine_channels, 600) * 0.3, torch.randn(2, _I.WIDE_CHANNELS, 120) * 0.3)
    fin = all(bool(torch.isfinite(v).all()) for v in o.values()
              if isinstance(v, torch.Tensor))
    ident = all(torch.equal(m7.state_dict()[k], m8.state_dict()[k])
                for k in m7.state_dict())
ok(7, "평균판이 저장·재적재되고 유한한 출력", fin and ident,
   "출력 %d 종 · 재적재 동일 %s · 평균 %d 점" % (len(o), ident, a7.n))

# ── [8] 체크포인트 키 + 배선 ─────────────────────────────────────
need = ("swa_start", "swa_lr", "swa_every", "swa_n")
have = [k for k in need if k in ck]
src = Path("src/run_train_cnn.py").read_text(encoding="utf-8")
wired = all('"%s":' % k in src for k in need)
swan = 'swa.state_dict(model)' in src and '_last.pt' in src
# 순서가 중요하다 — 끝점 저장이 **평균을 싣기 전에** 있어야 한다
i_last, i_load = src.find('_last.pt'), src.find('load_state_dict(swa.state_dict')
ok(8, "키 넷 + 배선 + 끝점 대조를 평균 적재 **전에** 저장",
   len(have) == 4 and wired and swan and 0 < i_last < i_load,
   "키 %d/4 · 배선 %s · 끝점저장 %d < 평균적재 %d" % (len(have), wired, i_last, i_load))

print("=" * 84)
print("실패 %d 개%s" % (len(FAIL), (" — " + str(FAIL)) if FAIL else ""))
raise SystemExit(1 if FAIL else 0)
