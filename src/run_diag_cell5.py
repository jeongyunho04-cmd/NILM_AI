# -*- coding: utf-8 -*-
"""헤드는 **어느 방향을 배웠나** — 뒤집힘의 기전을 가른다 (13.83.16 후속).

13.83.16: 실측 형제-ON 배경에 미니PC 를 주입하면 몸통 z probe 1.000 인데 게이트는
0.975 -> 0.236 으로 내려간다. 게이트 헤드도 z 위의 **선형** 사상이다
(`heads[J].weight[i_on]`, 256차원). 같은 z 위 같은 선형족인데 부호가 반대면
문제는 용량이 아니라 **방향**이다.

가르는 것:
  ① 로짓 분해   원시 로짓(w·z+b) 과 물리 프라이어를 갈라 어느 쪽이 내리는가
  ② 방향        cos(w_head, Δz_실측) · cos(w_head, w_probe) · |Δz|
  ③ 합성 대조   **같은 주입을 합성 배경에** 해서 Δz_합성 을 낸다.
                w·Δz_합성 > 0 이고 w·Δz_실측 < 0 이면 헤드는 자기가 배운 것에는
                맞고, **몸통이 두 영역에서 미니PC 를 다르게 인코딩**하는 것이다.
                cos(Δz_실측, Δz_합성) 이 그 값을 직접 준다.
  ④ 영역 이동   w·(z̄_실측배경 − z̄_합성배경) — 배경만으로 로짓이 얼마나 밀리나
  ⑤ 무엇을 켜나 같은 배경에 **충전기**·**저항성**을 대신 주입해 미니PC 게이트가
                어떻게 움직이는가. 오르면 헤드 방향은 '그 기기'가 아니라
                '그 전류/전력' 을 향한 것이다
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
import torch.nn.functional as F

from src.model.inputs import build_inputs
from src.model.net import P_CH_FINE, P_CH_WIDE
from src.model.realdata import RealWindows, WINDOW_CYCLES
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.synthesis.segment_pool import SegmentPool

CKPT = sys.argv[1] if len(sys.argv) > 1 else "results/cnn_v28.pt"
HD = "processed_data/holdout60_v22"
FILES = ["test_1", "test_2", "test_3", "test_4"]
SIB = ["laptop_charger", "beam_projector"]
W, STRIDE = WINDOW_CYCLES, 60
rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, apps, ck = load_model(CKPT, dev)
J = apps.index("minipc")
i_on = model.i_on
w_head = model.heads[J].weight[i_on].detach().float().cpu().numpy()
b_head = float(model.heads[J].bias[i_on].detach())
kappa = float(model.prior_kappa)
print("모델 %s · |w_head| %.3f · b %.3f · prior_kappa %.3g"
      % (CKPT, np.linalg.norm(w_head), b_head, kappa))

_Z = {}
model.trunk.register_forward_hook(
    lambda m, i, o: _Z.__setitem__("z", o.detach().float().cpu().numpy()))
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))


def pieces(app):
    out = []
    for a in pool.appliance_activations.get(app, []):
        hr = np.asarray(a.net_harmonics_ri, np.float32)
        pf = np.asarray(a.net_power_features, np.float32)
        for i in range(0, len(hr) - W + 1, W // 4):
            out.append((hr[i:i + W], pf[i:i + W]))
    return out


SEG = {a: pieces(a) for a in ("minipc", "laptop_charger", "hair_dryer", "fan")}
print("주입 조각 — %s" % {k: len(v) for k, v in SEG.items()})


def add(cut, seg):
    hr, pf = seg
    o = cut.copy()
    o[0:15] += hr[:, :, 0].T
    o[15:30] += hr[:, :, 1].T
    o[30] += pf[:, 0]
    o[31] += pf[:, 1]
    return o


@torch.no_grad()
def fwd(cuts):
    """z · 원시로짓(프라이어 전) · 프라이어 · 게이트 · 미니PC 전력."""
    fine, wide = build_inputs(np.stack(cuts))
    ft = torch.from_numpy(fine).to(dev)
    wt = torch.from_numpy(wide).to(dev)
    o = model(ft, wt)
    p_max = torch.maximum(ft[:, P_CH_FINE].amax(-1), wt[:, P_CH_WIDE].amax(-1))
    pri = (F.logsigmoid(kappa * (p_max[:, None] - model.on_threshold_asinh[None]))
           if kappa > 0 else torch.zeros(len(cuts), len(apps), device=dev))
    lg = o["on_logit"][:, J].float().cpu().numpy()
    return (_Z["z"], lg, pri[:, J].float().cpu().numpy(),
            torch.sigmoid(o["on_logit"][:, J]).float().cpu().numpy(),
            o["power"][:, J].float().cpu().numpy())


# ── 실측 배경 ──────────────────────────────────────────────────────────────
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
bg = []
for stem in FILES:
    spec = ev.get(stem)
    if spec is None or "minipc" not in spec.get("appliances_present", []):
        continue
    raw = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
    x = RealWindows._to_33ch(raw)
    iv = np.asarray(raw["is_valid"]).astype(bool)
    n = x.shape[1]

    def mk(app, key="on"):
        m = np.zeros(n, bool)
        for t0, t1 in spec["intervals"].get(app, {}).get(key, []):
            m[int(t0 * 60):int(min(t1 * 60, n))] = True
        return m

    y, unc = mk("minipc"), mk("minipc", "uncertain")
    s = np.zeros(n, bool)
    for a in SIB:
        if a in spec.get("appliances_present", []):
            s |= mk(a)
    for t0 in range(0, n - W + 1, STRIDE):
        sl = slice(t0, t0 + W)
        if iv[sl].all() and not y[sl].any() and not unc[sl].any() \
                and s[sl].std() == 0 and s[t0]:
            bg.append(x[:, sl])
print("실측 배경(형제ON · 미니PC 60초 내내 OFF) %d창" % len(bg))

# ── 합성 배경 (같은 무리 규칙: 미니PC OFF · 형제 SMPS ON) ──────────────────
Xh = np.load(HD + "/X.npy", mmap_mode="r")
Yh = np.load(HD + "/y_on.npy")
sel = (Yh[:, J] == 0) & ((Yh[:, apps.index("laptop_charger")] == 1)
                         | (Yh[:, apps.index("beam_projector")] == 1))
idx = np.flatnonzero(sel)[:len(bg) * 2]
sbg = [np.asarray(Xh[i], np.float32) for i in idx]
print("합성 배경(같은 규칙) %d창 / 후보 %d" % (len(sbg), int(sel.sum())))


def sweep(base, app, tag):
    """base 배경에 app 을 주입해 z·로짓 변화를 낸다."""
    dz, dlg, dpri, gA, gB, pA, pB = [], [], [], [], [], [], []
    segs = SEG[app]
    for i in range(0, len(base), 16):
        blk = base[i:i + 16]
        zA, lA, rA, sA, wA = fwd(blk)
        zB, lB, rB, sB, wB = fwd([add(c, segs[rng.integers(len(segs))]) for c in blk])
        dz.append(zB - zA); dlg.append((lB - rB) - (lA - rA)); dpri.append(rB - rA)
        gA.append(sA); gB.append(sB); pA.append(wA); pB.append(wB)
    return (np.concatenate(dz), np.concatenate(dlg), np.concatenate(dpri),
            np.concatenate(gA), np.concatenate(gB), np.concatenate(pA), np.concatenate(pB))


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


print("\n=== ①⑤ 같은 실측 배경에 무엇을 주입하면 미니PC 게이트가 어떻게 되나 ===")
print("  %-16s %10s %10s %10s %11s %11s"
      % ("주입", "게이트 전", "게이트 후", "Δ원시로짓", "Δ프라이어", "미니PC예측W"))
DR = {}
for app in ("minipc", "laptop_charger", "hair_dryer", "fan"):
    dz, dlg, dpri, gA, gB, pA, pB = sweep(bg, app, "real")
    DR[app] = dz.mean(0)
    print("  %-16s %10.3f %10.3f %10.2f %11.2f %6.1f->%5.1f"
          % (app, np.median(gA), np.median(gB), np.median(dlg), np.median(dpri),
             np.median(pA), np.median(pB)))

print("\n=== ③ 합성 배경에 같은 미니PC 를 주입하면 ===")
dzs, dlgs, dpris, gAs, gBs, pAs, pBs = sweep(sbg, "minipc", "syn")
print("  %-16s %10.3f %10.3f %10.2f %11.2f %6.1f->%5.1f"
      % ("minipc(합성배경)", np.median(gAs), np.median(gBs), np.median(dlgs),
         np.median(dpris), np.median(pAs), np.median(pBs)))

d_real, d_syn = DR["minipc"], dzs.mean(0)
print("\n=== ② 방향 ===")
print("  |Δz| 실측 %.3f · 합성 %.3f · cos(Δz실측, Δz합성) **%.3f**"
      % (np.linalg.norm(d_real), np.linalg.norm(d_syn), cos(d_real, d_syn)))
print("  w_head · Δz : 실측 **%+.3f** · 합성 **%+.3f**"
      % (w_head @ d_real, w_head @ d_syn))
print("  cos(w_head, Δz) : 실측 %+.3f · 합성 %+.3f"
      % (cos(w_head, d_real), cos(w_head, d_syn)))
for app in ("laptop_charger", "hair_dryer", "fan"):
    print("  cos(w_head, Δz_%s) %+.3f · cos(Δz_minipc실측, Δz_%s) %+.3f"
          % (app, cos(w_head, DR[app]), app, cos(d_real, DR[app])))

print("\n=== ④ 영역 이동 ===")
zr = np.concatenate([fwd(bg[i:i + 16])[0] for i in range(0, len(bg), 16)])
zsn = np.concatenate([fwd(sbg[i:i + 16])[0] for i in range(0, len(sbg), 16)])
shift = zr.mean(0) - zsn.mean(0)
print("  |z̄실측 − z̄합성| %.2f · w_head·이동 **%+.2f** (로짓 단위)"
      % (np.linalg.norm(shift), w_head @ shift))
print("  참고 — 배경 로짓 중앙: 실측 %+.2f · 합성 %+.2f"
      % (np.median(zr @ w_head + b_head), np.median(zsn @ w_head + b_head)))
