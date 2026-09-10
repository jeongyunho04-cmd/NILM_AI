# -*- coding: utf-8 -*-
"""영역인가 **동작점**인가 — 배경 전력을 맞춰 놓고 같은 주입을 다시.

`cell5.py` 가 낸 것 (같은 실측 미니PC 조각, 같은 헤드):
```
합성 배경 + 미니PC   게이트 0.003 -> 0.995   Δ원시로짓 **+10.16**   전력 0.0 -> 9.2W
실측 배경 + 미니PC   게이트 0.975 -> 0.191   Δ원시로짓  **-5.28**   전력 10.5 -> 1.5W
cos(Δz실측, Δz합성) **-0.129**       w·Δz  합성 +9.79 / 실측 -5.19
영역 이동 w·(z̄실측−z̄합성) **+9.18 로짓** (배경 로짓 실측 +3.67 대 합성 -5.69)
```
헤드는 **자기가 배운 것에는 완벽하다.** 문제는 몸통이 같은 물리적 주입을 두
배경에서 거의 반대 방향으로 인코딩한다는 것이다.

⚠ 그런데 두 배경은 **총전력이 다를 수 있다.** 실측 형제-ON·미니PC-OFF 배경은
   300W 대이고 합성 무리는 훨씬 낮을 수 있다. 입력이 asinh 라 같은 16W 라도
   배경이 크면 상대 변화가 작다. 그러면 원인은 '실측 대 합성' 이 아니라
   **동작점(배경 크기)** 이다 — 처방이 완전히 다르다.

가른다:
  ① 두 배경의 총전력 분포를 찍는다
  ② 합성에서 **실측과 같은 전력대**의 배경만 골라 같은 주입을 한다
  ③ 합성 학습 분포에 '배경 큰데 미니PC ON' 창이 있기는 한가 (커버리지)
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
KC, KP = apps.index("laptop_charger"), apps.index("beam_projector")
w_head = model.heads[J].weight[model.i_on].detach().float().cpu().numpy()
b_head = float(model.heads[J].bias[model.i_on].detach())
kappa = float(model.prior_kappa)
_Z = {}
model.trunk.register_forward_hook(
    lambda m, i, o: _Z.__setitem__("z", o.detach().float().cpu().numpy()))
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
MP = []
for a in pool.appliance_activations.get("minipc", []):
    hr = np.asarray(a.net_harmonics_ri, np.float32)
    pf = np.asarray(a.net_power_features, np.float32)
    for i in range(0, len(hr) - W + 1, W // 4):
        MP.append((hr[i:i + W], pf[i:i + W]))


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
    fine, wide = build_inputs(np.stack(cuts))
    ft, wt = torch.from_numpy(fine).to(dev), torch.from_numpy(wide).to(dev)
    o = model(ft, wt)
    p_max = torch.maximum(ft[:, P_CH_FINE].amax(-1), wt[:, P_CH_WIDE].amax(-1))
    pri = (F.logsigmoid(kappa * (p_max[:, None] - model.on_threshold_asinh[None]))[:, J]
           if kappa > 0 else torch.zeros(len(cuts), device=dev))
    return (_Z["z"], (o["on_logit"][:, J] - pri).float().cpu().numpy(),
            torch.sigmoid(o["on_logit"][:, J]).float().cpu().numpy(),
            o["power"][:, J].float().cpu().numpy())


def run(base, nm):
    dz, dl, gA, gB, pA, pB = [], [], [], [], [], []
    for i in range(0, len(base), 16):
        blk = base[i:i + 16]
        zA, lA, sA, wA = fwd(blk)
        zB, lB, sB, wB = fwd([add(c, MP[rng.integers(len(MP))]) for c in blk])
        dz.append(zB - zA); dl.append(lB - lA)
        gA.append(sA); gB.append(sB); pA.append(wA); pB.append(wB)
    dz = np.concatenate(dz); dl = np.concatenate(dl)
    gA, gB = np.concatenate(gA), np.concatenate(gB)
    pA, pB = np.concatenate(pA), np.concatenate(pB)
    d = dz.mean(0)
    print("  %-24s %4d %9.3f %9.3f %10.2f %9.2f %6.1f->%5.1f"
          % (nm, len(base), np.median(gA), np.median(gB), np.median(dl),
             w_head @ d, np.median(pA), np.median(pB)))
    return d


# ── 실측 배경 ──────────────────────────────────────────────────────────────
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
bg, bgp = [], []
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
            bg.append(x[:, sl]); bgp.append(float(np.median(x[30, sl])))
bgp = np.asarray(bgp)

Xh = np.load(HD + "/X.npy", mmap_mode="r")
Yh = np.load(HD + "/y_on.npy")
sib_h = (Yh[:, KC] == 1) | (Yh[:, KP] == 1)
off_h = Yh[:, J] == 0
cand = np.flatnonzero(sib_h & off_h)
php = np.array([float(np.median(Xh[i, 30])) for i in cand])

print("=== ① 배경 총전력 분포 (W, 창 중앙값) ===")
q = [5, 25, 50, 75, 95]
print("  실측 배경 %3d창  %s" % (len(bg), np.round(np.percentile(bgp, q), 1)))
print("  합성 후보 %3d창  %s" % (len(cand), np.round(np.percentile(php, q), 1)))
lo, hi = np.percentile(bgp, 10), np.percentile(bgp, 90)
mt = cand[(php >= lo) & (php <= hi)]
print("  실측 10~90%% 대역 %.0f~%.0fW 에 든 합성 배경 **%d창** (%.1f%%)"
      % (lo, hi, len(mt), 100 * len(mt) / len(cand)))

print("\n=== ② 같은 주입, 배경만 바꿔서 ===")
print("  %-24s %4s %9s %9s %10s %9s %11s"
      % ("배경", "창", "게이트전", "게이트후", "Δ원시로짓", "w·Δz", "미니PC예측W"))
d_real = run(bg, "실측 (형제ON·미니PC없음)")
d_syn = run([np.asarray(Xh[i], np.float32) for i in cand[:258]], "합성 (같은 규칙, 전부)")
d_mt = None
if len(mt) >= 32:
    d_mt = run([np.asarray(Xh[i], np.float32) for i in mt[:258]],
               "합성 (**전력 맞춤**)")
else:
    print("  %-24s %4d  (전력 맞춘 합성 배경이 거의 없다)" % ("합성 (전력 맞춤)", len(mt)))

# 실측 배경 중 전력이 낮은 쪽만 따로 (동작점 축을 실측 안에서도 본다)
if (bgp < np.median(bgp)).sum() >= 32:
    run([b for b, p in zip(bg, bgp) if p < np.median(bgp)], "실측 (저전력 절반)")
    run([b for b, p in zip(bg, bgp) if p >= np.median(bgp)], "실측 (고전력 절반)")


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


print("\n  cos(Δz실측, Δz합성전부) %+.3f%s"
      % (cos(d_real, d_syn),
         " · cos(Δz실측, Δz합성전력맞춤) %+.3f" % cos(d_real, d_mt) if d_mt is not None else ""))

print("\n=== ③ 커버리지 — 합성 학습 분포에 '배경 큰데 미니PC ON' 창이 있나 ===")
on_h = np.flatnonzero(sib_h & (Yh[:, J] == 1))
onp = np.array([float(np.median(Xh[i, 30])) for i in on_h[:3000]])
print("  합성 미니PC ON·형제ON 창 %d개 · 총전력 %s"
      % (len(on_h), np.round(np.percentile(onp, q), 1)))
for t in (100, 200, 300, 500):
    print("    총전력 > %4dW 인 몫 : 미니PC ON %5.1f%% · 미니PC OFF %5.1f%%"
          % (t, 100 * (onp > t).mean(), 100 * (php > t).mean()))
print("  실측 배경(미니PC OFF·형제ON) 중 > 300W 인 몫 %.1f%%" % (100 * (bgp > 300).mean()))
