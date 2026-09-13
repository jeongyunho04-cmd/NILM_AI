# -*- coding: utf-8 -*-
"""언제 뒤집혔나 — 판마다 **똑같은 주입 쌍**으로 훑는다 (13.83.17 후속).

13.83.17 이 남긴 것:
    cnn_v24b (1단계)          Δ원시로짓 **-0.67**  cos(Δz실측,Δz합성) **+0.152**
    adapt_v24b_site_s0 (2단계)          **-6.32**                   **-0.286**
    cnn_v28  (1단계)                    **-5.28**                   **-0.129**
1단계가 v24b 에서 v28 사이에 나빠졌다. 그 사이 판을 같은 자로 훑어 어디서인지 찾는다.

⚠ 공정하게 하려면 **주입 쌍이 판마다 같아야 한다.** 배경 창도, 더하는 미니PC
   조각도, `build_inputs` 결과도 모델과 무관하므로 **한 번만 만들어** 재사용한다.
   모델은 그 위를 지나가기만 한다.

내는 값 (전부 미니PC 열):
    배경 게이트 / 주입 후 게이트          실측·합성
    Δ원시로짓 (프라이어 뺀 값)            + 면 정상, − 면 뒤집힘
    cos(Δz실측, Δz합성)                  몸통이 같은 물리를 같은 방향으로 읽나
    w·(z̄실측 − z̄합성)                    영역 이동이 헤드 방향으로 몇 로짓인가
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

MODELS = sys.argv[1:] or ["cnn_v22", "cnn_v23", "cnn_v24a", "cnn_v24b", "cnn_v25",
                          "cnn_v26", "cnn_v27", "cnn_v28"]
HD = "processed_data/holdout60_v22"
FILES = ["test_1", "test_2", "test_3", "test_4"]
SIB = ["laptop_charger", "beam_projector"]
W, STRIDE = WINDOW_CYCLES, 60
rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

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


# ── 배경 만들기 (모델 무관, 한 번만) ───────────────────────────────────────
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
ah = json.load(open(HD + "/meta.json", encoding="utf-8"))["appliances"]
Jh, KC, KP = ah.index("minipc"), ah.index("laptop_charger"), ah.index("beam_projector")
cand = np.flatnonzero(((Yh[:, KC] == 1) | (Yh[:, KP] == 1)) & (Yh[:, Jh] == 0))
php = np.array([float(np.median(Xh[i, 30])) for i in cand])
lo, hi = np.percentile(bgp, 10), np.percentile(bgp, 90)
mt = cand[(php >= lo) & (php <= hi)][:258]
sbg = [np.asarray(Xh[i], np.float32) for i in mt]
print("실측 배경 %d창 (총전력 중앙 %.0fW) · 합성 전력맞춤 배경 %d창 (%.0fW)"
      % (len(bg), np.median(bgp), len(sbg), np.median(php[(php >= lo) & (php <= hi)][:258])))

picks_r = [MP[rng.integers(len(MP))] for _ in bg]
picks_s = [MP[rng.integers(len(MP))] for _ in sbg]


def pack(base, picks):
    fA, wA = build_inputs(np.stack(base))
    fB, wB = build_inputs(np.stack([add(c, p) for c, p in zip(base, picks)]))
    return (torch.from_numpy(fA), torch.from_numpy(wA),
            torch.from_numpy(fB), torch.from_numpy(wB))


RE, SY = pack(bg, picks_r), pack(sbg, picks_s)
print("주입 쌍 고정 완료 — 판마다 **같은 입력**을 쓴다\n")


def evaluate(name):
    model, apps, ck = load_model("results/%s.pt" % name, dev)
    J = apps.index("minipc")
    w = model.heads[J].weight[model.i_on].detach().float().cpu().numpy()
    kappa = float(model.prior_kappa)
    buf = []
    h = model.trunk.register_forward_hook(
        lambda m, i, o: buf.append(o.detach().float().cpu().numpy()))

    def go(P):
        out = []
        for f, wd in ((P[0], P[1]), (P[2], P[3])):
            buf.clear()
            L, G = [], []
            for i in range(0, len(f), 32):
                ft, wt = f[i:i + 32].to(dev), wd[i:i + 32].to(dev)
                with torch.no_grad():
                    o = model(ft, wt)
                p_max = torch.maximum(ft[:, P_CH_FINE].amax(-1), wt[:, P_CH_WIDE].amax(-1))
                pri = (F.logsigmoid(kappa * (p_max[:, None]
                                             - model.on_threshold_asinh[None]))[:, J]
                       if kappa > 0 else torch.zeros(len(ft), device=dev))
                L.append((o["on_logit"][:, J] - pri).float().cpu().numpy())
                G.append(torch.sigmoid(o["on_logit"][:, J]).float().cpu().numpy())
            out.append((np.concatenate(buf), np.concatenate(L), np.concatenate(G)))
        return out

    (zrA, lrA, grA), (zrB, lrB, grB) = go(RE)
    (zsA, lsA, gsA), (zsB, lsB, gsB) = go(SY)
    h.remove()
    dr, ds = (zrB - zrA).mean(0), (zsB - zsA).mean(0)
    shift = zrA.mean(0) - zsA.mean(0)
    cs = float(dr @ ds / (np.linalg.norm(dr) * np.linalg.norm(ds) + 1e-12))
    return dict(gr0=np.median(grA), gr1=np.median(grB), dlr=np.median(lrB - lrA),
                gs0=np.median(gsA), gs1=np.median(gsB), dls=np.median(lsB - lsA),
                cos=cs, shift=float(w @ shift), lr0=np.median(lrA), ls0=np.median(lsA))


print("  %-20s %17s %17s %8s %9s" % ("모델", "실측 배경(게이트 · Δ로짓)",
                                     "합성 배경(게이트 · Δ로짓)", "cos", "w·이동"))
for m in MODELS:
    try:
        r = evaluate(m)
    except Exception as e:
        print("  %-20s  건너뜀 (%s)" % (m, str(e)[:60]))
        continue
    print("  %-20s %.3f->%.3f %+7.2f  %.3f->%.3f %+7.2f %8.3f %+9.2f"
          % (m, r["gr0"], r["gr1"], r["dlr"], r["gs0"], r["gs1"], r["dls"],
             r["cos"], r["shift"]))
