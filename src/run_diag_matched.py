# -*- coding: utf-8 -*-
"""**같은 구성·같은 자리·창 내내 정상상태**인 합성 창을 합성기로 직접 만들어 실측 배경과 맞댄다 (13.83.22).

`run_diag_cell6`·`run_diag_trace` 의 "전력 맞춤" 합성 배경은 10~90% 대역 필터라 저전력 합성 창이
압도해 실제로는 중앙 66W 였다 (실측 472W). 여기서는 `synthesize_random_window(force_active=...,
full_window_placement=True)` 로 구성을 지정하고 자리 D(기저 전압 < 222V)·충전기 전력대·총전력을
실측 분포에 맞춰 걸러 낸다.

    python -X utf8 src/run_diag_matched.py float cnn_v29 cnn_v25     # test_1 부동 충전기 (13.83.22 ①)
    python -X utf8 src/run_diag_matched.py pair  cnn_v29 cnn_v25     # test_3/4 충전기+프로젝터 (13.83.22 ⑥)

내는 것:
  ① 실측 / 합성 각각에서 모델이 부르는 것 (미니PC·충전기 게이트와 전력) + 미니PC 주입 Δ로짓
  ② 차수별 |I_h|·절대 각도 (총합 · 합성 SMPS 정답 부분 · 합성 저항 부분 · 실측−합성저항)
  ③ 원시 수준 이식 — 합성에 실측 고조파(전부/h3-7/h9-15/크기만/위상만/전압)를 옮겨 심고 반대로도
  ④ (float 만) 짝 맞춘 IG 와 차수별 귀속

⚠ 합성기는 충전기를 28W 아래로 못 놓는다 (풀의 12~20W 구간이 36초, 연속 최장 1초). 그래서
   float 모드의 합성 충전기는 28~66W 다 — 그것이 바로 13.83.22 의 결론이다.
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

from src.model.inputs import build_inputs, VOLT_ORDERS
from src.model.net import P_CH_FINE, P_CH_WIDE
from src.model.realdata import RealWindows, WINDOW_CYCLES
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

MODE = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("float", "pair") else "float")
MODELS = [a for a in sys.argv[1:] if a not in ("float", "pair")] or ["cnn_v29", "cnn_v25"]
FILES = ["test_1", "test_2", "test_3", "test_4"] if MODE == "float" else ["test_3", "test_4"]
SIB = ["laptop_charger", "beam_projector"]
W, STRIDE = WINDOW_CYCLES, 60
N_SYN_TARGET = 129 if MODE == "float" else 60
rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ── 실측 배경 ──────────────────────────────────────────────────────────────
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
bg, bgp, bg_on = [], [], []
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

    on = {a: mk(a) for a in spec["appliances_present"]}
    y, unc = mk("minipc"), mk("minipc", "uncertain")
    s = np.zeros(n, bool)
    for a in SIB:
        if a in on:
            s |= on[a]
    for t0 in range(0, n - W + 1, STRIDE):
        sl = slice(t0, t0 + W)
        if iv[sl].all() and not y[sl].any() and not unc[sl].any() \
                and s[sl].std() == 0 and s[t0]:
            bg.append(x[:, sl]); bgp.append(float(np.median(x[30, sl])))
            bg_on.append({a: float(on[a][sl].mean()) for a in on})
bg = np.stack(bg); bgp = np.asarray(bgp)
print("실측 배경 %d창  P p10/50/90 %s  핫플 %.2f · 오븐 %.2f · 충전기 %.2f · 프로젝터 %.2f (ON 몫)"
      % (len(bg), np.round(np.percentile(bgp, [10, 50, 90])),
         np.mean([d.get("hotplate", 0) for d in bg_on]), np.mean([d.get("oven", 0) for d in bg_on]),
         np.mean([d.get("laptop_charger", 0) for d in bg_on]), np.mean([d.get("beam_projector", 0) for d in bg_on])))
# 현장 충전기 전력 추정 — |I3| 로 (충전기 ~4.2 mA/W, 핫플 h3 ~11 mA)
i3 = 1e3 * np.median(np.abs(bg[:, 2] + 1j * bg[:, 17]), axis=1)
p_chg_est = np.clip((i3 - 11.0) / 4.2, 0, None)
print("  현장 충전기 전력 추정(|I3|) p10/50/90 %s W" % np.round(np.percentile(p_chg_est, [10, 50, 90]), 1))

# ── 합성 (같은 구성 · 자리 D · 정상상태) ──────────────────────────────────
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
syn_gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=True)
np.random.seed(0)
syn, syn_meta, syn_smps = [], [], []
tries = 0
lo_p, hi_p = np.percentile(bgp, 5), np.percentile(bgp, 95)
lo_c, hi_c = ((np.percentile(p_chg_est, 5), np.percentile(p_chg_est, 95)) if MODE == "float" else (40.0, 66.0))
while len(syn) < N_SYN_TARGET and tries < 4000:
    tries += 1
    force = ["hotplate", "laptop_charger"]
    if MODE == "pair" and np.random.rand() < 0.7:
        force.append("beam_projector")
    if np.random.rand() < 0.5:
        force.append("oven")
    try:
        smp = syn_gen.synthesize_random_window(
            window_size_cycles=W, force_active=force, full_window_placement=True,
            compute_gt_harmonics=True, target_lookahead_cycles=360, sustained_power_limit_w=None)
    except Exception:
        continue
    if smp.metadata.get("base_voltage_v", 999) > 222.0:      # 자리 D 만
        continue
    if not smp.gt_is_on["laptop_charger"].all():
        continue
    hp = smp.gt_is_on["hotplate"]
    if not (hp[:60].all() and hp[-60:].all()):               # 핫플 듀티 라벨 결함(13.82) — 양 끝만
        continue
    if "minipc" in smp.active_appliances or (MODE == "float" and "beam_projector" in smp.active_appliances):
        continue
    pc = float(np.median(smp.gt_target_power_w["laptop_charger"]))
    pt = float(np.median(smp.power_features[:, 0]))
    if not (lo_c <= pc <= hi_c and lo_p <= pt <= hi_p):
        continue
    r_part = smp.harmonics_ri[:, :, 0].T; i_part = smp.harmonics_ri[:, :, 1].T
    vh = np.asarray(smp.voltage_harmonics_complex)[:, [h - 1 for h in VOLT_ORDERS]]
    X = np.concatenate([r_part, i_part, smp.power_features[:, 0:1].T, smp.power_features[:, 1:2].T,
                        smp.power_features[:, 4:5].T, vh.real.T, vh.imag.T], axis=0).astype(np.float32)
    g = np.asarray(smp.gt_harmonics_ri["laptop_charger"])
    if "beam_projector" in smp.active_appliances:
        g = g + np.asarray(smp.gt_harmonics_ri["beam_projector"])
    syn.append(X); syn_smps.append(g)
    syn_meta.append(dict(pc=pc, pt=pt, v=smp.metadata.get("base_voltage_v"), act=list(smp.active_appliances)))
syn = np.stack(syn); syn_smps = np.stack(syn_smps)
print("합성 %d창 (시도 %d)  P p10/50/90 %s  충전기 P p10/50/90 %s  오븐 %.2f · 프로젝터 %.2f  기저V 중앙 %.1f"
      % (len(syn), tries, np.round(np.percentile([m["pt"] for m in syn_meta], [10, 50, 90])),
         np.round(np.percentile([m["pc"] for m in syn_meta], [10, 50, 90]), 1),
         np.mean(["oven" in m["act"] for m in syn_meta]), np.mean(["beam_projector" in m["act"] for m in syn_meta]),
         np.median([m["v"] for m in syn_meta])))
n = min(len(bg), len(syn))
bg, syn, syn_smps = bg[:n], syn[:n], syn_smps[:n]
np.savez("results/_matched_%s.npz" % MODE, real=bg, syn=syn, syn_smps=syn_smps)


def ph(X):
    return X[:, 0:15] + 1j * X[:, 15:30]


def pstat(c, nm):
    mag = 1e3 * np.median(np.abs(c), axis=2)
    ang = np.rad2deg(np.angle(np.median(c.real, axis=2) + 1j * np.median(c.imag, axis=2)))
    o = [0, 2, 4, 6, 8, 10, 12, 14]
    print("  %-22s |I_h| mA %s" % (nm, np.round(np.median(mag, 0)[o], 1)))
    print("  %-22s 각도 °   %s   (IQR %s)" % ("", np.round(np.median(ang, 0)[o]).astype(int),
                                            np.round(np.percentile(ang, 75, 0)[o] - np.percentile(ang, 25, 0)[o]).astype(int)))


print("\n=== 차수별 페이저 h=1,3,…,15 (창 중앙 페이저 -> 창 사이 중앙) ===")
smps = syn_smps[:, :, :, 0].transpose(0, 2, 1) + 1j * syn_smps[:, :, :, 1].transpose(0, 2, 1)
pstat(ph(bg), "실측 총합")
pstat(ph(syn), "합성 총합")
pstat(smps, "합성 SMPS 정답 부분")
res_syn = ph(syn) - smps
pstat(res_syn, "합성 저항 부분")
pstat(ph(bg) - np.median(res_syn, axis=0, keepdims=True), "실측 − 합성 저항")
sg = lambda X: np.round(np.median(1e3 * np.std(np.abs(ph(X)), axis=2), 0)[[2, 4, 6, 8, 12]], 1)
print("  창 안 σ|I_h| mA h3,5,7,9,13  실측 %s  합성 %s" % (sg(bg), sg(syn)))

# ── 주입·이식 ──────────────────────────────────────────────────────────────
MP = []
for a in pool.appliance_activations.get("minipc", []):
    hr = np.asarray(a.net_harmonics_ri, np.float32); pf = np.asarray(a.net_power_features, np.float32)
    for i in range(0, len(hr) - W + 1, W // 4):
        MP.append((hr[i:i + W], pf[i:i + W]))
picks = [MP[rng.integers(len(MP))] for _ in range(n)]


def inject(X):
    o = X.copy()
    for i, (hr, pf) in enumerate(picks):
        o[i, 0:15] += hr[:, :, 0].T; o[i, 15:30] += hr[:, :, 1].T
        o[i, 30] += pf[:, 0]; o[i, 31] += pf[:, 1]
    return o


def swap(dst, src, idx):
    o = dst.copy()
    for k in idx:
        o[:, k] = src[:, k]; o[:, 15 + k] = src[:, 15 + k]
    return o


def set_mag(X, ref, idx):
    c = ph(X); r = ph(ref); o = c.copy()
    for k in idx:
        s = np.median(np.abs(r[:, k]), 1) / (np.median(np.abs(c[:, k]), 1) + 1e-9)
        o[:, k] = c[:, k] * s[:, None]
    out = X.copy(); out[:, 0:15] = o.real; out[:, 15:30] = o.imag
    return out


def set_arg(X, ref, idx):
    c = ph(X); r = ph(ref); o = c.copy()
    for k in idx:
        ar = np.angle(np.median(r[:, k].real, 1) + 1j * np.median(r[:, k].imag, 1))
        ac = np.angle(np.median(c[:, k].real, 1) + 1j * np.median(c[:, k].imag, 1))
        o[:, k] = c[:, k] * np.exp(1j * (ar - ac))[:, None]
    out = X.copy(); out[:, 0:15] = o.real; out[:, 15:30] = o.imag
    return out


def swap_volt(dst, src):
    o = dst.copy(); o[:, 32:45] = src[:, 32:45]; return o


h_all = list(range(1, 15)); h_low = [2, 4, 6]; h_hi = [8, 10, 12, 14]; h_odd = [2, 4, 6, 8, 10, 12, 14]
VARIANTS = [
    ("실측", bg), ("합성 (맞춤)", syn),
    ("합성 + 실측 I_h h2..15", swap(syn, bg, h_all)),
    ("합성 + 실측 I_h h3,5,7", swap(syn, bg, h_low)),
    ("합성 + 실측 I_h h9..15", swap(syn, bg, h_hi)),
    ("합성 + 실측 |I_h| h3..15 (크기만)", set_mag(syn, bg, h_odd)),
    ("합성 + 실측 arg I_h h3..15 (위상만)", set_arg(syn, bg, h_odd)),
    ("합성 + 실측 arg I_h h3,5,7", set_arg(syn, bg, h_low)),
    ("합성 + 실측 arg I_h h9..15", set_arg(syn, bg, h_hi)),
    ("합성 + 실측 V,Vh", swap_volt(syn, bg)),
    ("실측 + 합성 I_h h2..15", swap(bg, syn, h_all)),
    ("실측 + 합성 I_h h3,5,7", swap(bg, syn, h_low)),
    ("실측 + 합성 I_h h9..15", swap(bg, syn, h_hi)),
    ("실측 + 합성 |I_h| h3..15 (크기만)", set_mag(bg, syn, h_odd)),
    ("실측 + 합성 arg I_h h3..15 (위상만)", set_arg(bg, syn, h_odd)),
    ("실측 + 합성 arg I_h h9..15", set_arg(bg, syn, h_hi)),
    ("실측 + 합성 V,Vh", swap_volt(bg, syn)),
]
FG = {"홀수 Re/Im h1": [0, 8], "홀수 Re/Im h3-7": [1, 2, 3, 9, 10, 11], "홀수 Re/Im h9-15": [4, 5, 6, 7, 12, 13, 14, 15],
      "짝수 |I|": list(range(16, 23)), "P,Q,V": [23, 24, 25], "비": [26, 27, 28], "P 리플": [29, 30],
      "phi": list(range(31, 39)), "PF,I9/I3": [39, 40], "다단강하": [41, 42], "반파,|I2|": [43, 44],
      "Vh1": [45, 51], "Vh3-11": [46, 47, 48, 49, 50, 52, 53, 54, 55, 56]}
WG = {"P,Q,V": [0, 1, 2], "|I1,3,5,2|": [3, 4, 5, 6], "p_std": [7], "비": [8, 9], "p_dev": [10], "step": [11],
      "|I_h|": list(range(12, 27)), "phi": list(range(27, 35)), "Vh1": [35, 41], "Vh3-11": [36, 37, 38, 39, 40, 42, 43, 44, 45, 46]}


def run(name):
    model, apps, ck = load_model("results/%s.pt" % name, dev)
    J, KC = apps.index("minipc"), apps.index("laptop_charger")
    kappa = float(model.prior_kappa); thr = model.on_threshold_asinh

    def L(ft, wt, j=J):
        o = model(ft, wt)
        p_max = torch.maximum(ft[:, P_CH_FINE].amax(-1), wt[:, P_CH_WIDE].amax(-1))
        pri = F.logsigmoid(kappa * (p_max - thr[j])) if kappa > 0 else 0.0
        return o["on_logit"][:, j] - pri, o

    @torch.no_grad()
    def ev_(X):
        f, w = build_inputs(X); out = {"lm": [], "gm": [], "gc": [], "pm": [], "pc": []}
        for i in range(0, len(f), 32):
            ft = torch.from_numpy(f[i:i + 32]).to(dev); wt = torch.from_numpy(w[i:i + 32]).to(dev)
            lm, o = L(ft, wt)
            out["lm"].append(lm.cpu().numpy()); out["gm"].append(torch.sigmoid(o["on_logit"][:, J]).cpu().numpy())
            out["gc"].append(torch.sigmoid(o["on_logit"][:, KC]).cpu().numpy())
            out["pm"].append(o["power"][:, J].cpu().numpy()); out["pc"].append(o["power"][:, KC].cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    print("\n=== %s [%s] ===" % (name, MODE))
    print("  %-36s %8s %8s %8s %7s %7s | %9s %5s" % ("변형", "L미니PC", "게이트mpc", "게이트chg", "P mpc", "P chg", "Δ주입로짓", "양성%"))
    for nm, X in VARIANTS:
        a = ev_(X); b = ev_(inject(X)); d = b["lm"] - a["lm"]
        print("  %-36s %+8.2f %8.3f %8.3f %7.1f %7.1f | %+9.2f %4.0f%%"
              % (nm, np.median(a["lm"]), np.median(a["gm"]), np.median(a["gc"]), np.median(a["pm"]), np.median(a["pc"]),
                 np.median(d), 100 * (d > 0).mean()))
    if MODE != "float":
        return
    fR, wR = build_inputs(bg); fS, wS = build_inputs(syn)
    fR, wR, fS, wS = map(torch.from_numpy, (fR, wR, fS, wS))
    M = 48; AF = torch.zeros_like(fR); AW = torch.zeros_like(wR)
    for a_ in (torch.arange(M, dtype=torch.float32) + 0.5) / M:
        xf = (fS + a_ * (fR - fS)).to(dev).requires_grad_(True); xw = (wS + a_ * (wR - wS)).to(dev).requires_grad_(True)
        g1, g2 = torch.autograd.grad(L(xf, xw)[0].sum(), (xf, xw))
        AF += g1.cpu(); AW += g2.cpu()
    AF = ((fR - fS) * AF / M).numpy(); AW = ((wR - wS) * AW / M).numpy()
    with torch.no_grad():
        gap = (L(fR.to(dev), wR.to(dev))[0] - L(fS.to(dev), wS.to(dev))[0]).cpu().numpy()
    tot = AF.sum((1, 2)) + AW.sum((1, 2))
    print("  IG(짝 맞춤): 격차 %+.2f · IG 합 %+.2f · 완전성 오차 %.2f | 세밀 %+.2f 광역 %+.2f"
          % (gap.mean(), tot.mean(), np.abs(gap - tot).mean(), AF.sum((1, 2)).mean(), AW.sum((1, 2)).mean()))
    rows = [("세밀", k, AF[:, c].sum((1, 2))) for k, c in FG.items()] + [("광역", k, AW[:, c].sum((1, 2))) for k, c in WG.items()]
    rows.sort(key=lambda t: -abs(t[2].mean()))
    for br, k, v in rows[:12]:
        print("    %-5s %-16s %+6.2f (중앙 %+6.2f, 양성 %3.0f%%)" % (br, k, v.mean(), np.median(v), 100 * (v > 0).mean()))
    print("    세밀 홀수 차수별 h1,3,…,15: %s" % np.round([AF[:, [s, 8 + s]].sum((1, 2)).mean() for s in range(8)], 2))
    print("    광역 |I_h| 차수별 h1,3,…,15: %s" % np.round([AW[:, [12 + h - 1]].sum((1, 2)).mean() for h in (1, 3, 5, 7, 9, 11, 13, 15)], 2))


for m in MODELS:
    run(m)
