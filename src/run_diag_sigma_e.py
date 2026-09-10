# -*- coding: utf-8 -*-
"""자리 E 다기기 SMPS 창의 σ 격차와 미니PC 놓침(실패 ③, test_2)을 가른다 (13.83.25).

    python -X utf8 src/run_diag_sigma_e.py cnn_v31 cnn_v29

실측: test_2 에서 프로젝터+선풍기+충전기+미니PC 가 60초 내내 켜진 창 (격차표의 그 행).
합성: 같은 구성을 자리 E · 정상상태로 (`force_active`, `full_window_placement`), 정답 고조파 포함.

① σ|I_h| 를 절대값·차수별로 (실측 총합 / 합성 총합 / 합성 기기별 정답 부분 / 풀 원본 녹화의 정상 구간)
② 이식 — 실측 창의 **평균 페이저는 두고 합성의 창 안 편차만** 심는다 (그 반대도). σ 만 옮기는 시험이다.
   더해서 편차 이득(×k)으로 실측을 합성만큼 흔들면 미니PC 게이트가 켜지나 (섭동 — 축 표시용).
③ IG — 미니PC 원시 로짓의 실측−합성 격차를 채널 무리로 분해 (완전성 검사 포함).
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

MODELS = sys.argv[1:] or ["cnn_v31", "cnn_v29"]
COMP = ("beam_projector", "fan", "laptop_charger", "minipc")
W, STRIDE, N_SYN = WINDOW_CYCLES, 120, 30
O = [2, 4, 6, 8, 12]                      # h3,5,7,9,13
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ── 실측 (test_2) ──────────────────────────────────────────────────────────
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
spec = ev["test_2"]
raw = load_nilm_npz("processed_data/composite_eval/test_2.npz")
x = RealWindows._to_33ch(raw); iv = np.asarray(raw["is_valid"]).astype(bool); n = x.shape[1]


def mk(app, key="on"):
    m = np.zeros(n, bool)
    for t0, t1 in spec["intervals"].get(app, {}).get(key, []):
        m[int(t0 * 60):int(min(t1 * 60, n))] = True
    return m


on = {a: mk(a) for a in spec["appliances_present"]}
unc = np.zeros(n, bool)
for a in spec["appliances_present"]:
    unc |= mk(a, "uncertain")
R = []
for t0 in range(0, n - W + 1, STRIDE):
    sl = slice(t0, t0 + W)
    if not iv[sl].all() or unc[sl].any() or any(on[a][sl].std() != 0 for a in on):
        continue
    if tuple(sorted(a for a in on if on[a][t0])) == tuple(sorted(COMP)):
        R.append(x[:, sl])
R = np.stack(R)
print("실측 test_2 %s 정상상태 창 %d (P 중앙 %.0fW)" % ("+".join(a[:4] for a in COMP), len(R), np.median(R[:, 30])))

# ── 합성 (자리 E · 정상상태 · 정답 고조파) ──────────────────────────────────
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=True)
np.random.seed(0)
S, G, tries = [], {a: [] for a in COMP}, 0
while len(S) < N_SYN and tries < 3000:
    tries += 1
    try:
        s = gen.synthesize_random_window(window_size_cycles=W, force_active=list(COMP), full_window_placement=True,
                                         compute_gt_harmonics=True, target_lookahead_cycles=360, sustained_power_limit_w=None)
    except Exception:
        continue
    if s.metadata.get("base_voltage_v", 999) < 222.0 or set(s.active_appliances) != set(COMP):
        continue
    if not all(s.gt_is_on[a].all() for a in COMP):
        continue
    vh = np.asarray(s.voltage_harmonics_complex)[:, [h - 1 for h in VOLT_ORDERS]]
    S.append(np.concatenate([s.harmonics_ri[:, :, 0].T, s.harmonics_ri[:, :, 1].T, s.power_features[:, 0:1].T,
                             s.power_features[:, 1:2].T, s.power_features[:, 4:5].T, vh.real.T, vh.imag.T], 0).astype(np.float32))
    for a in COMP:
        g = np.asarray(s.gt_harmonics_ri[a]); G[a].append(g[:, :, 0].T + 1j * g[:, :, 1].T)
S = np.stack(S); G = {a: np.stack(v) for a, v in G.items()}
print("합성 %d창 (시도 %d, P 중앙 %.0fW)" % (len(S), tries, np.median(S[:, 30])))
n_ = min(len(R), len(S)); R, S = R[:n_], S[:n_]; G = {a: v[:n_] for a, v in G.items()}


def ph(X):
    return X[:, 0:15] + 1j * X[:, 15:30]


def sig(c):                                    # (B,15,T) -> 차수별 창 안 σ|I_h| 중앙 (mA)
    return np.median(1e3 * np.std(np.abs(c), axis=2), 0)


def mag(c):
    return np.median(1e3 * np.median(np.abs(c), axis=2), 0)


print("\n=== ① 창 안 σ|I_h| (mA), h3·5·7·9·13 ===")
print("  %-34s σ %s   |I_h| %s" % ("실측 총합", np.round(sig(ph(R))[O], 1), np.round(mag(ph(R))[O], 0)))
print("  %-34s σ %s   |I_h| %s" % ("합성 총합", np.round(sig(ph(S))[O], 1), np.round(mag(ph(S))[O], 0)))
for a in COMP:
    print("  %-34s σ %s   |I_h| %s" % ("합성 정답 부분 " + a, np.round(sig(G[a])[O], 1), np.round(mag(G[a])[O], 0)))
rest = ph(S) - sum(G[a] for a in COMP)
print("  %-34s σ %s   |I_h| %s" % ("합성 나머지 (배경·잡음·대기)", np.round(sig(rest)[O], 1), np.round(mag(rest)[O], 0)))
# 풀 원본 녹화의 정상 구간 σ (60초 창, 자리 E 녹화)
print("  --- 풀 원본 (자리 E 녹화, 60초 창 σ 중앙; 상태 일정 구간) ---")
for app, files, lo, hi in (("laptop_charger", ("laptop_charger_3", "laptop_charger_4", "laptop_charger_5"), 55, 70),
                           ("beam_projector", ("beam_projector_1", "beam_projector_2"), 40, 50),
                           ("fan", ("fan_1",), 20, 45), ("minipc", ("minipc_2", "minipc_3"), 8, 30)):
    ss = []
    for act in pool.appliance_activations[app]:
        if not any(f in str(act.source_file) for f in files):
            continue
        c = np.asarray(act.net_harmonics_complex); p = np.asarray(act.target_power_w)
        for i in range(0, len(c) - W + 1, W // 2):
            pp = p[i:i + W]
            if (pp >= lo).mean() > 0.9 and (pp < hi).mean() > 0.9:
                ss.append(1e3 * np.std(np.abs(c[i:i + W]), axis=0))
    if ss:
        print("  %-34s σ %s   (창 %d)" % ("풀 " + app + " %d~%dW" % (lo, hi), np.round(np.median(np.stack(ss), 0)[O], 1), len(ss)))

# ── ② 편차 이식 ────────────────────────────────────────────────────────────
def put(dst, src_c):
    o = dst.copy(); o[:, 0:15] = src_c.real; o[:, 15:30] = src_c.imag; return o


cR, cS = ph(R), ph(S)
mR, mS = cR.mean(2, keepdims=True), cS.mean(2, keepdims=True)
dev_swap_R = put(R, mR + (cS - mS))          # 실측 평균 + 합성 편차
dev_swap_S = put(S, mS + (cR - mR))          # 합성 평균 + 실측 편차
VAR = [("실측", R), ("합성", S),
       ("실측 평균 + 합성 편차 (σ 만 합성)", dev_swap_R),
       ("합성 평균 + 실측 편차 (σ 만 실측)", dev_swap_S),
       ("실측 + 합성 I_h h2..15 통째", put(R, np.concatenate([cR[:, :1], cS[:, 1:]], 1))),
       ("합성 + 실측 I_h h2..15 통째", put(S, np.concatenate([cS[:, :1], cR[:, 1:]], 1)))]
for k in (2.0, 4.0, 6.0):
    VAR.append(("실측 편차 ×%.0f (섭동)" % k, put(R, mR + (cR - mR) * k)))
for k in (0.5, 0.25, 0.16):
    VAR.append(("합성 편차 ×%.2f (섭동)" % k, put(S, mS + (cS - mS) * k)))

FG = {"홀수 Re/Im h1": [0, 8], "홀수 Re/Im h3-7": [1, 2, 3, 9, 10, 11], "홀수 Re/Im h9-15": [4, 5, 6, 7, 12, 13, 14, 15],
      "짝수 |I|": list(range(16, 23)), "P,Q,V": [23, 24, 25], "비": [26, 27, 28], "P 리플": [29, 30],
      "phi": list(range(31, 39)), "PF,I9/I3": [39, 40], "다단강하": [41, 42], "반파,|I2|": [43, 44],
      "Vh1": [45, 51], "Vh3-11": [46, 47, 48, 49, 50, 52, 53, 54, 55, 56]}
WG = {"P,Q,V": [0, 1, 2], "|I1,3,5,2|": [3, 4, 5, 6], "p_std": [7], "비": [8, 9], "p_dev": [10], "step": [11],
      "|I_h|": list(range(12, 27)), "phi": list(range(27, 35)), "Vh1": [35, 41], "Vh3-11": [36, 37, 38, 39, 40, 42, 43, 44, 45, 46]}

for nm in MODELS:
    model, apps, ck = load_model("results/%s.pt" % nm, dev)
    J = apps.index("minipc"); KC = apps.index("laptop_charger"); KP = apps.index("beam_projector"); KF = apps.index("fan")
    kappa = float(model.prior_kappa); thr = model.on_threshold_asinh[J]

    def L(ft, wt):
        o = model(ft, wt)
        p_max = torch.maximum(ft[:, P_CH_FINE].amax(-1), wt[:, P_CH_WIDE].amax(-1))
        pri = F.logsigmoid(kappa * (p_max - thr)) if kappa > 0 else 0.0
        return o["on_logit"][:, J] - pri, o

    print("\n=== ② %s — 편차 이식·섭동 (미니PC 게이트 중앙 · >0.5 몫 · 충전기/프로젝터/선풍기 게이트) ===" % nm)
    for tag, X in VAR:
        f, w = build_inputs(X)
        with torch.no_grad():
            lm, o = L(torch.from_numpy(f).to(dev), torch.from_numpy(w).to(dev))
            g = torch.sigmoid(o["on_logit"]).cpu().numpy(); pw = o["power"].cpu().numpy()
        print("  %-36s mpc %.3f (%3.0f%%)  L %+6.2f | chg %.2f proj %.2f fan %.2f | P mpc %4.1f"
              % (tag, np.median(g[:, J]), 100 * (g[:, J] > 0.5).mean(), np.median(lm.cpu().numpy()),
                 np.median(g[:, KC]), np.median(g[:, KP]), np.median(g[:, KF]), np.median(pw[:, J])))

    fR, wR = build_inputs(R); fS, wS = build_inputs(S)
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
    print("  ③ IG (실측−합성, 미니PC 로짓): 격차 %+.2f · IG 합 %+.2f · 완전성 오차 %.2f | 세밀 %+.2f 광역 %+.2f"
          % (gap.mean(), tot.mean(), np.abs(gap - tot).mean(), AF.sum((1, 2)).mean(), AW.sum((1, 2)).mean()))
    rows = [("세밀", k, AF[:, c].sum((1, 2))) for k, c in FG.items()] + [("광역", k, AW[:, c].sum((1, 2))) for k, c in WG.items()]
    rows.sort(key=lambda t: -abs(t[2].mean()))
    for br, k, v in rows[:10]:
        print("     %-5s %-16s %+6.2f (양성 %3.0f%%)" % (br, k, v.mean(), 100 * (v > 0).mean()))
    tp = AF.sum(1).mean(0)
    print("     세밀 시간축: 앞 5초 %+.2f | 뒤 5초 %+.2f | 타깃±10주기 %+.2f" % (tp[:300].sum(), tp[300:].sum(), tp[229:250].sum()))
