# -*- coding: utf-8 -*-
"""적분 기울기(IG)로 영역 이동 로짓을 채널 무리로 **정확히** 분해한다 (13.83.22).

13.83.17 이 남긴 것: 실측 형제-ON·미니PC-OFF 배경이 헤드 방향으로 +9.9 로짓 밀려 있다.
섭동(입력 흔들기)은 축을 가리킬 뿐 귀속이 못 된다 (13.83.21). IG 는 기준점 -> 목표점의
직선 경로에서 기울기를 적분하므로 **합이 정확히 로짓 차이**가 된다(완전성). 그래서 채널
무리별 합은 그 차이의 정확한 분해다 (경로 적분 오차는 따로 찍는다 — 이 자료에서 3% 안).

    python -X utf8 src/run_diag_ig.py cnn_v29 cnn_v28 cnn_v26 cnn_v25 cnn_v24b

배경은 `run_diag_cell6`·`run_diag_trace` 와 같은 실측 129창(형제 ON·미니PC 60초 내내 OFF)이고,
기준점은 `holdout60_v22` 에서 같은 규칙으로 고른 합성 창이다.

⚠ 이 기준점은 **전력이 안 맞는다** (합성 중앙 66W 대 실측 472W). cell6/trace 의 "전력 맞춤"
   10~90% 대역 필터는 저전력 합성 창이 압도해 전력을 못 맞춘다. 같은 구성·같은 전력의 기준점은
   `run_diag_matched.py` 가 합성기로 직접 만든다. 이 도구는 **판 사이 비교**(같은 입력)에 쓴다.

내는 것: 세밀 57채널·광역 47채널을 무리로 묶은 IG 합(평균·중앙·양성률), 세밀 시간축 분포,
판별 교차표. `results/_ig_attr.npz` 에 원시 귀속을 남긴다.
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

MODELS = sys.argv[1:] or ["cnn_v29", "cnn_v28", "cnn_v26", "cnn_v25", "cnn_v24b"]
HD = "processed_data/holdout60_v22"
FILES = ["test_1", "test_2", "test_3", "test_4"]
SIB = ["laptop_charger", "beam_projector"]
W, STRIDE = WINDOW_CYCLES, 60
M_STEPS = 64
rng = np.random.default_rng(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ── 배경 ─────────────────────────────────────────────────────────────────────
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
bg, bgp, bgf = [], [], []
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
            bg.append(x[:, sl]); bgp.append(float(np.median(x[30, sl]))); bgf.append(stem)
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
print("실측 배경 %d창 (P 중앙 %.0fW; %s) · 합성 기준점 %d창 (P 중앙 %.0fW — ⚠ 전력 미매칭)"
      % (len(bg), np.median(bgp), {f: bgf.count(f) for f in FILES}, len(sbg),
         np.median([np.median(c[30]) for c in sbg])))

fR, wR = build_inputs(np.stack(bg))
fS, wS = build_inputs(np.stack(sbg))

FG = {
    "홀수 Re/Im h1": [0, 8],
    "홀수 Re/Im h3-7": [1, 2, 3, 9, 10, 11],
    "홀수 Re/Im h9-15": [4, 5, 6, 7, 12, 13, 14, 15],
    "짝수 |I| h2-14": list(range(16, 23)),
    "P": [23], "Q": [24], "V": [25],
    "비 I3/I1,I5/I1,I2/I1": [26, 27, 28],
    "P 리플": [29, 30],
    "phi3,5,7,9 cos/sin": list(range(31, 39)),
    "PF, I9/I3": [39, 40],
    "다단강하": [41, 42],
    "반파, |I2| 평활": [43, 44],
    "Vh1 Re/Im": [45, 51],
    "Vh3-11 Re/Im": [46, 47, 48, 49, 50, 52, 53, 54, 55, 56],
}
WG = {
    "P,Q,V": [0, 1, 2],
    "|I1|,|I3|,|I5|,|I2|": [3, 4, 5, 6],
    "p_std": [7], "비": [8, 9], "p_dev": [10], "step_rate": [11],
    "|I_h| h1-15": list(range(12, 27)),
    "phi cos/sin": list(range(27, 35)),
    "Vh1 Re/Im": [35, 41],
    "Vh3-11 Re/Im": [36, 37, 38, 39, 40, 42, 43, 44, 45, 46],
}
assert sorted(sum(FG.values(), [])) == list(range(57))
assert sorted(sum(WG.values(), [])) == list(range(47))


def zshift(a, b):
    ma, mb = a.mean(-1), b.mean(-1)
    return (ma.mean(0) - mb.mean(0)) / (mb.std(0) + 1e-6)


zf, zw = zshift(fR, fS), zshift(wR, wS)
print("\n=== 서술 통계: (실측 평균 − 합성 평균)/합성 σ, 채널별 (무리 평균 |z| 와 부호) ===")
for nm, ch in FG.items():
    print("  세밀 %-22s |z| %5.2f  %s" % (nm, np.abs(zf[ch]).mean(), np.round(zf[ch], 1).tolist()[:8]))
for nm, ch in WG.items():
    print("  광역 %-22s |z| %5.2f  %s" % (nm, np.abs(zw[ch]).mean(), np.round(zw[ch], 1).tolist()[:8]))

perm = rng.permutation(len(sbg))
pair = perm[:len(bg)] if len(sbg) >= len(bg) else perm[np.arange(len(bg)) % len(sbg)]
fR_t, wR_t = torch.from_numpy(fR), torch.from_numpy(wR)
fS_t, wS_t = torch.from_numpy(fS[pair]), torch.from_numpy(wS[pair])
fM_t, wM_t = torch.from_numpy(fS.mean(0, keepdims=True)), torch.from_numpy(wS.mean(0, keepdims=True))


def run_model(name):
    model, apps, ck = load_model("results/%s.pt" % name, dev)
    J = apps.index("minipc")
    kappa = float(model.prior_kappa)
    thr = model.on_threshold_asinh[J]

    def L(ft, wt):
        o = model(ft, wt)
        p_max = torch.maximum(ft[:, P_CH_FINE].amax(-1), wt[:, P_CH_WIDE].amax(-1))
        pri = F.logsigmoid(kappa * (p_max - thr)) if kappa > 0 else 0.0
        return o["on_logit"][:, J] - pri

    def ig(ft_t, wt_t, ft_b, wt_b):
        B = ft_t.shape[0]
        ft_b = ft_b.expand(B, -1, -1) if ft_b.shape[0] == 1 else ft_b
        wt_b = wt_b.expand(B, -1, -1) if wt_b.shape[0] == 1 else wt_b
        gf = torch.zeros_like(ft_t); gw = torch.zeros_like(wt_t)
        for a in (torch.arange(M_STEPS, dtype=torch.float32) + 0.5) / M_STEPS:
            xf = (ft_b + a * (ft_t - ft_b)).to(dev).requires_grad_(True)
            xw = (wt_b + a * (wt_t - wt_b)).to(dev).requires_grad_(True)
            g1, g2 = torch.autograd.grad(L(xf, xw).sum(), (xf, xw))
            gf += g1.detach().cpu(); gw += g2.detach().cpu()
        gf /= M_STEPS; gw /= M_STEPS
        with torch.no_grad():
            Lt = L(ft_t.to(dev), wt_t.to(dev)).cpu()
            Lb = L(ft_b.to(dev), wt_b.to(dev)).cpu()
        return (ft_t - ft_b) * gf, (wt_t - wt_b) * gw, Lt, Lb

    res = {}
    for tag, (fb, wb) in (("짝", (fS_t, wS_t)), ("평균기준", (fM_t, wM_t))):
        AF, AW, LT, LB = [], [], [], []
        for i in range(0, len(bg), 16):
            sl = slice(i, i + 16)
            a_f, a_w, lt, lb = ig(fR_t[sl], wR_t[sl], fb[sl] if fb.shape[0] > 1 else fb,
                                  wb[sl] if wb.shape[0] > 1 else wb)
            AF.append(a_f); AW.append(a_w); LT.append(lt); LB.append(lb)
        AF = torch.cat(AF).numpy(); AW = torch.cat(AW).numpy()
        LT = torch.cat(LT).numpy(); LB = torch.cat(LB).numpy()
        res[tag] = dict(AF=AF, AW=AW, LT=LT, LB=LB, tot=AF.sum((1, 2)) + AW.sum((1, 2)), gap=LT - LB)
    return res


def report(name, r):
    for tag in ("짝", "평균기준"):
        d = r[tag]
        AF, AW, gap, tot = d["AF"], d["AW"], d["gap"], d["tot"]
        print("\n--- %s [%s]  L(실측) 중앙 %+.2f · L(기준) 중앙 %+.2f · 격차 평균 %+.2f | IG 합 %+.2f · "
              "완전성 오차 %.2f (%.0f%%)"
              % (name, tag, np.median(d["LT"]), np.median(d["LB"]), gap.mean(), tot.mean(),
                 np.abs(gap - tot).mean(), 100 * np.abs(gap - tot).mean() / (np.abs(gap).mean() + 1e-9)))
        rows = [("세밀", nm, AF[:, ch].sum((1, 2))) for nm, ch in FG.items()] \
            + [("광역", nm, AW[:, ch].sum((1, 2))) for nm, ch in WG.items()]
        rows.sort(key=lambda t: -abs(t[2].mean()))
        print("  세밀 합 %+6.2f · 광역 합 %+6.2f (짝 %d)" % (AF.sum((1, 2)).mean(), AW.sum((1, 2)).mean(), len(gap)))
        print("  %-5s %-24s %8s %8s %8s %6s" % ("갈래", "무리", "평균", "중앙", "sd", "양성%"))
        for br, nm, v in rows:
            print("  %-5s %-24s %+8.2f %+8.2f %8.2f %5.0f%%" % (br, nm, v.mean(), np.median(v), v.std(), 100 * (v > 0).mean()))
        tp = AF.sum(1).mean(0); t = 239
        print("  세밀 시간축: 앞 5초 %+.2f | 뒤 5초 %+.2f | 타깃±10주기 %+.2f | 타깃 뒤 60주기 %+.2f"
              % (tp[:300].sum(), tp[300:].sum(), tp[t - 10:t + 11].sum(), tp[540:].sum()))
        tw = AW.sum(1).mean(0)
        print("  광역 시간축: 0-20초 %+.2f | 20-40초 %+.2f | 40-60초 %+.2f" % (tw[:40].sum(), tw[40:80].sum(), tw[80:].sum()))


results = {}
for m in MODELS:
    try:
        results[m] = run_model(m)
    except Exception as e:
        print("  %s 건너뜀 (%s)" % (m, str(e)[:80]))
        continue
    report(m, results[m])

print("\n=== 판별 교차표 (짝 IG, 평균 로짓) ===")
keys = [("세밀", nm) for nm in FG] + [("광역", nm) for nm in WG]
print("  %-5s %-24s" % ("갈래", "무리") + "".join(" %9s" % m.replace("cnn_", "") for m in results))
for br, nm in keys:
    vals = []
    for m, r in results.items():
        A = r["짝"]["AF"] if br == "세밀" else r["짝"]["AW"]
        ch = FG[nm] if br == "세밀" else WG[nm]
        vals.append(A[:, ch].sum((1, 2)).mean())
    print("  %-5s %-24s" % (br, nm) + "".join(" %+9.2f" % v for v in vals))
print("  %-5s %-24s" % ("", "격차 (L실측 − L합성)")
      + "".join(" %+9.2f" % r["짝"]["gap"].mean() for r in results.values()))

np.savez("results/_ig_attr.npz",
         **{"%s_%s_%s" % (m, tag, k): v for m, r in results.items() for tag, d in r.items()
            for k, v in d.items()})
