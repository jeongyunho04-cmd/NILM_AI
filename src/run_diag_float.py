# -*- coding: utf-8 -*-
"""부동(float) 충전기 검정 셋 — 13.83.22 의 결론을 재현하는 값싼 자 (GPU 1분).

    python -X utf8 src/run_diag_float.py cnn_v29 cnn_v25 cnn_v28

`results/_matched_float.npz` (run_diag_matched.py float 가 남긴 실측 129창 / 합성 맞춤 창 / 합성
충전기 정답 부분)를 읽는다. 없으면 먼저 그것을 돌려라.

  ① test_1 의 충전기 OFF 계단 둘(137.1초 · 531.7초)에서 **현장 부동 충전기의 15차 복소 지문**을 뽑아
     풀의 충전기·미니PC 전력대별 지문과 |I_h|/|I3| 비와 φ 로 맞댄다 — 모양은 충전기다
  ② 합성 충전기 정답 부분을 k 배로 줄여 가며(50W -> 10W) 각 판이 어디서 "미니PC" 로 넘어가나
  ③ 풀의 12~20W 충전기 조각(연속 ≥1초, 총 5.4초)을 **이어 붙여** 15W 정상상태 충전기를 만들어 핫플 위에
     얹으면 모델이 실측 부동 충전기와 같은 판정을 내나 — 같으면 지금 재료로 실패 창을 재현할 수 있다
"""
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.inputs import build_inputs
from src.model.realdata import RealWindows
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.synthesis.segment_pool import SegmentPool

MODELS = sys.argv[1:] or ["cnn_v29", "cnn_v25", "cnn_v28"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = np.load("results/_matched_float.npz")
real, syn, chg = d["real"], d["syn"], d["syn_smps"]          # chg (B,W,15,2)
B, W = syn.shape[0], syn.shape[2]

# ── ① 현장 부동 충전기 지문 (ON−OFF 계단 차분) ───────────────────────────────
raw = load_nilm_npz("processed_data/composite_eval/test_1.npz")
x1 = RealWindows._to_33ch(raw)
c1 = x1[0:15] + 1j * x1[15:30]; P1 = x1[30]
o = [2, 4, 6, 8, 10, 12, 14]


def step_sig(t, w=2.0, gap=0.5):
    a = slice(int((t - gap - w) * 60), int((t - gap) * 60)); b = slice(int((t + gap) * 60), int((t + gap + w) * 60))
    ma = np.median(c1[:, a].real, 1) + 1j * np.median(c1[:, a].imag, 1)
    mb = np.median(c1[:, b].real, 1) + 1j * np.median(c1[:, b].imag, 1)
    return ma - mb, float(np.median(P1[a]) - np.median(P1[b]))


def show(nm, s):
    mag = 1e3 * np.abs(s)
    phi = np.rad2deg(np.angle(s[o] * np.exp(-1j * np.arange(1, 16)[o] * np.angle(s[0]))))
    print("  %-34s |I1| %5.0f |I_h| h3..15 %s  /|I3| %s  φ %s"
          % (nm, mag[0], np.round(mag[o], 1), np.round(mag[o] / mag[2], 2), np.round(phi).astype(int)))


print("=== ① test_1 부동 충전기 지문 (OFF 계단 차분, 자리 D) ===")
for t in (137.05, 531.68):
    s, dp = step_sig(t)
    show("계단 %.1f초 (ΔP %+.1fW)" % (t, dp), s)
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
print("  --- 풀 (net 페이저 중앙) ---")
for app, bands in [("laptop_charger", [(12, 20), (20, 30), (30, 45), (45, 70)]), ("minipc", [(7, 12), (12, 20), (20, 30)])]:
    acts = pool.appliance_activations[app]
    P = np.concatenate([np.asarray(a.target_power_w, np.float64) for a in acts])
    H = np.concatenate([np.asarray(a.net_harmonics_complex) for a in acts])
    for lo, hi in bands:
        m = (P >= lo) & (P < hi)
        if m.sum() < 100:
            continue
        show("%s %d-%dW (%d주기)" % (app, lo, hi, m.sum()), np.median(H[m].real, 0) + 1j * np.median(H[m].imag, 0))


# ── ② 합성 충전기 축소 · ③ 풀 12~20W 조각 이어 붙이기 ─────────────────────────
def scaled(k):
    out = syn.copy()
    out[:, 0:15] -= (1 - k) * chg[:, :, :, 0].transpose(0, 2, 1)
    out[:, 15:30] -= (1 - k) * chg[:, :, :, 1].transpose(0, 2, 1)
    out[:, 30] -= (1 - k) * np.abs(chg[:, :, 0, 0] + 1j * chg[:, :, 0, 1]) * out[:, 32]
    return out


runs = []
for a in pool.appliance_activations["laptop_charger"]:
    p = np.asarray(a.target_power_w, np.float64); m = (p >= 12) & (p < 20)
    hr = np.asarray(a.net_harmonics_ri, np.float32)
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i >= 60:
                runs.append((hr[i:j], p[i:j]))
            i = j
        else:
            i += 1
print("\n풀 12~20W 충전기 연속 조각(≥1초): %d개 · 총 %.1f초" % (len(runs), sum(len(r[0]) for r in runs) / 60))
rng = np.random.default_rng(0)
tiled = syn.copy()
for i in range(B):
    hs, ps = [], []
    while sum(len(h) for h in hs) < W:
        h, p = runs[rng.integers(len(runs))]; hs.append(h); ps.append(p)
    h = np.concatenate(hs)[:W]; p = np.concatenate(ps)[:W]
    tiled[i, 0:15] -= chg[i, :, :, 0].T; tiled[i, 15:30] -= chg[i, :, :, 1].T
    tiled[i, 30] -= np.abs(chg[i, :, 0, 0] + 1j * chg[i, :, 0, 1]) * syn[i, 32]
    tiled[i, 0:15] += h[:, :, 0].T; tiled[i, 15:30] += h[:, :, 1].T; tiled[i, 30] += p

for name in MODELS:
    model, apps, ck = load_model("results/%s.pt" % name, dev)
    J, KC = apps.index("minipc"), apps.index("laptop_charger")

    @torch.no_grad()
    def ev_(Z):
        f, w = build_inputs(Z); G, PW = [], []
        for i in range(0, len(f), 32):
            oo = model(torch.from_numpy(f[i:i + 32]).to(dev), torch.from_numpy(w[i:i + 32]).to(dev))
            G.append(torch.sigmoid(oo["on_logit"]).cpu().numpy()); PW.append(oo["power"].cpu().numpy())
        return np.concatenate(G), np.concatenate(PW)

    print("\n=== %s ===" % name)
    print("  ② 합성 충전기(~50W) x k      %8s %8s %7s %7s" % ("게이트mpc", "게이트chg", "P mpc", "P chg"))
    for k in (1.0, 0.8, 0.6, 0.45, 0.3, 0.2):
        G, PW = ev_(scaled(k))
        print("     k=%.2f (~%3.0fW)            %8.3f %8.3f %7.1f %7.1f"
              % (k, 50 * k, np.median(G[:, J]), np.median(G[:, KC]), np.median(PW[:, J]), np.median(PW[:, KC])))
    for nm, Z in (("③ 합성 핫플 + 풀 12~20W 충전기 이어붙임", tiled), ("   실측 (부동 충전기 ~14W)", real)):
        G, PW = ev_(Z)
        print("  %-36s %8.3f %8.3f %7.1f %7.1f  (mpc>0.5: %.0f%%)"
              % (nm, np.median(G[:, J]), np.median(G[:, KC]), np.median(PW[:, J]), np.median(PW[:, KC]), 100 * (G[:, J] > 0.5).mean()))
