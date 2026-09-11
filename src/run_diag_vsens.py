# -*- coding: utf-8 -*-
"""모델의 전력 헤드가 전압에 얼마나 반응하는가 (13.84.19 의 다음 수).

저항은 P = V²/R 이라 d ln P / d ln V = 2 다. 같은 합성 창의 **전압만** 바꿔(원시 32번 V 와
V_h 의 h1 실수부) 예측 전력의 탄성을 재고 2 와 견준다. 전류는 그대로 두는 반사실이므로
"헤드가 V 를 읽는가" 만 답한다 — 읽지 않으면 탄성이 0 이다.

    python -X utf8 src/run_diag_vsens.py cnn_v35 [N] [D|E]
"""
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.inputs import build_inputs, VOLT_ORDERS, VOLT_RE0
from src.run_gate_check import load_model
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

W, LOOK = 3600, 360
DV = (-8.0, -4.0, 0.0, +4.0, +8.0)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "cnn_v35"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    site = (sys.argv[3] if len(sys.argv) > 3 else "D").upper()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps = load_model("results/%s.pt" % tag, dev)[:2]
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
    gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    np.random.seed(0)
    ti = W - 1 - LOOK
    X, V0 = [], []
    tries = 0
    while len(X) < N and tries < 6000:
        tries += 1
        try:
            smp = gen.synthesize_random_window(
                window_size_cycles=W, force_active=["hair_dryer", "oven", "hotplate"],
                full_window_placement=True, compute_gt_harmonics=False,
                target_lookahead_cycles=LOOK, sustained_power_limit_w=None)
        except Exception:
            continue
        bv = smp.metadata.get("base_voltage_v", 999)
        if (bv > 222.0) if site == "D" else (bv <= 226.0):
            continue
        pf = np.asarray(smp.power_features)
        if pf[ti, 0] < 1500.0:
            continue
        r = smp.harmonics_ri[:, :, 0].T
        i_ = smp.harmonics_ri[:, :, 1].T
        vh = np.asarray(smp.voltage_harmonics_complex)[:, [h - 1 for h in VOLT_ORDERS]]
        X.append(np.concatenate([r, i_, pf[:, 0:1].T, pf[:, 1:2].T, pf[:, 4:5].T,
                                 vh.real.T, vh.imag.T], axis=0).astype(np.float32))
        V0.append(float(pf[ti, 4]))
    X = np.stack(X); V0 = np.asarray(V0)
    print("%s · 자리 %s · 창 %d · 전압 중앙 %.1fV  (전류는 고정한 반사실)" % (tag, site, len(X), np.median(V0)))
    out = {}
    for dv in DV:
        Y = X.copy()
        Y[:, 32] += dv                       # V 실효값
        Y[:, VOLT_RE0] += dv                 # Re(V_1) — V 실효값과 같은 양이다
        fine, wide = build_inputs(Y)
        with torch.no_grad():
            o = model(torch.from_numpy(fine).to(dev), torch.from_numpy(wide).to(dev))
            out[dv] = (torch.sigmoid(o["on_logit"]) * o["power_raw"]).float().cpu().numpy()
    lo, hi = DV[0], DV[-1]
    vmed = float(np.median(V0))
    dlnv = np.log((vmed + hi) / (vmed + lo))
    print("\n  기기               P(V−8)   P(V)    P(V+8)   탄성 dlnP/dlnV   이론 2.0 대비")
    for k, a in enumerate(apps):
        p0 = np.median(out[0.0][:, k])
        if p0 < 20:
            continue
        pl, ph = np.median(out[lo][:, k]), np.median(out[hi][:, k])
        el = np.log(max(ph, 1e-6) / max(pl, 1e-6)) / dlnv
        print("  %-18s %7.1f %7.1f %8.1f %12.2f %14s"
              % (a, pl, p0, ph, el, "%.0f%%" % (100 * el / 2.0)))
    tot = {d: np.median(out[d].sum(1)) for d in DV}
    el = np.log(tot[hi] / tot[lo]) / dlnv
    print("  %-18s %7.1f %7.1f %8.1f %12.2f %14s"
          % ("합계", tot[lo], tot[0.0], tot[hi], el, "%.0f%%" % (100 * el / 2.0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
