# -*- coding: utf-8 -*-
"""고전력 저항 다중 구성에서 **기기별** 전력 오차 (13.84.18 의 다음 수).

실측 파일에는 기기별 참값 와트가 없어 초과를 누구에게 물을지 못 정한다. 같은 구성을 합성으로
만들면 `gt_target_power_w` 가 있으니 바로 귀속된다. 자리 D · 드라이기+오븐+핫플(+포트) · 총 2200W 이상.

    python -X utf8 src/run_diag_highpower.py cnn_v35 [N]
"""
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.inputs import build_inputs, VOLT_ORDERS
from src.run_gate_check import load_model
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

W, TGT = 3600, 600
LOOK = 360


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "cnn_v35"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    site = (sys.argv[3] if len(sys.argv) > 3 else "D").upper()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps = load_model("results/%s.pt" % tag, dev)[:2]

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
    gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    np.random.seed(0)
    X, GT, POBS, V = [], [], [], []
    tries = 0
    ti = W - 1 - LOOK
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
        P = np.asarray(smp.power_features)[:, 0]
        if P[ti] < 2200.0:
            continue
        r = smp.harmonics_ri[:, :, 0].T
        i_ = smp.harmonics_ri[:, :, 1].T
        pf = np.asarray(smp.power_features)
        vh = np.asarray(smp.voltage_harmonics_complex)[:, [h - 1 for h in VOLT_ORDERS]]
        X.append(np.concatenate([r, i_, pf[:, 0:1].T, pf[:, 1:2].T, pf[:, 4:5].T,
                                 vh.real.T, vh.imag.T], axis=0).astype(np.float32))
        GT.append(np.array([float(smp.gt_target_power_w[a][ti]) if a in smp.gt_target_power_w else 0.0
                            for a in apps]))
        POBS.append(float(P[ti])); V.append(float(pf[ti, 4]))
    if not X:
        print("표본 0 — 구성을 못 만들었다"); return 1
    X = np.stack(X); GT = np.stack(GT); POBS = np.asarray(POBS); V = np.asarray(V)
    fine, wide = build_inputs(X)
    with torch.no_grad():
        o = model(torch.from_numpy(fine).to(dev), torch.from_numpy(wide).to(dev))
        pred = (torch.sigmoid(o["on_logit"]) * o["power_raw"]).float().cpu().numpy()
    print("%s · 자리 %s · 드라이기+오븐+핫플 · 총 2200W+ · 창 %d (시도 %d)" % (tag, site, len(X), tries))
    print("  관측 P 중앙 %.0fW · 전압 중앙 %.1fV · Σ예측 중앙 %.0fW · **초과 %+.0fW (%+.1f%%)**"
          % (np.median(POBS), np.median(V), np.median(pred.sum(1)),
             np.median(pred.sum(1) - POBS), 100 * np.median(pred.sum(1) - POBS) / np.median(POBS)))
    print("\n  기기               참값 중앙   예측 중앙    오차 중앙   오차/참값")
    order = np.argsort(-np.abs(np.median(pred - GT, axis=0)))
    for k in order:
        if abs(np.median(pred[:, k] - GT[:, k])) < 3 and np.median(GT[:, k]) < 3:
            continue
        g, p = np.median(GT[:, k]), np.median(pred[:, k])
        print("  %-18s %9.1f %11.1f %11.1f %10s"
              % (apps[k], g, p, np.median(pred[:, k] - GT[:, k]),
                 ("%+.1f%%" % (100 * (p - g) / g)) if g > 5 else "–"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
