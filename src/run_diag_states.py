# -*- coding: utf-8 -*-
"""상태별 전력 슬롯이 살아 있나 — **격리 녹화에 모델을 그대로 돌린다** (13.84.68).

`net.forward` 의 전력은 **곱**이다:

    p_raw = Σ_s mix[s] · p_states[s],   mix = softmax(state 로짓),  power = σ(on)·p_raw

`L_power` 는 그 **곱만** 본다. `L_state_power` 는 12.35 에서 **0 으로 껐다.** 그래서 슬롯
하나가 비어 있어도 다른 슬롯이 메우면 손실이 안 운다 — 혼합이 그쪽을 가리키는 동안은.
**혼합이 정확해지는 순간 비어 있던 슬롯이 드러난다.** 드라이기 약풍이 그 자리다.

이 자는 격리 녹화(참 상태를 아는 유일한 자료)에 모델을 그대로 돌려 `p_states`·`mix`·`power`
를 상태별로 편다. 복합에서는 못 본다 — 참 상태 라벨이 없기 때문이다.

    python -X utf8 src/run_diag_states.py [results/seq_h38_base.pt ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.realdata import RealWindows
from src.run_gate_check import load_model

#: (기기, 격리 녹화). 상태가 둘 이상인 기기만 본다.
PAIRS = (("hair_dryer", "hair_dryer_1"), ("minipc", "minipc_1"),
         ("laptop_charger", "laptop_charger_1"), ("beam_projector", "beam_projector_1"),
         ("fan", "fan_1"), ("oven", "oven_1"), ("hotplate", "hotplate_1"),
         ("electiric_kettle", "electric_kettle_1"), ("air_conditioner", "air_conditioner_1"))


def windows(stem, stride=600):
    """그 격리 녹화의 모델 입력과 참(상태, 전력). 창 중앙이 온전한 것만."""
    p = "processed_data/npz/%s.npz" % stem
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    S = np.asarray(d["state_id"]); P = np.asarray(d["p_denoised_w"])
    H = np.asarray(d["harmonics_complex"])
    rw = RealWindows(npz_dir="processed_data/npz", stems=[stem], stride=stride,
                     require_valid=False)
    tgt = rw.target_cycle
    sel = np.nonzero((tgt >= 800) & (tgt < len(P) - 800))[0]
    if len(sel) < 5:
        return None
    F, W = [], []
    for i in range(0, len(sel), 256):
        f, w, _, _, _ = rw.batch(sel[i:i + 256]); F.append(f); W.append(w)
    c = tgt[sel]
    r2 = np.abs(H[c, 1]) / np.maximum(np.abs(H[c, 0]), 1e-9)
    return np.concatenate(F), np.concatenate(W), S[c], P[c], r2


def main():
    cks = sys.argv[1:] or ["results/cnn_v37.pt", "results/seq_h38_base.pt"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = {app: windows(stem) for app, stem in PAIRS}

    for cp in cks:
        if not os.path.exists(cp):
            print("없다: %s" % cp); continue
        ck = torch.load(cp, map_location=dev, weights_only=False)
        apps = list(ck["appliances"])
        ref = ck.get("ref", "results/cnn_v37.pt") if "heads" in ck else cp
        model = load_model(ref, dev, weights=False,
                           mask=not bool(ck.get("no_mask", False)))[0]
        model.load_state_dict(ck["model"]); model.eval()
        print("\n=== %s ===" % os.path.basename(cp))
        print("  %-16s %-4s %5s %8s %8s   %-28s %-22s %8s"
              % ("기기", "상태", "창", "참 W", "|I2|/|I1|", "p_states[0..4]",
                 "mix[0..4]", "예측 W"))
        for app, _ in PAIRS:
            if cache.get(app) is None or app not in apps:
                continue
            F, W, S, P, r2 = cache[app]
            K = apps.index(app)
            PS, MX, PW = [], [], []
            with torch.no_grad():
                for i in range(0, len(F), 256):
                    o = model(torch.from_numpy(F[i:i + 256]).to(dev),
                              torch.from_numpy(W[i:i + 256]).to(dev))
                    PS.append(o["power_states"].float().cpu())
                    MX.append(o["power_mix"].float().cpu())
                    PW.append(o["power"].float().cpu())
            PS = torch.cat(PS).numpy(); MX = torch.cat(MX).numpy(); PW = torch.cat(PW).numpy()
            thr = 0.5 * np.median(P[P > 1]) if (P > 1).any() else 1.0
            for st in range(1, PS.shape[2]):
                m = (S == st) & (P > thr)
                if m.sum() < 3:
                    continue
                ps = np.median(PS[m][:, K, :], 0); mx = np.median(MX[m][:, K, :], 0)
                pred = float(np.median(PW[m][:, K])); true = float(np.median(P[m]))
                bad = " **비었다**" if (mx.argmax() > 0 and ps[mx.argmax()] < 0.05 * true) else ""
                print("  %-16s %-4d %5d %8.0f %8.3f   %-28s %-22s %8.0f%s"
                      % (app, st, m.sum(), true, np.median(r2[m]),
                         " ".join("%5.0f" % x for x in ps),
                         " ".join("%.2f" % x for x in mx), pred, bad))
    print("\n  읽는 법 — `mix` 가 가리키는 슬롯의 `p_states` 가 비어 있으면 **곱이 0** 이다.")
    print("  분류가 맞을수록 나빠진다 (혼합이 죽은 슬롯으로 더 몰린다). `L_state_power` 가")
    print("  꺼져 있어서 (12.35) 슬롯을 **따로** 붙잡는 항이 없다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
