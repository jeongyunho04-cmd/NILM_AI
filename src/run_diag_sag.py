# -*- coding: utf-8 -*-
"""고전력 합계 초과가 **전압 강하**로 설명되는가 (13.84.18).

13.84.17 ①: test_3·5(자리 D)에서 2200W 위 Σ예측−관측이 +6.6~9.2%. 커버리지 구멍은 아니었다.
저항 부하는 P = V²/R 이라 전압이 처지면 실제 전력이 공칭보다 작다. 모델이 공칭 전력을 내놓으면
그 차이가 그대로 초과가 된다. 예측 초과 대 (V_무부하/V_관측)² − 1 을 견준다.

    python -X utf8 src/run_diag_sag.py cnn_v35
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.run_gate_check import forward_file, load_model

FILES = ["test_1", "test_2", "test_3", "test_5"]
BANDS = [(200, 800), (800, 1500), (1500, 2200), (2200, 9999)]
STRIDE = 15


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "cnn_v35"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps = load_model("results/%s.pt" % tag, dev)[:2]
    print("%s · 무부하 전압 V0 = 그 파일의 관측 P 최저 5%% 구간 전압 중앙\n" % tag)
    print("파일     대역        n     V 중앙   V0     강하    Σ예측−관측   초과%   (V0/V)²−1   비")
    for stem in FILES:
        d = forward_file(model, stem, dev, stride=STRIDE)
        P = (d["gate"] * d["p_raw"]).sum(1)
        obs = d["p_observed"]
        V = d["v_rms"]
        lo_idx = obs <= np.percentile(obs, 5)
        V0 = float(np.median(V[lo_idx]))
        for lo, hi in BANDS:
            m = (obs >= lo) & (obs < hi)
            if m.sum() < 10:
                continue
            v = float(np.median(V[m]))
            err = float(np.median(P[m] - obs[m]))
            rel = 100 * err / max(np.median(obs[m]), 1.0)
            pred = 100 * ((V0 / v) ** 2 - 1.0)
            print("%-8s %4d~%-5d %5d  %7.1f %6.1f %6.1f  %+9.1f W %+7.1f%% %9.1f%% %7.2f"
                  % (stem, lo, hi, m.sum(), v, V0, V0 - v, err, rel, pred,
                     rel / pred if abs(pred) > 0.2 else float("nan")))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
