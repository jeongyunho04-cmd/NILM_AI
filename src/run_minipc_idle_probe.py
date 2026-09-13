"""미니PC IDLE 전력 예측 (12.159 의 겨냥, 12.166 의 판정용).

test_14~18 의 미니PC 는 **IDLE 전용**이다 (마우스 고장으로 켜놓기만 함).
격리 측정 참값 9.90W. 그 창에서 모델이 얼마를 내는지 잰다.

    python -m src.run_minipc_idle_probe --ckpt results/adapt_h2_s0.pt
"""
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.run_gate_check import forward_file, load_model

TRUE_W = 9.90
STEMS = ("test_14", "test_15", "test_16", "test_17", "test_18")


def _mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs:
        m[int(s * 60):min(int(e * 60), n)] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--events", default="processed_data/real_events_refined.json")
    a = ap.parse_args()
    ev = json.load(open(a.events, encoding="utf-8"))["files"]
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"미니PC IDLE 참값 {TRUE_W:.2f}W (test_14~18)")
    print(f"{'체크포인트':22s}{'파일':9s}{'ON창':>7s}{'예측W(중앙)':>12s}"
          f"{'게이트':>8s}{'p_raw':>9s}{'오차%':>9s}")
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        j = apps.index("minipc")
        tag = ck.split("/")[-1].replace(".pt", "")
        acc = []
        for stem in STEMS:
            if stem not in ev or "minipc" not in ev[stem]["intervals"]:
                continue
            n = int(ev[stem]["cycles"])
            on = _mask(ev[stem]["intervals"]["minipc"].get("on", []), n)
            d = forward_file(model, stem, dev, stride=30)
            m = on[d["targets"]]
            if m.sum() < 20:
                continue
            w = float(np.median((d["gate"][:, j] * d["p_raw"][:, j])[m]))
            acc.append(w)
            print(f"{tag:22s}{stem:9s}{int(m.sum()):7d}{w:12.2f}"
                  f"{float(np.median(d['gate'][m, j])):8.3f}"
                  f"{float(np.median(d['p_raw'][m, j])):9.2f}"
                  f"{(w / TRUE_W - 1) * 100:+8.1f}%")
        if acc:
            print(f"{tag:22s}{'** 평균 **':9s}{'':7s}{np.mean(acc):12.2f}"
                  f"{'':8s}{'':9s}{(np.mean(acc) / TRUE_W - 1) * 100:+8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
