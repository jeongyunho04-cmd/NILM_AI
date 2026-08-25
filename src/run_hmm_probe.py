"""
게이트 시계열에 지속성 사전을 걸어 본다 — 시간 상태 (설계 문서 12.85절)
==========================================================================
모델은 **창마다 독립으로** 판정한다. 5분 전에 켜진 기기는 창 안에 전이가 없어
정상 구간 판별력만으로 답해야 하는데, 그 d′ 가 경쟁 SMPS 가 있을 때 0.65 다
(12.84.4). 그런데 **가전은 한번 켜지면 분 단위로 켜져 있다** — 그 사전지식이
입력에도 손실에도 들어 있지 않다.

    python -m src.run_hmm_probe --ckpt results/adapt_ze1.pt --ckpt-smps results/cnn_ze1.pt

[왜 후처리로 먼저 재는가]
구조 변경(순환/상태) 전에 **얻을 것이 있는지** 를 재학습 없이 확인한다.
12.85 가 잰 미검출 연속구간 분포가 그 상한을 정한다 — 미니PC 는 미검출 시간의
39.9% 가 30초 초과라 평활로 못 메운다. 프로젝터·충전기는 30초 초과가 0% 다.

[모형]
2상태 HMM (OFF/ON) 을 기기마다 독립으로. 방출은 모델의 게이트를 그대로 쓰고
(log g / log(1−g)), 전이는 `p_stay` 하나로 둔다. Viterbi 로 복호한다.
`p_stay` 를 훑어 F1 이 최대가 되는 지점을 본다 — **그 최대값이 후처리로 얻을 수
있는 상한**이고, 그것이 작으면 구조 변경도 값이 없다.
"""
from typing import Dict, List
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.run_gate_check import forward_file, load_model, merge_smps


def viterbi(g: np.ndarray, p_stay: float) -> np.ndarray:
    """2상태 HMM 복호. g 는 P(ON|창). 반환은 0/1 상태열."""
    eps = 1e-6
    e1 = np.log(np.clip(g, eps, 1 - eps))          # ON 방출
    e0 = np.log(np.clip(1 - g, eps, 1 - eps))      # OFF 방출
    ls, lsw = np.log(p_stay), np.log1p(-p_stay)
    n = len(g)
    d0, d1 = e0[0], e1[0]
    bp = np.zeros((n, 2), np.int8)
    for i in range(1, n):
        c0 = (d0 + ls, d1 + lsw)                   # -> OFF
        c1 = (d0 + lsw, d1 + ls)                   # -> ON
        bp[i, 0] = int(c0[1] > c0[0]); bp[i, 1] = int(c1[1] > c1[0])
        d0, d1 = max(c0) + e0[i], max(c1) + e1[i]
    out = np.zeros(n, np.int8)
    out[-1] = int(d1 > d0)
    for i in range(n - 1, 0, -1):
        out[i - 1] = bp[i, out[i]]
    return out.astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser(description="게이트에 지속성 사전을 건다")
    ap.add_argument("--ckpt", default="results/adapt_ze1.pt")
    ap.add_argument("--ckpt-smps", default="results/cnn_ze1.pt")
    ap.add_argument("--stride", type=int, default=30, help="30 = 0.5초")
    ap.add_argument("--apps", nargs="*", default=["minipc", "beam_projector", "laptop_charger"])
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    mdl, apps, _ = load_model(a.ckpt, dev)
    ms, _, _ = load_model(a.ckpt_smps, dev)

    # 게이트와 정답을 한 번만 모은다
    G: Dict[str, List[np.ndarray]] = {}
    T: Dict[str, List[np.ndarray]] = {}
    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        d = merge_smps(forward_file(mdl, stem, dev, stride=a.stride),
                       forward_file(ms, stem, dev, stride=a.stride), apps)
        t = d["targets"] / 60.0
        for app in a.apps:
            if app not in ev[stem]["appliances_present"]:
                continue
            on = np.zeros(len(t), bool)
            for x, y in ev[stem]["intervals"][app]["on"]:
                on |= (t >= x) & (t <= y)
            G.setdefault(app, []).append(d["gate"][:, apps.index(app)])
            T.setdefault(app, []).append(on)

    dt = a.stride / 60.0
    print(f"창 간격 {dt:.2f}초 | 파일별 독립 복호\n")
    print(f"{'기기':16s}{'p_stay':>9s}{'평균 지속':>10s}{'재현율':>9s}{'정밀도':>9s}{'F1':>8s}")
    for app in a.apps:
        if app not in G:
            continue
        gs, ts = G[app], T[app]
        base_r = np.concatenate([(g > 0.5)[t] for g, t in zip(gs, ts)]).mean()
        base_p = np.concatenate([t[g > 0.5] for g, t in zip(gs, ts)]).mean()
        print(f"{app:16s}{'(원본)':>9s}{'-':>10s}{100*base_r:>8.1f}%{100*base_p:>8.1f}%"
              f"{2*base_r*base_p/max(base_r+base_p,1e-9):>8.3f}")
        for p_stay in (0.9, 0.99, 0.999, 0.9999, 0.99999):
            dec = [viterbi(g, p_stay) for g in gs]
            r = np.concatenate([d[t] for d, t in zip(dec, ts)]).mean()
            pr = np.concatenate([t[d] for d, t in zip(dec, ts)]).mean()
            f1 = 2 * r * pr / max(r + pr, 1e-9)
            mark = "  <-" if f1 > 2*base_r*base_p/max(base_r+base_p,1e-9) else ""
            print(f"{'':16s}{p_stay:>9.5f}{dt/(1-p_stay):>9.0f}s{100*r:>8.1f}%{100*pr:>8.1f}%"
                  f"{f1:>8.3f}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
