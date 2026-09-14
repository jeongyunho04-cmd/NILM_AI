# -*- coding: utf-8 -*-
"""전압 지수 0.66 이 **분포 밖 탐침의 산물인가** — 홀드아웃에서 직접 잰다 (14.17).

⚠⚠ **이 자는 치우쳐 있다 — 새 결론을 여기서 내지 마라** (14.21 · 14.29 ⑦).
    아래 95번 줄이 창을 `참 전력 봉우리 ±12%` 로 고른다. 그런데 `P = P_nom·(V/V_ref)²`
    이라 **전력으로 고르는 것이 곧 V 로 고르는 것**이고, 선택이 예측변수와 상관되어
    기울기를 끌어내린다. 그래서 **참값이 2.0 이 아니라 1.39~1.67 로 나온다**:
        치우친 자 (여기)          참값 e 1.39~1.67 · A ~1.0 · B ~1.95
        바른 자   (상태 라벨)      참값 e 1.97~2.12 · A ~1.0 · B ~1.95
    A/B 방향은 같지만 **참값이 안 맞는 자다.** 지수를 재려면
    `run_diag_sigv.exps_by_state` 를 써라 — `y_state` 라벨로 묶어 전력으로 고르지 않는다.
    [[selecting-on-the-outcome-biases-the-slope]] · [[check-the-ruler-against-a-known-value]]

사용자: *"상태 사이랑 상태 안이라니? seq2point 로 예측하는 거 아니야? 그리고 창캐시로
학습하잖아."*

그 지적으로 14.16 의 설명 ⓒ 가 무너졌다. 전력 손실이
    `_huber(out["power"] / s, tgt["y_power"] / s)`,  `s = gather(s_state, 참 상태)`
라 **상태 사이 차이는 이미 나눠져 없어진다**. 손실에 남는 것은 상태 안 상대오차뿐이다.

그리고 산수가 안 맞는다:
    전압이 만드는 전력 산포   7.1%   (창 전압 표준편차 7.90V · P ∝ V²)
    모델의 홀드아웃 MAE       6.9W  = 상대 ~2%
모델이 전압을 정말 무시했다면 오차가 **7% 밑으로 못 내려간다**. 내려갔다.

⇒ `k=0.66` 은 **α 훑기**(전류·전압·전력을 함께 α배)에서 나온 값이다. 훑기는 분포 밖으로
   나갈 수 있고 14.6 이 이미 그 함정을 한 번 밟았다 ([[sweep-axes-must-stay-in-distribution]]).
   여기서는 **훑지 않고** 홀드아웃 창을 그대로 써서, 같은 상태 안에서
   `log(예측/공칭)` 대 `log(V/V_ref)` 의 기울기를 잰다. 그것이 분포 안 지수다.

    python -X utf8 src/run_diag_vindist.py [--ckpt results/cnn_vexp_ctl.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.inputs import V_CENTER, V_SPAN, build_inputs
from src.run_gate_check import load_model

RES = ("electiric_kettle", "hair_dryer", "hotplate", "oven")
SMPS = ("beam_projector", "laptop_charger", "minipc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_vexp_ctl.pt")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32")
    a = ap.parse_args()

    d = Path(a.holdout)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    raw = np.load(d / "X.npy", mmap_mode="r")
    yp = np.asarray(np.load(d / "y_power.npy", mmap_mode="r"))
    yo = np.asarray(np.load(d / "y_on.npy", mmap_mode="r"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps_m = load_model(a.ckpt, dev)[:2]
    model.eval()
    print("체크포인트 %s · 기기 %d · 홀드아웃 창 %d" % (a.ckpt, len(apps_m), len(yp)))

    # ── 예측 ────────────────────────────────────────────────────────────
    fi, wi = build_inputs(np.asarray(raw))
    P = []
    with torch.no_grad():
        for i in range(0, len(fi), 256):
            o = model(torch.from_numpy(np.asarray(fi[i:i + 256])).to(dev),
                      torch.from_numpy(np.asarray(wi[i:i + 256])).to(dev))
            P.append(o["power"].float().cpu().numpy())
    pred = np.concatenate(P)

    # ── 창 전압 = 세밀 채널 25 의 타깃 부근 평균 ────────────────────────
    v = np.asarray(fi[:, 25]).mean(-1) * V_SPAN + V_CENTER

    print("\n창 전압 (채널 25): 중앙 %.1f V · 표준편차 %.2f V · p5~p95 %.1f~%.1f"
          % (np.median(v), v.std(), np.percentile(v, 5), np.percentile(v, 95)))

    print("\n**분포 안** 전압 지수 — 같은 상태 안에서 log(예측) ~ e·log(V)")
    print("  %-18s %6s %8s %10s %10s %8s"
          % ("기기", "상태", "창", "참값 e", "예측 e", "r(예측)"))
    for a_ in RES + SMPS:
        if a_ not in apps:
            continue
        k = apps.index(a_)
        km = apps_m.index(a_)
        on = (yo[:, k] == 1) & (yp[:, k] > 5) & (pred[:, km] > 5)
        if on.sum() < 300:
            continue
        lw = np.log(yp[on, k])
        hist, edges = np.histogram(lw, bins=40)
        for st, bi in enumerate(np.argsort(hist)[::-1][:3], 1):
            c = 0.5 * (edges[bi] + edges[bi + 1])
            # 봉우리 +-12% — 전압이 만드는 +-7% 를 덮되 이웃 상태는 안 든다
            m = on & (np.abs(np.log(np.maximum(yp[:, k], 1e-9)) - c) < 0.12)
            if m.sum() < 200:
                continue
            lv = np.log(v[m] / np.median(v[m]))
            lt = np.log(yp[m, k] / np.median(yp[m, k]))
            lp = np.log(pred[m, km] / np.median(pred[m, km]))
            et = np.polyfit(lv, lt, 1)[0]
            ep = np.polyfit(lv, lp, 1)[0]
            r = float(np.corrcoef(lv, lp)[0, 1])
            print("  %-18s %6d %8d %10.2f %10.2f %8.2f"
                  % (a_, st, m.sum(), et, ep, r))
    print("\n  참값 e 가 2 인데 예측 e 도 2 에 가까우면 **모델은 전압을 이미 읽는다**")
    print("  -> 그러면 α 훑기의 0.66 은 **분포 밖 탐침의 산물**이고 --vexp 는 고칠 것이 없는 데")
    print("     2.0 을 더한 것이 된다 (14.16 의 실측 악화와 맞는다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
