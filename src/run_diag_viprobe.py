# -*- coding: utf-8 -*-
"""몸통이 **V·I 곱**을 표현하나 — 재학습 없이 선형 탐침 (14.18).

14.17 이 남긴 질문: 모델은 전압 응답의 **65% 만** 읽는다. 왜인가.
저항 부하의 전력은 `P = V·I` 인데 **곱**이다. 몸통(`self.trunk`, 444행 `z = self.trunk(x)`)의
표현 `z` 에서 그 곱을 선형으로 읽을 수 있어야 헤드(`nn.Linear`)가 쓸 수 있다.

  ① log|I1|      전류 크기 — 이건 당연히 읽혀야 한다
  ② log V        전압 — 채널이 하나(세밀 25)뿐이다
  ③ log(V·|I1|)  **곱**. 선형 헤드가 쓰려면 몸통이 이미 만들어 둬야 한다
  ④ log(V²·..)   참고

  탐침은 **능선 회귀**이고 **창을 시간 블록으로 가르지 않는다** — 홀드아웃 창은 서로
  독립이라(캐시가 독립 창이다) 누출 걱정이 없다. 그래도 절반 나눠 검정한다.

⚠ ③ 이 ① 보다 눈에 띄게 낮으면 **곱이 표현에 없다**는 뜻이고, 그것이 65% 의 원인 후보다.
⚠ ③ 이 높은데도 모델이 못 쓰면 원인은 몸통이 아니라 **헤드나 손실** 쪽이다.

    python -X utf8 src/run_diag_viprobe.py [--ckpt results/cnn_vexp_ctl.pt]
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


def ridge_r2(Z, y, lam=1e-2):
    """절반으로 나눠 학습/검정. 표준화 능선."""
    n = len(Z)
    idx = np.arange(n)
    rng = np.random.default_rng(0); rng.shuffle(idx)
    tr, te = idx[:n // 2], idx[n // 2:]
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
    Xtr, Xte = (Z[tr] - mu) / sd, (Z[te] - mu) / sd
    ym = y[tr].mean()
    w = np.linalg.solve(Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1]),
                        Xtr.T @ (y[tr] - ym))
    p = Xte @ w + ym
    return float(1 - ((y[te] - p) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_vexp_ctl.pt")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32")
    a = ap.parse_args()

    d = Path(a.holdout)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    raw = np.asarray(np.load(d / "X.npy", mmap_mode="r"))
    yp = np.asarray(np.load(d / "y_power.npy", mmap_mode="r"))
    yo = np.asarray(np.load(d / "y_on.npy", mmap_mode="r"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps_m = load_model(a.ckpt, dev)[:2]
    model.eval()

    fi, wi = build_inputs(raw)
    # ── 몸통 표현을 갈고리로 받는다 ─────────────────────────────────────
    feats = []
    h = model.trunk.register_forward_hook(lambda m, i, o: feats.append(o.detach().float().cpu().numpy()))
    with torch.no_grad():
        for i in range(0, len(fi), 256):
            model(torch.from_numpy(fi[i:i + 256]).to(dev),
                  torch.from_numpy(wi[i:i + 256]).to(dev))
    h.remove()
    Z = np.concatenate(feats)
    print("체크포인트 %s · 창 %d · 몸통 표현 %d차원" % (a.ckpt, len(Z), Z.shape[1]))

    # ── 목표 ────────────────────────────────────────────────────────────
    #   |I1| 은 원시 채널 0 (기본파 전류 크기), V 는 세밀 채널 25
    ti = fi.shape[-1] // 2
    I1 = np.asarray(raw[:, 0]).mean(-1)
    V = np.asarray(fi[:, 25]).mean(-1) * V_SPAN + V_CENTER
    ok = (I1 > 0.02) & np.isfinite(I1) & np.isfinite(V)
    Zo, I1o, Vo = Z[ok], I1[ok], V[ok]
    print("   유효 창 %d · |I1| 중앙 %.3f A · V 중앙 %.1f V (표준편차 %.2f)"
          % (ok.sum(), np.median(I1o), np.median(Vo), Vo.std()))

    tg = {
        "log|I1|": np.log(I1o),
        "log V": np.log(Vo),
        "log(V·|I1|)": np.log(Vo * I1o),
        "log(V²·|I1|)": np.log(Vo ** 2 * I1o),
        "log(관측 전력)": np.log(np.maximum(np.asarray(raw[ok, 30]).mean(-1), 1e-3)),
    }
    print("\n**선형 탐침** — 몸통 표현 z 에서 얼마나 읽히나 (능선, 절반 검정)")
    print("  %-16s %10s %10s" % ("목표", "R²", "표준편차"))
    res = {}
    for n_, y in tg.items():
        r = ridge_r2(Zo, y)
        res[n_] = r
        print("  %-16s %10.3f %10.4f" % (n_, r, y.std()))

    print("\n  읽는 법")
    print("   · log V 가 낮으면 **몸통이 전압을 안 들고 있다** -> 헤드가 쓸 수가 없다")
    print("   · log(V·|I1|) 이 log|I1| 보다 눈에 띄게 낮으면 **곱이 표현에 없다**")
    print("   · 둘 다 높은데 예측 지수가 0.65 면 원인은 몸통이 아니라 헤드/손실이다")
    print("\n  차이: log(V·I) − log|I1| = %+.3f · log V = %.3f"
          % (res["log(V·|I1|)"] - res["log|I1|"], res["log V"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
