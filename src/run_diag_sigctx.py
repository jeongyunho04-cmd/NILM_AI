# -*- coding: utf-8 -*-
"""**문맥으로 지문을 설명할 수 있나** — `sig` 상수의 오차를 회귀로 깎아 본다 (14.414).

왜 이것부터인가
--------------
14.413 이 `y_harm`(기기별 참 고조파, 위상 포함)을 캐시에 담았다. 그 자료가 말한 것:

```
  상수 `sig` 의 오차 (백색화 홀수차, `sig_state` 로 재도)
    충전기 **68.0%** · 미니PC **71.0%** · 빔 **86.5%**   ·   저항 넷 3.7~15.0%
  와트당 지문 산포   합성 86~98% · **실측 녹화 간 49~68%** · 녹화 안 11~21%
```

여기서 곧장 "손실을 다시 짜자" 로 가면 안 된다. 먼저 물을 것은 **그 오차가 문맥으로
설명되는 양인가** 다. 설명되면 `sig(문맥)` 으로 고칠 수 있고, 안 되면 그 오차는
문맥으로도 못 잡는 것이라 방향이 죽는다 — 그리고 그것이 (나) `--load-rot` 이
진 이유를 설명해 준다 (§59~61: 실측 AUC 전부 |t|<1).

⚠ **적합 자료로 판정하지 않는다.** 창을 나눠 **안 본 창**에서만 읽는다. 문맥 변수를
   늘리면 적합 오차는 반드시 준다 — 그건 아무 말도 아니다.

무엇을 넣나 (겹쳐 쌓는다)
```
  M0 상수            기기당 페이저 하나 = 지금 `sig` 와 같은 꼴
  M1 + 상태          `sig_state` 와 같은 꼴 (13.11)
  M2 + log P         부하 의존 (§14.386 의 `k = a·ln P` 가 이 자리다)
  M3 + 선로 Z        r_grid · x_grid — 자리/세션이 여기로 들어온다
  M4 + 형제 전력     다른 기기가 얼마나 켜져 있나 = **결합 문맥**
```

    python -X utf8 -m src.run_diag_sigctx <캐시경로>
"""
from typing import List
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import json  # noqa: E402
import os  # noqa: E402

from src.model.lossbuild import build_loss  # noqa: E402

APPS = ("air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven")
ODD = np.array([h % 2 == 0 for h in range(15)])
SMPS = ("minipc", "laptop_charger", "beam_projector")


def ridge_fit(X, Y, lam=1e-3):
    """(n,f) -> (n,d) 능형 회귀. 절편은 X 에 1 열로 넣어 온다."""
    A = X.T @ X + lam * np.eye(X.shape[1]) * max(np.trace(X.T @ X) / X.shape[1], 1e-9)
    return np.linalg.solve(A, X.T @ Y)


def main() -> int:
    d = sys.argv[1] if len(sys.argv) > 1 else "cache/train60_v52"
    yh = np.load(os.path.join(d, "y_harm.npy")).astype(np.float64)
    yp = np.load(os.path.join(d, "y_power.npy")).astype(np.float64)
    yon = np.load(os.path.join(d, "y_on.npy"))
    ys = np.load(os.path.join(d, "y_state.npy")).astype(int)
    zg = np.load(os.path.join(d, "z_grid.npy")).astype(np.float64)
    apps = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))["appliances"]
    loss = build_loss(list(APPS), "cpu", harm_even_magnitude=True,
                      state_signatures=True, standby_operating="session")
    hs = loss.harm_scale.numpy().astype(np.float64)
    sig = loss.sig.numpy().astype(np.float64)
    sigs = loss.sig_state.numpy().astype(np.float64)
    S = sigs.shape[1]

    #: 백색화 홀수차 실벡터 (d=16). 손실이 보는 공간이다.
    def flat(T):
        v = (T / hs[None, :, None])[:, ODD]
        return v.reshape(len(v), -1)

    rng = np.random.default_rng(0)
    n = len(yh)
    te = rng.random(n) < 0.3                      # **안 본 창** 30%
    print("  캐시 %s · 창 %d (적합 %d · 판정 %d)" % (d, n, (~te).sum(), te.sum()))
    print("\n  **안 본 창**에서의 상대오차 ‖참 − 예측‖/‖참‖ (백색화 홀수차)")
    print("  %-16s %6s | %7s %7s | M0상수 M1+상태 M2+logP M3+Z  M4+형제"
          % ("기기", "표본", "배관sig", "sig_st"))
    for k, a in enumerate(apps):
        m = yon[:, k].astype(bool) & (yp[:, k] > 3.0)
        if m.sum() < 60:
            continue
        T = flat(yh[m, k])                                       # (n,16) 참
        P = yp[m, k]
        nrm = np.linalg.norm(T, axis=1)
        # 배관이 실제로 쓰는 상수 둘
        e_pipe = np.median(np.linalg.norm(T - flat(P[:, None, None] * sig[k][None]), axis=1)
                           / np.maximum(nrm, 1e-12))
        st = np.clip(ys[m, k], 0, S - 1)
        e_st = np.median(np.linalg.norm(T - flat(P[:, None, None] * sigs[k][st]), axis=1)
                         / np.maximum(nrm, 1e-12))
        # 회귀 목표 = **와트당** 지문 (전력 축을 먼저 뺀다)
        Yw = T / np.maximum(P[:, None], 1e-6)
        one = np.ones((m.sum(), 1))
        oh_st = np.zeros((m.sum(), S)); oh_st[np.arange(m.sum()), st] = 1.0
        lp = np.log(np.maximum(P, 1e-3))[:, None]
        z = zg[m]
        sib = yp[m][:, [j for j in range(len(apps)) if j != k]]
        sib = np.concatenate([np.log1p(sib), np.log1p(sib.sum(1, keepdims=True))], 1)
        feats = [one, np.concatenate([one, oh_st], 1),
                 np.concatenate([one, oh_st, lp], 1),
                 np.concatenate([one, oh_st, lp, z], 1),
                 np.concatenate([one, oh_st, lp, z, sib], 1)]
        errs: List[float] = []
        tr = ~te[m]
        for X in feats:
            Xs = X / np.maximum(np.abs(X[tr]).max(0, keepdims=True), 1e-9)
            W = ridge_fit(Xs[tr], Yw[tr])
            pred = (Xs[~tr] @ W) * P[~tr][:, None]
            errs.append(float(np.median(np.linalg.norm(T[~tr] - pred, axis=1)
                                        / np.maximum(nrm[~tr], 1e-12))))
        star = "★" if a in SMPS else " "
        print(" %s%-16s %6d | %6.1f%% %6.1f%% | %5.1f%% %6.1f%% %6.1f%% %5.1f%% %6.1f%%"
              % (star, a, m.sum(), 100 * e_pipe, 100 * e_st, *[100 * x for x in errs]))
    print("\n  ⇒ M0 대 M4 가 크게 갈리면 그 오차는 **문맥으로 설명된다** — `sig(문맥)` 이 선다.")
    print("     안 갈리면 문맥으로도 못 잡는 몫이고, §59~61 이 진 이유가 설명된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
