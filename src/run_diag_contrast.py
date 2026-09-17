# -*- coding: utf-8 -*-
"""**대조 손실이 설 수 있나** — 정답 배분이 어긋난 배분보다 잔차가 작은가 (14.412).

사용자 지적: *"상수로 손실을 가려내는 게 이중으로 왜곡을 만들 것 같다."*

맞는 지적이다. 지금 `L_harm` 은 상수 넷을 쌓아 올린다 (`sig_state` · `harm_scale` ·
`HARM_DEADZONE_PROFILE` · `HARM_ORDER_DPRIME`) — 뒤의 셋은 앞 상수의 오차를 덮으려고
얹힌 것이다. 그리고 14.411 이 쟀다: **잔차의 95% 가 순방향 모델 오차**다.

```
  obs = A·P_참 + e
  절대   ‖obs − A·P̂‖ = ‖A(P_참 − P̂) + e‖   -> ‖e‖ 가 지배하고 배분이 e 를 흡수한다
  대조   ‖obs − A·P̂‖ 대 ‖obs − A·P̃‖        -> e 가 양쪽에 들어가 **상쇄**된다
```

대조로 가려면 전제가 하나 필요하다 — **순서가 살아 있어야 한다.** 절대적합이 95%
틀린 상태에서도 정답이 오답보다 낮은가? 그것만 잰다. 손실을 짓기 전에.

```
  [1] 맞바꿈 방향   ON 인 SMPS 두 기기 사이에서 δ 와트를 옮긴다 (참 -> 오답)
  [2] 순서          잔차가 **오르는** 창의 비율. 0.5 면 동전이고 대조는 못 선다
  [3] 여유          (오답 잔차 − 참 잔차) / 창 간 산포 = 대조 문제의 d'
  [4] 대조군        **저항** 쌍으로도 같은 것을 잰다 (§34.3 은 저항 사전 오차를 0.8% 로 쟀다)
```

    python -X utf8 -m src.run_diag_contrast
    python -X utf8 -m src.run_diag_contrast --frac 0.25 0.5 1.0

⚠ `processed_data/composite_eval` 을 읽는다 — 채점 전용. HPC 에 올리지 않는다.
"""
from typing import Dict, List
import argparse
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from src.model.lossbuild import build_loss  # noqa: E402
from src.model.realdata import dense_targets  # noqa: E402
from src.run_diag_harmresid import ALIAS, on_set, partial  # noqa: E402
from src.run_scorecard import load_events  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ("air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven")
SMPS = ("minipc", "laptop_charger", "beam_projector")
RES = ("oven", "hotplate", "electiric_kettle", "hair_dryer")
STEMS = ("test_1", "test_2", "test_3", "test_4", "test_5", "test_7")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=120)
    ap.add_argument("--frac", type=float, nargs="*", default=[0.25, 0.5, 1.0],
                    help="옮기는 와트의 비율 (주는 기기 전력 대비)")
    a = ap.parse_args()

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    loss = build_loss(list(APPS), "cpu", harm_even_magnitude=True,
                      state_signatures=True, standby_operating="session")
    hs = loss.harm_scale.numpy().astype(np.float64)
    sigs = loss.sig_state.numpy().astype(np.float64)     # (K,S,H,2)
    sig = loss.sig.numpy().astype(np.float64)            # (K,H,2)
    sb = loss.standby_sig.numpy().astype(np.float64)
    nz = loss.noise_sig.numpy().astype(np.float64)
    H = sig.shape[1]
    ev = load_events()

    def lharm(pred, obs):
        e = loss._harm_err(torch.from_numpy(pred[None]).float(),
                           torch.from_numpy(obs[None]).float()).numpy()[0]
        m = loss.harm_mask.numpy()
        return float((e * m[:, None]).mean() / max(m.mean(), 1e-9))

    rows: Dict[str, List[List[float]]] = {}
    n_win = 0
    for stem in STEMS:
        if stem not in ev:
            continue
        rw = dense_targets(stem, stride=a.stride, site_transfer=None)
        tgt = np.asarray(rw.target_cycle, float) / 60.0
        obs_all = np.asarray(rw.obs_harm, float)
        for i in range(len(rw)):
            t1 = float(tgt[i]); t0 = t1 - 60.0
            if t0 < 0 or partial(ev, stem, t0, t1):
                continue
            on = on_set(ev, stem, t0, t1)
            if len(on) < 2:
                continue
            k_on = [APPS.index(x) for x in on]
            off = [j for j in range(len(APPS)) if j not in k_on]
            base = nz + sb[off].sum(0)
            obs = obs_all[i]
            y = obs - base
            cols = sigs[k_on].reshape(-1, H, 2)
            A = np.concatenate([cols[:, :, 0] / hs, cols[:, :, 1] / hs], axis=1).T
            b = np.concatenate([y[:, 0] / hs, y[:, 1] / hs])
            try:
                q, _ = nnls(A, b)
            except Exception:
                continue
            S = sigs.shape[1]
            P = q.reshape(len(k_on), S).sum(1)                 # 기기별 와트
            L_true = lharm(base + np.einsum("k,khc->hc", q, cols), obs)
            n_win += 1
            for grp, names in (("SMPS", SMPS), ("저항", RES)):
                idx = [n for n, x in enumerate(on) if x in names]
                for u, v in itertools.permutations(idx, 2):
                    if P[u] < 1.0:
                        continue
                    for f in a.frac:
                        d = P[u] * f
                        Pn = P.copy(); Pn[u] -= d; Pn[v] += d
                        #: ⚠ 기기별 와트를 **그 기기 안의 상태 비율 그대로** 다시 편다.
                        #   그래야 바뀐 것이 **배분 하나**다 (상태까지 다시 풀면 두 개다).
                        qn = q.reshape(len(k_on), S).copy()
                        for kk in (u, v):
                            s0 = qn[kk].sum()
                            qn[kk] = qn[kk] * (Pn[kk] / s0) if s0 > 1e-9 else qn[kk]
                            if s0 <= 1e-9:
                                qn[kk, 0] = Pn[kk]
                        L_bad = lharm(base + np.einsum("k,khc->hc", qn.reshape(-1), cols), obs)
                        rows.setdefault("%s f=%.2f" % (grp, f), []).append(
                            [L_true, L_bad, P[u], d])

    print("\n  창 **%d개** (전이 없는 60초 · ON 기기 2종 이상) · stride %.1f초"
          % (n_win, a.stride / 60.0))
    print("\n  %-14s %7s | %9s %9s | **순서 맞음** | 여유 d'  | 옮긴 W 중앙"
          % ("무리 · 비율", "표본", "참 L_harm", "오답 L_harm"))
    for k in sorted(rows):
        r = np.asarray(rows[k], float)
        if len(r) < 20:
            continue
        up = float((r[:, 1] > r[:, 0]).mean())
        gap = r[:, 1] - r[:, 0]
        dp = float(gap.mean() / max(gap.std(ddof=1), 1e-12))
        print("  %-14s %7d | %9.4f %9.4f |    **%5.1f%%**   | %7.2f  | %8.1f"
              % (k, len(r), r[:, 0].mean(), r[:, 1].mean(), 100 * up, dp,
                 float(np.median(r[:, 3]))))
    print("\n  ⇒ 순서가 50%% 면 동전이고 **대조 손실은 원리상 못 선다**.")
    print("     여유 d' 는 대조 문제 자체의 판별력이다 — §63 의 절대 문제 d' 1.24 와 견줘라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
