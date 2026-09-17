# -*- coding: utf-8 -*-
"""충전기의 **전력 되감기 구간**으로 공선을 깬다 (14.399, 사용자 지시).

§56 이 남긴 문제: 충전기는 충전이 진행되면 전력이 **단조 감소**해 `ln P` 와 시간이
`corr −0.65` 다. 그래서 *"부하 반응"* 과 *"시간 표류"* 를 못 가른다
([[distinguish-drift-from-response]]).

★ **부호 시험** — 이것이 결정적이다:
```
  내림 구간 (P↓, t↑)   부하 가설이면 겉보기 a>0 · 표류(b<0) 가설이어도 겉보기 a>0
  오름 구간 (P↑, t↑)   부하 가설이면 **a>0 그대로** · 표류 가설이면 겉보기 **a<0**
  ⇒ 두 구간의 a 부호가 **같으면 부하** · **뒤집히면 표류**
```

★ 그리고 둘째 대조 — **토막 고정효과**:
```
  (녹화 x B초 토막) 마다 위상과 lnP 를 **각자 평균을 빼고** 적합한다.
  그러면 B 초보다 느린 **어떤 모양의 표류든** 통째로 흡수된다 (모양을 가정 안 한다).
  남는 `a` 는 **토막 안에서 전력이 흔들린 몫**만으로 맞춘 것이다.
  ⚠ 식별 점검을 같이 찍는다 — 토막 안 `lnP` 표준편차가 0 에 가까우면 못 맞춘다
```

    python -X utf8 -m src.run_diag_chgrev
"""
import argparse
from typing import List

import numpy as np

from src import env_guard  # noqa: F401

from src.model.net import state_power_w  # noqa: E402
from src.model.sigalign import rec_of  # noqa: E402
from src.run_diag_cyclelaw import DISC, ORD, resid, sweep  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

FS = 60.0                      # 주기/초


def acts_of(pool, app: str, sids):
    """활성화마다 `(z, P, 녹화, 활성화번호)` — **시간 차례를 지킨다**."""
    out = []
    for n, a in enumerate(pool.appliance_activations.get(app, [])):
        st = getattr(a, "state_id", None)
        if st is None:
            continue
        pw = state_power_w(a, False)
        m = (pw > 1.0) & np.isin(np.asarray(st), list(sids))
        if m.sum() < 600:
            continue
        out.append((np.asarray(a.net_harmonics_complex)[m]
                    / np.maximum(pw[m], 1e-6)[:, None], pw[m], rec_of(a), n))
    return out


def trend(p: np.ndarray, win: int) -> np.ndarray:
    """중앙값 필터 — 되감기를 **잡음이 아니라 추세**에서 찾는다."""
    k = max(3, win // 2 * 2 + 1)
    pad = np.pad(p, k // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), -1)


def fit(z, x, tag) -> str:
    if len(x) < 2000 or np.std(x) < 1e-3:
        return "%s n=%d **못 맞춘다** (lnP 표준편차 %.4f)" % (tag, len(x), np.std(x))
    a = sweep(z, x, ORD)[0]
    d0 = resid(z, x, 0.0, DISC)[0].mean()
    d1 = resid(z, x, a, DISC)[0].mean()
    return ("%s n=%6d  σ(lnP) %.3f  **a = %+6.2f**  잔차 %.4f -> %.4f (%+.1f%%)"
            % (tag, len(x), np.std(x), a, d0, d1, 100 * (d1 - d0) / max(d0, 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="laptop_charger")
    ap.add_argument("--states", default="1,2")
    ap.add_argument("--smooth-s", type=float, default=2.0, help="추세 중앙값 필터 (초)")
    ap.add_argument("--horizon-s", type=float, default=10.0, help="기울기를 재는 시평 (초)")
    ap.add_argument("--thr", type=float, default=0.02, help="|Δ lnP| 문턱")
    ap.add_argument("--blocks", default="10,30,60,300", help="고정효과 토막 (초)")
    a_ = ap.parse_args()
    sids = [int(x) for x in a_.states.split(",")]

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    acts = acts_of(pool, a_.app, sids)
    print("충전기 되감기로 공선 깨기 (14.399) — %s s%s · 활성화 %d개\n"
          % (a_.app, a_.states, len(acts)))

    W = int(a_.smooth_s * FS)
    H = int(a_.horizon_s * FS)
    zs, xs, ts, bl, grp = [], [], [], [], []
    nrise = nfall = nflat = 0
    for z, p, rec, n in acts:
        lp = np.log(np.maximum(trend(p, W), 1e-9))
        d = np.full(len(lp), 0.0)
        d[:-H] = lp[H:] - lp[:-H]
        d[-H:] = d[max(0, len(d) - H - 1)]
        g = np.where(d > a_.thr, 1, np.where(d < -a_.thr, -1, 0))
        nrise += int((g == 1).sum()); nfall += int((g == -1).sum()); nflat += int((g == 0).sum())
        zs.append(z); xs.append(np.log(np.maximum(p, 1e-9)))
        ts.append(np.arange(len(p)) / FS)
        bl.append(np.full(len(p), "%s#%d" % (rec, n)))
        grp.append(g)
    Z = np.concatenate(zs); X = np.concatenate(xs); T = np.concatenate(ts)
    B = np.concatenate(bl); G = np.concatenate(grp)
    print("  주기 %d개 — **오름 %d (%.1f%%)** · 내림 %d (%.1f%%) · 평평 %d (%.1f%%)"
          % (len(X), nrise, 100 * nrise / len(X), nfall, 100 * nfall / len(X),
             nflat, 100 * nflat / len(X)))
    print("  (추세 %.0fs 중앙값 · 기울기 시평 %.0fs · 문턱 |ΔlnP| %.3f)\n"
          % (a_.smooth_s, a_.horizon_s, a_.thr))

    # ── ★ 부호 시험 ────────────────────────────────────────────────────────
    print("  ★★ **부호 시험** — 같으면 부하, 뒤집히면 표류")
    rows: List[str] = []
    for tag, m in (("  내림 (P↓ t↑)", G == -1), ("  오름 (P↑ t↑)", G == 1),
                   ("  평평       ", G == 0), ("  전체       ", np.ones(len(G), bool))):
        if m.sum() < 2000:
            print("    %s n=%d — 표본 부족" % (tag, m.sum()))
            continue
        x = X[m] - np.median(X[m])
        r = float(np.corrcoef(x, T[m])[0, 1]) if np.std(x) > 0 else float("nan")
        rows.append("    " + fit(Z[m], x, tag) + "   corr(lnP,t) %+.3f" % r)
    print("\n".join(rows))

    # ── ★ 토막 고정효과 ────────────────────────────────────────────────────
    print("\n  ★★ **토막 고정효과** — B초보다 느린 표류를 모양 가정 없이 흡수한다")
    for bs in [float(v) for v in a_.blocks.split(",")]:
        key = np.char.add(B, np.char.mod("|%d", (T / bs).astype(int)))
        _, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(float)
        mean = np.bincount(inv, weights=X) / cnt
        xd = X - mean[inv]                       # 토막 안 평균을 뺀 lnP
        keep = cnt[inv] >= 60                    # 1초 미만 토막은 뺀다
        print("    " + fit(Z[keep], xd[keep], "  B=%4.0fs 토막 %5d개" % (bs, len(cnt))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
