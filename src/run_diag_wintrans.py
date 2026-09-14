# -*- coding: utf-8 -*-
"""14.43 ④ — **창 안 전환이 캐시/홀드아웃에 정말 없나.** 전력 계단으로 넓게 센다.

14.43 은 `|I2|`(반파 증거) 하나만, 그것도 오븐 통전 창에서만 봤다. 여기서는 **전력 계단**을
센다 — 기기 종류를 안 가리고, 타깃 앞/뒤를 갈라서.

세밀 채널 23 = `asinh(P/100)` 이므로 `P = 100·sinh(x)` 로 되돌린다. 600사이클(10초) 안에서
`|ΔP| > --step` 인 지점을 세고, 타깃(239) 앞/뒤로 나눈다.

⚠ 핫플 릴레이는 **주기 2초**다 (14.39: 통전 0.47초 · 진폭 455W). 통전 중인 핫플이 든 창이면
  10초에 **전환이 ~10번** 있어야 한다. 0 에 가까우면 생성기가 듀티를 안 만든 것이다.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model.inputs import POWER_SCALE, build_inputs, fine_target_index  # noqa: E402
from src.model.net import P_CH_FINE                                    # noqa: E402

TGT = fine_target_index()


def _count(fine, step):
    """창마다 (앞 전환수, 뒤 전환수, 전체 최대 |ΔP|)."""
    p = POWER_SCALE * np.sinh(fine[:, P_CH_FINE].astype(np.float64))
    d = np.abs(np.diff(p, axis=-1))                       # (n, 599)
    hit = d > step
    return hit[:, :TGT].sum(-1), hit[:, TGT:].sum(-1), d.max(-1)


I2_CH = 16          #: 짝수차 |I2| (세밀 배치 v2)


def _hw(tag, fine, lo, hi):
    """**날카로운 계수** — 반파 기기가 창 **중간에 처음 켜지는** 배치가 몇 % 인가.

    14.43 ② 가 `|I2|` 의 '미래 평균 − 과거 평균' 으로 쟀는데 그것은 **이미 켜져 있고
    값이 조금 오르는 것**도 같이 센다 (합성이 정확히 그랬다: 타깃 |I2| 3.493 · 미래 점유 100%).
    여기서는 **타깃에서 없고(<lo) 미래에 있는(>hi)** 것만 센다.
    """
    i2 = fine[:, I2_CH]
    at = i2[:, TGT]
    fut = i2[:, TGT + 1:]
    pre = i2[:, :TGT + 1].max(-1)
    on_now = at > hi
    on_future = fut.max(-1) > hi
    # 창 중간에 **처음** 켜짐: 타깃에서도 과거 전체에서도 없었는데 미래에 있다
    fresh = (pre < lo) & on_future
    print(f"  [날카로운 계수] 타깃 |I2| < {lo} **이면서** 미래 |I2| > {hi} 인 창: "
          f"**{100 * fresh.mean():.2f}%** ({int(fresh.sum())}/{len(fine)})")
    print(f"     비교 — 타깃에 이미 켜져 있는 창 {100 * on_now.mean():.2f}%"
          f" · 미래 어딘가에 있는 창 {100 * on_future.mean():.2f}%"
          f" · 과거 최대 |I2| 중앙 {np.median(pre):.3f}")
    if fresh.any():
        frac = (fut[fresh] > hi).mean(-1)
        print(f"     그 창의 미래 구간 점유 중앙 {np.median(frac):.0%}"
              f" · 미래 최대 |I2| 중앙 {np.median(fut[fresh].max(-1)):.3f}")


def _report(tag, fine, step, sel=None, names=None):
    a, b, mx = _count(fine, step)
    tot = a + b
    n = len(fine)
    print(f"\n=== {tag} · {n:,}창 · 계단 문턱 {step:.0f}W ===")
    print(f"  전환이 **하나라도** 있는 창: {100*(tot>0).mean():5.1f}%"
          f"   │ 앞(타깃 이전)만 {100*((a>0)&(b==0)).mean():5.1f}%"
          f"   │ 뒤(타깃 이후)만 {100*((a==0)&(b>0)).mean():5.1f}%"
          f"   │ 양쪽 {100*((a>0)&(b>0)).mean():5.1f}%")
    print(f"  창당 전환 수: 중앙 {np.median(tot):.0f} · 평균 {tot.mean():.2f}"
          f" · p90 {np.percentile(tot,90):.0f} · 최대 {tot.max():.0f}"
          f"   │ 최대 |ΔP| 중앙 {np.median(mx):.0f}W")
    if sel is not None:
        for nm, m in zip(names, sel):
            if m.sum() < 20:
                continue
            t2 = tot[m]
            print(f"    {nm:>16s} {int(m.sum()):>6}창 · 전환 있는 창 {100*(t2>0).mean():5.1f}%"
                  f" · 창당 중앙 {np.median(t2):.0f} · 평균 {t2.mean():.2f}"
                  f" · p90 {np.percentile(t2,90):.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--step", type=float, default=100.0)
    ap.add_argument("--hw-lo", type=float, default=0.5, help="타깃에서 '반파 없음' 문턱")
    ap.add_argument("--hw-hi", type=float, default=2.0, help="미래에서 '반파 있음' 문턱")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    print(f"타깃 {TGT}/600 · 뒤 {599-TGT}사이클 ({(599-TGT)/60:.1f}초)")

    if a.holdout:
        from src.evaluation.holdout import load_holdout
        hs = load_holdout(a.holdout)
        F = []
        for i in range(0, len(hs), a.batch):
            f, _ = build_inputs(np.asarray(hs.X[i:i + a.batch]))
            F.append(f)
        F = np.concatenate(F)
        apps = list(hs.appliances)
        yon = np.asarray(hs.y_on)
        sel, nm = [], []
        for app in ("hotplate", "oven", "electiric_kettle", "hair_dryer"):
            if app in apps:
                sel.append(yon[:, apps.index(app)] > 0); nm.append(app)
        _report(f"합성 홀드아웃 {a.holdout}", F, a.step, sel, nm)
        _hw(f"합성 홀드아웃 {a.holdout}", F, a.hw_lo, a.hw_hi)

    if a.real:
        from src.evaluation.sealing import is_sealed
        from src.run_plot_real import dense_targets, load_events
        ev = load_events()
        FF, LAB = [], []
        apps = ("hotplate", "oven", "electiric_kettle", "hair_dryer")
        for s in sorted(ev):
            if is_sealed(s):
                continue
            rw = dense_targets(s, stride=a.stride)
            t = rw.target_cycle / 60.0
            L = np.zeros((len(t), len(apps)), bool)
            for k, app in enumerate(apps):
                for x, y in ev[s]["intervals"].get(app, {}).get("on", []):
                    L[(t >= x) & (t < y), k] = True
            FF.append(rw.fine); LAB.append(L)
        F = np.concatenate(FF); L = np.concatenate(LAB)
        _report("실측 복합 (봉인 제외)", F, a.step,
                [L[:, k] for k in range(len(apps))], list(apps))
        _hw("실측 복합 (봉인 제외)", F, a.hw_lo, a.hw_hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
