# -*- coding: utf-8 -*-
"""14.42 ③ 의 남은 물음 — **실측의 '미래 반파 증거' 가 합성의 것과 무엇이 다른가.**

14.42 가 확정한 것:
  · 오븐 혼합 붕괴는 실측 **7.3±1.1%**(조인 자) 대 합성 **0.4~0.7%**(참 y_state) — ~12배.
  · '타깃 이후 |I2| 상승>0.5' 창은 합성 **4.3%** · 실측 **1.5%** — 생성기가 오히려 더 만든다.
⇒ 빈도가 아니다. 그러면 **증거의 성질**이다. 여기서 그것을 맞댄다.

자를 양쪽에 같게:
  합성 — 오븐이 **참으로 통전**(`y_state[오븐]==2`)인 창
  실측 — 오븐 라벨 ON · **다른 저항기기 전부 라벨 OFF** · `P >= 0.9·V²/R_오븐`
          (다른 큰 부하가 없으니 그 전력을 낼 수 있는 것은 오븐뿐이다)
둘 다 '미래 전환' 으로 쪼개고, 그 창에서 **반파 증거의 크기·지속·동시성**을 찍는다.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model.inputs import build_inputs, fine_target_index   # noqa: E402
from src.model.postproc import RESISTIVE_OHM                   # noqa: E402
from src.run_gate_check import load_model                      # noqa: E402

I2_CH, I1_RE, I1_IM = 16, 0, 8      #: 세밀 배치 v2 — |I2| · Re(I1) · Im(I1)
TGT = fine_target_index()
RES = ("electiric_kettle", "oven", "hotplate", "hair_dryer")
OVEN_COND, DRYER_HW = 2, 1          #: 오븐 HEATING · 드라이 약풍(반파)


def _stats(fine, mask, ok, tag, hi_thr=2.0):
    """**실제로 쓴 마스크 그대로** 반파 증거의 성질을 찍는다 (14.47).

    ⚠ 14.43 에서 이 함수가 **무딘 자(평균차이)의 부분집합**을 서술하고 있었다 —
      판정은 날카로운 자로 하면서 서술은 다른 창을 보고 있었던 것이다. 마스크를 받는다.
    """
    if mask.sum() < 5:
        print(f"    {tag}: 표본 {int(mask.sum())}개 — 못 읽는다")
        return
    i2 = fine[mask][:, I2_CH]
    i1 = np.hypot(fine[mask][:, I1_RE], fine[mask][:, I1_IM])
    fut = i2[:, TGT + 1:]
    on = fut > hi_thr
    n = len(i2)
    # 반파가 미래에서 **처음 켜지는 시점** (타깃 기준 초)
    first = np.full(n, np.nan)
    any_on = on.any(-1)
    first[any_on] = on[any_on].argmax(-1) / 60.0
    occ = on.mean(-1)                                    # 미래 구간 점유
    run = on.sum(-1) / 60.0                              # 켜져 있는 총 시간(초)
    r21 = (i2[:, TGT + 1:] / np.maximum(i1[:, TGT + 1:], 1e-6))
    print(f"    {tag}  n={n} · 붕괴 {100 * (~ok[mask]).mean():.1f}%")
    print(f"      타깃 |I2| {np.median(i2[:, TGT]):.3f}"
          f" · 과거 최대 {np.median(i2[:, :TGT + 1].max(-1)):.3f}"
          f" · 미래 최대 {np.median(fut.max(-1)):.3f}")
    print(f"      **미래 점유 {np.median(occ):.0%}** (p25 {np.percentile(occ, 25):.0%}"
          f" · p75 {np.percentile(occ, 75):.0%})"
          f" · 켜진 시간 중앙 **{np.median(run):.2f}초** / 6.0초")
    print(f"      **켜지는 시점** 타깃 +{np.nanmedian(first):.2f}초"
          f" (p25 +{np.nanpercentile(first, 25):.2f} · p75 +{np.nanpercentile(first, 75):.2f})")
    print(f"      미래 |I2|/|I1| 최대 중앙 {np.median(r21.max(-1)):.3f}"
          f" · 타깃 |I1| {np.median(i1[:, TGT]):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--onset", type=float, default=0.5)
    ap.add_argument("--sharp", action="store_true",
                    help="'미래 전환' 을 **날카롭게** 정의한다 — 과거 전체에서 |I2| 최대가 "
                         "--hw-lo 미만이고 미래 최대가 --hw-hi 초과. 평균차이 자는 "
                         "'이미 켜져 있는데 값이 조금 오르는 것' 도 같이 세었다 (14.43 ②).")
    ap.add_argument("--hw-lo", type=float, default=0.5)
    ap.add_argument("--hw-hi", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()

    for ck in a.ckpt:
        model, apps, _ = load_model(ck, a.dev); model.eval()
        jo = apps.index("oven")
        print(f"\n{'=' * 78}\n{ck}")

        if a.holdout:
            from src.evaluation.holdout import load_holdout
            hs = load_holdout(a.holdout)
            F, W = [], []
            for i in range(0, len(hs), a.batch):
                f, w = build_inputs(np.asarray(hs.X[i:i + a.batch]))
                F.append(f); W.append(w)
            F = np.concatenate(F); W = np.concatenate(W)
            MX = []
            with torch.no_grad():
                for i in range(0, len(F), a.batch):
                    o = model(torch.from_numpy(np.ascontiguousarray(F[i:i + a.batch])).to(a.dev),
                              torch.from_numpy(np.ascontiguousarray(W[i:i + a.batch])).to(a.dev))
                    MX.append(o["power_mix"].float().cpu().numpy())
            MX = np.concatenate(MX)
            m = (np.asarray(hs.y_on)[:, jo] > 0) & (np.asarray(hs.y_state)[:, jo] == OVEN_COND)
            ok = MX[:, jo, OVEN_COND] >= 0.5
            ons = F[:, I2_CH, TGT + 1:].mean(-1) - F[:, I2_CH, :TGT + 1].mean(-1)
            if a.sharp:
                fresh = ((F[:, I2_CH, :TGT + 1].max(-1) < a.hw_lo)
                         & (F[:, I2_CH, TGT + 1:].max(-1) > a.hw_hi))
            else:
                fresh = ons > a.onset
            hi = m & fresh; lo = m & ~fresh
            f = lambda s, b: (100 * b.sum() / s.sum()) if s.sum() else float("nan")
            print(f"  [합성 · 참 y_state] 오븐 통전 {int(m.sum())}창 · 붕괴 {f(m, m & ~ok):.1f}%"
                  f"  │ 미래 전환 있음 {int(hi.sum())}창 **{f(hi, hi & ~ok):.1f}%**"
                  f"  │ 없음 {int(lo.sum())}창 {f(lo, lo & ~ok):.1f}%")
            jd = apps.index("hair_dryer")
            co = m & (np.asarray(hs.y_on)[:, jd] > 0) & (np.asarray(hs.y_state)[:, jd] == DRYER_HW)
            print(f"  [동시성] 오븐 HEATING **과 동시에** 드라이 약풍(반파) 인 창: "
                  f"**{int(co.sum())}** / {int(m.sum())} = {100*co.sum()/max(m.sum(),1):.1f}%")
            _stats(F, hi, ok, "합성 · 결합 창(오븐 통전 × 반파 미래)")
            _stats(F, lo, ok, "합성 · 전환 없는 창")

        if a.real:
            from src.evaluation.sealing import is_sealed
            from src.run_plot_real import dense_targets, load_events
            ev = load_events()
            FA, MA, SEL = [], [], []
            for s in sorted(ev):
                if is_sealed(s):
                    continue
                rw = dense_targets(s, stride=a.stride); t = rw.target_cycle / 60.0
                P = rw.p_observed.astype(np.float64); V = rw.v_observed.astype(np.float64)
                MX = []
                with torch.no_grad():
                    for i in range(0, len(P), a.batch):
                        o = model(torch.from_numpy(np.ascontiguousarray(rw.fine[i:i+a.batch])).to(a.dev),
                                  torch.from_numpy(np.ascontiguousarray(rw.wide[i:i+a.batch])).to(a.dev))
                        MX.append(o["power_mix"].float().cpu().numpy())
                MX = np.concatenate(MX)
                lab = {}
                for x in RES:
                    mm = np.zeros(len(t), bool)
                    for p, q in ev[s]["intervals"].get(x, {}).get("on", []):
                        mm |= (t >= p) & (t < q)
                    lab[x] = mm
                others = np.zeros(len(t), bool)
                for x in RES:
                    if x != "oven":
                        others |= lab[x]
                sel = lab["oven"] & ~others & (P >= 0.9 * V ** 2 / RESISTIVE_OHM["oven"])
                FA.append(rw.fine); MA.append(MX); SEL.append(sel)
            F = np.concatenate(FA); MX = np.concatenate(MA); m = np.concatenate(SEL)
            ok = MX[:, jo, OVEN_COND] >= 0.5
            ons = F[:, I2_CH, TGT + 1:].mean(-1) - F[:, I2_CH, :TGT + 1].mean(-1)
            if a.sharp:
                fresh = ((F[:, I2_CH, :TGT + 1].max(-1) < a.hw_lo)
                         & (F[:, I2_CH, TGT + 1:].max(-1) > a.hw_hi))
            else:
                fresh = ons > a.onset
            hi = m & fresh; lo = m & ~fresh
            f = lambda s, b: (100 * b.sum() / s.sum()) if s.sum() else float("nan")
            print(f"  [실측 · 조인 자] 오븐 단독통전 {int(m.sum())}창 · 붕괴 {f(m, m & ~ok):.1f}%"
                  f"  │ 미래 전환 있음 {int(hi.sum())}창 **{f(hi, hi & ~ok):.1f}%**"
                  f"  │ 없음 {int(lo.sum())}창 {f(lo, lo & ~ok):.1f}%")
            _stats(F, hi, ok, "실측 · 결합 창(오븐 단독통전 × 반파 미래)")
            _stats(F, lo, ok, "실측 · 전환 없는 창")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
