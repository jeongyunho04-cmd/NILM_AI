# -*- coding: utf-8 -*-
"""14.41 ⑤ — **상태 혼합 붕괴가 '타깃 이후 전환' 으로 설명되는가.** 합성 대 실측, 같은 자.

14.41 이 확정한 것: 세밀 창 600사이클 중 **360(6초)이 타깃보다 뒤**이고, 그 뒤쪽에 들어온
반파 증거가 현재 상태를 뒤집는다 (미래를 지우면 mix 0.137 -> 0.939).

14.40 ⑤ 가 남긴 물음: 실측 8.0% 대 합성 0.5% 의 **13배 차이**가
  (가) 생성기가 '타깃 이후 전환' 창을 덜 만들어서인가        -> 생성기를 고친다
  (나) 아니면 실측에만 있는 다른 것인가                      -> 구조를 고친다

**자를 양쪽에 같게 댄다.** 선택은 물리 대리자로 통일한다:
    게이트>0.5 **이고** `P관측 >= 0.9·V²/R`  ->  그 기기는 통전 중일 수밖에 없다
붕괴 = `mix[통전상태] < 0.5`. 합성에서는 참 `y_state` 로 이 대리자를 **검증**까지 한다.

'타깃 이후 전환' = 세밀 채널 16(`|I2|`, arcsinh)의 **타깃 이후 평균 − 이전 평균** 이
`--onset` 을 넘는 것. 반파 기기가 타깃 뒤에 켜지는 창이다.
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

#: 그 기기의 **통전** 상태 번호 (`--res-cond-state` 규약, 14.32).
COND = {"oven": 2, "hotplate": 2, "electiric_kettle": 1, "hair_dryer": 2}
KO = {"electiric_kettle": "포트", "oven": "오븐", "hotplate": "핫플", "hair_dryer": "드라이"}
I2_CH = 16                       #: 세밀에서 짝수차 |I2| 채널 (13.12 배치 v2)
TGT = fine_target_index()        #: 600−1−360 = 239


def _fwd(model, fine, wide, dev, batch):
    MX, G, PW = [], [], []
    with torch.no_grad():
        for i in range(0, len(fine), batch):
            o = model(torch.from_numpy(np.ascontiguousarray(fine[i:i + batch])).to(dev),
                      torch.from_numpy(np.ascontiguousarray(wide[i:i + batch])).to(dev))
            MX.append(o["power_mix"].float().cpu().numpy())
            G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
            PW.append(o["power"].float().cpu().numpy())
    return np.concatenate(MX), np.concatenate(G), np.concatenate(PW)


def _onset(fine):
    """세밀 창의 `|I2|` 가 **타깃 이후**에 얼마나 오르는가 (창마다 스칼라)."""
    a = fine[:, I2_CH, :TGT + 1].mean(-1)
    b = fine[:, I2_CH, TGT + 1:].mean(-1)
    return b - a


def _report(tag, apps, fine, mix, gate, pw, P, V, on_true, y_state, y_power, onset_thr):
    ons = _onset(fine)
    print(f"\n=== {tag} · {len(fine):,}창 ===")
    print(f"  '타깃 이후 |I2| 상승 > {onset_thr}' 인 창: "
          f"**{100 * (ons > onset_thr).mean():.1f}%**   "
          f"(중앙 {np.median(ons):+.3f} · p90 {np.percentile(ons, 90):+.3f} "
          f"· p99 {np.percentile(ons, 99):+.3f})")
    print(f"  {'기기':>6}{'통전 창':>9}{'붕괴':>7}{'비율':>8}"
          f"{'│ 전환 있는 창':>14}{'붕괴':>6}{'비율':>8}"
          f"{'│ 전환 없는 창':>14}{'붕괴':>6}{'비율':>8}")
    for app, sid in COND.items():
        if app not in apps:
            continue
        j = apps.index(app)
        need = V ** 2 / RESISTIVE_OHM[app]
        m = (on_true[:, j] > 0) & (gate[:, j] > 0.5) & (P >= 0.9 * need)
        if m.sum() < 20:
            continue
        bad = m & (mix[:, j, sid] < 0.5)
        hi = m & (ons > onset_thr); lo = m & (ons <= onset_thr)
        f = lambda a, b: (100 * b.sum() / a.sum()) if a.sum() else float("nan")
        print(f"  {KO[app]:>6}{int(m.sum()):>9}{int(bad.sum()):>7}{f(m, bad):>7.1f}%"
              f"{int(hi.sum()):>14}{int((hi & bad).sum()):>6}{f(hi, hi & bad):>7.1f}%"
              f"{int(lo.sum()):>14}{int((lo & bad).sum()):>6}{f(lo, lo & bad):>7.1f}%")
        if y_state is not None:
            t = (on_true[:, j] > 0) & (y_state[:, j] == sid)
            agree = (m & t).sum() / max(t.sum(), 1)
            print(f"         └ 대리자 검증: 참 통전 {int(t.sum())}창 중 대리자가 잡은 것 "
                  f"{100 * agree:.1f}%  ·  참 통전에서의 붕괴 "
                  f"{100 * (t & (mix[:, j, sid] < 0.5)).sum() / max(t.sum(), 1):.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--real", action="store_true", help="실측 복합 파일로 (봉인 제외)")
    ap.add_argument("--onset", type=float, default=0.5,
                    help="'타깃 이후 |I2| 상승' 문턱 (arcsinh 채널 단위)")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--even-median", type=int, default=0,
                    help="짝수차 중앙값을 **강제**한다 (14.160) — 분포 밖 시험용. 안 주면 체크포인트를 따라간다")
    a = ap.parse_args()
    from src.run_gate_check import sync_even_median
    sync_even_median(a.ckpt, a.even_median)   #: 창을 짓기 **전에** 전역을 맞춘다
    if not a.holdout and not a.real:
        raise SystemExit("--holdout 또는 --real 중 하나를 줘라")

    print(f"타깃 위치 {TGT}/600 · 타깃 이후 {600 - 1 - TGT}사이클 "
          f"({(600 - 1 - TGT) / 60:.1f}초) · 문턱 {a.onset}")

    for ck in a.ckpt:
        model, apps, _ = load_model(ck, a.dev)
        model.eval()
        print(f"\n{'=' * 78}\n{ck}")
        if a.holdout:
            from src.evaluation.holdout import load_holdout
            hs = load_holdout(a.holdout)
            ti = int(hs.meta["target_index"])
            F, W = [], []
            for i in range(0, len(hs), a.batch):
                f, w = build_inputs(np.asarray(hs.X[i:i + a.batch]))
                F.append(f); W.append(w)
            F = np.concatenate(F); W = np.concatenate(W)
            mix, g, pw = _fwd(model, F, W, a.dev, a.batch)
            V = np.asarray(hs.X[:, 32, ti]).astype(np.float64)
            _report(f"합성 홀드아웃 {a.holdout}", apps, F, mix, g, pw,
                    np.asarray(hs.p_observed, np.float64), V,
                    np.asarray(hs.y_on), np.asarray(hs.y_state), np.asarray(hs.y_power),
                    a.onset)
        if a.real:
            from src.evaluation.sealing import is_sealed
            from src.run_plot_real import dense_targets, load_events
            ev = load_events()
            FF, WW, PP, VV, ON = [], [], [], [], []
            for s in sorted(ev):
                if is_sealed(s):
                    continue
                rw = dense_targets(s, stride=a.stride)
                t = rw.target_cycle / 60.0
                lab = np.zeros((len(t), len(apps)), np.int8)
                for k, app in enumerate(apps):
                    for x, y in ev[s]["intervals"].get(app, {}).get("on", []):
                        lab[(t >= x) & (t < y), k] = 1
                FF.append(rw.fine); WW.append(rw.wide)
                PP.append(rw.p_observed.astype(np.float64))
                VV.append(rw.v_observed.astype(np.float64)); ON.append(lab)
            F = np.concatenate(FF); W = np.concatenate(WW)
            mix, g, pw = _fwd(model, F, W, a.dev, a.batch)
            _report("실측 복합 (봉인 제외)", apps, F, mix, g, pw,
                    np.concatenate(PP), np.concatenate(VV), np.concatenate(ON),
                    None, None, a.onset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
