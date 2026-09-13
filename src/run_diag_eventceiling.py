# -*- coding: utf-8 -*-
"""사건 구조의 천장 — 합성 전이로 배워 실측 전이를 가를 수 있는가 (13.84.21).

13.84.10 은 **학습 없는** 최근접 템플릿으로 SMPS3 0.50 을 얻었다. 여기서는 배운다:
합성 복합 창에서 전이 Δ 를 대량으로 만들어 적합하고, 실측의 깨끗한 사건에서 채점한다.
그 다음 미니PC 타임라인을 사건만으로 복원해 v35/v37 창별 판정과 견준다.

    python -X utf8 src/run_diag_eventceiling.py [N_SYN]
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from scipy.stats import trim_mean
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.run_diag_transition import collect          # 실측 사건 (오염 규칙 포함)
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

FS = 60
PRE, POST, GUARD = 13, 3, 15
W = 3600
N_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 1500


from src.model.transition import feat   # 정의는 한 곳에 (13.84.24)


def syn_events(n_target):
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
    gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    np.random.seed(0)
    out, tries = [], 0
    while len(out) < n_target and tries < 40000:
        tries += 1
        try:
            smp = gen.synthesize_random_window(
                window_size_cycles=W, full_window_placement=False,
                compute_gt_harmonics=False, target_lookahead_cycles=360,
                sustained_power_limit_w=None)
        except Exception:
            continue
        H = np.asarray(smp.harmonics_complex)
        P = np.asarray(smp.power_features)[:, 0]
        on = {a: np.asarray(smp.gt_is_on[a]).astype(bool) for a in smp.gt_is_on}
        # 창 안의 모든 전이 시각
        ts = []
        for a, m in on.items():
            d = np.diff(m.astype(np.int8))
            for i in np.nonzero(d != 0)[0]:
                ts.append((i + 1, a, "on" if d[i] > 0 else "off"))
        if not ts:
            continue
        allt = np.array([t for t, _, _ in ts])
        for i0, a, kind in ts:
            if np.sum(np.abs(allt - i0) < GUARD * FS) > 1:
                continue
            lo, hi = i0 - PRE * FS, i0 - POST * FS
            lo2, hi2 = i0 + POST * FS, i0 + PRE * FS
            if lo < 0 or hi2 > W:
                continue
            seg = np.concatenate([P[lo:hi], P[lo2:hi2]])
            dh = trim_mean(H[lo2:hi2], 0.2, axis=0) - trim_mean(H[lo:hi], 0.2, axis=0)
            dp = float(trim_mean(P[lo2:hi2], 0.2) - trim_mean(P[lo:hi], 0.2))
            if kind == "off":
                dh, dp = -dh, -dp
            if abs(dp) < 3.0:
                continue
            out.append((a, dh, dp))
    return out, tries


def main():
    E, _ = collect()
    real = [(app, dh, dp) for stem, app, kind, dh, dp in E]
    print("실측 깨끗한 사건 %d개 · 기기별 %s"
          % (len(real), {a: sum(1 for x in real if x[0] == a) for a in sorted({x[0] for x in real})}))
    syn, tries = syn_events(N_SYN)
    print("합성 사건 %d개 (창 시도 %d) · 기기별 %s"
          % (len(syn), tries, {a: sum(1 for x in syn if x[0] == a) for a in sorted({x[0] for x in syn})}))

    apps = sorted({x[0] for x in syn} | {x[0] for x in real})
    idx = {a: i for i, a in enumerate(apps)}
    Xs = np.stack([feat(d, p) for _, d, p in syn]); ys = np.array([idx[a] for a, _, _ in syn])
    Xr = np.stack([feat(d, p) for _, d, p in real]); yr = np.array([idx[a] for a, _, _ in real])
    SM = [idx[a] for a in ("beam_projector", "laptop_charger", "minipc") if a in idx]

    for nm, clf in (("로지스틱", make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))),
                    ("GBM", HistGradientBoostingClassifier(max_iter=300, random_state=0))):
        clf.fit(Xs, ys)
        pr = clf.predict(Xr)
        acc = (pr == yr).mean()
        sm = np.isin(yr, SM)
        acc_sm = (pr[sm] == yr[sm]).mean() if sm.any() else float("nan")
        in_sm = np.isin(pr[sm], SM).mean() if sm.any() else float("nan")
        mp = yr == idx.get("minipc", -1)
        acc_mp = (pr[mp] == yr[mp]).mean() if mp.any() else float("nan")
        print("  %-8s 9종 %.2f (%d/%d) · SMPS 사건 정확 %.2f (족 안에 드는 비율 %.2f) · 미니PC %.2f (%d개)"
              % (nm, acc, (pr == yr).sum(), len(yr), acc_sm, in_sm, acc_mp, mp.sum()))
        if nm == "GBM":
            print("     실측 혼동 (참값 -> 예측):")
            for a in apps:
                m = yr == idx[a]
                if not m.any():
                    continue
                cnt = {}
                for q in pr[m]:
                    cnt[apps[q]] = cnt.get(apps[q], 0) + 1
                print("       %-18s n=%2d  %s" % (a, m.sum(),
                      ", ".join("%s %d" % kv for kv in sorted(cnt.items(), key=lambda x: -x[1]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
