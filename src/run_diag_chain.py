# -*- coding: utf-8 -*-
"""축소판 사슬 — 인코더를 얼리고 **계수 셋만** 자료로 맞춘다 (13.84.23).

13.84.21~22 가 남긴 것: 전이 축은 실패 ② 를 푼다(참 시각에서 1.000). 그런데 방출이 과확신이라
손으로 섞으면 안 움직이고, 딱딱한 검출은 정밀도 0.60 이라 무너진다. 그래서 **검출하지 않고**
매 초에 전이 점수를 매겨 사슬에 넣고, 방출·전이·전환벌점 세 계수를 맞춘다.

⚠ 계수를 합성에서 맞추면 안 된다 — 방출은 합성에서 이미 잘 맞아 w_em ~ 1 이 뽑히는데 실측에서는
   그게 틀린 값이다. 그래서 **실측 파일 하나 빼기**로 맞추고 뺀 파일에서 채점한다(계수 3개, 파일 5개).

전이 점수에는 **'사건 아님' 부류**가 필요하다. 사건만으로 배운 분류기는 아무 시각에서나 기기를
자신 있게 지목한다. 그래서 합성에서 사건이 아닌 시각도 같이 뽑아 10번째 부류로 넣는다.

    python -X utf8 src/run_diag_chain.py [N_SYN]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
from scipy.stats import trim_mean
from sklearn.ensemble import HistGradientBoostingClassifier

from src.preprocessing import load_nilm_npz
from src.run_diag_eventceiling import feat
from src.run_gate_check import forward_file, load_model
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

FS, PRE, POST, GUARD = 60, 13, 3, 15
W = 3600
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
N_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
NONE = "__none__"
CACHE = Path("results/_syn_events_none_%d.npz" % N_SYN)


def _delta(H, P, i0):
    lo, hi = i0 - PRE * FS, i0 - POST * FS
    lo2, hi2 = i0 + POST * FS, i0 + PRE * FS
    dh = trim_mean(H[lo2:hi2], 0.2, axis=0) - trim_mean(H[lo:hi], 0.2, axis=0)
    dp = float(trim_mean(P[lo2:hi2], 0.2) - trim_mean(P[lo:hi], 0.2))
    return dh, dp


def syn_with_none(n_target):
    """사건 + '사건 아님' 표본. 뒤엣것이 없으면 분류기가 아무 데서나 기기를 지목한다."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return list(d["apps"]), d["X"], d["y"]
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
    gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    np.random.seed(0)
    rows, labs = [], []
    tries = 0
    while sum(1 for l in labs if l != NONE) < n_target and tries < 30000:
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
        ts = []
        for a, m in on.items():
            d = np.diff(m.astype(np.int8))
            for i in np.nonzero(d != 0)[0]:
                ts.append((i + 1, a, d[i] > 0))
        allt = np.array([t for t, _, _ in ts]) if ts else np.array([-10 ** 9])
        for i0, a, up in ts:
            if np.sum(np.abs(allt - i0) < GUARD * FS) > 1:
                continue
            if i0 - PRE * FS < 0 or i0 + PRE * FS > W:
                continue
            dh, dp = _delta(H, P, i0)
            if not up:
                dh, dp = -dh, -dp
            if abs(dp) < 3.0:
                continue
            rows.append(feat(dh, dp)); labs.append(a)
        # 사건 아님: 어떤 전이에서도 20초 넘게 떨어진 시각
        for _ in range(3):
            i0 = int(np.random.randint(PRE * FS + 1, W - PRE * FS - 1))
            if np.min(np.abs(allt - i0)) < 20 * FS:
                continue
            dh, dp = _delta(H, P, i0)
            up = dp > 0
            rows.append(feat(dh if up else -dh, dp if up else -dp)); labs.append(NONE)
    apps = sorted(set(labs))
    ix = {a: i for i, a in enumerate(apps)}
    X = np.stack(rows); y = np.array([ix[l] for l in labs])
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, apps=np.array(apps, object), X=X, y=y)
    print("합성 표본 %d (창 시도 %d) · 사건 아님 %d" % (len(y), tries, int((y == ix[NONE]).sum())))
    return apps, X, y


def viterbi(em1, sw_on, sw_off):
    """2상태. em1 (T,) 은 상태1 의 방출 이득, sw_on/sw_off (T,) 는 그 시각의 전환 이득."""
    T = len(em1)
    dp = np.zeros((T, 2)); bk = np.zeros((T, 2), np.int8)
    dp[0, 1] = em1[0]
    for t in range(1, T):
        # 상태 0 으로: 유지(0->0) 대 꺼짐 전이(1->0)
        a0, b0 = dp[t - 1, 0], dp[t - 1, 1] + sw_off[t]
        # 상태 1 로: 유지(1->1) 대 켜짐 전이(0->1)
        a1, b1 = dp[t - 1, 1], dp[t - 1, 0] + sw_on[t]
        dp[t, 0] = max(a0, b0); bk[t, 0] = 0 if a0 >= b0 else 1
        dp[t, 1] = max(a1, b1) + em1[t]; bk[t, 1] = 1 if a1 >= b1 else 0
    path = np.zeros(T, np.int8); path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bk[t, path[t]]
    return path.astype(bool)


def main():
    apps_s, X, y = syn_with_none(N_SYN)
    inone = apps_s.index(NONE)
    # ⚠ 사건 아님이 82%% 라 가중 없이 적합하면 분류기가 **참 사건에서도 사건 아님**이라 한다
    # (실측 미니PC 11개 중 9개, 로그비 −20.7). 부류 가중으로 균형을 맞춘다.
    cnt = np.bincount(y, minlength=len(apps_s)).astype(float)
    sw = (cnt.sum() / np.maximum(cnt, 1))[y]
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y, sample_weight=sw)
    print("전이 분류기 · 부류 %d (사건 아님 포함) · 표본 %d · 부류 가중 적용" % (len(apps_s), len(y)))

    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    TAG = "cnn_v37"
    model, apps_m = load_model("results/%s.pt" % TAG, dev)[:2]

    data = {}
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"]); P = np.asarray(r["power_features"])[:, 0]
        n = len(P); spec = ev[stem]
        d_ = forward_file(model, stem, dev, stride=15)
        t_ = d_["targets"] / 60.0
        grid = np.arange(PRE + 1, n / FS - PRE - 1, 1.0)          # 1초 격자
        # 방출: 게이트 로그오즈를 1초 격자에 평균으로 내린다
        em = {}
        for a in spec["appliances_present"]:
            if a not in apps_m:
                continue
            g = np.clip(d_["gate"][:, apps_m.index(a)], 1e-4, 1 - 1e-4)
            lo = np.log(g / (1 - g))
            em[a] = np.array([lo[np.abs(t_ - t) <= 0.5].mean() if np.any(np.abs(t_ - t) <= 0.5)
                              else 0.0 for t in grid])
        # 전이: 매 초 Δ 를 내고 '사건 아님' 대비 로그비
        F, DP = [], []
        for t in grid:
            dh, dp = _delta(H, P, int(round(t * FS)))
            up = dp > 0
            F.append(feat(dh if up else -dh, dp if up else -dp)); DP.append(dp)
        proba = clf.predict_proba(np.stack(F))
        DP = np.asarray(DP)
        lr = np.log(np.clip(proba, 1e-9, 1) / np.clip(proba[:, inone:inone + 1], 1e-9, 1))
        truth = {}
        for a in em:
            yy = np.zeros(len(grid), bool)
            for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                yy |= (grid >= t0) & (grid < t1)
            truth[a] = yy
        data[stem] = dict(grid=grid, em=em, lr=lr, dp=DP, truth=truth,
                          base={a: (em[a] > 0) for a in em})
        print("  %s 준비 · 격자 %d초 · 기기 %d" % (stem, len(grid), len(em)))

    def run(stem, w_em, w_tr, pen):
        d = data[stem]
        out = {}
        for a in d["em"]:
            j = apps_s.index(a) if a in apps_s else None
            s = d["lr"][:, j] if j is not None else np.zeros(len(d["grid"]))
            on = np.where(d["dp"] > 0, w_tr * s - pen, -pen)
            off = np.where(d["dp"] < 0, w_tr * s - pen, -pen)
            out[a] = viterbi(w_em * d["em"][a], on, off)
        return out

    def score(stems, w_em, w_tr, pen, only=None):
        accs = []
        for stem in stems:
            p = run(stem, w_em, w_tr, pen)
            for a in p:
                if only and a != only:
                    continue
                accs.append((p[a] == data[stem]["truth"][a]).mean())
        return float(np.mean(accs)) if accs else float("nan")

    GR_EM = [0.01, 0.03, 0.1, 0.3, 1.0]
    GR_TR = [0.3, 1.0, 3.0, 10.0]
    GR_PN = [1.0, 3.0, 10.0, 30.0]
    print("\n파일 하나 빼기 — 계수 3개를 나머지 4파일에서 맞추고 뺀 파일에서 채점 (%s)" % TAG)
    print("  파일      창별 단독   사슬(뺀 파일)   고른 계수 (w_em, w_tr, pen)")
    held, base_all = [], []
    for stem in FILES:
        tr = [s for s in FILES if s != stem]
        best, bp = -1, None
        for we in GR_EM:
            for wt in GR_TR:
                for pn in GR_PN:
                    v = score(tr, we, wt, pn)
                    if v > best:
                        best, bp = v, (we, wt, pn)
        b = float(np.mean([(data[stem]["base"][a] == data[stem]["truth"][a]).mean()
                           for a in data[stem]["em"]]))
        h = score([stem], *bp)
        held.append(h); base_all.append(b)
        print("  %-8s %9.3f %13.3f    %s" % (stem, b, h, bp))
    print("  평균     %9.3f %13.3f" % (float(np.mean(base_all)), float(np.mean(held))))

    print("\n미니PC 만")
    print("  파일      창별 단독   사슬(뺀 파일)")
    for stem in FILES:
        if "minipc" not in data[stem]["em"]:
            continue
        tr = [s for s in FILES if s != stem]
        best, bp = -1, None
        for we in GR_EM:
            for wt in GR_TR:
                for pn in GR_PN:
                    v = score(tr, we, wt, pn, only="minipc")
                    if v > best:
                        best, bp = v, (we, wt, pn)
        b = float((data[stem]["base"]["minipc"] == data[stem]["truth"]["minipc"]).mean())
        h = score([stem], *bp, only="minipc")
        print("  %-8s %9.3f %13.3f    %s" % (stem, b, h, bp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
