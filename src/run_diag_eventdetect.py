# -*- coding: utf-8 -*-
"""사건 구조의 두 구멍을 메운다 (13.84.22).

13.84.21 의 천장은 (a) 참 전이 시각을 주고 (b) 합성 사건이 SMPS 로 쏠린 채로 얻었다. 여기서는
  1. 기기를 돌아가며 강제해 합성 사건 표본의 균형을 맞추고 다시 배운다
  2. 검출기를 만든다 - 시각을 안 주고 1초마다 델타를 훑어 국소 최대를 사건으로 잡는다
  3. 검출한 사건으로 타임라인을 복원해 참 시각판·창별 판정과 견준다

    python -X utf8 src/run_diag_eventdetect.py [N_SYN] [THR_W]
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
APPS9 = ("air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
         "hotplate", "laptop_charger", "minipc", "oven")
N_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
THR_W = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
MATCH_S = 5.0
CACHE = Path("results/_syn_events_bal_%d.npz" % N_SYN)


def _delta(H, P, i0):
    lo, hi = i0 - PRE * FS, i0 - POST * FS
    lo2, hi2 = i0 + POST * FS, i0 + PRE * FS
    dh = trim_mean(H[lo2:hi2], 0.2, axis=0) - trim_mean(H[lo:hi], 0.2, axis=0)
    dp = float(trim_mean(P[lo2:hi2], 0.2) - trim_mean(P[lo:hi], 0.2))
    return dh, dp


def syn_balanced(n_target):
    """기기를 돌아가며 강제해 사건 표본의 균형을 맞춘다."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return list(d["apps"]), d["X"], d["y"]
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
    gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    np.random.seed(0)
    per = {a: [] for a in APPS9}
    quota = max(1, n_target // len(APPS9))
    tries = 0
    while min(len(v) for v in per.values()) < quota and tries < 60000:
        tries += 1
        want = min(per, key=lambda a: len(per[a]))
        try:
            smp = gen.synthesize_random_window(
                window_size_cycles=W, force_active=[want], full_window_placement=False,
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
        if not ts:
            continue
        allt = np.array([t for t, _, _ in ts])
        for i0, a, up in ts:
            if a not in per or len(per[a]) >= quota * 2:
                continue
            if np.sum(np.abs(allt - i0) < GUARD * FS) > 1:
                continue
            if i0 - PRE * FS < 0 or i0 + PRE * FS > W:
                continue
            dh, dp = _delta(H, P, i0)
            if not up:
                dh, dp = -dh, -dp
            if abs(dp) < 3.0:
                continue
            per[a].append(feat(dh, dp))
    apps = [a for a in APPS9 if per[a]]
    X = np.concatenate([np.stack(per[a]) for a in apps])
    y = np.concatenate([np.full(len(per[a]), i) for i, a in enumerate(apps)])
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, apps=np.array(apps, object), X=X, y=y)
    print("  합성 사건 %d개 (창 시도 %d)" % (len(y), tries))
    for i, a in enumerate(apps):
        print("     %-18s %d" % (a, int((y == i).sum())))
    return apps, X, y


def detect(H, P, n):
    """1초마다 델타를 훑어 |dP| 국소 최대를 사건으로 잡는다."""
    cand = np.arange(PRE * FS + 1, n - PRE * FS - 1, FS)
    sc = np.zeros(len(cand))
    for j, i0 in enumerate(cand):
        lo, hi = i0 - PRE * FS, i0 - POST * FS
        lo2, hi2 = i0 + POST * FS, i0 + PRE * FS
        sc[j] = abs(float(trim_mean(P[lo2:hi2], 0.2) - trim_mean(P[lo:hi], 0.2)))
    keep, used = [], np.zeros(len(cand), bool)
    for j in np.argsort(-sc):
        if sc[j] < THR_W or used[j]:
            continue
        keep.append(int(cand[j]))
        used |= np.abs(cand - cand[j]) < GUARD * FS
    return sorted(keep)


def main():
    # 균형 생성은 실패했다(강제 활성이 깨끗한 전이를 거의 안 만든다 — 60000 시도에 5개).
    # 13.84.21 의 쏠린 표본을 그대로 쓰되 **클래스 가중**으로 균형을 흉내낸다.
    from src.run_diag_eventceiling import syn_events as _se
    cp = Path("results/_syn_events_1500.npz")
    if cp.exists():
        d = np.load(cp, allow_pickle=True)
        apps_s, X, y = list(d["apps"]), d["X"], d["y"]
    else:
        syn, _ = _se(1500)
        apps_s = sorted({a for a, _, _ in syn})
        ix = {a: i for i, a in enumerate(apps_s)}
        X = np.stack([feat(dh, dp) for _, dh, dp in syn])
        y = np.array([ix[a] for a, _, _ in syn])
    cnt = np.bincount(y, minlength=len(apps_s)).astype(float)
    w = (cnt.sum() / np.maximum(cnt, 1))[y]
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y, sample_weight=w)
    print("균형 합성 사건으로 적합 · 기기 %d종 · 표본 %d" % (len(apps_s), len(y)))
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {t: load_model("results/%s.pt" % t, dev)[:2] for t in ("cnn_v35", "cnn_v37")}
    tot = {k: [0.0, 0] for k in ("검출+추적", "참시각+추적", "cnn_v35", "cnn_v37")}
    mp = {k: [] for k in ("검출+추적", "참시각+추적", "cnn_v35", "cnn_v37")}
    PR, RC = [], []
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        spec = ev[stem]
        truth_t = sorted(t for a, d in spec["intervals"].items() for iv in d.get("on", [])
                         for t in iv if PRE < t < n / FS - PRE)
        det = detect(H, P, n)
        dt = np.array([d / FS for d in det])
        tt = np.array(truth_t)
        ok = len(dt) > 0 and len(tt) > 0
        hit = np.array([np.abs(tt - x).min() <= MATCH_S for x in dt]) if ok else np.zeros(len(dt), bool)
        rec = np.array([np.abs(dt - x).min() <= MATCH_S for x in tt]) if ok else np.zeros(len(tt), bool)
        prec = float(hit.mean()) if len(dt) else 0.0
        recall = float(rec.mean()) if len(tt) else 0.0
        err = float(np.median([np.abs(tt - x).min() for x in dt[hit]])) if hit.any() else float("nan")
        PR.append(prec)
        RC.append(recall)
        print("%-8s 참 사건 %2d · 검출 %2d · 정밀도 %.2f · 재현율 %.2f · 시각오차 중앙 %.1fs"
              % (stem, len(tt), len(dt), prec, recall, err))
        for nm, tlist in (("검출+추적", dt), ("참시각+추적", tt)):
            state = {a: np.zeros(n, bool) for a in apps_s}
            for t in tlist:
                i0 = int(round(t * FS))
                if i0 - PRE * FS < 0 or i0 + PRE * FS > n:
                    continue
                dh, dp = _delta(H, P, i0)
                up = dp > 0
                a = apps_s[int(clf.predict(feat(dh if up else -dh, dp if up else -dp)[None])[0])]
                state[a][i0:] = up
            pres = [a for a in apps_s if a in spec["appliances_present"]]
            accs = []
            for a in pres:
                yy = np.zeros(n, bool)
                for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                    yy[int(t0 * FS):int(min(t1 * FS, n))] = True
                acc = float((state[a] == yy).mean())
                accs.append(acc)
                if a == "minipc":
                    mp[nm].append((stem, acc))
            tot[nm][0] += float(np.mean(accs))
            tot[nm][1] += 1
        for tg in models:
            model, apps_m = models[tg]
            d_ = forward_file(model, stem, dev, stride=15)
            t_ = d_["targets"] / 60.0
            accs = []
            for a in [x for x in spec["appliances_present"] if x in apps_m]:
                g = d_["gate"][:, apps_m.index(a)] > 0.5
                yy = np.zeros(len(g), bool)
                for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                    yy |= (t_ >= t0) & (t_ < t1)
                acc = float((g == yy).mean())
                accs.append(acc)
                if a == "minipc":
                    mp[tg].append((stem, acc))
            tot[tg][0] += float(np.mean(accs))
            tot[tg][1] += 1
    print("")
    print("검출 평균: 정밀도 %.2f · 재현율 %.2f (문턱 %.0fW, 맞춤 허용 %.0fs)"
          % (float(np.mean(PR)), float(np.mean(RC)), THR_W, MATCH_S))
    print("평균 시간 정확도: " + " · ".join("%s %.3f" % (k, v[0] / max(v[1], 1)) for k, v in tot.items()))
    print("")
    print("미니PC 만")
    stems = [s for s, _ in mp["참시각+추적"]]
    print("  파일     " + "".join("%14s" % k for k in mp))
    for i, s in enumerate(stems):
        print("  %-8s " % s + "".join("%14.3f" % mp[k][i][1] for k in mp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
