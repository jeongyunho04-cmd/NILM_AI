# -*- coding: utf-8 -*-
"""창별 방출 + 사건 전이를 한 상태 모형으로 묶으면 둘 다 이기는가 (13.84.21 셋째 반).

기기마다 2상태 사슬:
    방출 = w · (CNN 게이트 로그오즈)          <- w 를 훑는다. w=0 이면 사건만, w=1 이면 지금 그대로
    전이 = 참 전이 시각에서 사건 분류기가 그 기기를 지목하면 +BONUS, 아니면 −STAY
Viterbi 로 푼다. 검출은 완벽하다고 놓은 **천장** 측정이다.

    python -X utf8 src/run_diag_fusion.py [N_SYN] [BONUS]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
from scipy.stats import trim_mean
from sklearn.ensemble import HistGradientBoostingClassifier

from src.preprocessing import load_nilm_npz
from src.run_diag_eventceiling import feat, syn_events
from src.run_gate_check import forward_file, load_model

FS, PRE, POST = 60, 13, 3
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
N_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
BONUS = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
STAY = 3.0
EM_W = (1.0, 0.3, 0.1, 0.03, 0.0)
CACHE = Path("results/_syn_events_%d.npz" % N_SYN)


def get_syn():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return list(d["apps"]), d["X"], d["y"]
    syn, _ = syn_events(N_SYN)
    apps = sorted({a for a, _, _ in syn})
    idx = {a: i for i, a in enumerate(apps)}
    X = np.stack([feat(dh, dp) for _, dh, dp in syn])
    y = np.array([idx[a] for a, _, _ in syn])
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, apps=np.array(apps, object), X=X, y=y)
    return apps, X, y


def viterbi(em, sw):
    T = len(em)
    dp = np.full((T, 2), -1e18)
    bk = np.zeros((T, 2), int)
    dp[0] = em[0]
    for t in range(1, T):
        for s in (0, 1):
            stay = dp[t - 1, s]
            flip = dp[t - 1, 1 - s] + sw[t]
            if flip > stay:
                dp[t, s] = flip + em[t, s]
                bk[t, s] = 1 - s
            else:
                dp[t, s] = stay + em[t, s]
                bk[t, s] = s
    path = np.zeros(T, int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bk[t, path[t]]
    return path.astype(bool)


def main():
    apps_s, X, y = get_syn()
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y)
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tags = ["cnn_v35", "cnn_v37"]
    models = {t: load_model("results/%s.pt" % t, dev)[:2] for t in tags}
    agg, mp_rows = {}, []
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        spec = ev[stem]
        times = sorted(t for a, d in spec["intervals"].items() for iv in d.get("on", [])
                       for t in iv if PRE < t < n / FS - PRE)
        called = []
        for t in times:
            i0 = int(round(t * FS))
            dh = (trim_mean(H[i0 + POST * FS:i0 + PRE * FS], 0.2, axis=0)
                  - trim_mean(H[i0 - PRE * FS:i0 - POST * FS], 0.2, axis=0))
            dp_ = float(trim_mean(P[i0 + POST * FS:i0 + PRE * FS], 0.2)
                        - trim_mean(P[i0 - PRE * FS:i0 - POST * FS], 0.2))
            up = dp_ > 0
            a = apps_s[int(clf.predict(feat(dh if up else -dh, dp_ if up else -dp_)[None])[0])]
            called.append((t, a))
        for tag in tags:
            model, apps_m = models[tag]
            d_ = forward_file(model, stem, dev, stride=15)
            t_ = d_["targets"] / 60.0
            for a in [x for x in spec["appliances_present"] if x in apps_m]:
                g = np.clip(d_["gate"][:, apps_m.index(a)], 1e-4, 1 - 1e-4)
                lo = np.log(g / (1 - g))
                yy = np.zeros(len(g), bool)
                for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                    yy |= (t_ >= t0) & (t_ < t1)
                sw = np.full(len(lo), -STAY)
                for t, aa in called:
                    if aa == a:
                        sw[int(np.argmin(np.abs(t_ - t)))] = +BONUS
                base = ((g > 0.5) == yy).mean()
                row = {w: (viterbi(np.stack([np.zeros_like(lo), w * lo], 1), sw) == yy).mean()
                       for w in EM_W}
                agg.setdefault(tag, {}).setdefault("base", []).append(base)
                for w in EM_W:
                    agg[tag].setdefault(w, []).append(row[w])
                if a == "minipc":
                    mp_rows.append((stem, tag, base, row))
    hdr = "".join("  w=%-5g" % w for w in EM_W)
    print("기기·파일 평균 시간 정확도 (참 전이 시각을 준 천장) — 방출 가중 훑기")
    print("  모델      창별 단독" + hdr)
    for tag in tags:
        b = float(np.mean(agg[tag]["base"]))
        print("  %-8s %9.3f" % (tag, b) + "".join("%8.3f" % np.mean(agg[tag][w]) for w in EM_W))
    print("")
    print("미니PC 만")
    print("  파일     모델      창별 단독" + hdr)
    for stem, tag, b, row in mp_rows:
        print("  %-8s %-9s %8.3f" % (stem, tag, b) + "".join("%8.3f" % row[w] for w in EM_W))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
