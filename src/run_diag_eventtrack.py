# -*- coding: utf-8 -*-
"""사건만으로 타임라인을 복원하면 창별 판정보다 나은가 (13.84.21 둘째 반).

참 전이 시각(사람 라벨)을 **그냥 준 상태**에서 — 즉 검출은 완벽하다고 놓고 — 각 사건을
합성에서 배운 분류기로 가른 뒤 기기별 ON/OFF 를 토글해 타임라인을 만든다. 방향은 ΔP 부호로 안다.
그 타임라인의 시간 정확도를 v35/v37 의 창별 게이트와 견준다. 이것이 사건 구조의 **천장**이다.

    python -X utf8 src/run_diag_eventtrack.py [N_SYN]
"""
import json
import sys

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


def main():
    syn, _ = syn_events(N_SYN)
    apps_s = sorted({a for a, _, _ in syn})
    idx = {a: i for i, a in enumerate(apps_s)}
    X = np.stack([feat(d, p) for _, d, p in syn])
    y = np.array([idx[a] for a, _, _ in syn])
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y)
    print("합성 사건 %d개로 적합 · 기기 %d종\n" % (len(syn), len(apps_s)))

    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {t: load_model("results/%s.pt" % t, dev)[:2] for t in ("cnn_v35", "cnn_v37")}
    tot = {"사건 추적": [0, 0], "cnn_v35": [0, 0], "cnn_v37": [0, 0]}
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        spec = ev[stem]
        # 참 전이 시각 (검출은 완벽하다고 놓는다)
        times = []
        for a, d in spec["intervals"].items():
            for t0, t1 in d.get("on", []):
                times.append(float(t0)); times.append(float(t1))
        times = sorted(t for t in times if PRE < t < n / FS - PRE)
        state = {a: np.zeros(n, bool) for a in apps_s}
        for t in times:
            i0 = int(round(t * FS))
            lo, hi = i0 - PRE * FS, i0 - POST * FS
            lo2, hi2 = i0 + POST * FS, i0 + PRE * FS
            dh = trim_mean(H[lo2:hi2], 0.2, axis=0) - trim_mean(H[lo:hi], 0.2, axis=0)
            dp = float(trim_mean(P[lo2:hi2], 0.2) - trim_mean(P[lo:hi], 0.2))
            up = dp > 0
            f = feat(dh if up else -dh, dp if up else -dp)
            a = apps_s[int(clf.predict(f[None])[0])]
            state[a][i0:] = up
        truth = {a: np.zeros(n, bool) for a in apps_s}
        for a, d in spec["intervals"].items():
            if a not in truth:
                continue
            for t0, t1 in d.get("on", []):
                truth[a][int(t0 * FS):int(min(t1 * FS, n))] = True
        pres = [a for a in apps_s if a in spec["appliances_present"]]
        acc = np.mean([ (state[a] == truth[a]).mean() for a in pres ])
        tot["사건 추적"][0] += acc; tot["사건 추적"][1] += 1
        line = "%-8s 사건 %2d · 기기 %d · 사건추적 시간정확 %.3f" % (stem, len(times), len(pres), acc)
        for tg, (model, apps_m) in models.items():
            d_ = forward_file(model, stem, dev, stride=15)
            t_ = d_["targets"] / 60.0
            accs = []
            for a in pres:
                if a not in apps_m:
                    continue
                g = d_["gate"][:, apps_m.index(a)] > 0.5
                yy = np.zeros(len(g), bool)
                for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                    yy |= (t_ >= t0) & (t_ < t1)
                accs.append((g == yy).mean())
            m_ = float(np.mean(accs))
            tot[tg][0] += m_; tot[tg][1] += 1
            line += " · %s %.3f" % (tg, m_)
        print(line)
        # 미니PC 만 따로
        if "minipc" in pres:
            a = "minipc"
            s_ = (state[a] == truth[a]).mean()
            print("           미니PC만: 사건추적 %.3f" % s_, end="")
            for tg, (model, apps_m) in models.items():
                d_ = forward_file(model, stem, dev, stride=15)
                t_ = d_["targets"] / 60.0
                g = d_["gate"][:, apps_m.index(a)] > 0.5
                yy = np.zeros(len(g), bool)
                for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                    yy |= (t_ >= t0) & (t_ < t1)
                print(" · %s %.3f" % (tg, (g == yy).mean()), end="")
            print()
    print("\n평균 시간 정확도: " + " · ".join("%s %.3f" % (k, v[0] / max(v[1], 1)) for k, v in tot.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
