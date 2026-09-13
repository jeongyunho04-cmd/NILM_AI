# -*- coding: utf-8 -*-
"""실측 창이 몸통 z 공간에서 합성 학습 분포 **안**에 있는가, 그리고 이웃의 합성 라벨이 실측 참값과 맞는가 (13.84 ⑤).

    python -X utf8 src/run_diag_zknn.py <syn_train_dir> <syn_holdout_dir> cnn_v32

합성 train 소표본의 z 를 참조로, (합성 holdout / 실측 test_1~4) 각 창의 k=5 최근접 거리(차원별 표준화)와
이웃의 미니PC 라벨 투표를 낸다. 거리가 합성 holdout 보다 훨씬 크면 헤드는 **외삽**하고 있고, 이웃 투표가
헤드보다 실측 참값과 잘 맞으면 z 에는 합성 라벨로 읽어도 옮겨지는 비선형 구조가 있다는 뜻이다.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.model.realdata import dense_targets
from src.model.traincache import CachedWindows
from src.run_gate_check import load_model


def zs_of(model, fine, wide, j, bs=256):
    store = {}
    h = model.trunk.register_forward_hook(lambda m, i, o: store.__setitem__("z", o.detach()))
    Z, G = [], []
    for i in range(0, len(fine), bs):
        with torch.no_grad():
            o = model(torch.from_numpy(np.ascontiguousarray(fine[i:i + bs])),
                      torch.from_numpy(np.ascontiguousarray(wide[i:i + bs])))
        Z.append(store["z"].numpy()); G.append(torch.sigmoid(o["on_logit"][:, j]).numpy())
    h.remove()
    return np.concatenate(Z), np.concatenate(G)


def main():
    torch.set_num_threads(8)
    tr, ho = CachedWindows(sys.argv[1]), CachedWindows(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else "cnn_v32"
    model, apps, _ = load_model("results/%s.pt" % name, "cpu")
    j, kc, kp = apps.index("minipc"), apps.index("laptop_charger"), apps.index("beam_projector")
    btr, bho = tr.batch(np.arange(len(tr))), ho.batch(np.arange(len(ho)))
    Ztr, Gtr = zs_of(model, btr[0], btr[1], j); ytr = btr[3][:, j] > 0.5
    Zho, Gho = zs_of(model, bho[0], bho[1], j); yho = bho[3][:, j] > 0.5
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    nn_ = NearestNeighbors(n_neighbors=5).fit((Ztr - mu) / sd)

    def probe(Z):
        d, idx = nn_.kneighbors((Z - mu) / sd)
        return d.mean(1), ytr[idx].mean(1)

    d_ho, v_ho = probe(Zho)
    print("== %s · 몸통 z 최근접 이웃 (참조: 합성 train %d창, k=5, 차원 표준화)" % (name, len(Ztr)))
    print("   %-28s %5s | %8s %8s | %10s %10s" % ("집합", "창", "거리p50", "거리p90", "헤드 AUC", "이웃투표 AUC"))
    print("   %-28s %5d | %8.2f %8.2f | %10.3f %10.3f" % (
        "합성 holdout", len(Zho), np.median(d_ho), np.percentile(d_ho, 90),
        roc_auc_score(yho, Gho), roc_auc_score(yho, v_ho)))
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    for stem in ("test_1", "test_2", "test_3", "test_4"):
        rw = dense_targets(stem, stride=30)
        Z, G = zs_of(model, rw.fine, rw.wide, j)
        spec = ev[stem]; t = rw.target_cycle; n = int(t.max()) + 1

        def mask(app, key="on"):
            m = np.zeros(n + 3600, bool)
            for a, b in spec["intervals"].get(app, {}).get(key, []):
                m[int(a * 60):int(b * 60)] = True
            return m[t]
        y = mask("minipc"); unc = mask("minipc", "uncertain")
        sib = mask("laptop_charger") | mask("beam_projector")
        d, v = probe(Z)
        for lab, m in (("전체", ~unc), ("형제ON·미니PC OFF", sib & ~y & ~unc), ("미니PC ON", y & ~unc)):
            if m.sum() < 5:
                continue
            auc_h = roc_auc_score(y[m], G[m]) if len(set(y[m])) == 2 else float("nan")
            auc_v = roc_auc_score(y[m], v[m]) if len(set(y[m])) == 2 else float("nan")
            print("   %-28s %5d | %8.2f %8.2f | %10.3f %10.3f   (헤드 게이트 %.3f · 이웃 ON 몫 %.3f)" % (
                stem + " " + lab, m.sum(), np.median(d[m]), np.percentile(d[m], 90), auc_h, auc_v,
                G[m].mean(), v[m].mean()))


if __name__ == "__main__":
    main()
