# -*- coding: utf-8 -*-
"""합성 라벨이 **어떤 단서로** 풀리는지 모델 없이 잰다 (13.84 ②).

    python -X utf8 src/run_diag_shortcut.py <syn_train_dir> <syn_holdout_dir>

창 수준 손 특징 몇 개(타깃 시점 값 + 창 평균)로 미니PC ON/OFF 를 합성(train 소표본)에서 맞추는
선형 분류기를 만들고, 같은 분류기를 합성 holdout 소표본과 **실측 창(test_1~4)** 에 건다.
합성에서 높고 실측에서 무너지는 특징 무리가 "BCE 가 고르는 지름길" 이다.

⚠ 이것은 *데이터의 성질* 을 재는 것이다 — 1단계 BCE 는 이 지름길로 손실을 거의 0 으로 만들 수
   있으므로 그것을 버릴 이유가 없다는 뜻이다. 모델 특유의 귀속(IG)은 13.83.22 에 있다.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.model.inputs import fine_target_index
from src.model.realdata import dense_targets
from src.model.traincache import CachedWindows

T = fine_target_index()
# 세밀 배치 v3 (inputs.py). 그룹마다 타깃 시점 값과 뒤 10초 평균을 둘 다 쓴다.
GROUPS = {
    "P만":           [23],
    "P·Q·V":         [23, 24, 25],
    "|I1|(Re/Im h1)": [0, 8],
    "홀수 Re/Im h3~15": list(range(1, 8)) + list(range(9, 16)),
    "짝수 크기":       list(range(16, 23)),
    "크기비":         [26, 27, 28, 39, 40],
    "위상 φ3~9":      list(range(31, 39)),
    "전압 고조파":     list(range(45, 57)),
}
GROUPS["전부"] = sorted(set(sum(GROUPS.values(), [])))


def feats(fine, chans):
    x_t = fine[:, chans, T]
    x_m = fine[:, chans, :].mean(-1)
    return np.concatenate([x_t, x_m], axis=1)


def real_set(stems, apps):
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    j = apps.index("minipc")
    F, Y, M, S = [], [], [], []
    for stem in stems:
        rw = dense_targets(stem, stride=30)
        spec = ev[stem]; t = rw.target_cycle; n = int(t.max()) + 1

        def mask(app, key="on"):
            m = np.zeros(n + 3600, bool)
            for a, b in spec["intervals"].get(app, {}).get(key, []):
                m[int(a * 60):int(b * 60)] = True
            return m[t]
        y = mask("minipc") if "minipc" in spec["appliances_present"] else np.zeros(len(t), bool)
        unc = mask("minipc", "uncertain") if "minipc" in spec["appliances_present"] else np.zeros(len(t), bool)
        F.append(rw.fine); Y.append(y); M.append(~unc); S += [stem] * len(t)
    return np.concatenate(F), np.concatenate(Y), np.concatenate(M), np.asarray(S)


def main():
    tr, ho = CachedWindows(sys.argv[1]), CachedWindows(sys.argv[2])
    apps = tr.appliances; j = apps.index("minipc")
    ftr = tr.batch(np.arange(len(tr))); fho = ho.batch(np.arange(len(ho)))
    Xtr, ytr = ftr[0], ftr[3][:, j] > 0.5
    Xho, yho = fho[0], fho[3][:, j] > 0.5
    Xre, yre, mre, sre = real_set(["test_1", "test_2", "test_3", "test_4"], apps)
    print("합성 train %d창 (미니PC ON %.2f) · holdout %d창 (%.2f) · 실측 %d창 (%.2f)" % (
        len(ytr), ytr.mean(), len(yho), yho.mean(), mre.sum(), yre[mre].mean()))
    print("\n특징 무리별 미니PC ON/OFF AUC — 선형(로지스틱) / 비선형(GBM)")
    print("   %-18s | %-23s | %-23s | %s" % ("무리", "선형: 합성ho  실측", "GBM: 합성ho  실측", "실측 파일별 (GBM) t1 t2 t3 t4"))
    for name, ch in GROUPS.items():
        A, B, C = feats(Xtr, ch), feats(Xho, ch), feats(Xre, ch)
        lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(A, ytr)
        gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1).fit(A, ytr)
        a1, a2 = roc_auc_score(yho, lin.predict_proba(B)[:, 1]), roc_auc_score(yre[mre], lin.predict_proba(C[mre])[:, 1])
        b1, b2 = roc_auc_score(yho, gbm.predict_proba(B)[:, 1]), roc_auc_score(yre[mre], gbm.predict_proba(C[mre])[:, 1])
        per = []
        pg = gbm.predict_proba(C)[:, 1]
        for st in ("test_1", "test_2", "test_3", "test_4"):
            m = mre & (sre == st)
            per.append(roc_auc_score(yre[m], pg[m]) if len(set(yre[m])) == 2 else float("nan"))
        print("   %-18s |   %.3f   %.3f            |   %.3f   %.3f            | " % (name, a1, a2, b1, b2)
              + " ".join("%.2f" % v for v in per))


if __name__ == "__main__":
    main()
