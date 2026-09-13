# -*- coding: utf-8 -*-
"""결정 시험 — **실측 배경 + 실측 미니PC 주입**. 라벨이 구조적으로 깨끗하다.

`ceil4.py`: 배경이 **충전기 단독**이면 A 대 A+B 가 활성화 교차로도 AUC 1.000 이다
(미니PC 142.6mA 가 충전기 414.8mA 의 34%). 즉 미니PC↔충전기 자체는 안 어렵다.
(⚠ ceil4 는 `source_file` 대신 활성화 인덱스로 갈라서 '녹화 교차'가 아니었다.
   여기서는 그 실수를 고치고, 배경을 실측 복합으로 바꾼다.)

실측 복합 4파일의 형제 ON 창은 총전력 95~642W 다. 배경은 충전기 하나가 아니라
**켜져 있는 전부**다. 그것이 미니PC 9W 를 삼키는지 직접 잰다:

    A    = 실측 창 (미니PC OFF 라벨, 다른 것은 무엇이든 켜져 있음)
    A+B  = 같은 창 + 격리 녹화의 미니PC 60사이클

라벨은 구성으로 정해지므로 교락도 없고 표본도 창 수만큼이다 (19사건 문제 해소).
⚠ 미니PC 조각은 **학습/채점에서 녹화를 갈라** 쓴다 (파형 암기 방지).
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from src.preprocessing import load_nilm_npz
from src.synthesis.segment_pool import SegmentPool

CYC, NH, MA, W, STR = 60.0, 15, 1000.0, 60, 30
FILES = ["test_1", "test_2", "test_3", "test_4"]
SIB = ["laptop_charger", "beam_projector"]
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
rng = np.random.default_rng(0)

pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
MP = []
for a in pool.appliance_activations.get("minipc", []):
    z = np.asarray(a.net_harmonics_complex)
    for i in range(0, len(z) - W + 1, W):
        MP.append((a.source_file, z[i:i + W]))
srcs = sorted({s for s, _ in MP})
print("미니PC 조각 %d · 녹화 %s" % (len(MP), srcs))
half = set(srcs[::2])                       # 학습용 녹화 / 채점용 녹화
MP_tr = [z for s, z in MP if s in half]
MP_te = [z for s, z in MP if s not in half]
print("  학습쪽 녹화 %s (%d조각) · 채점쪽 %s (%d조각)"
      % (sorted(half), len(MP_tr), sorted(set(srcs) - half), len(MP_te)))

BG, FID, SIBON, PT = [], [], [], []
for fi, stem in enumerate(FILES):
    spec = ev.get(stem)
    if spec is None or "minipc" not in spec.get("appliances_present", []):
        continue
    raw = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
    C = np.asarray(raw["harmonics_complex"])[:, :NH]
    P = np.asarray(raw["power_features"])
    ok = np.asarray(raw["is_valid"]).astype(bool)
    n = len(C)

    def mk(app, key="on"):
        m = np.zeros(n, bool)
        for t0, t1 in spec["intervals"].get(app, {}).get(key, []):
            m[int(t0 * CYC):int(min(t1 * CYC, n))] = True
        return m

    y = mk("minipc")
    ok &= ~mk("minipc", "uncertain") & ~y     # 미니PC 가 **꺼진** 창만 배경으로
    s = np.zeros(n, bool)
    for a in SIB:
        if a in spec.get("appliances_present", []):
            s |= mk(a)
    for i in range(0, n - W + 1, STR):
        sl = slice(i, i + W)
        if not ok[sl].all() or s[sl].std() > 0:
            continue
        BG.append(C[sl]); FID.append(fi); SIBON.append(bool(s[i]))
        PT.append(float(P[sl, 0].mean()))
FID, SIBON, PT = np.asarray(FID), np.asarray(SIBON), np.asarray(PT)
print("배경 창 %d (형제 ON %d · 형제 OFF %d) · 총전력 중앙 %.0fW"
      % (len(BG), int(SIBON.sum()), int((~SIBON).sum()), np.median(PT)))


MEAN_ONLY = "--mean-only" in sys.argv


def feat(z):
    """⚠ 표준편차를 쓰면 **주입 자체가 분산을 키우는 것**을 읽을 수 있다
    (var(A+B) = var(A)+var(B)). `--mean-only` 로 그 통로를 막고 다시 잰다."""
    m = [np.real(z).mean(0), np.imag(z).mean(0)]
    if not MEAN_ONLY:
        m += [np.real(z).std(0), np.imag(z).std(0)]
    return np.concatenate(m) * MA


def build(sel, mps):
    X, Y = [], []
    for i in np.where(sel)[0]:
        b = BG[i]
        X.append(feat(b)); Y.append(0)
        X.append(feat(b + mps[rng.integers(len(mps))])); Y.append(1)
    return np.asarray(X), np.asarray(Y)


def cross(sel, label):
    print("\n=== %s ===" % label)
    out = []
    for k in range(len(FILES)):
        tr, te = sel & (FID != k), sel & (FID == k)
        if te.sum() < 40 or tr.sum() < 100:
            print("  %-24s 창 %d (부족)" % ("나머지 -> %s" % FILES[k], int(te.sum())))
            continue
        Xtr, Ytr = build(tr, MP_tr)
        Xte, Yte = build(te, MP_te)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        c = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                           max_depth=4, random_state=0)
        c.fit((Xtr - mu) / sd, Ytr)
        a = roc_auc_score(Yte, c.predict_proba((Xte - mu) / sd)[:, 1])
        out.append(a)
        print("  %-24s 배경 %4d · 총W중앙 %6.0f · AUC %.3f"
              % ("나머지 -> %s" % FILES[k], int(te.sum()), np.median(PT[te]), a))
    if out:
        print("  **파일 교차 평균 %.3f**" % np.mean(out))
    return out


cross(SIBON, "형제 SMPS ON 배경에 미니PC 주입 — 파일 교차")
cross(~SIBON, "대조: 형제 SMPS OFF 배경 — 파일 교차")

print("\n=== 배경 총전력대별 (형제 ON, 파일 교차 평균) ===")
print("  %-14s %8s %8s" % ("배경 총W", "창", "AUC"))
for lo, hi in ((0, 60), (60, 120), (120, 300), (300, 1e9)):
    m = SIBON & (PT >= lo) & (PT < hi)
    if m.sum() < 120:
        print("  %-14s %8d  (부족)" % ("%d~%d" % (lo, min(hi, 9999)), int(m.sum())))
        continue
    out = []
    for k in range(len(FILES)):
        tr, te = m & (FID != k), m & (FID == k)
        if te.sum() < 40 or tr.sum() < 100:
            continue
        Xtr, Ytr = build(tr, MP_tr)
        Xte, Yte = build(te, MP_te)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        c = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                           max_depth=4, random_state=0)
        c.fit((Xtr - mu) / sd, Ytr)
        out.append(roc_auc_score(Yte, c.predict_proba((Xte - mu) / sd)[:, 1]))
    print("  %-14s %8d %8s" % ("%d~%d" % (lo, min(hi, 9999)), int(m.sum()),
                               "  -  " if not out else "%.3f" % np.mean(out)))
