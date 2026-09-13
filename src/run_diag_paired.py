# -*- coding: utf-8 -*-
"""짝지은 합성 창 — 시간축 특징이 합성에서 미니PC 를 가르고 **실측으로 옮겨가는가** (13.84.15).

13.84.14 의 남은 물음. 추세 제거/창내 상대 표현이 사건 축의 정보를 나르는 것은 맞는데,
그 정보가 (a) 합성 창에도 있고 (b) 합성에서 적합한 결정면이 실측에서도 서는가.

짝은 **정답 분해로 정확히** 만든다 — 씨앗 맞추기는 강제 활성이 난수 순서를 바꿔 못 쓴다.
   X_on  = 합성 총전류(충전기+프로젝터+미니PC)
   X_off = X_on − gt_harmonics_ri[minipc],  P_off = P − gt_target_power_w[minipc]
같은 자리·Z·텍스처·형제 전력·형제 씨앗에서 미니PC 만 뺀 창이다. 여기에 **전이가 있는** 음성
(창 앞에서 미니PC 가 꺼지는 자연 창)도 같이 모은다 — 실측 test_4 의 오탐 창이 그 꼴이다.

    python -X utf8 src/run_diag_paired.py [N]
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_nilm_npz
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

FS = 60
W = 3600                      # 60초
TGT = 10 * FS                 # 타깃 10초 (세밀 갈래가 보는 구간)
HEAD = 10 * FS                # 창 앞 10초
HALF = int(2.5 * FS)          # 세밀에서 계산 가능한 추세 제거 폭
N_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 250
P_LO, P_HI = 70.0, 110.0


def movavg(a, half):
    k = 2 * half + 1
    pad = np.pad(a, (half, half), mode="edge")
    c = np.cumsum(np.insert(pad, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def feats(Hc, P):
    """(W,15) complex + (W,) P -> 특징 무리 셋."""
    i1 = Hc[:, 0]
    ang = np.degrees(np.angle(i1))
    ang = (ang - np.median(ang) + 180.0) % 360.0 - 180.0 + np.median(ang)   # 감김 제거
    mag1 = np.abs(i1)
    r31 = np.abs(Hc[:, 2]) / (mag1 + 1e-12)
    t, h = slice(-TGT, None), slice(0, HEAD)
    A = [ang[t].mean(), mag1[t].mean(), r31[t].mean(), P[t].mean()]
    B = [ang[t].mean() - ang[h].mean(), mag1[t].mean() - mag1[h].mean(),
         r31[t].mean() - r31[h].mean(), P[t].mean() - P[h].mean()]
    C = []
    for x in (ang, r31, P):
        d = x - movavg(x, HALF)
        j = int(np.argmax(np.abs(d)))
        C += [float(d[j]), float(np.abs(d).max())]
    return np.array(A), np.array(B), np.array(C)


GROUPS = {"A 절대(현행)": (0,), "A− P 제외": (0,), "B 창내 상대": (1,),
          "C 추세제거 ±2.5초": (2,), "A+B": (0, 1), "A+C": (0, 2)}


def pack(rows, key):
    A = np.stack([r[0] for r in rows]); B = np.stack([r[1] for r in rows]); C = np.stack([r[2] for r in rows])
    if key == "A− P 제외":
        return A[:, :3]
    parts = {0: A, 1: B, 2: C}
    return np.concatenate([parts[i] for i in GROUPS[key]], axis=1)


# ── 합성 ──────────────────────────────────────────────────────────────────
pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
gen = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=True)
np.random.seed(0)
pos, neg_sub, neg_nat, steps = [], [], [], []
tries = 0
while len(pos) < N_TARGET and tries < 20000:
    tries += 1
    try:
        smp = gen.synthesize_random_window(
            window_size_cycles=W, force_active=["laptop_charger", "beam_projector", "minipc"],
            full_window_placement=False, compute_gt_harmonics=True,
            target_lookahead_cycles=360, sustained_power_limit_w=None)
    except Exception:
        continue
    if smp.metadata.get("base_voltage_v", 999) > 222.0:        # 자리 D 만
        continue
    on = {a: np.asarray(smp.gt_is_on[a]).astype(bool) for a in smp.gt_is_on}
    if not (on["laptop_charger"][-TGT:].all() and on["beam_projector"][-TGT:].all()):
        continue
    Hc = np.asarray(smp.harmonics_complex)
    P = np.asarray(smp.power_features)[:, 0]
    if not (P_LO <= P[-TGT:].mean() <= P_HI):
        continue
    m = on.get("minipc", np.zeros(W, bool))
    if m[-TGT:].all():
        g = np.asarray(smp.gt_harmonics_ri["minipc"])
        gc = g[:, :, 0] + 1j * g[:, :, 1]
        gp = np.asarray(smp.gt_target_power_w["minipc"])
        pos.append(feats(Hc, P))
        neg_sub.append(feats(Hc - gc, P - gp))
        steps.append((float(np.degrees(np.angle(Hc[-TGT:, 0].mean()))
                            - np.degrees(np.angle((Hc - gc)[-TGT:, 0].mean()))),
                      float(gp[-TGT:].mean()), float(P[-TGT:].mean())))
    elif (not m[-TGT:].any()) and m.any():
        neg_nat.append(feats(Hc, P))

print("합성: 양성 %d · 뺄셈 음성 %d · 전이 음성 %d  (시도 %d)"
      % (len(pos), len(neg_sub), len(neg_nat), tries))
if steps:
    s = np.array(steps)
    print("  미니PC 유발 Δ∠I1 (타깃 10초, 짝 안에서)  중앙 %+.3f°  IQR %.3f~%.3f   미니PC P 중앙 %.1fW  총 P 중앙 %.1fW"
          % (np.median(s[:, 0]), *np.percentile(s[:, 0], [25, 75]), np.median(s[:, 1]), np.median(s[:, 2])))
    print("  실측 test_4 미니PC 계단 Δ∠I1 = 1.475° (13.84.14)  -> 비 %.2f" % (np.median(s[:, 0]) / 1.475))

# ── 실측 ──────────────────────────────────────────────────────────────────
ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
real = {}
for stem in ["test_4", "test_3", "test_2", "test_1"]:
    r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
    Hc = np.asarray(r["harmonics_complex"]); P = np.asarray(r["power_features"])[:, 0]
    n = len(P); valid = np.asarray(r["is_valid"]).astype(bool)

    def mk(a):
        z = np.zeros(n, bool)
        for t0, t1 in ev[stem]["intervals"].get(a, {}).get("on", []):
            z[int(t0 * FS):int(min(t1 * FS, n))] = True
        return z

    y = mk("minipc"); cond = mk("laptop_charger")
    if "beam_projector" in ev[stem]["intervals"]:
        cond &= mk("beam_projector")
    Xs, ys = [], []
    for t0 in range(0, n - W + 1, 5 * FS):
        sl = slice(t0, t0 + W); tl = slice(t0 + W - TGT, t0 + W)
        if not valid[sl].all() or not cond[tl].all():
            continue
        if y[tl].all():
            lab = 1
        elif not y[tl].any():
            lab = 0
        else:
            continue
        Xs.append(feats(Hc[sl], P[sl])); ys.append(lab)
    if len(set(ys)) == 2:
        real[stem] = (Xs, np.array(ys))
        print("실측 %s: 조건창 양성 %d · 음성 %d" % (stem, sum(ys), len(ys) - sum(ys)))

# ── 합성에서 적합 → 실측에서 채점 ────────────────────────────────────────
neg = neg_sub + neg_nat
ysyn = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
rs = np.random.RandomState(0)
sh = rs.permutation(len(ysyn)); cut = int(0.7 * len(sh))
tr, te = sh[:cut], sh[cut:]
print("\n%-20s %8s %s" % ("특징 무리", "합성내", "  ".join("%8s" % k for k in real)))
for key in GROUPS:
    Xs = np.concatenate([pack(pos, key), pack(neg, key)], axis=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(Xs[tr], ysyn[tr])
    row = [roc_auc_score(ysyn[te], clf.predict_proba(Xs[te])[:, 1])]
    for stem, (Xr, yr) in real.items():
        row.append(roc_auc_score(yr, clf.predict_proba(pack(Xr, key))[:, 1]))
    print("%-20s %8.3f %s" % (key, row[0], "  ".join("%8.3f" % v for v in row[1:])))
