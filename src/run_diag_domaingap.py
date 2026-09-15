# -*- coding: utf-8 -*-
"""**도메인 간극을 채널로 짚는다** (14.141) — 추측 말고 이름을 뽑는다.

14.140 이 가른 것: 합성 포트+드라이 창 339개에서 오븐 통전 헛율이 **0.3~4.4%** 인데
실측 같은 조합에서는 **59.6%** 다. 모델은 둘을 **가를 수 있다** — 식별 불가가 아니라
**도메인 간극**이다.

그래서 *왜* 를 또 추측하지 않는다. **같은 조합의 창을 실측과 합성에서 모으고, 입력
채널마다 "이 값 하나로 실측/합성을 얼마나 잘 가르나"(AUC)를 재서 줄 세운다.**
AUC 가 1.0 에 가까운 채널이 합성이 못 만들고 있는 것이다.

⚠ HPC_RULES §0 — `processed_data/composite_eval`(실측)은 **HPC 에 안 올린다.**
   그래서 두 쪽을 따로 요약하고 **요약본만** 주고받는다 (창당 322개 숫자).

```
  --mode synth   HPC 에서 홀드아웃 -> 요약 npz
  --mode real    로컬에서 실측     -> 요약 npz
  --compare A B  둘을 읽어 채널별 AUC 로 줄 세운다
```

    (HPC)  python -X utf8 -m src.run_diag_domaingap --mode synth --out synth.npz
    (로컬) python -X utf8 src/run_diag_domaingap.py --mode real  --out real.npz
    (로컬) python -X utf8 src/run_diag_domaingap.py --compare real.npz synth.npz
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.model.inputs import (  # noqa: E402
    EVEN2_CH, EVEN_MAG0, EVEN_ORDERS, FINE_VOLT0, HALFWAVE_CH, ODD_ORDERS,
    PHI0, PHI_ORDERS, VOLT_ORDERS, build_inputs, fine_target_index,
)

DRYER_STATE = 1          #: 0 이면 안 맞춘다 (옛 경로)
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
STATS = ("타깃", "평균", "최대", "최소")


def fine_names():
    """세밀 57채널 이름 — 뜻으로 읽히게."""
    n = {}
    for i, h in enumerate(ODD_ORDERS):
        n[2 * i], n[2 * i + 1] = "Re(I%d)" % h, "Im(I%d)" % h
    for i, h in enumerate(EVEN_ORDERS):
        n[EVEN_MAG0 + i] = "|I%d|" % h
    n.update({23: "asinh(P/100)", 24: "Q", 25: "V",
              26: "|I3|/|I1|", 27: "|I5|/|I1|", 28: "|I2|/|I1|",
              29: "P-이동평균(±0.5s)", 30: "P-이동평균(±2.5s)",
              39: "역률", 40: "|I9|/|I3|",
              41: "P-P(t+3.0s)", 42: "P-P(t+5.5s)",
              HALFWAVE_CH: "**반파** |I2|-|I4|", EVEN2_CH: "**짝수2차** |I2|"})
    for i, h in enumerate(PHI_ORDERS):
        n[PHI0 + 2 * i], n[PHI0 + 2 * i + 1] = "cosφ%d" % h, "sinφ%d" % h
    for i, h in enumerate(VOLT_ORDERS):
        n[FINE_VOLT0 + i] = "Re(V%d)" % h
        n[FINE_VOLT0 + len(VOLT_ORDERS) + i] = "Im(V%d)" % h
    return [n.get(i, "ch%d" % i) for i in range(57)]


def summarize(fine, wide):
    """창마다 채널별 (타깃, 평균, 최대, 최소). 세밀 57x4 + 광역 47x2."""
    t = fine_target_index()
    f = np.stack([fine[:, :, t], fine.mean(-1), fine.max(-1), fine.min(-1)], -1)
    w = np.stack([wide[:, :, -1], wide.mean(-1)], -1)
    return f.astype(np.float32), w.astype(np.float32)


def auc(a, b):
    """a(실측) 가 b(합성)보다 큰 쌍의 비율. 0.5 면 못 가른다, 1.0/0.0 이면 완전분리."""
    x = np.concatenate([a, b])
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    #: 동점 처리 — 순위 평균
    o = np.argsort(x, kind="mergesort")
    xs = x[o]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    na, nb = len(a), len(b)
    return (r[:na].sum() - na * (na - 1) / 2.0) / (na * nb)


def collect_real():
    from src.evaluation.real_events import load_events
    from src.model.realdata import dense_targets
    F, W, HP = [], [], []
    for s in FILES:
        rw = dense_targets(s, stride=30)
        t = np.asarray(rw.target_cycle, float) / 60.0
        iv = load_events()[s]["intervals"]

        def mk(app):
            m = np.zeros(len(t), bool)
            for a, b in iv.get(app, {}).get("on", []):
                m |= (t >= a) & (t <= b)
            return m
        keep = mk("electiric_kettle") & mk("hair_dryer") & ~mk("oven")
        if not keep.any():
            continue
        idx = np.flatnonzero(keep)
        hp = mk("hotplate")[idx]
        for i in range(0, len(idx), 256):
            f, w, *_ = rw.batch(idx[i:i + 256])
            F.append(np.ascontiguousarray(f)); W.append(np.ascontiguousarray(w))
        HP.append(hp)
    return np.concatenate(F), np.concatenate(W), np.concatenate(HP)


def collect_synth(path):
    from src.evaluation.holdout import load_holdout
    hs = load_holdout(path)
    apps = list(hs.appliances)
    jo, jk, jd, jh = (apps.index(x) for x in
                      ("oven", "electiric_kettle", "hair_dryer", "hotplate"))
    on = hs.y_on.astype(bool)
    yst = np.asarray(hs.y_state)
    keep = on[:, jk] & on[:, jd] & ~(on[:, jo] & (yst[:, jo] == 2))
    #: ⚠ 14.143 — **드라이기 상태를 맞춘다.** 안 맞추면 s1(반파)/s2(전파)가 70:30 으로
    #:   섞여 |I2| 중앙값이 절반으로 눌리고, 그 혼합이 **모든 마진널 비교를 오염**시킨다.
    #:   실측 포트+드라이 창은 사실상 전부 반파다 (|I2| 0.844A · 사전 0.4326x2.07=0.90A).
    #:   맞춘 뒤 합성 276창 |I2| 0.876A 로 실측과 같아진다 — 그런데 헛율은 0~3.3% 대 30~66%.
    if DRYER_STATE:
        keep &= (yst[:, jd] == DRYER_STATE)
    idx = np.flatnonzero(keep)
    F, W = [], []
    for i in range(0, len(idx), 256):
        f, w = build_inputs(np.asarray(hs.X[idx[i:i + 256]]))
        F.append(f); W.append(w)
    return np.concatenate(F), np.concatenate(W), on[idx, jh]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("real", "synth"))
    ap.add_argument("--out", default="")
    ap.add_argument("--holdout", default="processed_data/holdout60_v32h")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("실측npz", "합성npz"))
    ap.add_argument("--hotplate", choices=("any", "on", "off"), default="any")
    a = ap.parse_args()

    if a.mode:
        F, W, HP = collect_real() if a.mode == "real" else collect_synth(a.holdout)
        f, w = summarize(F, W)
        np.savez_compressed(a.out, fine=f, wide=w, hotplate=HP)
        print("%s 창 %d개 -> %s  (세밀 %s · 광역 %s)"
              % (a.mode, len(f), a.out, f.shape, w.shape))
        return 0

    R, S = (np.load(p) for p in a.compare)
    hr, hsy = R["hotplate"].astype(bool), S["hotplate"].astype(bool)
    if a.hotplate != "any":
        want = a.hotplate == "on"
        R = {"fine": R["fine"][hr == want], "wide": R["wide"][hr == want]}
        S = {"fine": S["fine"][hsy == want], "wide": S["wide"][hsy == want]}
    rf, sf, rw_, sw = R["fine"], S["fine"], R["wide"], S["wide"]
    print("도메인 간극 — 포트+드라이 창 · 실측 **%d** 대 합성 **%d** (핫플: %s)"
          % (len(rf), len(sf), a.hotplate))
    print("  AUC 1.0 = 그 값 하나로 실측/합성이 **완전히 갈린다** · 0.5 = 못 가른다\n")
    nm = fine_names()
    rows = []
    for c in range(rf.shape[1]):
        for k, st in enumerate(STATS):
            v = auc(rf[:, c, k], sf[:, c, k])
            rows.append((max(v, 1 - v), v, "세밀 ch%-2d %-18s %s" % (c, nm[c], st),
                         np.median(rf[:, c, k]), np.median(sf[:, c, k])))
    for c in range(rw_.shape[1]):
        for k, st in enumerate(("타깃", "평균")):
            v = auc(rw_[:, c, k], sw[:, c, k])
            rows.append((max(v, 1 - v), v, "광역 ch%-2d %-18s %s" % (c, "", st),
                         np.median(rw_[:, c, k]), np.median(sw[:, c, k])))
    rows.sort(reverse=True)
    print("  %-40s %8s %11s %11s" % ("채널", "AUC", "실측 중앙", "합성 중앙"))
    for sc, v, lab, mr, ms in rows[:28]:
        print("  %-40s %8.3f %11.4f %11.4f" % (lab, v, mr, ms))
    print("")
    ge = sum(1 for r in rows if r[0] >= 0.95)
    print("  AUC>=0.95 (사실상 완전분리) 인 항목 **%d/%d**" % (ge, len(rows)))
    print("  AUC>=0.80 인 항목 %d/%d" % (sum(1 for r in rows if r[0] >= 0.80), len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
