# -*- coding: utf-8 -*-
"""전이 **짝 맞추기** 제약에 이가 있는가 (13.84.42).

13.84.41 이 본 실패의 모양은 *가짜 switch-off* 다 — test_2 에서 3.8분의 가짜 OFF 하나가
5분을 통째로 뒤집었다. 물리적으로 OFF 는 앞선 ON 을 **되돌리는** 것이므로
`Δ_off ≈ −Δ_on` 이어야 하는데, 지금 모델은 `sw_on`·`sw_off` 를 따로 점수 매기고
**한 번도 견주지 않는다.**

굽기 전에 그 제약에 이가 있는지 셋으로 본다.
  ① 참 짝    같은 통전 구간의 ON 과 OFF 가 실제로 얼마나 반대인가 (크기비·cos)
  ② 귀무     서로 다른 구간의 ON/OFF 를 짝지으면 얼마나 되는가 — ①이 이보다 나아야 한다
  ③ **본 시험**  모델(Viterbi)이 낸 전이에서 참/거짓을 이 점수로 가를 수 있는가 (AUC)

⚠ Δ 는 **관측 총전류**의 차분이라 그 사이 다른 기기가 움직이면 오염된다. 그래서 ③ 이 본 시험이다 —
  ①②가 좋아도 ③ 이 안 되면 제약을 넣을 근거가 없다.

    python -X utf8 src/run_diag_pair.py [results/seq_h38_base.pt]
"""
import json
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

FS = 60
WIN = 8          # 전후 평균 구간 (초) — 13.84.31 ⑥ 과 같은 규약
MARGIN = 3       # 전이에서 띄울 초
ORD = [1, 3, 5, 7, 9, 11, 13, 15]
TOL_S = 10.0     # 모델 전이를 참 전이로 볼 시간 허용오차


def delta(H, P, t_s):
    """t_s(초)에서의 Δ — (앞 8초 평균) 대비 (뒤 8초 평균). (페이저 (8,), ΔP)"""
    i = int(round(t_s * FS))
    a0, a1 = i - (MARGIN + WIN) * FS, i - MARGIN * FS
    b0, b1 = i + MARGIN * FS, i + (MARGIN + WIN) * FS
    if a0 < 0 or b1 > len(H):
        return None, None
    oi = [o - 1 for o in ORD]
    d = H[b0:b1, oi].mean(0) - H[a0:a1, oi].mean(0)
    return d, float(P[b0:b1].mean() - P[a0:a1].mean())


def pair_score(d_on, d_off):
    """짝 점수 — 0 이면 완벽히 반대(짝이 맞다), 1 이면 같은 방향(안 맞다).

    `|Δ_on + Δ_off| / (|Δ_on| + |Δ_off|)`. 크기와 방향을 한꺼번에 본다.
    """
    n = np.linalg.norm(d_on) + np.linalg.norm(d_off)
    return float(np.linalg.norm(d_on + d_off) / n) if n > 0 else np.nan


def auc(pos, neg):
    """pos 가 neg 보다 **작을수록** 좋은 점수라고 보고 AUC (0.5=무의미)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan, len(pos), len(neg)
    w = (pos[:, None] < neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(w / (len(pos) * len(neg))), len(pos), len(neg)


def main():
    ck_path = sys.argv[1] if len(sys.argv) > 1 else "results/seq_h38_base.pt"
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    raw = {}
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        raw[stem] = (np.asarray(r["harmonics_complex"]),
                     np.asarray(r["power_features"])[:, 0])

    # ── ① 참 짝 · ② 귀무 ──────────────────────────────────────────────────
    print("① 참 짝 — 같은 통전 구간의 ON 과 OFF 가 얼마나 반대인가")
    print("   (점수 0 = 완벽히 반대 · 1 = 같은 방향. 전후 %d초 평균, 여유 %d초)" % (WIN, MARGIN))
    true_pair, null_pair = {}, {}
    for stem in FILES:
        H, P = raw[stem]
        for app, spec in ev[stem]["intervals"].items():
            ons = []
            for t0, t1 in spec.get("on", []):
                d0, p0 = delta(H, P, t0)
                d1, p1 = delta(H, P, t1)
                if d0 is None or d1 is None:
                    continue
                ons.append((d0, d1, p0, p1))
            if len(ons) < 1:
                continue
            for d0, d1, _, _ in ons:
                true_pair.setdefault(app, []).append(pair_score(d0, d1))
            # 귀무 — 서로 **다른** 구간의 ON 과 OFF
            for (a0, _, _, _), (_, b1, _, _) in combinations(ons, 2):
                null_pair.setdefault(app, []).append(pair_score(a0, b1))
    print("   %-18s %7s %10s %10s | %7s %10s" % ("기기", "참짝", "중앙", "p90", "귀무", "중앙"))
    for app in sorted(true_pair):
        t = np.array(true_pair[app], float); t = t[~np.isnan(t)]
        n = np.array(null_pair.get(app, []), float); n = n[~np.isnan(n)]
        print("   %-18s %7d %10.3f %10.3f | %7d %10.3f"
              % (app, len(t), np.median(t) if len(t) else np.nan,
                 np.percentile(t, 90) if len(t) else np.nan,
                 len(n), np.median(n) if len(n) else np.nan))
    tm = np.array(true_pair.get("minipc", []), float); tm = tm[~np.isnan(tm)]
    nm = np.array(null_pair.get("minipc", []), float); nm = nm[~np.isnan(nm)]
    a, np_, nn = auc(tm, nm)
    print("   미니PC 참짝 대 귀무 AUC %.3f  (참 %d · 귀무 %d)" % (a, np_, nn))

    # ── ③ 본 시험 — 모델의 전이에서 참/거짓을 가르는가 ────────────────────
    print("\n③ 본 시험 — 모델(Viterbi)이 낸 **switch-off** 에서 참과 거짓을 가르는가")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)
    km = apps.index("minipc")

    # ── ③b 넓혀서 — **전 기기**의 모델 전이에서 ΔP 정합이 참/거짓을 가르는가 ───
    # ③ 의 미니PC 표본이 5개뿐이라 닫을 수 없다. 같은 질문을 전 기기로 넓힌다.
    wide = {"off": {"pair": ([], []), "dp": ([], [])},
            "on": {"dp": ([], [])}}
    with torch.no_grad():
        for stem, d in cache.items():
            H, P = raw[stem]
            Z, GL = [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                     torch.cat(GL)[None])
            pth = viterbi(em, on, off, ini)[0].cpu().numpy()
            t = d["t"]
            for k in np.nonzero(d["present"])[0]:
                edges = []
                for t0, t1 in ev[stem]["intervals"].get(apps[k], {}).get("on", []):
                    edges += [t0, t1]
                lo = None
                for i in np.nonzero(np.diff(pth[:, k].astype(np.int8)))[0] + 1:
                    ts = float(t[i])
                    dd, dp = delta(H, P, ts)
                    if dd is None:
                        continue
                    ok = (min(abs(ts - e) for e in edges) <= TOL_S) if edges else False
                    if pth[i, k]:
                        lo = (ts, dd)
                        # 켜짐이면 ΔP 가 **양수**여야 한다
                        wide["on"]["dp"][0 if ok else 1].append(-dp)
                    else:
                        # 꺼짐이면 ΔP 가 **음수**여야 한다 -> 점수는 +dp (작을수록 참답다)
                        wide["off"]["dp"][0 if ok else 1].append(dp)
                        if lo is not None:
                            wide["off"]["pair"][0 if ok else 1].append(pair_score(lo[1], dd))
    print("\n③b 전 기기로 넓혀 — 점수가 작을수록 '참답다'")
    print("   %-22s %8s %8s %10s %10s %8s"
          % ("", "참", "거짓", "참 중앙", "거짓 중앙", "AUC"))
    for lbl, (g, b) in (("꺼짐 · ΔP (W)", wide["off"]["dp"]),
                        ("꺼짐 · 짝점수", wide["off"]["pair"]),
                        ("켜짐 · −ΔP (W)", wide["on"]["dp"])):
        aa, ng, nb = auc(g, b)
        print("   %-22s %8d %8d %10.1f %10.1f %8.3f"
              % (lbl, ng, nb,
                 np.median(g) if ng else np.nan, np.median(b) if nb else np.nan, aa))

    good, bad, rows = [], [], []
    with torch.no_grad():
        for stem, d in cache.items():
            if not d["present"][km]:
                continue
            H, P = raw[stem]
            Z, GL = [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                     torch.cat(GL)[None])
            path = viterbi(em, on, off, ini)[0].cpu().numpy()[:, km]
            t = d["t"]
            flips = np.nonzero(np.diff(path.astype(np.int8)))[0] + 1
            true_edges = []
            for t0, t1 in ev[stem]["intervals"].get("minipc", {}).get("on", []):
                true_edges += [t0, t1]
            last_on = None
            for i in flips:
                ts = float(t[i])
                dd, dp = delta(H, P, ts)
                if dd is None:
                    continue
                if path[i]:                 # OFF -> ON
                    last_on = (ts, dd)
                    continue
                if last_on is None:         # 앞선 ON 이 없는 OFF (파일 첫머리)
                    continue
                sc = pair_score(last_on[1], dd)
                ok = min(abs(ts - e) for e in true_edges) <= TOL_S if true_edges else False
                (good if ok else bad).append(sc)
                rows.append((stem, ts / 60.0, last_on[0] / 60.0, sc, "참" if ok else "거짓", dp))
    print("   %-8s %9s %9s %8s %6s %9s" % ("파일", "OFF(분)", "짝ON(분)", "짝점수", "참/거짓", "ΔP(W)"))
    for r in sorted(rows):
        print("   %-8s %9.2f %9.2f %8.3f %6s %9.1f" % r)
    a, ng, nb = auc(good, bad)
    print("   참 %d개 중앙 %.3f · 거짓 %d개 중앙 %.3f · **AUC %.3f**"
          % (ng, np.median(good) if ng else np.nan, nb,
             np.median(bad) if nb else np.nan, a))
    print("   AUC 0.5 = 못 가른다. 이 표본으로는 ±0.15 쯤 흔들린다 — 자릿수만 읽어라")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
