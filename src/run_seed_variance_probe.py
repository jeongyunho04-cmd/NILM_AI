"""
시드 분산의 소재 진단 — 무엇이 흔들리고 무엇이 안 흔들리는가 (설계 문서 12.41절)
==================================================================================
12.39 가 같은 설정의 두 실행에서 이런 폭을 봤다.

    유령 21.78W   핫플 F1 0.120                              <- 흔들린다
    미니PC 0.017  프로젝터 0.005  오븐 0.001  합성 MAE 0.058W  <- 안 흔들린다

**두 실행 사이 차이를 창 단위로 갈라 그 폭이 어디 사는지 찾는다.** 학습하지 않는다 —
저장된 체크포인트로 추론만 한다.

    python -m src.run_seed_variance_probe --ckpt results/cnn_ov1.pt results/cnn_ov1_s1.pt

[가설 — 돌리기 전에 적는다]
12.38.1 이 지목한 미결정 구간은 **관측 >=1300W & 전기포트 없음 & 오븐·핫플 동시
통전** 이다. 유령의 주범이 전기포트고 그 구간에서 포트·핫플·오븐이 서로를
대신할 수 있으므로, 예측하는 바는:

    (a) 실행 간 차이가 >=1300W 구간에 몰린다
    (b) 그 차이의 대부분이 전기포트(없는 기기)와 핫플(있는 기기) **사이의 맞교환**이다
        - 즉 창마다 Δ(포트) 와 Δ(핫플) 의 부호가 반대이고 크기가 비슷하다
    (c) 프로젝터·미니PC·오븐은 어느 구간에서도 안 흔들린다

(b) 가 맞으면 유령과 핫플 F1 은 **같은 하나의 미결정**이고 지표가 둘로 보일 뿐이다.
틀리면 원인이 둘이고 따로 고쳐야 한다.
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.run_gate_check import forward_file, load_model
from src.run_live import KOR

# 12.38.1 / 인수인계 2.1 이 쓰는 구간
BINS = [(0.0, 300.0), (300.0, 800.0), (800.0, 1300.0), (1300.0, 1e9)]
BIN_LAB = ["<300W", "300-800", "800-1300", ">=1300"]


def _mask_from(pairs, n_cycles: int, targets: np.ndarray) -> np.ndarray:
    m = np.zeros(n_cycles, bool)
    for s, e in pairs or []:
        m[int(float(s) * 60):int(float(e) * 60)] = True
    return m[targets]


def collect(ckpts: List[str], stems: List[str], dev: str, stride: int) -> dict:
    """체크포인트마다 파일마다 창 단위 예측을 모은다."""
    out = {}
    for c in ckpts:
        model, apps, _ = load_model(c, dev)
        per = {}
        for stem in stems:
            d = forward_file(model, stem, dev, stride=stride)
            per[stem] = {"P": d["gate"] * d["p_raw"], "gate": d["gate"],
                         "pobs": d["p_observed"], "targets": d["targets"]}
        out[Path(c).stem] = {"apps": apps, "files": per}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="시드 분산의 소재 진단 (12.41절)")
    ap.add_argument("--ckpt", nargs=2,
                    default=["results/cnn_ov1.pt", "results/cnn_ov1_s1.pt"],
                    help="같은 설정·다른 시드 두 개")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/seed_variance_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    data = collect(a.ckpt, stems, dev, a.stride)
    tags = list(data)
    A, B = data[tags[0]], data[tags[1]]
    apps = A["apps"]
    payload: Dict[str, object] = {"ckpt": tags, "stems": stems}

    print("=" * 92)
    print(f"[시드 분산의 소재]  {tags[0]}  vs  {tags[1]}   "
          f"({len(stems)}파일, stride {a.stride})")
    print("=" * 92)

    # ── (1) 유령을 기기별로 가른다 ────────────────────────────────────────
    print()
    print("  [1] 유령(없는 기기에 붙은 전력)을 기기별로 — 파일 평균 W")
    print(f"    {'기기':<12s}{tags[0]:>14s}{tags[1]:>14s}{'차이':>10s}{'없는파일':>9s}")
    print("    " + "-" * 59)
    ghost_rows, tot_a, tot_b = {}, 0.0, 0.0
    for j, app in enumerate(apps):
        va, vb = [], []
        for stem in stems:
            if app in ev[stem]["appliances_present"]:
                continue
            va.append(float(A["files"][stem]["P"][:, j].mean()))
            vb.append(float(B["files"][stem]["P"][:, j].mean()))
        if not va:
            continue
        # gate_check 와 같은 정의: 파일별 합계를 **전 파일 수**로 나눈다
        ma, mb = float(np.sum(va) / len(stems)), float(np.sum(vb) / len(stems))
        tot_a += ma
        tot_b += mb
        ghost_rows[app] = {"a": ma, "b": mb, "n_files": len(va)}
        print(f"    {KOR.get(app, app):<12s}{ma:>14.2f}{mb:>14.2f}"
              f"{mb - ma:>+10.2f}{len(va):>9d}")
    print("    " + "-" * 59)
    print(f"    {'합계':<12s}{tot_a:>14.2f}{tot_b:>14.2f}{tot_b - tot_a:>+10.2f}")
    payload["ghost_by_app"] = ghost_rows

    # ── (2) 관측 전력 구간별 ──────────────────────────────────────────────
    print()
    print("  [2] 그 차이가 어느 관측 구간에서 오는가 — 창 가중 유령 W")
    print(f"    {'구간':<10s}{'창비율':>8s}{tags[0]:>14s}{tags[1]:>14s}"
          f"{'차이':>10s}{'전체차이중':>11s}")
    print("    " + "-" * 68)
    binrows, dtot = [], 0.0
    for (lo, hi), lab in zip(BINS, BIN_LAB):
        sa = sb = nw = 0.0
        for stem in stems:
            fa, fb = A["files"][stem], B["files"][stem]
            m = (fa["pobs"] >= lo) & (fa["pobs"] < hi)
            if not m.any():
                continue
            absent = [j for j, x in enumerate(apps)
                      if x not in ev[stem]["appliances_present"]]
            sa += float(fa["P"][np.ix_(m, absent)].sum())
            sb += float(fb["P"][np.ix_(m, absent)].sum())
            nw += float(m.sum())
        binrows.append({"bin": lab, "n_windows": nw, "sum_a": sa, "sum_b": sb,
                        "a_w": sa / nw if nw else 0.0, "b_w": sb / nw if nw else 0.0})
        dtot += sb - sa
    n_all = sum(r["n_windows"] for r in binrows)
    for r in binrows:
        share = (r["sum_b"] - r["sum_a"]) / dtot if dtot else float("nan")
        print(f"    {r['bin']:<10s}{100 * r['n_windows'] / n_all:>7.1f}%"
              f"{r['a_w']:>14.2f}{r['b_w']:>14.2f}{r['b_w'] - r['a_w']:>+10.2f}"
              f"{100 * share:>10.1f}%")
    payload["ghost_by_bin"] = binrows

    # ── (3) >=1300W 안에서 구성별 ─────────────────────────────────────────
    print()
    print("  [3] >=1300W 안에서 — 오븐 히터·핫플 통전 구성별 (test_4/5/6)")
    print(f"    {'구성':<20s}{'창':>7s}{tags[0]:>14s}{tags[1]:>14s}{'차이':>10s}")
    print("    " + "-" * 65)
    comp = {"오븐통전+핫플통전": [0.0, 0.0, 0.0], "오븐통전+핫플휴지": [0.0, 0.0, 0.0],
            "오븐휴지+핫플통전": [0.0, 0.0, 0.0], "오븐휴지+핫플휴지": [0.0, 0.0, 0.0]}
    for stem in ("test_4", "test_5", "test_6"):
        if stem not in stems:
            continue
        fa, fb = A["files"][stem], B["files"][stem]
        n_cyc, tgt = int(ev[stem]["cycles"]), fa["targets"]
        iv = ev[stem]["intervals"]
        ovm = _mask_from(iv.get("oven", {}).get("_heater_pulses"), n_cyc, tgt)
        hpm = _mask_from(iv.get("hotplate", {}).get("on"), n_cyc, tgt)
        hi = fa["pobs"] >= 1300.0
        absent = [j for j, x in enumerate(apps)
                  if x not in ev[stem]["appliances_present"]]
        for lab, m in (("오븐통전+핫플통전", ovm & hpm), ("오븐통전+핫플휴지", ovm & ~hpm),
                       ("오븐휴지+핫플통전", ~ovm & hpm), ("오븐휴지+핫플휴지", ~ovm & ~hpm)):
            mm = m & hi
            if not mm.any():
                continue
            comp[lab][0] += float(mm.sum())
            comp[lab][1] += float(fa["P"][np.ix_(mm, absent)].sum())
            comp[lab][2] += float(fb["P"][np.ix_(mm, absent)].sum())
    for lab, (nw, sa, sb) in comp.items():
        if nw == 0:
            continue
        print(f"    {lab:<20s}{int(nw):>7d}{sa / nw:>14.2f}{sb / nw:>14.2f}"
              f"{(sb - sa) / nw:>+10.2f}")
    payload["ghost_by_composition"] = {k: v for k, v in comp.items() if v[0]}

    # ── (4) 있는 기기는 정말 안 흔들리는가 ────────────────────────────────
    print()
    print("  [4] 창 단위 불일치 — 두 실행의 예측 전력 차 (그 기기가 있는 파일에서만)")
    print(f"    {'기기':<12s}{'평균|ΔW|':>11s}{'p95|ΔW|':>10s}"
          f"{'게이트불일치':>13s}{'평균예측W':>11s}")
    print("    " + "-" * 58)
    dis = {}
    for j, app in enumerate(apps):
        da, gg, pm = [], [], []
        for stem in stems:
            if app not in ev[stem]["appliances_present"]:
                continue
            fa, fb = A["files"][stem], B["files"][stem]
            da.append(np.abs(fa["P"][:, j] - fb["P"][:, j]))
            gg.append((fa["gate"][:, j] > 0.5) != (fb["gate"][:, j] > 0.5))
            pm.append(fa["P"][:, j])
        if not da:
            continue
        da, gg, pm = np.concatenate(da), np.concatenate(gg), np.concatenate(pm)
        dis[app] = {"mean_abs_dw": float(da.mean()),
                    "p95_dw": float(np.percentile(da, 95)),
                    "gate_disagree": float(gg.mean()),
                    "mean_pred_w": float(pm.mean())}
        print(f"    {KOR.get(app, app):<12s}{da.mean():>11.2f}"
              f"{np.percentile(da, 95):>10.2f}{100 * gg.mean():>12.1f}%"
              f"{pm.mean():>11.2f}")
    payload["disagreement"] = dis

    # ── (5) 포트 <-> 핫플 맞교환인가 ──────────────────────────────────────
    print()
    print("  [5] 가설 (b) — Δ(포트, 없는 기기) 와 Δ(핫플, 있는 기기) 가 맞교환인가")
    print("      맞교환이면 r 이 -1 에 가깝고 |Δ포트+Δ핫플| 이 |Δ포트| 보다 훨씬 작다")
    jk, jh = apps.index("electiric_kettle"), apps.index("hotplate")
    swap = {}
    for stem in ("test_4", "test_5", "test_6"):
        if stem not in stems:
            continue
        fa, fb = A["files"][stem], B["files"][stem]
        hi = fa["pobs"] >= 1300.0
        if hi.sum() < 20:
            print(f"    {stem:8s} >=1300W 창 {int(hi.sum())}개 — 건너뜀")
            continue
        dk = fb["P"][hi, jk] - fa["P"][hi, jk]
        dh = fb["P"][hi, jh] - fa["P"][hi, jh]
        r = (float(np.corrcoef(dk, dh)[0, 1])
             if dk.std() > 0 and dh.std() > 0 else float("nan"))
        print(f"    {stem:8s} 창 {int(hi.sum()):>5d}  Δ포트 {dk.mean():>+8.1f}W"
              f"  Δ핫플 {dh.mean():>+8.1f}W  r = {r:>+.3f}"
              f"  |Δ포트+Δ핫플| {np.abs(dk + dh).mean():>6.1f}W"
              f"  vs |Δ포트| {np.abs(dk).mean():>6.1f}W")
        swap[stem] = {"n": int(hi.sum()), "d_kettle_w": float(dk.mean()),
                      "d_hotplate_w": float(dh.mean()), "corr": r,
                      "abs_sum_w": float(np.abs(dk + dh).mean()),
                      "abs_kettle_w": float(np.abs(dk).mean())}
    payload["swap_test"] = swap

    # ── (6) >=1300W 구간의 총결산 — 재분배인가 덧붙임인가 ─────────────────
    print()
    print("  [6] >=1300W 구간 총결산 — 유령이 남의 몫을 뺏은 것인가, 위에 얹은 것인가")
    hi_all = {s: (A["files"][s]["pobs"] >= 1300.0) for s in stems}
    nw = sum(int(m.sum()) for m in hi_all.values())
    print(f"      창 {nw}개 ({100 * nw / sum(len(m) for m in hi_all.values()):.1f}%)")
    print(f"    {'항목':<14s}{tags[0]:>14s}{tags[1]:>14s}{'차이':>10s}")
    print("    " + "-" * 52)

    def _hi_mean(side, col):
        num = sum(float(col(side["files"][s])[hi_all[s]].sum())
                  for s in stems if hi_all[s].any())
        return num / nw if nw else 0.0

    rows6 = {}
    for lab, col in (("관측 P", lambda f: f["pobs"]),
                     ("예측 기기합", lambda f: f["P"].sum(1))):
        va, vb = _hi_mean(A, col), _hi_mean(B, col)
        rows6[lab] = {"a": va, "b": vb}
        print(f"    {lab:<14s}{va:>14.1f}{vb:>14.1f}{vb - va:>+10.1f}")
    for j, app in enumerate(apps):
        va = _hi_mean(A, lambda f, j=j: f["P"][:, j])
        vb = _hi_mean(B, lambda f, j=j: f["P"][:, j])
        if max(abs(va), abs(vb)) < 1.0:
            continue
        rows6[app] = {"a": va, "b": vb}
        print(f"      {KOR.get(app, app):<12s}{va:>14.1f}{vb:>14.1f}{vb - va:>+10.1f}")
    payload["hi_budget"] = rows6

    # 유령이 '망설임'인가 '확신하며 틀림'인가 (12.9.14 의 구분)
    print()
    print("      유령 포트의 모양 — 하드 게이트로 폭이 안 줄었다면 확신하며 틀리는 것이다")
    print(f"    {'':14s}{'게이트>0.5 창':>14s}{'그때 중앙 W':>13s}{'평균 게이트':>12s}")
    for tg, side in ((tags[0], A), (tags[1], B)):
        g = np.concatenate([side["files"][s]["gate"][hi_all[s], jk]
                            for s in stems if hi_all[s].any()])
        pw = np.concatenate([side["files"][s]["P"][hi_all[s], jk]
                             for s in stems if hi_all[s].any()])
        on = g > 0.5
        med = float(np.median(pw[on])) if on.any() else 0.0
        print(f"    {tg:<14s}{100 * on.mean():>13.1f}%{med:>13.1f}{g.mean():>12.3f}")
        payload.setdefault("kettle_shape", {})[tg] = {
            "on_rate": float(on.mean()), "median_w_when_on": med,
            "mean_gate": float(g.mean())}

    # ── (7) 핫플 F1 을 구간으로 가른다 ────────────────────────────────────
    print()
    print("  [7] 핫플 F1 0.120 은 어디서 오는가 — 관측 구간별 정밀도/재현율")
    print(f"    {'파일':<8s}{'구간':<10s}{'참ON창':>7s}"
          f"{'  ' + tags[0] + ' P/R/F1':<26s}{'  ' + tags[1] + ' P/R/F1':<26s}")
    print("    " + "-" * 78)
    hp_rows = {}
    for stem in ("test_4", "test_5", "test_6"):
        if stem not in stems:
            continue
        fa, fb = A["files"][stem], B["files"][stem]
        truth = _mask_from(ev[stem]["intervals"].get("hotplate", {}).get("on"),
                           int(ev[stem]["cycles"]), fa["targets"])
        for lo, lab in ((0.0, "<1300"), (1300.0, ">=1300")):
            m = (fa["pobs"] >= lo) if lo else (fa["pobs"] < 1300.0)
            if m.sum() < 20:
                continue
            cells = []
            for f in (fa, fb):
                pred = f["gate"][m, jh] > 0.5
                t = truth[m]
                tp = float((pred & t).sum())
                pr = tp / max(pred.sum(), 1)
                rc = tp / max(t.sum(), 1)
                f1 = 2 * pr * rc / max(pr + rc, 1e-9)
                cells.append((pr, rc, f1))
            hp_rows[f"{stem}/{lab}"] = {"n": int(m.sum()), "n_true_on": int(truth[m].sum()),
                                        "a": cells[0], "b": cells[1]}
            print(f"    {stem:<8s}{lab:<10s}{int(truth[m].sum()):>7d}  "
                  + "  ".join(f"{p:.3f}/{r:.3f}/{f:.3f}" for p, r, f in cells)
                  + f"    ΔF1 {cells[1][2] - cells[0][2]:+.3f}")
    payload["hotplate_by_bin"] = hp_rows

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print()
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
