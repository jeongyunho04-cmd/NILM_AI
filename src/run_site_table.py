# -*- coding: utf-8 -*-
"""run_gate_check 의 JSON 을 장소별 표로 접는다.

인수인계 표(hwO 행)와 **같은 자**다 — soft, 후처리 off, 파일별 창 수 가중.
장소 A = test_5~13 (SMPS 위주 + 오븐), 장소 B = test_15~18 (저항 위주, 오븐 없음).
test_14 는 어느 쪽에도 안 넣는다 (두 장소가 섞인 이사 당일 녹화).
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np

SITE = {
    "A": ("test_5", "test_6", "test_7", "test_8", "test_11", "test_12", "test_13"),
    "B": ("test_15", "test_16", "test_17", "test_18"),
}
COLS = ("hair_dryer", "electiric_kettle", "hotplate", "minipc", "beam_projector",
        "oven", "laptop_charger", "fan", "air_conditioner")
SHORT = {"hair_dryer": "드라이기", "electiric_kettle": "포트", "hotplate": "핫플",
         "minipc": "미니PC", "beam_projector": "프로젝터", "oven": "오븐",
         "laptop_charger": "충전기", "fan": "선풍기", "air_conditioner": "에어컨"}


def fold(det, stems, mode="soft"):
    stems = [s for s in stems if s in det]
    if not stems:
        return None
    # 파일 가중은 채점된 사이클 수로 준다 (per_app_f1 의 n_scored).
    w = np.array([max(v.get("n_scored", 0)
                      for v in det[s][mode]["per_app_f1"].values())
                  for s in stems], float)
    row = {"n": int(w.sum()), "stems": stems}
    for app in COLS:
        # **그 파일에 없는 기기는 F1 에 안 넣는다.** `n_true_on == 0` 이면
        # gate_check 이 f1 을 0.0 으로 적는데(오탐이 하나라도 있으면), 그것은
        # 검출력이 아니라 유령이고 `ghost`/`absent` 열이 따로 센다. 안 거르면
        # test_5 의 포트 오탐 2창(1초)이 장소 A 포트 F1 을 0.949 -> 0.502 로
        # 끌어내려 "시드 다중해" 로 잘못 읽힌다 (12.179 에서 겪었다).
        f = np.array([det[s][mode]["per_app_f1"].get(app, {}).get("f1", np.nan)
                      if det[s][mode]["per_app_f1"].get(app, {}).get("n_true_on", 0) > 0
                      else np.nan
                      for s in stems], float)
        m = np.isfinite(f)
        row[app] = float(np.average(f[m], weights=w[m])) if m.any() else float("nan")
    # 전체 F1 은 gate_check 이 파일마다 내는 `on_off_f1_mean` 을 접는다.
    # 열 평균과 다르다 — 저쪽은 그 파일에 실제로 있는 기기만 센다.
    row["F1"] = float(np.average([det[s][mode]["on_off_f1_mean"] for s in stems],
                                 weights=w))
    row["colF1"] = float(np.nanmean([row[a] for a in COLS]))
    row["resid"] = float(np.average([det[s][mode]["residual_abs_w"] for s in stems],
                                    weights=w))
    row["ghost"] = float(np.average([det[s][mode]["absent_sum_w"] for s in stems],
                                    weights=w))
    row["absent"] = {}
    for a in COLS:
        v = [(det[s][mode]["absent"].get(a, {}).get("mean_w", np.nan), wi)
             for s, wi in zip(stems, w)]
        v = [(x, n) for x, n in v if np.isfinite(x)]
        if v:
            row["absent"][a] = float(np.average([x for x, _ in v],
                                                weights=[n for _, n in v]))
    return row


def main():
    ap = argparse.ArgumentParser(description="장소별 표")
    ap.add_argument("json", nargs="+")
    ap.add_argument("--site", default="B", choices=("A", "B", "both"))
    ap.add_argument("--mode", default="soft", choices=("soft", "hard"))
    ap.add_argument("--median", action="store_true", help="시드 중앙값 한 줄로 접는다")
    ap.add_argument("--absent", action="store_true", help="없는 기기별 유령W 도 낸다")
    a = ap.parse_args()

    sites = ("A", "B") if a.site == "both" else (a.site,)
    for site in sites:
        rows = []
        for f in a.json:
            blob = json.load(open(f, encoding="utf-8"))
            for tag, det in blob.items():
                if tag.startswith("_"):
                    continue
                r = fold(det, SITE[site], a.mode)
                if r:
                    rows.append((tag, r))
        if not rows:
            continue
        print(f"\n=== 장소 {site} ({a.mode}, 창 {rows[0][1]['n']:,}) "
              f"{'/'.join(rows[0][1]['stems'])} ===")
        head = f"{'':22s}" + "".join(f"{SHORT[c]:>9s}" for c in COLS) \
               + f"{'전체F1':>9s}{'잔차':>8s}{'유령W':>8s}"
        print(head)
        for tag, r in rows:
            print(f"{tag:22s}" + "".join(
                ("      .   " if math.isnan(r[c]) else f"{r[c]:9.3f}") for c in COLS)
                + f"{r['F1']:9.3f}{r['resid']:8.2f}{r['ghost']:8.2f}")
        if a.median and len(rows) > 1:
            med = {k: float(np.median([r[k] for _, r in rows]))
                   for k in list(COLS) + ["F1", "colF1", "resid", "ghost"]}
            print(f"{'** 중앙 **':22s}" + "".join(
                ("      .   " if math.isnan(med[c]) else f"{med[c]:9.3f}") for c in COLS)
                + f"{med['F1']:9.3f}{med['resid']:8.2f}{med['ghost']:8.2f}")
        if a.absent:
            keys = sorted({k for _, r in rows for k in r["absent"]})
            print("  [없는 기기에 준 전력 W]  " + "  ".join(
                f"{SHORT[k]}" for k in keys))
            for tag, r in rows:
                print(f"  {tag:20s}" + "".join(
                    f"{r['absent'].get(k, float('nan')):9.2f}" for k in keys))


if __name__ == "__main__":
    main()
