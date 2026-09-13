"""판 표를 손으로 옮겨 적지 않는다 — `pc_*` + `gc_*` 를 인수인계 표 한 장으로

설계 문서와 인수인계가 12.122 이래 같은 표를 손으로 옮기고 있다. 열이 여덟 개고
**출처가 둘**이라(`run_power_check` 와 `run_gate_check`) 옮길 때마다 어느 파일
집합의 평균인지 다시 찾아야 했다 (규칙 34).

    프로젝터W                    pc_*.json  summary.beam_projector.mean_err_w
    유령8 / 잔차 / F1 / 기기별    gc_*.json  겨냥 8파일 평균
    유령대조                     gc_*.json  **대조 3파일**(test_9/11/12) 평균

⚠ **`유령대조` 만 대조 3파일이고 나머지는 전부 겨냥 8파일이다** (12.122.17 의
   표 관례). 전 11파일로 평균 내면 기준선 F1 이 0.846 이 아니라 0.839 로 나온다.

    python -X utf8 -m src.run_summarize_pan --pc results/pc_ss.json --gc results/gc_ss.json
    python -X utf8 -m src.run_summarize_pan --pc ... --gc ... --base "기준선 w4.0"
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import re
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

CTRL = ("test_9", "test_11", "test_12")
SEED = re.compile(r"_s\d+$")

#: 태그 접두사 -> 표시 이름. 12.137.1 의 판 이름을 그대로 쓴다.
NAMES = {"adapt_ovh": "기준선 w4.0", "adapt_hw2": "가중 1/h²",
         "adapt_isig3": "in-situ 지문", "adapt_wi": "가중+지문",
         "adapt_ss": "단일기기(대조청정)", "adapt_sa": "단일기기(전파일)",
         "adapt_pref": "전력사전", "adapt_nt": "Norton 크기만",
         "adapt_zi": "복소 Z·I", "adapt_zl0": "Z·I, λ=0 (replay 없음)",
         "adapt_zc30": "Z·I, 캐시 30k(2.2GB)", "adapt_zc10": "Z·I, 캐시 10k(0.7GB)",
         "adapt_zc3": "Z·I, 캐시 3k(0.2GB)", "adapt_zv": "Z·I + h1 전압보정",
         "adapt_vs": "h1 전압보정만 (Z 불필요)",
         "adapt_znr": "Z·I + h1 정규화",
         "adapt_zvnr": "Z·I + h1 정규화 + 추종"}


def group(tags: List[str]) -> Dict[str, List[str]]:
    """`_s<N>` 을 떼어 설정별로 묶는다. `adapt_ovh` 는 접미사가 없다 (시드 0)."""
    out: Dict[str, List[str]] = {}
    for t in tags:
        out.setdefault(SEED.sub("", t), []).append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pc", required=True, help="run_power_check 산출물")
    ap.add_argument("--gc", required=True, help="run_gate_check 산출물")
    ap.add_argument("--gate", default="soft", choices=("soft", "hard"))
    # `run_gate_check` 의 후처리 접미사. 12.149 가 `+sq0.5+ab1` 을 더했다.
    ap.add_argument("--suffix", default="", metavar="S",
                    help="gc 태그 접미사. 기본은 산출물에서 자동으로 찾는다 "
                         "(예: '+pp+rm0.02', '+pp+rm0.02+sq0.5+ab1')")
    ap.add_argument("--base", default="adapt_ovh", help="Δ 를 재는 기준 판의 태그 접두사")
    # ── 사람 라벨 층 (2026-09-02, 규칙 24) ──────────────────────────────
    # 겨냥 8파일에는 `ai_inferred_signal_checked` 인 셋(test.2/test3/test_4)이
    # 섞여 있다. 그 라벨은 신호에서 추정한 뒤 재대조한 것이라 **모델과 완전히
    # 독립이 아니다.** 이 깃발은 `human_switching_log` 인 파일만 남긴다:
    #     겨냥  test_5 / test_6 / test_7 / test_8 / test_13   (5)
    #     대조  test_11 / test_12                            (2. test_9 는 신호추정이라 빠진다)
    ap.add_argument("--human-only", action="store_true",
                    help="사람이 스위치를 적은 파일만으로 층화한다 (규칙 24)")
    a = ap.parse_args()

    pc = json.loads(Path(a.pc).read_text(encoding="utf-8"))
    gc = json.loads(Path(a.gc).read_text(encoding="utf-8"))
    print(f"  프로젝터W: {a.pc}")
    print(f"  나머지   : {a.gc}")
    print(f"  그 명령  : {' '.join(gc.get('_config', {}).get('argv', ['(없음)'])[1:])}\n")

    gtags = [t for t in gc if not t.startswith("_")]
    # 접미사를 **산출물에서 읽는다.** 손으로 적으면 파이프라인이 바뀔 때마다
    # 조용히 어긋난다 (규칙 33 — 명령이 산출물에 있다).
    sfx = a.suffix
    if not sfx:
        cand = {t[t.index("+"):] for t in gtags if "+" in t}
        if len(cand) != 1:
            raise SystemExit(f"접미사가 하나가 아닙니다: {sorted(cand)}. --suffix 로 고르십시오")
        sfx = cand.pop()
    print(f"  후처리   : {sfx}")

    ctrl, keep = CTRL, None
    if a.human_only:
        from src.evaluation.real_events import load_events
        from src.run_summarize_gate import human_stems
        keep = set(human_stems(load_events()))
        ctrl = tuple(s for s in CTRL if s in keep)
        print(f"  층       : 사람 라벨만 — 겨냥 {sorted(keep - set(CTRL))} / 대조 {list(ctrl)}")
    print()
    groups = group([t.split("+")[0] for t in gtags])

    rows: Dict[str, dict] = {}
    for pre, tags in groups.items():
        r: Dict[str, np.ndarray] = {}
        psfx = "" if any(t in pc for t in tags) else             sorted({k[k.index("+"):] for k in pc if not k.startswith("_") and "+" in k})[0]
        if keep is None:
            r["proj_w"] = np.array(
                [pc[t + psfx]["summary"]["beam_projector"]["mean_err_w"]
                 for t in tags if t + psfx in pc])
        else:
            # 층을 바꾸면 요약도 그 층에서 다시 낸다 — `summary` 를 그대로 쓰면
            # 다른 파일 집합의 평균을 인용하게 된다 (규칙 34).
            from src.evaluation.power_ref import summarize_power_ref
            r["proj_w"] = np.array([
                summarize_power_ref({s: v for s, v in pc[t + psfx]["per_file"].items()
                                     if s in keep}).get(
                    "beam_projector", {"mean_err_w": np.nan})["mean_err_w"]
                for t in tags if t + psfx in pc])
        per = [gc[t + sfx] for t in tags]
        stems = [s for s in per[0] if not s.startswith("_")
                 if keep is None or s in keep]
        tgt = [s for s in stems if s not in ctrl]
        r["ghost8"] = np.array([np.mean([p[s][a.gate]["absent_sum_w"] for s in tgt])
                                for p in per])
        r["ghostC"] = np.array([np.mean([p[s][a.gate]["absent_sum_w"] for s in ctrl])
                                for p in per])
        r["resid"] = np.array([np.mean([p[s][a.gate]["residual_abs_w"] for s in tgt])
                               for p in per])

        # 12.122.17 의 표 관례 — `on_off_f1_mean` 의 **겨냥 8파일** 평균이다.
        # 기기별 F1 을 5종 평균하면 0.846 이 아니라 0.848 이 나온다 (규칙 34).
        r["f1"] = np.array([np.mean([p[s][a.gate]["on_off_f1_mean"] for s in tgt])
                            for p in per])

        def app_metric(p, key, field):
            v = [p[s][a.gate]["per_app_f1"].get(key) for s in tgt]
            v = [x[field] for x in v
                 if x and x.get("n_true_on", 0) > 0 and x[field] == x[field]]
            return float(np.mean(v)) if v else float("nan")

        for nm, key, fld in (("pj_p", "beam_projector", "precision"),
                             ("lc_r", "laptop_charger", "recall"),
                             ("mp_p", "minipc", "precision")):
            r[nm] = np.array([app_metric(p, key, fld) for p in per])
        rows[NAMES.get(pre, pre)] = r

    cols = [("proj_w", "프로젝터W"), ("ghost8", "유령8"), ("ghostC", "유령대조"),
            ("resid", "잔차"), ("f1", "F1"), ("pj_p", "프로젝정밀"),
            ("lc_r", "충전재현"), ("mp_p", "미니정밀")]
    print(f"  {'판':<20s}{'n':>3s}" + "".join(f"{lab:>16s}" for _, lab in cols))
    print("  " + "-" * (23 + 16 * len(cols)))
    order = [NAMES[k] for k in NAMES if NAMES[k] in rows] + \
            [k for k in rows if k not in NAMES.values()]
    for name in order:
        r = rows[name]
        cells = []
        for key, _ in cols:
            v = r[key][~np.isnan(r[key])]
            fmt = ".2f" if key in ("proj_w", "ghost8", "ghostC", "resid") else ".3f"
            cells.append(f"{v.mean():>9{fmt}} ±{v.max() - v.min():>5.2f}"
                         if len(v) > 1 else f"{v[0]:>9{fmt}}{'':>6s}")
        print(f"  {name:<20s}{len(r['proj_w']):>3d}" + "".join(f"{c:>16s}" for c in cells))

    base = NAMES.get(a.base, a.base)
    if base in rows:
        print(f"\n  Δ vs {base}  (± 는 실행 간 폭 max−min. 표준편차가 아니다)\n")
        print(f"  {'판':<20s}" + "".join(f"{lab:>12s}" for _, lab in cols))
        print("  " + "-" * (20 + 12 * len(cols)))
        for name in order:
            if name == base:
                continue
            print(f"  {name:<20s}" + "".join(
                f"{np.nanmean(rows[name][k]) - np.nanmean(rows[base][k]):>+12.3f}"
                for k, _ in cols))
    print(f"\n  ⚠ `유령대조` 만 대조 {len(ctrl)}파일({'/'.join(ctrl)}), "
          f"나머지는 겨냥 파일 평균이다"
          + (" (사람 라벨 5파일)." if a.human_only else " (8파일)."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
