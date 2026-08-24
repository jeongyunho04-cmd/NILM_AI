"""
`run_gate_check` 산출물을 한 줄로 요약한다 (설계 문서 12.39절)
================================================================
설계 문서와 인수인계가 계속 같은 표를 손으로 옮겨 적고 있었다. 그 표를 만든다.

    | 태그 | 유령 | 잔차 | 미니PC | 프로젝터 | 충전기 | 핫플 | 오븐 |

**유령·잔차는 비봉인 전 파일 평균 W, F1 은 사람 기록 파일 평균이다.**
사람 기록(`human_switching_log`)과 AI 추정(`ai_inferred_signal_checked`)을 섞어
평균 내면 정답 등급이 다른 것을 한 숫자로 뭉갠다 (12.30.4절).

    # 한 파일 안의 모든 체크포인트
    python -m src.run_summarize_gate results/gate_check_ov.json

    # 여러 파일을 한 표로
    python -m src.run_summarize_gate results/gate_check_{pf,ov}.json

    # 시드 묶음 — `_s1` `_s2` 접미사를 떼어 설정별 평균±폭을 낸다 (12.39)
    python -m src.run_summarize_gate results/gate_check_seed.json --by-seed

[왜 폭까지 찍는가]
12.38.5 가 남긴 부채다. 설정당 실행이 1회면 설정 효과와 안착 운이 구분되지 않는다.
`--by-seed` 는 **같은 설정의 실행 간 폭**을 옆에 붙여, 설정 차이가 그 폭보다
큰지 눈으로 바로 보게 한다.
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

from src.evaluation.real_events import load_events

# 표에 세울 기기와 표시 이름. 순서가 인수인계 표와 같아야 옮겨 적기 쉽다.
APPS = [("minipc", "미니PC"), ("beam_projector", "프로젝터"),
        ("laptop_charger", "충전기"), ("hotplate", "핫플"), ("oven", "오븐")]

SEED_SUFFIX = re.compile(r"_s\d+$")


def human_stems(events: dict) -> List[str]:
    """사람이 스위치를 적은 파일만. 없으면 전부 (옛 정답 파일 대비)."""
    f = events.get("files", events)
    hs = [k for k, v in f.items()
          if isinstance(v, dict) and v.get("_label_provenance") == "human_switching_log"]
    return sorted(hs)


def summarize_one(per_file: dict, gate: str, hs: List[str]) -> dict:
    """체크포인트 하나 -> 표 한 줄."""
    stems = [s for s in per_file if not s.startswith("_")]
    row = {
        "ghost_w": float(np.mean([per_file[s][gate]["absent_sum_w"] for s in stems])),
        "resid_w": float(np.mean([per_file[s][gate]["residual_abs_w"] for s in stems])),
        "n_files": len(stems),
    }
    use = [s for s in hs if s in per_file] or stems
    row["n_f1_files"] = len(use)
    for key, _ in APPS:
        vals = []
        for s in use:
            v = per_file[s][gate]["per_app_f1"].get(key)
            # n_true_on == 0 이면 F1 이 정의되지 않는다. 0 으로 세면 안 된다.
            if v and v.get("n_true_on", 0) > 0 and v["f1"] == v["f1"]:
                vals.append(v["f1"])
        row[key] = float(np.mean(vals)) if vals else float("nan")
    return row


def fmt_table(rows: Dict[str, dict]) -> str:
    head = f"  {'태그':<18s}{'유령W':>9s}{'잔차W':>9s}" + "".join(
        f"{lab:>10s}" for _, lab in APPS) + f"{'파일':>7s}"
    out = [head, "  " + "-" * (len(head) - 2)]
    for tag, r in rows.items():
        out.append(f"  {tag:<18s}{r['ghost_w']:>9.2f}{r['resid_w']:>9.2f}" + "".join(
            f"{r[k]:>10.3f}" for k, _ in APPS) + f"{r['n_files']:>4d}/{r['n_f1_files']:<2d}")
    return "\n".join(out)


def fmt_seed_table(groups: Dict[str, List[dict]]) -> str:
    """설정별 평균과 **실행 간 폭**. 폭이 설정 차이보다 크면 비교가 성립하지 않는다."""
    cols = [("ghost_w", "유령W"), ("resid_w", "잔차W")] + [(k, l) for k, l in APPS]
    head = f"  {'설정':<16s}{'n':>3s}" + "".join(f"{lab:>20s}" for _, lab in cols)
    out = [head, "  " + "-" * (len(head) - 2)]
    for tag, rs in groups.items():
        cells = []
        for key, _ in cols:
            v = np.array([r[key] for r in rs], float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                cells.append(f"{'-':>20s}")
            elif len(v) == 1:
                cells.append(f"{v[0]:>13.3f}{'':>7s}")
            else:
                cells.append(f"{v.mean():>13.3f} ±{v.max() - v.min():>5.2f}")
        out.append(f"  {tag:<16s}{len(rs):>3d}" + "".join(cells))
    out.append("")
    out.append("  ± 는 실행 간 **폭**(max-min)이다. 표준편차가 아니다 — n=3 에서는"
               " 폭이 더 정직하다.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="gate_check 결과 요약 (12.39절)")
    ap.add_argument("json", nargs="+", help="run_gate_check --out 산출물")
    ap.add_argument("--gate", choices=("soft", "hard"), default="soft")
    ap.add_argument("--by-seed", action="store_true",
                    help="`_s<N>` 접미사를 떼어 설정별로 묶고 실행 간 폭을 낸다")
    ap.add_argument("--csv", default=None, help="같은 표를 CSV 로도 저장")
    a = ap.parse_args()

    hs = human_stems(load_events())
    rows: Dict[str, dict] = {}
    for path in a.json:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for tag, per_file in payload.items():
            if tag in rows:
                print(f"  ⚠ 태그 중복 '{tag}' — 나중 파일({path})로 덮어씁니다.")
            rows[tag] = summarize_one(per_file, a.gate, hs)

    print(f"게이트 {a.gate} | F1 은 사람 기록 {', '.join(hs) or '(없음)'} 평균")
    print()
    print(fmt_table(rows))

    if a.by_seed:
        groups: Dict[str, List[dict]] = {}
        for tag, r in rows.items():
            groups.setdefault(SEED_SUFFIX.sub("", tag), []).append(r)
        print()
        print(fmt_seed_table(groups))

    if a.csv:
        keys = ["ghost_w", "resid_w"] + [k for k, _ in APPS]
        lines = ["tag," + ",".join(keys)]
        lines += [t + "," + ",".join(f"{r[k]:.6g}" for k in keys) for t, r in rows.items()]
        Path(a.csv).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n저장: {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
