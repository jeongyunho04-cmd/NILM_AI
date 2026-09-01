"""
사람 스위칭 로그 지도 — Leave-One-File-Out (SMPS_PLAN 4.5절)
===============================================================
`test_5/6/7/8/13` 은 **스위치를 누른 사람이 그 자리에서 적은** on/off 정답을
갖고 있다 (`_label_provenance: human_switching_log`, 3,499초 = 58.3분).
그런데 이 저장소는 그것을 **채점에만 썼다** — `run_adapt` 는 `crit.unlabeled`
하나만 부르고 그 손실에는 기기별 정답이 한 항도 없다.

이 스크립트가 묻는 것은 하나다:

    기기별 on/off 정답을 직접 줘도 SMPS 배분이 안 갈리는가?

    갈린다   -> 12.87.3 의 '배분 미결정' 이 틀렸다. 정보가 없던 게 아니라
                **학습 신호가 없었던 것**이다. 처방은 커미셔닝 녹화다
    안 갈린다 -> 12.87.3 이 가장 강한 형태로 확정된다. 관측에 없는 것이다
                -> 계측 축(SMPS_PLAN 4.6)만 남고 그 판단이 확실해진다

**어느 쪽이 나와도 결정적이다.** 그것이 이 항목의 값어치다.

[왜 LOFO 인가]
라벨을 학습에 넣고 **같은 파일로 채점하면** 시험 문제를 보여주고 시험을 본
것이라 아무것도 증명하지 못한다. 파일 하나를 빼고 적응한 뒤 그 파일로 채점한다.
`run_adapt --holdout-real` 이 이미 *"뺀 파일도 채점은 한다"* 라서 폴드 코드를
새로 짤 게 없다.

[대조 — 규칙 20]
`test_9`(사람 라벨 없음)와 `test_11/12`(SMPS 0종)는 거의 안 변해야 한다.
`realdata.HUMAN_ON_DEFAULT_STEMS` 가 그 셋을 지도에서 빼 놨다. 그래도 재학습
잡음은 도므로 **대조 파일의 변화량을 같이 찍고**, 그것이 겨냥한 파일만큼
움직이면 판정을 보류한다 (12.114 의 형태).

[⚠ 잡음 바닥을 먼저 재야 한다]
12.115 가 잰 2단계 시드 폭은 **집계 지표만**이다 (유령 6.05W / F1 0.011 /
전이 9). 이 실험의 주 판정은 **기기별** 충전기 재현·프로젝터 정밀인데 그
지표의 폭은 이 저장소가 한 번도 안 쟀다. `--baseline-only` 로 w_real_on=0
폴드를 먼저 돌려 폭을 만든 뒤 그것과 비교한다. 폭 없이 나온 +0.08 은 근거가
아니다.

    # 1) 잡음 바닥 (지도 끔, 시드 3개 x 5폴드)
    python -m src.run_real_on_lofo --baseline-only --seeds 0,1,2

    # 2) 본 실험 (w 세 점)
    python -m src.run_real_on_lofo --w 0.1,0.3,1.0 --seeds 0

    # 3) 요약만 다시
    python -m src.run_real_on_lofo --summarize-only
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import argparse
import json
import subprocess
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model.realdata import HUMAN_ON_CONTROL_STEMS, HUMAN_ON_DEFAULT_STEMS

#: 판정을 보는 기기. 프로젝터↔충전기가 이 저장소 최악의 쌍이다 (OPERATING_POINT 7절).
SMPS = ("beam_projector", "laptop_charger", "minipc")

#: 운영점 그대로 (OPERATING_POINT / `results/_adapt_ovh.log`).
#: `--w-cons 0.1 --w-harm 4.0` 은 run_adapt 기본값이라 따로 안 넘긴다.
DEFAULT_INIT = "results/cnn_ovh.pt"
DEFAULT_CACHE = "cache/train60_ovh"


def fold_tag(w: float, seed: int, held: str) -> str:
    return f"ronl_w{w:g}_s{seed}_{held}".replace(".", "p")


def run_fold(a, w: float, seed: int, held: str) -> Path:
    """폴드 하나. `held` 를 적응에서 빼고 `held` 로 채점한다."""
    tag = fold_tag(w, seed, held)
    out = Path(a.out) / f"{tag}.json"
    if out.exists() and not a.force:
        print(f"  [skip] {tag} — 이미 있음")
        return out
    cmd = [sys.executable, "-m", "src.run_adapt",
           "--init", a.init, "--tag", tag, "--out", a.out,
           "--steps", str(a.steps), "--seed", str(seed),
           "--holdout-real", held,
           "--w-real-on", f"{w:g}", "--real-on-scope", a.scope]
    if a.cache:
        cmd += ["--cache", a.cache]
    print(f"  [run ] {tag}\n         {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"폴드 실패: {tag} (exit {r.returncode})")
    return out


def read_fold(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pick(payload: dict, stem: str) -> Dict[str, float]:
    """홀드아웃 파일 하나에서 판정 지표를 뽑는다.

    `real_detail[stem]["on_off"][app]` 에 기기별 precision/recall/f1 이 있다.
    **적응에 쓴 파일 점수는 안 본다** — 그쪽은 라벨을 본 파일이다.
    """
    det = (payload.get("real_detail") or {}).get(stem)
    if det is None:
        return {}
    oo = det.get("on_off", {})
    row: Dict[str, float] = {
        "residual_abs_w": det.get("residual_abs_w", float("nan")),
        "absent_sum_w": (det.get("absent") or {}).get("absent_sum_w", float("nan")),
    }
    for app in SMPS:
        s = oo.get(app)
        if not s:
            continue
        row[f"{app}.recall"] = s.get("recall", float("nan"))
        row[f"{app}.precision"] = s.get("precision", float("nan"))
        row[f"{app}.f1"] = s.get("f1", float("nan"))
    return row


def controls(payload: dict) -> Dict[str, float]:
    """대조 파일의 오귀속·잔차. 규칙 20 — 겨냥한 파일만큼 움직이면 판정 보류."""
    det = payload.get("real_detail") or {}
    out: Dict[str, float] = {}
    for stem in HUMAN_ON_CONTROL_STEMS:
        d = det.get(stem)
        if not d:
            continue
        out[f"{stem}.absent_sum_w"] = (d.get("absent") or {}).get("absent_sum_w", float("nan"))
        out[f"{stem}.residual_abs_w"] = d.get("residual_abs_w", float("nan"))
    return out


def mean_std(xs: Sequence[float]) -> str:
    xs = [x for x in xs if x == x]
    if not xs:
        return "     —      "
    if len(xs) == 1:
        return f"{xs[0]:.3f}       "
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return f"{m:.3f} ±{sd:.3f}"


def summarize(a, ws: List[float], seeds: List[int], stems: Sequence[str]) -> None:
    print()
    print("=" * 92)
    print("LOFO 요약 — 홀드아웃 파일에서만 잰 값 (적응에 안 쓴 파일)")
    print("=" * 92)

    table: Dict[float, Dict[str, List[float]]] = {}
    ctl: Dict[float, Dict[str, List[float]]] = {}
    for w in ws:
        table[w], ctl[w] = {}, {}
        for seed in seeds:
            for held in stems:
                p = read_fold(Path(a.out) / f"{fold_tag(w, seed, held)}.json")
                if p is None:
                    continue
                for k, v in pick(p, held).items():
                    table[w].setdefault(k, []).append(v)
                for k, v in controls(p).items():
                    ctl[w].setdefault(k, []).append(v)

    keys = [f"{app}.{m}" for app in SMPS for m in ("recall", "precision", "f1")]
    print(f"\n{'지표':28s}" + "".join(f"{'w=' + f'{w:g}':>16s}" for w in ws))
    print("-" * 92)
    for k in keys:
        row = "".join(f"{mean_std(table[w].get(k, [])):>16s}" for w in ws)
        # **주 판정은 프로젝터 정밀도다** (2026-08-31 기준선 15폴드로 정정).
        # SMPS_PLAN 4.5 는 '충전기 재현 0.641' 을 주 판정으로 적었는데, 사람 라벨
        # 5파일에서 실제로 재면 **0.961 (시드 SD 0.013)** 로 이미 포화다.
        # 0.641 은 다른 파일 집합의 값이고, `test.2/test3/test_4` 는 충전기 ON
        # 구간이 라벨에 없어(n_true_on=0) 재현율을 잴 수조차 없다.
        #
        # 실제로 무너져 있는 것은 **프로젝터 정밀도 0.794** 이고, 그것도 SMPS 가
        # 많은 파일에 몰려 있다 (test_5 0.710 / test_6 0.706 / test_7 0.755 vs
        # AI 라벨 파일 0.958~0.994). 프로젝터가 과대검출된다는 뜻이고,
        # `run_sig_conditioning` (6)절의 NNLS 가 프로젝터로 몰리는 것과 같은 방향이다.
        star = "  <- 주 판정" if k == "beam_projector.precision" else ""
        print(f"{k:28s}{row}{star}")
    print("-" * 92)
    for k in ("absent_sum_w", "residual_abs_w"):
        print(f"{k:28s}" + "".join(f"{mean_std(table[w].get(k, [])):>16s}" for w in ws))

    print(f"\n{'대조 (규칙 20)':28s}" + "".join(f"{'w=' + f'{w:g}':>16s}" for w in ws))
    print("-" * 92)
    for k in sorted({k for w in ws for k in ctl[w]}):
        print(f"{k:28s}" + "".join(f"{mean_std(ctl[w].get(k, [])):>16s}" for w in ws))

    print("\n" + "=" * 92)
    print("판정 (SMPS_PLAN 4.5 — 돌리기 전에 적은 것)")
    print("=" * 92)
    print("  프로젝터 정밀이 w=0 폭 밖으로 오른다     -> 12.87.3 이 틀렸다. 커미셔닝 녹화로 간다")
    print("     (시드 폭 SD 0.006, 파일 폭 SD 0.105 — 2026-08-31 기준선 15폴드)")
    print("  거의 안 움직인다                          -> 12.87.3 확정. 계측 축(4.6)만 남는다")
    print("  대조가 겨냥한 파일만큼 움직인다            -> **판정 보류**. 재학습 잡음이다 (12.114)")
    print("  ⚠ w=0 열이 그 폭이다. 그 열이 비어 있으면 --baseline-only 를 먼저 돌릴 것")

    out = Path(a.out) / "ronl_summary.json"
    out.write_text(json.dumps(
        {"scope": a.scope, "w": ws, "seeds": seeds, "stems": list(stems),
         "held_out_metrics": {str(w): table[w] for w in ws},
         "controls": {str(w): ctl[w] for w in ws}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w", default="0.3",
                    help="w_real_on 값들 (쉼표). 0 은 자동으로 기준선으로 들어간다")
    ap.add_argument("--seeds", default="0", help="2단계 시드 (쉼표)")
    ap.add_argument("--scope", default="smps", choices=("smps", "present", "all"))
    ap.add_argument("--stems", default=",".join(HUMAN_ON_DEFAULT_STEMS),
                    help="LOFO 폴드로 돌 파일")
    ap.add_argument("--init", default=DEFAULT_INIT, help="1단계 체크포인트")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="합성 replay 캐시")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--baseline-only", action="store_true",
                    help="w=0 만 돌린다. **기기별 지표의 시드 폭을 먼저 만드는 용도**")
    ap.add_argument("--summarize-only", action="store_true", help="이미 있는 폴드만 모아 요약")
    ap.add_argument("--force", action="store_true", help="있는 폴드도 다시 돌린다")
    a = ap.parse_args()

    stems = [x.strip() for x in a.stems.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    ws = [0.0] if a.baseline_only else \
        sorted({0.0} | {float(x) for x in a.w.split(",") if x.strip()})

    bad = set(stems) & set(HUMAN_ON_CONTROL_STEMS)
    if bad:
        raise SystemExit(f"대조 파일을 폴드로 넣을 수 없습니다: {sorted(bad)} (규칙 20)")

    n = len(ws) * len(seeds) * len(stems)
    print("=" * 92)
    print("사람 스위칭 로그 지도 — LOFO (SMPS_PLAN 4.5절)")
    print("=" * 92)
    print(f"  w      {ws}")
    print(f"  시드   {seeds}")
    print(f"  폴드   {stems}")
    print(f"  범위   scope={a.scope}   1단계 {a.init}   {a.steps} step")
    print(f"  ** 폴드 {n}회. 2단계만이라 폴드당 약 3분 -> 대략 {n * 3}분 **")
    if not a.summarize_only:
        for w in ws:
            for seed in seeds:
                for held in stems:
                    run_fold(a, w, seed, held)
    summarize(a, ws, seeds, stems)


if __name__ == "__main__":
    main()
