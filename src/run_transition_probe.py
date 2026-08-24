"""
전이 귀속 진단 — 기기가 하나 바뀔 때 모델은 **어느 기기를** 움직이는가 (12.35.2절)
=====================================================================================
`run_gate_check` 의 F1 은 창 단위 정오만 센다. 그런데 12.33~12.36 이 좁혀 온 실패는
**맞바꿈**이다 — 충전기를 껐는데 프로젝터를 껐다고 답한다. 켜져 있는 창 수는 거의
맞으므로 F1 은 그것을 흐릿하게만 보여 준다. 여기서는 전이 하나하나를 본다.

    python -m src.run_transition_probe --ckpt results/adapt_ph1.pt \
        --ckpt-smps results/cnn_ov1.pt

`--ckpt-smps` 를 주면 SMPS 3종만 그쪽에서 가져온다 (`run_live` 운영 조합과 동일).

[무엇을 세는가]
정답의 `on` 구간 경계마다 앞뒤 창의 기기별 예측 전력 중앙값 차 `Δ_j` 를 잰다.

    정확   |Δ| 가 가장 큰 기기가 실제로 바뀐 기기이고, 부호도 맞다
    오귀속 가장 큰 |Δ| 가 `--min-delta` 이상인데 엉뚱한 기기다
    미검출 가장 큰 |Δ| 가 `--min-delta` 미만이다 — 아무것도 안 움직였다

12.35.2 가 손으로 낸 `test_7` 8/13 이 (미검출 구분 없는) 이 정의다.
12.40.2 ② 가 이 수를 판정 기준으로 쓴다.

> **미검출을 갈라내는 이유.** 실측 실패 19개 중 4개는 모든 기기의 |Δ| 가 1W 도
> 안 되는 창이다. 거기서 argmax 는 잡음에 대한 동전 던지기다. 그것을 '엉뚱한
> 기기로 귀속했다' 로 세면 12.33~12.36 이 좁혀 온 **맞바꿈**과 뒤섞인다.
> 둘은 원인도 처방도 다르다 — 하나는 감도, 하나는 판별이다.

[⚠ 기본이 SMPS 3종인 이유]
핫플·오븐의 정답은 **통전 단위**라 전이가 파일당 100개를 넘고, 그 전이는 사람이
스위치를 누른 것이 아니라 서모스탯이 끊은 것이다 (인수인계 2.5절). 섞으면 세고자
하는 것이 묻힌다. `--apps` 로 바꿀 수 있다.

[⚠ 창이 60초인데 앞뒤 5초를 보는 이유]
타깃 시점은 창 끝에서 6초 안쪽이고(`TARGET_LOOKAHEAD`), 예측은 그 시점에 대한
것이다. 앞뒤 5초 중앙값이면 전이를 사이에 두고 서로 다른 구성을 본다.
`--guard` 로 전이 직전·직후 1초를 도려낸다.

[⚠ 라벨 시각에 스냅을 건다 — 이것이 없으면 전이를 통째로 놓친다]
`test_7` 의 라벨은 I3 계단으로 정밀화했는데도 관측 P 계단과 최대 5.5초 어긋난다.

    라벨 145.2 -> 관측 계단 143.7    라벨 410.7 -> 416.2    라벨 600.7 -> 606.2

라벨을 그대로 중심으로 삼으면 410.7 의 앞뒤 5초가 **둘 다 전이 앞쪽**이라
관측 ΔP 가 -4.1W 로 나온다 (실제로는 -45W 다). 전이를 놓친 것을 "모델이 못 맞혔다"
로 세게 된다. 그래서 `[t0-snap, t0+snap]` 안에서 **관측 계단이 가장 큰 지점**으로
옮긴 뒤 잰다. 옮긴 거리를 표에 함께 찍는다 — 크면 라벨을 의심할 근거다.
`--snap 0` 으로 끄면 옛 동작이다.

[⚠ SMPS 전이는 P 가 아니라 |I3| 로 스냅한다 — 복합 파일에서 결정적이다]
`test_5`/`test_6` 은 오븐·핫플이 같이 돈다. P 로 스냅하면 ±6초 안의 1,000W 짜리
히터 계단으로 끌려가 45W 짜리 SMPS 전이를 통째로 놓친다 (관측 ΔW 가 ±1,100W 로
찍힌다). **오븐 히터는 3차를 거의 안 흘린다** — 1156W 에서 |I3| 0.0124A,
I3/I1 = 0.0023 이다 (12.37.2·인수인계 2.3). 그래서 |I3| 로 스냅하면 저항 계단이
사라지고 SMPS 계단만 남는다. `--snap-on` 기본값 `auto` 가 SMPS 3종은 `i3`,
나머지는 `p` 를 쓴다.
"""
from pathlib import Path
from typing import Dict, List, Tuple
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
from src.run_gate_check import EVEN_CHANNELS, forward_file, load_model
from src.run_live import KOR, SMPS_GROUP

CYCLES_PER_S = 60.0


def transitions(intervals: dict, apps: List[str],
                t_lo: float, t_hi: float) -> List[Tuple[float, str, int]]:
    """정답 `on` 구간 -> [(시각, 기기, +1 켜짐 / -1 꺼짐)].

    **`t_lo`/`t_hi` 는 파일 길이가 아니라 예측이 존재하는 구간이다.** 첫 타깃은
    창 하나가 지난 뒤에야 나오므로(`test_7` 은 54.0초), 파일 길이로 자르면
    앞뒤 창이 반쪽인 전이를 '틀렸다' 로 세게 된다.
    """
    out = []
    for app in apps:
        for s, e in intervals.get(app, {}).get("on", []):
            if t_lo <= float(s) <= t_hi:
                out.append((float(s), app, +1))
            if t_lo <= float(e) <= t_hi:
                out.append((float(e), app, -1))
    return sorted(out)


def _i3_ma(obs_harm: np.ndarray) -> np.ndarray:
    """관측 3차 고조파 크기 (mA). (n,15,2) Re/Im -> (n,)."""
    h3 = np.asarray(obs_harm)[:, 2]
    return np.hypot(h3[:, 0], h3[:, 1]) * 1000.0


def _snap(t: np.ndarray, sig: np.ndarray, t0: float, half: float, guard: float,
          snap: float) -> float:
    """관측 신호의 계단이 가장 큰 지점으로 전이 시각을 옮긴다. 후보는 예측 격자."""
    if snap <= 0:
        return t0
    cand = t[(t >= t0 - snap) & (t <= t0 + snap)]
    best, best_d = t0, -1.0
    for c in cand:
        pre = (t >= c - half) & (t <= c - guard)
        post = (t >= c + guard) & (t <= c + half)
        if pre.sum() < 2 or post.sum() < 2:
            continue
        dmag = abs(float(np.median(sig[post]) - np.median(sig[pre])))
        if dmag > best_d:
            best, best_d = float(c), dmag
    return best


def probe_file(d: dict, stem: str, apps_model: List[str], ev: dict,
               apps: List[str], half: float, guard: float,
               edge_guard: float, snap: float, snap_on: str,
               min_delta: float) -> dict:
    """전이마다 어느 기기가 얼마나 움직였는지."""
    P = d["gate"] * d["p_raw"]
    t = d["targets"] / CYCLES_PER_S
    i3 = _i3_ma(d["obs_harm"])
    info = ev[stem]
    present = [a for a in apps if a in info.get("appliances_present", [])]
    js = {a: apps_model.index(a) for a in present}
    tr = transitions(info["intervals"], present,
                     float(t[0]) + half + edge_guard,
                     min(float(t[-1]) - half - edge_guard,
                         float(info["duration_s"]) - edge_guard))

    rows, skipped = [], []
    for t_lab, app, sign in tr:
        use_i3 = snap_on == "i3" or (snap_on == "auto" and app in SMPS_GROUP)
        t0 = _snap(t, i3 if use_i3 else d["p_observed"], t_lab, half, guard, snap)
        pre = (t >= t0 - half) & (t <= t0 - guard)
        post = (t >= t0 + guard) & (t <= t0 + half)
        if pre.sum() < 2 or post.sum() < 2:
            # 파일 시작·끝이라 앞뒤 창이 없다. 못 잰 것이지 틀린 것이 아니다.
            skipped.append({"t_s": t_lab, "app": app, "sign": sign})
            continue
        delta = {a: float(np.median(P[post, j]) - np.median(P[pre, j]))
                 for a, j in js.items()}
        picked = max(delta, key=lambda a: abs(delta[a]))
        if abs(delta[picked]) < min_delta:
            verdict = "미검출"
        elif picked == app and np.sign(delta[picked]) == sign:
            verdict = "정확"
        else:
            verdict = "오귀속"
        rows.append({
            "t_s": t_lab, "t_snapped_s": t0, "snap_s": t0 - t_lab,
            "true_app": app, "sign": sign,
            "delta_w": delta,
            "picked": picked,
            "picked_delta_w": delta[picked],
            "correct": verdict == "정확",
            "verdict": verdict,
            "obs_delta_w": float(np.median(d["p_observed"][post])
                                 - np.median(d["p_observed"][pre])),
            "obs_delta_i3_ma": float(np.median(i3[post]) - np.median(i3[pre])),
            "snapped_on": "i3" if use_i3 else "p",
        })
    n_ok = sum(r["correct"] for r in rows)
    n_miss = sum(r["verdict"] == "미검출" for r in rows)
    n_wrong = sum(r["verdict"] == "오귀속" for r in rows)
    # 맞바꿈: 실제 기기와 고른 기기가 둘 다 SMPS 3종이고 서로 다르다
    n_swap = sum(1 for r in rows if r["verdict"] == "오귀속"
                 and r["true_app"] in SMPS_GROUP and r["picked"] in SMPS_GROUP)
    return {"n": len(rows), "n_correct": n_ok, "n_wrong": n_wrong,
            "n_undetected": n_miss, "n_swap": n_swap,
            "accuracy": n_ok / len(rows) if rows else float("nan"),
            "n_skipped": len(skipped), "skipped": skipped,
            "apps": present, "rows": rows}


def print_file(stem: str, res: dict) -> None:
    apps = res["apps"]
    skip = f", 못 잼 {res['n_skipped']}" if res["n_skipped"] else ""
    print()
    print(f"  [{stem}]  {res['n_correct']}/{res['n']} 정확"
          f"  (오귀속 {res['n_wrong']} · 미검출 {res['n_undetected']}{skip})"
          f"  {', '.join(KOR.get(a, a) for a in apps)}")
    head = (f"    {'t(s)':>8s}{'스냅':>6s}  {'실제':<14s}{'관측ΔW':>9s}{'ΔI3mA':>8s}"
            + "".join(f"{KOR.get(a, a):>10s}" for a in apps) + "   판정")
    print(head)
    print("    " + "-" * (len(head) - 4))
    for r in res["rows"]:
        lab = f"{KOR.get(r['true_app'], r['true_app'])} {'on' if r['sign'] > 0 else 'off'}"
        mark = ("정확" if r["correct"] else "미검출" if r["verdict"] == "미검출"
                else f"-> {KOR.get(r['picked'], r['picked'])}")
        print(f"    {r['t_s']:>8.1f}{r['snap_s']:>+6.1f}  {lab:<14s}"
              f"{r['obs_delta_w']:>9.1f}{r['obs_delta_i3_ma']:>8.0f}"
              + "".join(f"{r['delta_w'][a]:>10.1f}" for a in apps)
              + f"   {mark}")


def main() -> int:
    ap = argparse.ArgumentParser(description="전이 귀속 진단 (12.35.2절)")
    ap.add_argument("--ckpt", default="results/adapt_ph1.pt")
    ap.add_argument("--ckpt-smps", default=None, metavar="PT",
                    help="SMPS 3종만 이 체크포인트로 (운영 조합)")
    ap.add_argument("--zero-even", action="store_true",
                    help="SMPS 체크포인트 입력의 짝수차 채널을 0 으로 (12.74절). "
                         "짝수차는 계측 인공물이다 (12.72)")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="기본: 사람 기록 파일 전부")
    ap.add_argument("--apps", nargs="*", default=list(SMPS_GROUP))
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--half", type=float, default=5.0, help="전이 앞뒤로 보는 폭(초)")
    ap.add_argument("--guard", type=float, default=1.0, help="전이 직근 도려내기(초)")
    ap.add_argument("--edge-guard", type=float, default=1.0,
                    help="예측이 있는 구간의 양 끝에서 이만큼 더 물러난다(초)")
    ap.add_argument("--snap", type=float, default=6.0,
                    help="라벨 시각을 관측 계단으로 옮기는 최대 거리(초). 0=끄기")
    ap.add_argument("--snap-on", choices=("auto", "p", "i3"), default="auto",
                    help="스냅에 쓸 관측 신호. auto=SMPS 3종은 i3, 나머지는 p")
    ap.add_argument("--min-delta", type=float, default=5.0, metavar="W",
                    help="가장 큰 |Δ| 가 이 아래면 '미검출' 로 센다. 잡음 argmax 제외")
    ap.add_argument("--out", default="results/transition_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = a.stems or [s for s in sorted(ev)
                        if not is_sealed(s)
                        and ev[s].get("_label_provenance") == "human_switching_log"]

    model, apps_model, _ = load_model(a.ckpt, dev)
    model_s = None
    if a.ckpt_smps:
        model_s, apps_s, _ = load_model(a.ckpt_smps, dev)
        if apps_s != apps_model:
            print("  ⚠ 두 체크포인트의 기기 목록이 다릅니다. 중단합니다.")
            return 1

    tag = Path(a.ckpt).stem + (f"+{Path(a.ckpt_smps).stem}" if a.ckpt_smps else "")
    print("=" * 84)
    print(f"[전이 귀속] {tag} | 앞뒤 {a.half:.0f}초 중앙값, 직근 {a.guard:.0f}초 제외")
    print("=" * 84)

    out: Dict[str, dict] = {}
    tot = ok = wrong = miss = swap = 0
    for stem in stems:
        d = forward_file(model, stem, dev, stride=a.stride)
        if model_s is not None:
            ds = forward_file(model_s, stem, dev, stride=a.stride,
                              zero_ch=EVEN_CHANNELS if a.zero_even else None)
            six = [apps_model.index(x) for x in SMPS_GROUP if x in apps_model]
            for k in ("gate", "p_raw", "standby"):
                d[k][:, six] = ds[k][:, six]
        res = probe_file(d, stem, apps_model, ev, a.apps, a.half, a.guard,
                         a.edge_guard, a.snap, a.snap_on, a.min_delta)
        out[stem] = res
        print_file(stem, res)
        tot += res["n"]; ok += res["n_correct"]; wrong += res["n_wrong"]
        miss += res["n_undetected"]; swap += res["n_swap"]

    print()
    print(f"  합계 {ok}/{tot} = {ok / tot if tot else float('nan'):.3f} 정확"
          f"  |  오귀속 {wrong} (그중 SMPS 맞바꿈 {swap})  |  미검출 {miss}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({tag: {"total": {
                               "n": tot, "n_correct": ok, "n_wrong": wrong,
                               "n_undetected": miss, "n_swap": swap}, **out}},
                                      ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
