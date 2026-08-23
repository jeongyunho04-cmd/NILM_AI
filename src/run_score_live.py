"""
실시간 로그 채점 — `run_live` 가 남긴 예측 스트림을 4.3절 지표로 읽는다
=======================================================================
`run_live.py` 는 기록만 하고 채점하지 않는다. 이 파일이 그 짝이다.

    python -m src.run_score_live                              # results/live_log.jsonl
    python -m src.run_score_live --log results/run2.jsonl --json results/live_score.json
    python -m src.run_score_live --all-sessions               # 이어붙은 옛 세션까지

[왜 추론 루프 밖에서 채점하는가]
사람은 기기를 바꾸고 **몇 초 뒤에** 확정한다. 확정은 언제나 사후에 온다.
로그를 다 쌓은 뒤 한 번에 읽어야 전이 구간을 제대로 도려낼 수 있다.

[⚠ 6초 어긋남 — 이 파일이 존재하는 첫 번째 이유]
`run_live` 가 남기는 `t_s` 는 **창의 끝**(가장 최근에 읽은 행)이다. 그런데 모델의
타깃 시점은 `inputs.TARGET_LOOKAHEAD = 360` 사이클, 즉 **창 끝에서 6초 전**이다
(12.9.12절. 설계 문서 5.1절의 "1초" 는 그 뒤 갱신되지 않은 서술이다).

    로그의 t_s = 1000.0   ->   이 예측이 말하는 시점은 994.0

맞추지 않고 채점하면 모든 전이가 6초씩 밀린다. 한 번 켜면 10초 안에 끝나는
포트·드라이기(0.4절 중앙값 9.2s / 9.8s)에서는 **정답과 오답이 통째로 뒤바뀐다.**

[정답을 어떻게 세우는가]
`type:"actual"` 레코드(사람이 space 로 확정한 것)만 정답으로 쓴다. `pred` 안에
박혀 있는 `actual` 필드는 **확정 전의 토글 상태**라, 키를 누르다 만 중간 상태가
섞인다. 확정 기록이 하나도 없을 때만 그쪽으로 떨어진다 (경고를 찍는다).

확정 시각 사이를 계단으로 채우고, 각 확정 시각 ±`--guard` 초는 도려낸다.
사람의 반응 지연이 거기 들어 있다.

**확정 dict 에 없는 기기는 OFF 가 아니라 '모름' 이다.** `run_live` 의 `actual` 은
빈 dict 로 시작해 누른 기기만 채워지므로, 없는 키를 OFF 로 읽으면 켜져 있던
기기가 통째로 오답이 된다. 측정 전에 **`0`(전부 꺼짐)을 한 번 눌러** 모든 키를
채워 두는 편이 낫다.

[무엇을 재는가]
라벨 없이 나오는 것과 확정이 있어야 나오는 것을 나눠 찍는다. 전자는 보드를
켜 두기만 해도 나오므로 **첫 세션부터 볼 수 있다.**
"""
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from src.evaluation.metrics import (
    HIGH_LOAD_THRESHOLD_W,
    RESISTIVE_MIN_TRUE_W,
    RESISTIVE_TRIO,
)
from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.inputs import TARGET_LOOKAHEAD
from src.run_baseline import LOW_LOAD, S_I

SAMPLING_HZ = 60.0
LOOKAHEAD_S = TARGET_LOOKAHEAD / SAMPLING_HZ      # 6.0초. 위 주석 참조
HEDGE_LO, HEDGE_HI = 0.05, 0.95                   # run_gate_check 와 같은 폭
GATE_ON = 0.5

# 0.1절의 세 무리. `file_registry` 는 선풍기를 MOTOR 로 두지만 고조파 무리로는
# 저항성 쪽에 붙는다(0.1절 표). 여기서는 문서 0.1절의 무리를 따른다.
GROUPS = {
    "저항성 5종": ("electiric_kettle", "oven", "hair_dryer", "hotplate", "fan"),
    "SMPS 3종": ("beam_projector", "laptop_charger", "minipc"),
    "인버터": ("air_conditioner",),
}
KOR = {"oven": "오븐", "hotplate": "핫플", "electiric_kettle": "포트",
       "hair_dryer": "드라이기", "minipc": "미니PC", "beam_projector": "프로젝터",
       "laptop_charger": "충전기", "fan": "선풍기", "air_conditioner": "에어컨"}


# ── 로그 읽기 ──────────────────────────────────────────────────────────────
def load_log(path: Path) -> List[dict]:
    """JSONL 을 읽는다. 깨진 줄은 건너뛴다 (Ctrl+C 로 끊긴 마지막 줄)."""
    if not path.exists():
        raise FileNotFoundError(f"로그가 없습니다: {path.resolve()}")
    out, bad = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  ⚠ 읽지 못한 줄 {bad}개 (끊긴 기록으로 보입니다)")
    return out


def split_sessions(recs: List[dict], gap_s: float = 30.0) -> List[List[dict]]:
    """`t_s` 가 크게 되감기면 새 세션으로 본다.

    `run_live` 는 로그를 **이어쓴다**(`open(..., "a")`). 여러 번 측정하면 한 파일에
    쌓이는데 세션마다 `t_s` 가 0 부터 다시 시작하므로, 통째로 채점하면 옛 세션의
    확정이 새 세션의 예측에 붙는다.

    작은 역전(재생 시 323.5 -> 322.5, 12.29.4절)은 세션 경계가 아니다.
    """
    sessions: List[List[dict]] = []
    cur: List[dict] = []
    hi: Optional[float] = None
    for r in recs:
        t = r.get("t_s")
        if t is None:
            continue
        if hi is not None and t < hi - gap_s and cur:
            sessions.append(cur)
            cur, hi = [], None
        cur.append(r)
        hi = t if hi is None else max(hi, t)
    if cur:
        sessions.append(cur)
    return sessions


# ── 정답 타임라인 ──────────────────────────────────────────────────────────
class Truth:
    """확정 시각의 계단 함수. 확정 근방 `guard` 초는 '모름' 으로 도려낸다."""

    def __init__(self, confirms: List[Tuple[float, Dict[str, bool]]],
                 guard: float, hold_last: bool = True):
        self.c = sorted(confirms, key=lambda x: x[0])
        self.t = [x[0] for x in self.c]
        self.guard = guard
        self.hold_last = hold_last

    def __len__(self) -> int:
        return len(self.c)

    def at(self, t: float) -> Optional[Dict[str, bool]]:
        """`t` 시점의 확정 상태. 모르면 None."""
        if not self.c or t < self.t[0]:
            return None                      # 첫 확정 이전은 아무것도 모른다
        i = bisect_right(self.t, t) - 1
        if t - self.t[i] < self.guard:
            return None                      # 확정 직후 — 사람 반응 지연 구간
        if i + 1 < len(self.t) and self.t[i + 1] - t < self.guard:
            return None                      # 다음 확정 직전 — 이미 바뀌었을 수 있다
        if i == len(self.t) - 1 and not self.hold_last:
            return None
        return self.c[i][1]


def build_truth(session: List[dict], guard: float, hold_last: bool,
                use_embedded: bool) -> Tuple[Truth, str]:
    """확정 레코드로 정답을 세운다. 없으면 pred 에 박힌 토글 상태로 떨어진다."""
    conf = [(r["t_s"], {k: bool(v) for k, v in (r.get("state") or {}).items()})
            for r in session if r.get("type") == "actual"]
    if conf:
        return Truth(conf, guard, hold_last), "확정(space)"
    if not use_embedded:
        return Truth([], guard, hold_last), "없음"
    # 떨어짐: pred 에 박힌 토글 상태가 바뀌는 지점을 확정으로 친다
    emb: List[Tuple[float, Dict[str, bool]]] = []
    prev: Optional[Dict[str, bool]] = None
    for r in session:
        if r.get("type") != "pred" or "actual" not in r:
            continue
        st = {k: bool(v) for k, v in r["actual"].items()}
        if st != prev:
            emb.append((r["t_s"], st))
            prev = st
    return Truth(emb, guard, hold_last), "토글(미확정)" if emb else "없음"


# ── 표 유틸 ────────────────────────────────────────────────────────────────
def kor(a: str) -> str:
    return KOR.get(a, a)


def fmt(v: float, w: int = 7, p: int = 2, dash: str = "—") -> str:
    return f"{dash:>{w}s}" if v != v else f"{v:>{w}.{p}f}"


# ── A. 라벨 없이 나오는 진단 ───────────────────────────────────────────────
def diagnose_unlabeled(apps: List[str], gate: np.ndarray, power: np.ndarray,
                       p_obs: np.ndarray, p_pred: np.ndarray, t: np.ndarray,
                       low_watt: float) -> dict:
    """확정이 하나도 없어도 나오는 지표. 첫 세션부터 볼 수 있다."""
    resid = p_obs - p_pred
    dur_min = max((t.max() - t.min()) / 60.0, 1e-9)

    rows = []
    for j, a in enumerate(apps):
        g, pw = gate[:, j], power[:, j]
        on = g > GATE_ON
        # 게이트 채터: 0.5 를 넘나든 횟수. **실시간에서만 보이는 실패다.**
        # 합성 홀드아웃은 창을 무작위로 뽑아 섞으므로 이 축이 아예 없다.
        cross = int(np.count_nonzero(np.diff(on.astype(np.int8)) != 0))
        rows.append({
            "appliance": a,
            "hedge_rate": float(np.mean((g > HEDGE_LO) & (g < HEDGE_HI))),
            "on_rate": float(on.mean()),
            "mean_w": float(pw.mean()),
            "max_w": float(pw.max()),
            "chatter_per_min": cross / dur_min,
        })

    # 저전력 창의 '추첨' (12.23.2). 없는 기기가 이기면 오귀속이 수십 W 뛴다.
    low = p_obs < low_watt
    share: Dict[str, float] = {}
    if low.any():
        win = np.asarray(apps)[np.argmax(power[low], axis=1)]
        n = int(low.sum())
        share = {a: c / n for a, c in Counter(win.tolist()).most_common()}

    return {
        "residual": {
            "median_w": float(np.median(resid)),
            "mae_w": float(np.mean(np.abs(resid))),
            "p5_w": float(np.percentile(resid, 5)),
            "p95_w": float(np.percentile(resid, 95)),
            "p_observed_median_w": float(np.median(p_obs)),
        },
        "per_appliance": rows,
        "low_power_share": share,
        "low_power_windows": int(low.sum()),
        "low_watt": low_watt,
    }


def print_unlabeled(d: dict) -> None:
    r = d["residual"]
    base = r["p_observed_median_w"]
    rel = r["median_w"] / base * 100 if base else float("nan")
    print("\n── A. 라벨 없이 나오는 진단 ─────────────────────────────────────────────")
    print("\n  총전력 잔차  P_관측 − Σ P̂     (라벨이 필요 없는 유일한 정합성 검사)")
    print(f"    중앙값 {r['median_w']:+7.1f}W    MAE {r['mae_w']:6.1f}W"
          f"    p5 {r['p5_w']:+7.1f}W   p95 {r['p95_w']:+7.1f}W"
          f"    (관측 중앙 {base:.1f}W 의 {rel:+.1f}%)")

    print("\n  게이트 — 헤지율은 '결정을 못 한' 비율 (0.05<σ<0.95), 채터는 0.5 교차/분")
    h = f"    {'가전':<12s}{'헤지율':>9s}{'켜짐율':>9s}{'평균W':>9s}{'최대W':>9s}{'채터/분':>10s}"
    print(h)
    print("    " + "-" * (len(h) - 4))
    for x in sorted(d["per_appliance"], key=lambda z: -z["mean_w"]):
        print(f"    {kor(x['appliance']):<12s}{x['hedge_rate'] * 100:>8.1f}%"
              f"{x['on_rate'] * 100:>8.1f}%{x['mean_w']:>9.1f}{x['max_w']:>9.1f}"
              f"{x['chatter_per_min']:>10.2f}")

    if d["low_power_share"]:
        print(f"\n  저전력 창(관측 < {d['low_watt']:.0f}W, {d['low_power_windows']}개)에서"
              f" 최대 지분을 가져간 기기   ← 12.23.2 의 '추첨'")
        top = list(d["low_power_share"].items())[:5]
        print("    " + "   ".join(f"{kor(a)} {v * 100:.1f}%" for a, v in top))


# ── B. 확정 라벨로 채점 ────────────────────────────────────────────────────
def score_labeled(apps: List[str], gate: np.ndarray, power: np.ndarray,
                  p_obs: np.ndarray, truth: np.ndarray, known: np.ndarray,
                  high_watt: float) -> dict:
    """4.3절 지표. `known` 이 False 인 (창, 기기) 는 채점하지 않는다."""
    pred_on = gate > GATE_ON
    high = p_obs >= high_watt

    per, absent, unknown = [], [], []
    for j, a in enumerate(apps):
        m = known[:, j]
        if not m.any():
            unknown.append(a)
            continue
        t_on, p_on = truth[m, j], pred_on[m, j]
        tp = float((p_on & t_on).sum())
        fp = float((p_on & ~t_on).sum())
        fn = float((~p_on & t_on).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else 0.0

        # FA_i — 꺼져 있을 때 붙인 전력. 고부하 동시 / 없음으로 나눈다 (4.3절)
        off = m & ~truth[:, j]
        on_m = m & truth[:, j]
        fa_hi = float(power[off & high, j].mean()) if (off & high).any() else float("nan")
        fa_lo = float(power[off & ~high, j].mean()) if (off & ~high).any() else float("nan")
        fa = float(power[off, j].mean()) if off.any() else float("nan")
        s = S_I.get(a, float("nan"))

        row = {
            "appliance": a, "f1": f1, "precision": prec, "recall": rec,
            "n_scored": int(m.sum()), "n_true_on": int(t_on.sum()),
            "fa_w": fa, "fa_rel": fa / s,
            "fa_high_w": fa_hi, "fa_low_w": fa_lo, "fa_high_rel": fa_hi / s,
            "transfer_w": fa_hi - fa_lo,
            # 켜져 있을 때 평균 예측. 정답 전력이 없으니 RE_i 는 못 낸다
            # (`real_events` 상단 참조). 대신 정격 대비로 찍어 0.6절의
            # 체계적 과소 예측이 눈에 띄게 한다.
            "on_mean_w": float(power[on_m, j].mean()) if on_m.any() else float("nan"),
            "s_i": s,
        }
        per.append(row)
        if t_on.sum() == 0:            # 세션 내내 한 번도 안 켠 기기 = 정답 0 확정
            absent.append(row)

    return {"per_appliance": per, "absent": absent, "unknown": unknown,
            "high_watt": high_watt,
            "n_scorable": int(known.any(axis=1).sum())}


def resistive_confusion_live(apps: List[str], power: np.ndarray, p_obs: np.ndarray,
                             truth: np.ndarray, known: np.ndarray,
                             min_true_w: float) -> Optional[dict]:
    """저항 3종 혼동행렬 — 불리언 정답용 변형.

    `metrics.resistive_confusion` 은 정답 **전력**으로 발열 구간을 골라내는데
    (`min_true_w`), 실측 로그에는 정답 전력이 없다. 대신 **관측 총전력**이 그
    문턱을 넘는 창만 쓴다. 오븐 팬/조명(16W) 구간을 히터로 세는 것을 막는 것이
    목적이므로 같은 역할을 한다.
    """
    idx = [apps.index(a) for a in RESISTIVE_TRIO if a in apps]
    if len(idx) < 2:
        return None
    kn = known[:, idx].all(axis=1)
    t = truth[:, idx]
    solo = kn & (t.sum(axis=1) == 1) & (p_obs >= min_true_w)
    if not solo.any():
        return None
    a_true = np.argmax(t[solo], axis=1)
    a_pred = np.argmax(power[solo][:, idx], axis=1)
    m = np.zeros((len(idx), len(idx)), int)
    for x, y in zip(a_true, a_pred):
        m[x, y] += 1
    return {"labels": [apps[i] for i in idx], "matrix": m.tolist(),
            "n_windows": int(solo.sum()),
            "accuracy": float(np.mean(a_true == a_pred)),
            "min_true_w": min_true_w}


def print_labeled(d: dict, conf: Optional[dict], source: str) -> None:
    print("\n── B. 확정 라벨로 채점 ──────────────────────────────────────────────────")
    print(f"  정답 출처: {source}   채점 가능 창 {d['n_scorable']:,}개"
          f"   고부하 조건: 관측 ≥ {d['high_watt']:.0f}W")

    print("\n  기기별 on/off  (4.3절)")
    h = (f"    {'가전':<12s}{'F1':>8s}{'정밀도':>9s}{'재현율':>9s}"
         f"{'채점창':>9s}{'참ON':>8s}{'ON평균W':>10s}{'정격W':>8s}")
    print(h)
    print("    " + "-" * (len(h) - 4))
    for x in sorted(d["per_appliance"], key=lambda z: -z["n_true_on"]):
        print(f"    {kor(x['appliance']):<12s}{fmt(x['f1'], 8, 3)}{fmt(x['precision'], 9, 3)}"
              f"{fmt(x['recall'], 9, 3)}{x['n_scored']:>9,d}{x['n_true_on']:>8,d}"
              f"{fmt(x['on_mean_w'], 10, 1)}{x['s_i']:>8.0f}")
    if d["unknown"]:
        names = ", ".join(kor(a) for a in d["unknown"])
        print(f"    (확정에 한 번도 안 나온 기기 = 채점 제외: {names})")

    print("\n  오귀속 FA — 꺼져 있을 때 붙인 전력. 전체 MAE 에는 안 보인다 (4.3절)")
    print("  저부하 4종 목표: FA_rel(고부하 동시) < 0.15")
    h = (f"    {'가전':<12s}{'FA(고부하)':>12s}{'FA(없음)':>11s}"
         f"{'전가량':>10s}{'FA_rel':>9s}{'판정':>7s}")
    print(h)
    print("    " + "-" * (len(h) - 4))
    order = sorted(d["per_appliance"],
                   key=lambda z: -(z["fa_w"] if z["fa_w"] == z["fa_w"] else -1))
    for x in order:
        v = x["fa_high_rel"]
        if x["appliance"] not in LOW_LOAD or v != v:
            mark = "—"
        else:
            mark = "OK" if v < 0.15 else "초과"
        print(f"    {kor(x['appliance']):<12s}{fmt(x['fa_high_w'], 12, 2)}"
              f"{fmt(x['fa_low_w'], 11, 2)}{fmt(x['transfer_w'], 10, 2)}"
              f"{fmt(v, 9, 3)}{mark:>7s}")

    if d["absent"]:
        tot = sum(x["fa_w"] for x in d["absent"] if x["fa_w"] == x["fa_w"])
        items = "   ".join(f"{kor(x['appliance'])} {x['fa_w']:.1f}W" for x in d["absent"])
        print("\n  세션 내내 한 번도 안 켠 기기 — 정답이 0 으로 확정된다"
              " (`score_absent` 와 같은 정의)")
        print(f"    {items}")
        print(f"    합계 {tot:.1f}W  ← 이 전력은 전부 오답이다")

    if conf:
        print(f"\n  저항 3종 혼동행렬  (3종 중 한 대만 켜진 창 {conf['n_windows']}개,"
              f" 관측 ≥ {conf['min_true_w']:.0f}W)")
        names = [kor(a) for a in conf["labels"]]
        print("    " + " " * 12 + "".join(f"{'→' + n:>10s}" for n in names))
        for i, n in enumerate(names):
            print(f"    {n:<12s}" + "".join(f"{v:>10d}" for v in conf["matrix"][i]))
        print(f"    정확도 {conf['accuracy'] * 100:.1f}%"
              f"   (합성 홀드아웃 1.000 / 문서 실측 1.2~5.8%)")
    else:
        print("\n  저항 3종 혼동행렬: 3종 중 정확히 한 대만 켜진 구간이 없어 못 잽니다.")

    print("\n  무리별 요약   ← 12.24: 저항과 저부하가 반대로 움직여 집계에서 상쇄된다")
    by = {x["appliance"]: x for x in d["per_appliance"]}
    h = f"    {'무리':<14s}{'평균F1':>9s}{'FA 합(W)':>11s}{'기기수':>8s}"
    print(h)
    print("    " + "-" * (len(h) - 4))
    for g, members in GROUPS.items():
        xs = [by[a] for a in members if a in by]
        if not xs:
            continue
        f1s = [x["f1"] for x in xs if x["f1"] == x["f1"]]
        fas = [x["fa_w"] for x in xs if x["fa_w"] == x["fa_w"]]
        mf1 = float(np.mean(f1s)) if f1s else float("nan")
        print(f"    {g:<14s}{fmt(mf1, 9, 3)}{sum(fas):>11.1f}{len(xs):>8d}")


# ── 본체 ───────────────────────────────────────────────────────────────────
def score_session(session: List[dict], a) -> Optional[dict]:
    preds = [r for r in session if r.get("type") == "pred"]
    if not preds:
        print("  이 세션에는 예측 레코드가 없습니다.")
        return None

    # **시간 순으로 정렬한다.** 수신기는 프레임을 도착 순서대로 쓰므로(펌웨어
    # 선택적 재전송, `run_live.CycleRing` 참조) 예측 레코드의 `t_s` 도 국소적으로
    # 거꾸로 갈 수 있다. 채터 지표는 이웃한 두 예측의 차분이라 정렬하지 않으면
    # 없는 진동이 보인다.
    n_rev = sum(1 for a, b in zip(preds, preds[1:]) if b["t_s"] < a["t_s"])
    if n_rev:
        print(f"  ⚠ 시간이 거꾸로 가는 예측 레코드 {n_rev}개 — 정렬해서 채점합니다")
        preds.sort(key=lambda r: r["t_s"])

    apps = list(preds[0]["gate"].keys())
    t = np.array([r["t_s"] for r in preds], float)
    gate = np.array([[r["gate"].get(x, 0.0) for x in apps] for r in preds], float)
    power = np.array([[r["power_w"].get(x, 0.0) for x in apps] for r in preds], float)
    p_obs = np.array([r.get("p_observed", np.nan) for r in preds], float)
    p_pred = np.array([r.get("pred_total", np.nan) for r in preds], float)

    # 모델이 말하는 시점은 창 끝이 아니라 6초 전이다 (모듈 상단 주석)
    t_target = t - LOOKAHEAD_S

    print("=" * 88)
    print(f"  추론 {len(preds):,}회   t = {t.min():.1f} ~ {t.max():.1f}s"
          f" ({(t.max() - t.min()) / 60:.1f}분)   기기 {len(apps)}종")
    print(f"  타깃 보정 −{LOOKAHEAD_S:.1f}s 적용 (TARGET_LOOKAHEAD={TARGET_LOOKAHEAD} 사이클)")
    # 입력이 얼마나 온전했는지를 성능과 나란히 봐야 한다. 되꽂음이 많았다면
    # 그 세션의 지표는 순서 보정 덕을 본 것이고, --no-reorder 와 비교할 거리가 된다.
    rs = next((r for r in reversed(session) if r.get("type") == "ring_stats"), None)
    if rs:
        print(f"  입력 순서: 되꽂음 {rs.get('n_backfill', 0):,}행"
              f"  버림 {rs.get('n_stale', 0):,}행"
              f"  = {rs.get('reorder_rate', 0) * 100:.2f}%"
              f"  최대 역전 {rs.get('max_backfill_s', 0):.2f}초")
    print("=" * 88)

    un = diagnose_unlabeled(apps, gate, power, p_obs, p_pred, t, a.low_watt)
    print_unlabeled(un)
    out = {"n_pred": len(preds), "t_start": float(t.min()), "t_end": float(t.max()),
           "appliances": apps, "lookahead_s": LOOKAHEAD_S, "unlabeled": un,
           "ring_stats": rs, "n_time_reversed_preds": n_rev}

    if a.truth_events:
        # 재생 검증 경로. 사람 확정 대신 `real_events.json` 의 타임라인을 쓴다.
        # uncertain 구간은 빠진다 (오븐 팬/조명처럼 정답이 모호한 자리.
        # `build_on_off_truth` 주석 참조).
        #
        # **⚠ 시간축이 다를 수 있다.** 정답은 전처리된(세션 분리 + seq 정렬 +
        # 60Hz 결측 보간) 사이클 인덱스 기준인데, 재생은 원본 CSV 의 `t_s` 를
        # 그대로 쓴다. 유실을 보간한 만큼 정답 쪽이 길어진다:
        #     test_4  701.5s vs 원본 701.5s   일치
        #     test.2  466.0s vs 원본 463.0s   3.0s 어긋남
        #     test3   413.0s vs 원본 407.5s   5.5s 어긋남
        # 어긋난 채 채점하면 전이가 통째로 밀린 숫자가 조용히 나온다. 그래서
        # 길이를 검사하고, 안 맞으면 멈춘다. 정식 경로는 `run_reeval` 이다.
        dur = float(load_events()[a.truth_events].get("duration_s", 0.0))
        drift = abs(dur - t.max())
        print(f"\n  정답 타임라인 {dur:.1f}s vs 재생 {t.max():.1f}s"
              f"  → 어긋남 {drift:.1f}s (허용 {a.max_drift:.1f}s)")
        if drift > a.max_drift:
            print("\n  ⚠ 시간축이 안 맞아 채점하지 않습니다. 이대로 채점하면 전이가")
            print("    밀린 숫자가 나옵니다. 전처리된 데이터로 재는 정식 경로를 쓰십시오:")
            print(f"      python -m src.run_reeval --ckpt {'<ckpt>'} --files {a.truth_events}")
            print("    (--max-drift 로 강제할 수는 있지만 권하지 않습니다)")
            return out
        n_cyc = int(max(t.max(), dur) * SAMPLING_HZ) + 1
        on, scorable = build_on_off_truth(a.truth_events, apps, n_cyc)
        ix = np.clip((t_target * SAMPLING_HZ).astype(int), 0, n_cyc - 1)
        T, K = on[ix], scorable[ix]
        K = K & (t_target >= 0)[:, None]        # 예열 구간은 정답 시각 이전이다
        sc = score_labeled(apps, gate, power, p_obs, T, K, a.high_watt)
        conf = resistive_confusion_live(apps, power, p_obs, T, K, a.min_true_w)
        print_labeled(sc, conf, f"real_events.json / {a.truth_events}")
        out["labeled"] = sc
        out["resistive_confusion"] = conf
        out["truth_source"] = f"events:{a.truth_events}"
        return out

    truth_fn, source = build_truth(session, a.guard, not a.no_hold_last, not a.no_embedded)
    if len(truth_fn) == 0:
        print("\n── B. 확정 라벨로 채점 ──────────────────────────────────────────────────")
        print("  확정 기록이 없습니다. 측정 중에 기기 번호를 누르고 space 를 누르면")
        print("  이 절이 채워집니다. 처음에 0(전부 꺼짐)을 한 번 눌러 두면 모든 기기가")
        print("  채점 대상이 됩니다 — 안 누른 기기는 'OFF' 가 아니라 '모름' 입니다.")
        return out

    if source == "토글(미확정)":
        print("\n  ⚠ space 확정이 없어 pred 에 박힌 토글 상태로 채점합니다.")
        print("    키를 누르다 만 중간 상태가 섞일 수 있습니다.")

    T = np.zeros_like(gate, bool)
    K = np.zeros_like(gate, bool)
    for i, tt in enumerate(t_target):
        st = truth_fn.at(float(tt))
        if st is None:
            continue
        for j, app in enumerate(apps):
            if app in st:
                K[i, j] = True
                T[i, j] = st[app]

    if not K.any():
        print("\n── B. 확정 라벨로 채점 ──────────────────────────────────────────────────")
        print(f"  확정 {len(truth_fn)}회가 있지만 가드밴드(±{a.guard:.1f}s) 밖의 예측이")
        print("  없습니다. --guard 를 줄이거나 확정 후 조금 더 켜 두십시오.")
        return out

    sc = score_labeled(apps, gate, power, p_obs, T, K, a.high_watt)
    conf = resistive_confusion_live(apps, power, p_obs, T, K, a.min_true_w)
    print_labeled(sc, conf, f"{source} {len(truth_fn)}회, 가드 ±{a.guard:.1f}s")
    out["labeled"] = sc
    out["resistive_confusion"] = conf
    out["n_confirm"] = len(truth_fn)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="run_live 로그 채점 (4.3절 지표)")
    ap.add_argument("--log", default="results/live_log.jsonl")
    ap.add_argument("--json", default=None, help="지표를 JSON 으로도 저장")
    ap.add_argument("--all-sessions", action="store_true",
                    help="이어붙은 옛 세션까지 전부 (기본: 마지막 세션만)")
    ap.add_argument("--truth-events", default=None, metavar="STEM",
                    help="사람 확정 대신 real_events.json 의 타임라인으로 채점한다. "
                         "옛 파일 재생 검증용 (test.2 / test3 / test_4). "
                         "test 는 봉인되어 있어 막힌다")
    ap.add_argument("--max-drift", type=float, default=1.0, metavar="SEC",
                    help="--truth-events 에서 허용할 시간축 어긋남 (기본 1초). "
                         "넘으면 채점하지 않는다")
    ap.add_argument("--guard", type=float, default=5.0,
                    help="확정 시각 ±N초 제외. 사람 반응 지연 (기본 5)")
    ap.add_argument("--no-hold-last", action="store_true",
                    help="마지막 확정 이후 구간을 채점하지 않는다")
    ap.add_argument("--no-embedded", action="store_true",
                    help="space 확정이 없을 때 토글 상태로 떨어지지 않는다")
    ap.add_argument("--high-watt", type=float, default=HIGH_LOAD_THRESHOLD_W,
                    help=f"고부하 동시 조건 (기본 관측 {HIGH_LOAD_THRESHOLD_W:.0f}W)")
    ap.add_argument("--low-watt", type=float, default=150.0,
                    help="'저전력 창' 문턱 (기본 150W)")
    ap.add_argument("--min-true-w", type=float, default=RESISTIVE_MIN_TRUE_W,
                    help=f"저항3종 혼동의 발열 구간 문턱 (기본 {RESISTIVE_MIN_TRUE_W:.0f}W)")
    a = ap.parse_args()

    path = Path(a.log)
    recs = load_log(path)
    sessions = split_sessions(recs)
    if not sessions:
        print(f"  {path} 에 읽을 레코드가 없습니다.")
        return 1

    pick = sessions if a.all_sessions else sessions[-1:]
    tail = "" if a.all_sessions else " (마지막 것만 채점. 전부 보려면 --all-sessions)"
    print(f"\n로그 {path}   세션 {len(sessions)}개{tail}")

    results = []
    for k, s in enumerate(pick, start=len(sessions) - len(pick) + 1):
        print(f"\n\n[세션 {k}/{len(sessions)}]")
        r = score_session(s, a)
        if r:
            r["session"] = k
            results.append(r)

    if a.json and results:
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  저장: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
