"""
실측 복합 부하 평가 (Real Composite-Load Evaluation)
=====================================================
실측에는 **기기별 정답 전력이 없다.** 설계 문서 4.3절이 "기기별 MAE" 를 지표로
적어 두었지만 계산할 수가 없다. 여기서는 계산 가능한 것만 잰다.

    1. 총전력 잔차      P_관측 − Σ P̂        항상 가능. 3.3절의 진단값
    2. on/off F1        타임라인의 확실한 구간에서만. uncertain 구간은 제외
    3. 이벤트 ΔP        타임라인이 시각·ΔP 는 정확하다고 명시한 전이만

`processed_data/real_events.json` 이 정답 원본이다. 4.2절대로 정상상태 귀속은
옮기지 않았다 — 그 부분에 오류가 확인되었기 때문이다.

`test.csv` 는 `sealing.py` 가 막는다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import json
import numpy as np

from .sealing import assert_not_sealed

EVENTS_PATH = Path("processed_data/real_events.json")
SAMPLING_HZ = 60.0
# 이벤트 시각 오차 허용폭. 상태 분류기의 dwell-time(0.3~1.0초)과 타임라인
# 판독 오차를 함께 감안한 값이다.
EVENT_TOLERANCE_S = 3.0


def load_events(path: Union[str, Path] = EVENTS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"실측 이벤트 정답이 없습니다: {p.resolve()}")
    return json.loads(p.read_text(encoding="utf-8"))["files"]


def build_on_off_truth(
    stem: str,
    appliances: Sequence[str],
    n_cycles: int,
    events: Optional[dict] = None,
) -> tuple:
    """타임라인에서 (on, scorable) 마스크를 만든다.

    Returns:
        on:       (n_cycles, K) bool  — 확실히 켜져 있던 구간
        scorable: (n_cycles, K) bool  — 채점해도 되는 구간 (uncertain 제외)

    uncertain 을 OFF 로 채점하면 안 된다. 오븐이 대표적인데, 타임라인은 히터 통전만
    적어 두었고 그 사이에 팬/조명(is_on=1)이 도는 구간이 섞여 있다. 그것을 OFF 로
    치면 모델이 맞게 예측해도 오답이 된다.
    """
    assert_not_sealed(stem)
    files = events if events is not None else load_events()
    if stem not in files:
        raise KeyError(f"'{stem}' 의 타임라인 정답이 없습니다. 있는 것: {sorted(files)}")
    spec = files[stem]["intervals"]

    on = np.zeros((n_cycles, len(appliances)), dtype=bool)
    scorable = np.ones((n_cycles, len(appliances)), dtype=bool)

    def to_idx(t0, t1):
        return max(0, int(t0 * SAMPLING_HZ)), min(n_cycles, int(t1 * SAMPLING_HZ))

    for j, app in enumerate(appliances):
        s = spec.get(app)
        if s is None:      # 타임라인에 없는 기기 = 그 녹화에 아예 없었다
            continue
        for t0, t1 in s.get("on", []):
            a, b = to_idx(t0, t1)
            on[a:b, j] = True
        for t0, t1 in s.get("uncertain", []):
            a, b = to_idx(t0, t1)
            scorable[a:b, j] = False
        # 확실히 켜진 구간은 uncertain 이어도 채점한다
        scorable[on[:, j], j] = True
    return on, scorable


def score_on_off(
    pred_on: np.ndarray,             # (n_cycles, K) bool
    stem: str,
    appliances: Sequence[str],
    events: Optional[dict] = None,
) -> Dict[str, dict]:
    """실측 on/off F1. uncertain 구간은 세지 않는다."""
    pred_on = np.asarray(pred_on, bool)
    truth, scorable = build_on_off_truth(stem, appliances, len(pred_on), events)
    out = {}
    for j, app in enumerate(appliances):
        m = scorable[:, j]
        if not m.any():
            continue
        p, t = pred_on[m, j], truth[m, j]
        tp, fp, fn = float((p & t).sum()), float((p & ~t).sum()), float((~p & t).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        out[app] = {
            "f1": 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else 0.0,
            "precision": prec, "recall": rec,
            "n_scored": int(m.sum()), "n_true_on": int(t.sum()),
            "ignored_uncertain": int((~scorable[:, j]).sum()),
        }
    return out


def score_absent(
    pred_power: np.ndarray,          # (n, K) 기기별 예측 전력 (창 단위여도 된다)
    stem: str,
    appliances: Sequence[str],
    pred_on: Optional[np.ndarray] = None,    # (n, K) bool
    s_i: Optional[Dict[str, float]] = None,
    events: Optional[dict] = None,
) -> dict:
    """**그 파일에 없던 기기에 붙인 전력** — 라벨 없이 잴 수 있는 오귀속 지표.

    12.4절 표는 "기기별 FA 는 실측에서 못 잰다 (라벨 없음)" 고 적었는데,
    **그 파일에 없는 기기는 정답이 0 으로 확정이다.** `appliances_present` 가
    어느 기기가 없었는지 알려주므로 FA 를 정확히 계산할 수 있다.

    이것이 2단계(4.2절)의 주 판정 지표다. `L_cons` 는 합만 보므로 오귀속을
    전혀 못 본다 (12.5절). 실제로 1차 적응에서 모자란 합을 채우라고 시키자
    모델이 이미 틀린 기기(핫플레이트)에 16.8W 를 더 붙였다.
    """
    assert_not_sealed(stem)
    files = events if events is not None else load_events()
    present = set(files[stem]["appliances_present"])
    p = np.asarray(pred_power, dtype=np.float64)
    rows, total = {}, 0.0
    for j, app in enumerate(appliances):
        if app in present:
            continue
        mu = float(p[:, j].mean())
        total += mu
        rows[app] = {
            "mean_w": mu,
            "p95_w": float(np.percentile(p[:, j], 95)),
            "max_w": float(p[:, j].max()),
            "on_rate": float(np.asarray(pred_on)[:, j].mean()) if pred_on is not None else float("nan"),
            "fa_rel": mu / s_i[app] if s_i and app in s_i else float("nan"),
        }
    tot_pred = float(p.sum(1).mean())
    return {
        "absent": rows,
        "absent_sum_w": total,
        "pred_total_w": tot_pred,
        "absent_share": total / tot_pred if tot_pred > 0 else float("nan"),
        "n_absent": len(rows),
    }


def score_events(
    pred_power: np.ndarray,          # (n_cycles, K) 기기별 예측 전력
    stem: str,
    appliances: Sequence[str],
    tolerance_s: float = EVENT_TOLERANCE_S,
    settle_s: float = 2.0,
    events: Optional[dict] = None,
) -> List[dict]:
    """이벤트 시각에서 그 기기의 예측 전력이 옳은 만큼 뛰었는지 본다.

    타임라인이 "이벤트 시각과 ΔP 는 정확하다" 고 밝힌 부분만 쓴다 (4.2절).
    전이 앞뒤 `settle_s` 구간의 중앙값 차이를 예측 ΔP 로 본다.
    """
    assert_not_sealed(stem)
    files = events if events is not None else load_events()
    spec = files[stem]["events"]
    pred_power = np.asarray(pred_power, dtype=np.float64)
    n = len(pred_power)
    w = int(settle_s * SAMPLING_HZ)
    tol = int(tolerance_s * SAMPLING_HZ)

    out = []
    for ev in spec:
        app = ev["appliance"]
        if app not in appliances:
            continue
        j = list(appliances).index(app)
        c = int(ev["t_s"] * SAMPLING_HZ)
        pre = pred_power[max(0, c - tol - w):max(1, c - tol), j]
        post = pred_power[min(n - 1, c + tol):min(n, c + tol + w), j]
        if len(pre) == 0 or len(post) == 0:
            continue
        got = float(np.median(post) - np.median(pre))
        want = float(ev["delta_p_w"])
        out.append({
            "t_s": ev["t_s"], "appliance": app, "kind": ev["kind"],
            "delta_p_true_w": want, "delta_p_pred_w": got,
            "error_w": got - want,
            "error_rel": (got - want) / abs(want) if want else float("nan"),
            "sign_correct": bool(np.sign(got) == np.sign(want)),
        })
    return out


def format_event_table(rows: List[dict]) -> str:
    h = f"{'시각(s)':>9s}{'가전':>18s}{'전이':>6s}{'참 ΔP':>10s}{'예측 ΔP':>11s}{'오차':>10s}{'부호':>6s}"
    lines = [h, "-" * len(h)]
    for r in rows:
        lines.append(f"{r['t_s']:>9.1f}{r['appliance']:>18s}{r['kind']:>6s}"
                     f"{r['delta_p_true_w']:>+10.1f}{r['delta_p_pred_w']:>+11.1f}"
                     f"{r['error_w']:>+10.1f}{'O' if r['sign_correct'] else 'X':>6s}")
    return "\n".join(lines)
