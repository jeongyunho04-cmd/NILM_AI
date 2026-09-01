"""
기기별 전력 오차 — **참값을 아는 기기에 대해서만** (12.122.6, 2026-09-01)
============================================================================
4.2절이 *"실측에는 기기별 정답이 없다"* 고 적었고 그 말이 이 저장소의 채점을
총합(잔차)과 없는 기기(유령) 둘로 묶어 놓았다. **그런데 넷은 있다.**

격리 녹화의 60초 창 통전 전력이 좁은 기기가 있다:

```
기기                 창      p5      중앙     p95      폭    폭/중앙
electiric_kettle    15  1252.5  1256.7  1262.3    9.7     0.8%
hotplate             8   457.6   460.1   461.9    4.3     0.9%
oven                12  1123.5  1143.2  1172.9   49.5     4.3%
beam_projector      95    44.2    46.9    47.8    3.6     7.7%
--- 못 쓴다 ---
hair_dryer          11   637.4   841.8   985.2  347.8    41.3%   강/약 상태
laptop_charger     257    33.3    48.7    68.4   35.1    72.1%
minipc             280     8.2    12.4    22.7   14.4   116.1%
```

**왜 이 지표가 필요한가.** 12.122.6 이 프로젝터 스냅으로 예측을 88W -> 47W 로
**참값에 맞췄는데**, 저장소가 재는 것이 잔차와 유령뿐이라 그 개선이 어느 지표에도
안 잡히고 부작용만 잡혔다. 배분을 겨냥한 처방이 **원리적으로 판정 불가**였다.

지금 모델은 프로젝터 ON 창에서 74~99W 를 낸다. **27~52W 오차이고 아무도 안 쟀다.**

[검출 실패와 배분 오차를 반드시 가른다]
`P̂ = σ(on)·p_raw` 라 게이트가 꺼지면 전력이 0 이 된다. 그것을 배분 오차로 세면
검출 문제가 배분 지표로 새어 든다. 그래서 정답이 ON 인 창을 둘로 가른다:

    검출됨 (게이트 ON)   -> **배분 오차**를 잰다     <- 이 지표의 본체
    놓침   (게이트 OFF)  -> 개수만 세고 오차에서 뺀다

[전이 창을 뺀다]
예측은 0.5초 stride 창에서 올라오고 창 자체가 60초다. 스위치 전후의 창은 두
상태가 섞이므로 ON 구간 가장자리 `EDGE_GUARD_CYCLES` 를 양쪽에서 잘라낸다.

[⚠ 프로젝터의 저전력 상태]
격리에서 프로젝터 ON 사이클의 3.7% 가 3~5W 대다 (램프 꺼지고 팬만 도는 구간).
60초 창 평균에서는 거의 안 보이지만(창 최소 41.8W) 완전히 0 은 아니다.
그래서 **중앙 |오차| 와 '격리 폭 안에 든 비율' 을 함께** 낸다 — 뒤쪽이 소수
상태에 둔감하다.

[⚠ 모델을 견줄 때는 **상한 후처리를 끄고** 재라]
`apply_postproc` 의 상한(프로젝터 55W)이 켜져 있으면 예측이 그 값에 눌려
**모든 모델이 같은 오차를 낸다** — 55.0 − 46.9 = 8.1W 다. 2026-09-01 에
2단계 절제 넷(기준선·기울기균등·헤지0·짝수차제외)을 이 상태로 재서 전부
"8.1 ±0.0" 이 나왔다. 후처리가 모델 차이를 통째로 가린 것이지 차이가
없는 것이 아니다.

    처방이 **모델**이면      --postproc off 로 잰다
    처방이 **후처리**면      운영 조합 그대로 잰다 (그게 배포되는 것이므로)

[규칙 14 — 이 표는 격리에서 잰 것이다]
`REFERENCE_W` 는 격리 녹화의 값이다. 복합에서 그 기기가 같은 전력을 쓴다는
가정이 들어간다. 저항 3종은 `V^2/R` 이라 전압만 같으면 성립하고(12.112 가
0.53% 안에서 확인), 프로젝터는 12.120.1 이 복합 격리 양쪽에서 확인했다.
**이 가정이 깨지는 기기는 표에 넣으면 안 된다** — 드라이기·충전기·미니PC 가 그렇다.
"""
from typing import Dict, Optional, Sequence
import numpy as np

from .real_events import build_on_off_truth, load_events

#: 전이 앞뒤로 잘라낼 사이클 수. 창이 60초이므로 그 절반을 뺀다.
EDGE_GUARD_CYCLES = 1800

#: 참값을 아는 기기 — (중앙W, p5, p95). 격리 녹화 60초 창, 2026-09-01 측정.
#: `python -m src.run_power_check --recompute-ref` 가 이 표를 다시 낸다.
REFERENCE_W: Dict[str, tuple] = {
    "electiric_kettle": (1276.9, 1253.1, 1285.9),   # 녹화 2개, 창 33, 폭/중앙 0.026
    "hotplate": (460.1, 457.6, 461.9),              # 녹화 3개, 창  8, 폭/중앙 0.009
    "oven": (1143.2, 1123.5, 1172.9),               # 녹화 3개, 창 12, 폭/중앙 0.043
    "beam_projector": (46.9, 44.2, 47.8),           # 녹화 3개, 창 95, 폭/중앙 0.077
}

#: 표에 **안 넣은** 기기와 그 이유 (규칙 14 — 안 잰 것을 측정처럼 쓰지 않는다).
#: 폭/중앙이 0.10 을 넘으면 상태가 여럿이라 단일 참값을 못 쓴다.
EXCLUDED_REASON: Dict[str, str] = {
    "air_conditioner": "폭/중앙 1.454 (15.5~790W). 인버터라 연속 가변",
    "minipc": "폭/중앙 1.162 (8.2~22.7W). 유휴/작업 3.3배 (12.32.1)",
    "laptop_charger": "폭/중앙 0.722 (33~68W). 고속/트리클",
    "fan": "폭/중앙 0.558 (22~38.5W). 풍량 단계",
    "hair_dryer": "폭/중앙 0.413 (637~985W). 강/약",
}


def _ref_at(app: str, ref: Dict[str, tuple], v_rms) -> np.ndarray:
    """전압 보정한 참값. `P = V^2/R` 인 기기만 보정하고 나머지는 상수다."""
    r = ref[app][0]
    v0 = REFERENCE_V.get(app)
    if v0 is None or v_rms is None:
        return r
    return r * (np.asarray(v_rms, float) / v0) ** 2


def _trim_edges(on: np.ndarray, guard: int) -> np.ndarray:
    """ON 구간의 양 끝 `guard` 사이클을 잘라낸 마스크."""
    if guard <= 0 or not on.any():
        return on.copy()
    d = np.diff(np.concatenate([[False], on, [False]]).astype(np.int8))
    out = on.copy()
    for a in np.flatnonzero(d == 1):
        out[a:a + guard] = False
    for b in np.flatnonzero(d == -1):
        out[max(0, b - guard):b] = False
    return out


#: **물리 가드** — 라벨상 켜진 기기들의 참값 **합**을 관측 총전력이 감당해야 한다.
#: 못 감당하면 그 창의 라벨은 관측과 모순이므로 안 센다.
#:
#: 라벨이 세션 단위인 기기(오븐, 12.119)는 "ON" 이라도 히터가 안 흐른다.
#: 2026-09-01: 오븐만 켜졌다는 20,421 창의 관측 P 중앙이 **131W** 였다 (참값 1143W).
#:
#: ⚠ **기기 하나만 보는 가드로는 모자란다.** 처음에 `P관측 >= 그 기기 참값 x 0.8`
#: 로 걸었더니 거의 안 걸러졌다 — 저항이 둘 켜졌다는 창에서 하나만 흐르면 그
#: 하나의 참값은 넘기 때문이다. 그 상태로 재면 저항 과소예측이 **3배 부풀려진다**:
#:
#:     오븐 +다른저항    합 가드 없음 -192.3W (-16.8%)  ->  합 가드 -66.5W (-5.8%)
#:     포트 +다른저항    합 가드 없음 -223.6W (-17.5%)  ->  합 가드 -28.4W (-2.2%)
OBSERVED_GUARD_FRAC = 0.9

#: 전압 보정. 저항은 `P = V^2/R` 이라 참값이 측정 전압에 매인다.
#: 격리 녹화의 전압(아래)과 복합의 전압이 다르면 그만큼 참값이 틀린다 —
#: 2026-09-01 에 포트 참값이 −15W 어긋났고 V^2 로 정확히 설명됐다
#: (격리 213.0V -> 복합 211.7V, 예측 1262.0 = 실측 1262.0).
REFERENCE_V: Dict[str, float] = {
    "electiric_kettle": 213.0,
    "hotplate": 218.0,
    "oven": 215.8,
}


def score_power_ref(
    P: np.ndarray,                    # (n_cycles, K) 예측 전력 (게이트 곱한 값)
    pred_on: np.ndarray,              # (n_cycles, K) 게이트 판정
    stem: str,
    appliances: Sequence[str],
    events: Optional[dict] = None,
    reference: Optional[Dict[str, tuple]] = None,
    guard: int = EDGE_GUARD_CYCLES,
    p_observed: Optional[np.ndarray] = None,
    v_rms: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """정답이 ON 인 구간에서 **배분 오차**를 잰다.

    Returns:
        {기기: {n, n_detected, detect_rate, median_abs_err_w, mean_err_w,
                p90_abs_err_w, within_band, ref_w}}

        `mean_err_w` 는 **부호가 있다** — 과대예측이 양수다. 방향을 봐야
        처방(스냅·재배분)의 부호를 정할 수 있다.
        `within_band` 는 격리 p5~p95 안에 든 비율이다 (소수 상태에 둔감).
    """
    ref = REFERENCE_W if reference is None else reference
    ev = events if events is not None else load_events()
    truth, scorable = build_on_off_truth(stem, appliances, len(P), ev)
    pred_on = np.asarray(pred_on, bool)

    # 라벨상 켜진 기기들의 참값 합 (전압 보정). 관측이 이것을 못 감당하면
    # 그 창의 라벨은 관측과 모순이다 — 세션 단위 라벨이 대표적이다.
    consistent = None
    if p_observed is not None:
        claim = np.zeros(len(P))
        for j, app in enumerate(appliances):
            if app not in ref:
                continue
            claim = claim + np.where(truth[:, j], _ref_at(app, ref, v_rms), 0.0)
        consistent = np.asarray(p_observed) >= claim * OBSERVED_GUARD_FRAC

    out: Dict[str, dict] = {}
    for j, app in enumerate(appliances):
        if app not in ref:
            continue
        m = _trim_edges(truth[:, j], guard) & scorable[:, j]
        r0, lo0, hi0 = ref[app]
        rv = _ref_at(app, ref, v_rms)
        r = float(np.mean(rv)) if np.ndim(rv) else float(rv)
        sc = r / max(r0, 1e-9)
        lo, hi = lo0 * sc, hi0 * sc
        if consistent is not None:
            m = m & consistent
        if not m.any():
            continue
        det = m & pred_on[:, j]
        row = {"ref_w": float(r), "n": int(m.sum()), "n_detected": int(det.sum()),
               "detect_rate": float(det.sum() / m.sum()),
               "guarded": p_observed is not None}
        if det.any():
            e = P[det, j] - (rv[det] if np.ndim(rv) else rv)
            row.update({
                "median_abs_err_w": float(np.median(np.abs(e))),
                "mean_err_w": float(e.mean()),
                "p90_abs_err_w": float(np.percentile(np.abs(e), 90)),
                "within_band": float(((P[det, j] >= lo) & (P[det, j] <= hi)).mean()),
            })
        else:
            row.update({"median_abs_err_w": float("nan"), "mean_err_w": float("nan"),
                        "p90_abs_err_w": float("nan"), "within_band": float("nan")})
        out[app] = row
    return out


def summarize_power_ref(per_file: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    """파일별 결과를 기기별로 합친다. **사이클 수로 가중한다.**"""
    acc: Dict[str, list] = {}
    for rows in per_file.values():
        for app, s in rows.items():
            if s.get("n_detected", 0) > 0:
                acc.setdefault(app, []).append(s)
    out: Dict[str, dict] = {}
    for app, rows in acc.items():
        w = np.array([r["n_detected"] for r in rows], float)
        out[app] = {k: float(np.nansum([r[k] for r in rows] * w) / w.sum())
                    for k in ("median_abs_err_w", "mean_err_w", "p90_abs_err_w", "within_band")}
        out[app]["ref_w"] = rows[0]["ref_w"]
        out[app]["n"] = int(sum(r["n"] for r in rows))
        out[app]["n_detected"] = int(w.sum())
        out[app]["detect_rate"] = float(w.sum() / max(sum(r["n"] for r in rows), 1))
        out[app]["n_files"] = len(rows)
    return out


def format_power_ref(summary: Dict[str, dict]) -> str:
    """사람이 읽는 표. `run_gate_check` / `run_adapt` 가 그대로 찍는다."""
    if not summary:
        return "  기기별 전력 오차: 잴 수 있는 기기가 없다"
    lines = [f"  {'기기':16s}{'참값W':>8s}{'중앙|오차|':>11s}{'평균오차':>10s}"
             f"{'p90':>8s}{'폭안':>8s}{'검출률':>8s}{'창':>9s}"]
    for app, s in sorted(summary.items()):
        lines.append(f"  {app:16s}{s['ref_w']:8.1f}{s['median_abs_err_w']:11.1f}"
                     f"{s['mean_err_w']:+10.1f}{s['p90_abs_err_w']:8.1f}"
                     f"{s['within_band']:8.3f}{s['detect_rate']:8.3f}{s['n_detected']:9,}")
    return "\n".join(lines)
