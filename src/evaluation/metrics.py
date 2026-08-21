"""
평가 지표 (Evaluation Metrics)
===============================
설계 문서 4.3절의 지표를 구현한다. **모델보다 먼저 만들어야 하는 것**이다 —
비교할 자가 없으면 baseline 도 CNN 도 잘한 건지 알 수 없다.

[합성에서만 잴 수 있는 것과 실측에서도 잴 수 있는 것을 갈라 둘 것]
실측 복합 부하에는 **기기별 정답이 없다**. 4.3절이 "기기별 MAE" 를 지표로 적어
두었지만 실측에서는 계산 자체가 불가능하다.

    지표                    합성 홀드아웃   실측(test.2/test3)
    기기별 MAE (W)               O                X  라벨 없음
    FA_i / RE_i                  O                X
    저항 3종 혼동행렬             O                X
    총전력 잔차                   O                O
    on/off F1                    O           ~  이벤트 시각 기반 (real_events.py)
    이벤트 ΔP 정확도              O                O  (real_events.py)

[FA 와 RE 를 반드시 함께 볼 것]
`FA` 는 **꺼져 있을 때** 잘못 붙인 전력만 잰다. 0.6절의 저부하 체계적 과소 예측은
기기가 **켜져 있을 때** 일어나므로 FA 에 안 잡힌다. FA 만 보면 "저부하에 헛것을
안 붙였으니 성공" 으로 읽히는데 실제로는 켜진 것도 34% 낮게 읽고 있을 수 있다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np

# 0.7절의 "고부하". 이 기기들의 참 전력 합이 임계를 넘으면 고부하 동시 구간으로 본다.
HIGH_LOAD_APPLIANCES = ("electiric_kettle", "oven", "hair_dryer", "hotplate", "air_conditioner")
HIGH_LOAD_THRESHOLD_W = 500.0

# 0.2절에서 고조파 지문이 사실상 같은 무리. 실질 난이도가 여기서 결정된다.
RESISTIVE_TRIO = ("electiric_kettle", "oven", "hair_dryer")

# 저항3종 혼동을 채점할 최소 참 전력.
#
# **이 하한이 없으면 지표가 히터가 아니라 팬/조명을 센다.** 오븐의 `is_on` 에는
# 히터가 꺼진 팬/조명 구간(약 16W)이 섞여 있어서(0.2절), 홀드아웃에서 오븐 단독
# 창의 74.7% 가 그 상태였다. 16W 참값에 세 기기 예측이 전부 0 근처면 argmax 는
# 난수이고, 그 잡음이 모델 간 차이로 보였다.
#   결함 지표: cnn_v2 0.953 / cnn_v6 0.756 / GBM 0.970
#   300W 하한: cnn_v2 0.996 / cnn_v6 0.994 / GBM 0.988   <- 사실상 동률
RESISTIVE_MIN_TRUE_W = 300.0

# 이 값보다 큰 전력이면 '켜졌다'고 본다 (예측 on/off 를 전력에서 유도할 때)
ON_POWER_THRESHOLD_W = 5.0

# RE(상대 오차) 의 분모로 인정할 최소 참 전력.
# 켜짐 라벨이 붙어 있어도 상태 전이 순간에는 참 전력이 0 에 가까울 수 있고,
# 그때 |err|/P 가 발산해 평균을 통째로 날려 버린다.
# (실제로 드라이기 RE 가 31,177 로 나왔다 - 전이 샘플 몇 개 때문이었다)
RE_MIN_TRUE_W = 1.0


def _safe(x: np.ndarray) -> float:
    return float(x) if np.isfinite(x) else float("nan")


@dataclass
class ApplianceScore:
    """기기 1종의 성적."""
    appliance: str
    n_on: int
    n_off: int
    mae_w: float                 # 전 구간 평균 절대 오차 (W)
    mae_on_w: float              # 켜져 있을 때만
    nmae_on: float               # MAE(on)/s_i - 정격 대비. 발산하지 않는다
    re_on: float                 # RE_i - 상대 오차 (참 전력 RE_MIN_TRUE_W 이상만)
    re_on_median: float          # RE 중앙값. 꼬리에 안 끌린다
    n_re: int                    # RE 계산에 들어간 표본 수
    bias_on_w: float             # 켜져 있을 때 평균 부호 오차 (음수 = 과소 예측)
    fa_w: float                  # FA_i - 꺼져 있을 때 잘못 붙인 평균 전력 (W)
    fa_rel: float                # FA_i / s_i
    fa_high_w: float             # 고부하가 같이 켜진 구간에서의 FA
    fa_low_w: float              # 고부하가 없는 구간에서의 FA
    fa_high_rel: float
    transfer_w: float            # fa_high - fa_low. 곧 전가량
    f1: float
    precision: float
    recall: float
    sae: float                   # 총 에너지 상대 오차 (부호 있음)

    def as_row(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _f1(pred_on: np.ndarray, true_on: np.ndarray) -> tuple:
    tp = float(np.sum(pred_on & true_on))
    fp = float(np.sum(pred_on & ~true_on))
    fn = float(np.sum(~pred_on & true_on))
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec > 0 and rec > 0) else 0.0
    return f1, prec, rec


def score_appliances(
    y_true: np.ndarray,                 # (N, K) 참 전력 (W)
    y_pred: np.ndarray,                 # (N, K) 예측 전력 (W)
    appliances: Sequence[str],
    s_i: Optional[Dict[str, float]] = None,
    on_true: Optional[np.ndarray] = None,   # (N, K) bool. 없으면 y_true > 임계
    on_pred: Optional[np.ndarray] = None,   # (N, K) bool. 없으면 y_pred > 임계
    high_load_mask: Optional[np.ndarray] = None,  # (N,) bool. 없으면 y_true 에서 유도
) -> List[ApplianceScore]:
    """기기별 성적표. 4.3절의 지표를 한 번에 낸다."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"모양이 다릅니다: {y_true.shape} vs {y_pred.shape}")
    n, k = y_true.shape
    if k != len(appliances):
        raise ValueError(f"기기 수가 맞지 않습니다: {k} vs {len(appliances)}")

    on_t = (y_true > ON_POWER_THRESHOLD_W) if on_true is None else np.asarray(on_true, bool)
    on_p = (y_pred > ON_POWER_THRESHOLD_W) if on_pred is None else np.asarray(on_pred, bool)

    if high_load_mask is None:
        idx = [i for i, a in enumerate(appliances) if a in HIGH_LOAD_APPLIANCES]
        high_load_mask = (y_true[:, idx].sum(axis=1) > HIGH_LOAD_THRESHOLD_W) if idx \
            else np.zeros(n, dtype=bool)
    high_load_mask = np.asarray(high_load_mask, bool)

    out: List[ApplianceScore] = []
    for j, app in enumerate(appliances):
        t, p = y_true[:, j], y_pred[:, j]
        err = p - t
        m_on, m_off = on_t[:, j], ~on_t[:, j]
        off_hi = m_off & high_load_mask
        off_lo = m_off & ~high_load_mask
        scale = float((s_i or {}).get(app, np.nan))

        # 상대 오차는 참 전력이 충분히 큰 표본에서만 잰다 (RE_MIN_TRUE_W 참조)
        m_re = m_on & (t > RE_MIN_TRUE_W)
        rel = np.abs(err[m_re]) / t[m_re] if m_re.any() else np.zeros(0)

        fa = _safe(np.mean(p[m_off])) if m_off.any() else float("nan")
        fa_hi = _safe(np.mean(p[off_hi])) if off_hi.any() else float("nan")
        fa_lo = _safe(np.mean(p[off_lo])) if off_lo.any() else float("nan")
        f1, prec, rec = _f1(on_p[:, j], m_on)
        tot_t = float(t.sum())

        out.append(ApplianceScore(
            appliance=app,
            n_on=int(m_on.sum()), n_off=int(m_off.sum()),
            mae_w=_safe(np.mean(np.abs(err))),
            mae_on_w=_safe(np.mean(np.abs(err[m_on]))) if m_on.any() else float("nan"),
            nmae_on=(_safe(np.mean(np.abs(err[m_on]))) / scale
                     if (m_on.any() and scale == scale) else float("nan")),
            re_on=_safe(np.mean(rel)) if len(rel) else float("nan"),
            re_on_median=_safe(np.median(rel)) if len(rel) else float("nan"),
            n_re=int(m_re.sum()),
            bias_on_w=_safe(np.mean(err[m_on])) if m_on.any() else float("nan"),
            fa_w=fa, fa_rel=fa / scale if scale == scale else float("nan"),
            fa_high_w=fa_hi, fa_low_w=fa_lo,
            fa_high_rel=fa_hi / scale if scale == scale else float("nan"),
            transfer_w=fa_hi - fa_lo if (fa_hi == fa_hi and fa_lo == fa_lo) else float("nan"),
            f1=f1, precision=prec, recall=rec,
            sae=(float(p.sum()) - tot_t) / tot_t if tot_t > 0 else float("nan"),
        ))
    return out


def resistive_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    appliances: Sequence[str],
    trio: Sequence[str] = RESISTIVE_TRIO,
    min_true_w: float = RESISTIVE_MIN_TRUE_W,
) -> Optional[dict]:
    """저항 3종 혼동행렬. 실질 난이도는 여기서 결정된다 (4.3절).

    다중 라벨 회귀라 '혼동' 을 이렇게 정의한다:
    **3종 중 정확히 1대만, 그것도 발열 전력으로 켜진 창**을 골라, 모델이 셋 중
    가장 큰 전력을 붙인 기기를 예측으로 본다.

    `min_true_w` 하한이 핵심이다. 없으면 오븐의 팬/조명(16W) 구간까지 세게 되어
    지표가 난수가 된다 (RESISTIVE_MIN_TRUE_W 주석 참조).
    """
    idx = [appliances.index(a) for a in trio if a in appliances]
    if len(idx) < 2:
        return None
    t, p = np.asarray(y_true)[:, idx], np.asarray(y_pred)[:, idx]
    on = t > ON_POWER_THRESHOLD_W
    solo = on.sum(axis=1) == 1
    # 발열 전력 구간만. 부속 저전력 상태(오븐 팬/조명)는 이 지표의 대상이 아니다.
    solo = solo & (t.max(axis=1) >= min_true_w)
    if not solo.any():
        return None
    truth = np.argmax(on[solo], axis=1)
    guess = np.argmax(p[solo], axis=1)
    names = [appliances[i] for i in idx]
    m = np.zeros((len(idx), len(idx)), dtype=int)
    for a, b in zip(truth, guess):
        m[a, b] += 1
    return {
        "labels": names,
        "matrix": m.tolist(),                       # matrix[참][예측]
        "n_windows": int(solo.sum()),
        "accuracy": float(np.mean(truth == guess)),
        "min_true_w": float(min_true_w),
    }


def state_breakdown(
    y_true: np.ndarray, y_pred: np.ndarray, y_on: np.ndarray, y_state: np.ndarray,
    appliances: Sequence[str], min_windows: int = 10,
) -> List[dict]:
    """기기별 **동작 상태별** 성적.

    `is_on` 은 물리적으로 매우 다른 상태를 하나로 묶는다. 오븐은 팬/조명(16W)과
    히터(1,200W)가 둘 다 `is_on=1` 이고, 에어컨도 송풍(14W)과 냉방(750W)이 그렇다.
    집계 지표만 보면 **저전력 부속 상태의 실패가 통째로 묻힌다.**

    실제로 cnn_v2 는 오븐 팬/조명을 2배로(15W -> 30W), 에어컨 송풍을 0 으로
    예측했는데, 기기 단위 MAE 로는 보이지 않았다. 이 표에서만 드러난다.
    """
    from src.labeling.state_definitions import get_appliance_config
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rows: List[dict] = []
    for j, app in enumerate(appliances):
        names = {s.state_id: s.name for s in get_appliance_config(app).states}
        on = np.asarray(y_on)[:, j] == 1
        if not on.any():
            continue
        for sid in sorted(set(np.asarray(y_state)[on, j].tolist())):
            m = on & (np.asarray(y_state)[:, j] == sid)
            if int(m.sum()) < min_windows:
                continue
            t, p = float(np.median(y_true[m, j])), float(np.median(y_pred[m, j]))
            rows.append({
                "appliance": app, "state_id": int(sid),
                "state": names.get(int(sid), f"STATE_{sid}"),
                "n": int(m.sum()), "true_median_w": t, "pred_median_w": p,
                "median_rel_err": abs(p - t) / max(t, 1e-9),
                "mae_w": float(np.mean(np.abs(y_pred[m, j] - y_true[m, j]))),
            })
    return rows


def format_state_table(rows: List[dict], flag_rel: float = 0.5) -> str:
    h = f"{'가전':17s}{'상태':22s}{'창수':>6s}{'참W':>9s}{'예측W':>9s}{'오차':>8s}"
    out = [h, "-" * len(h)]
    for r in rows:
        flag = " <<<" if r["median_rel_err"] > flag_rel else ""
        out.append(f"{r['appliance']:17s}{r['state'][:20]:22s}{r['n']:>6d}"
                   f"{r['true_median_w']:>9.1f}{r['pred_median_w']:>9.1f}"
                   f"{100*r['median_rel_err']:>7.0f}%{flag}")
    return chr(10).join(out)


def total_power_residual(
    y_pred: np.ndarray,              # (N, K) 기기별 예측 전력
    p_observed: np.ndarray,          # (N,) 관측 총전력
    standby_pred: Optional[np.ndarray] = None,   # (N, K)
    p_noise: Optional[np.ndarray] = None,        # (N,)
) -> dict:
    """설명하지 못한 전력. **실측에서도 잴 수 있는 몇 안 되는 지표다.**

    설계 문서 3.3절에서 잔차 헤드를 뺀 대신 쓰기로 한 진단값이 바로 이것이다.
    학습된 파라미터가 아니라 산술이라 부호가 있고, 과대 예측도 드러난다.
    """
    recon = np.asarray(y_pred, dtype=np.float64).sum(axis=1)
    if standby_pred is not None:
        recon = recon + np.asarray(standby_pred, dtype=np.float64).sum(axis=1)
    if p_noise is not None:
        recon = recon + np.asarray(p_noise, dtype=np.float64)
    r = np.asarray(p_observed, dtype=np.float64) - recon
    return {
        "mean_w": _safe(np.mean(r)),
        "mean_abs_w": _safe(np.mean(np.abs(r))),
        "p95_abs_w": _safe(np.percentile(np.abs(r), 95)),
        "max_abs_w": _safe(np.max(np.abs(r))),
        "mean_rel": _safe(np.mean(np.abs(r)) / max(float(np.mean(p_observed)), 1e-9)),
    }


def summarize(
    scores: List[ApplianceScore],
    low_load: Sequence[str] = ("beam_projector", "laptop_charger", "fan", "minipc"),
    fa_target: float = 0.15,
) -> dict:
    """4.3절 목표치 통과 여부까지 포함한 요약."""
    by = {s.appliance: s for s in scores}
    lows = [by[a] for a in low_load if a in by]
    fa_ok = [s for s in lows if s.fa_high_rel == s.fa_high_rel and s.fa_high_rel < fa_target]
    return {
        "mae_w_mean": float(np.nanmean([s.mae_w for s in scores])),
        "f1_mean": float(np.nanmean([s.f1 for s in scores])),
        "worst_f1": min(((s.f1, s.appliance) for s in scores), default=(float("nan"), None)),
        "low_load_fa_high_rel": {s.appliance: s.fa_high_rel for s in lows},
        "low_load_re_on": {s.appliance: s.re_on for s in lows},
        "low_load_nmae_on": {s.appliance: s.nmae_on for s in lows},
        "low_load_bias_on_w": {s.appliance: s.bias_on_w for s in lows},
        "fa_target": fa_target,
        "fa_target_pass": f"{len(fa_ok)}/{len(lows)}",
        "max_transfer_w": float(np.nanmax([s.transfer_w for s in scores])) if scores else float("nan"),
    }


def format_table(scores: List[ApplianceScore]) -> str:
    """사람이 읽는 표."""
    h = (f"{'가전':18s}{'양성':>7s}{'MAE':>9s}{'MAE(on)':>10s}{'nMAE':>8s}{'RE중앙':>8s}"
         f"{'편향(on)':>10s}{'FA':>8s}{'FA(고부하)':>12s}{'전가':>8s}{'F1':>7s}")
    lines = [h, "-" * len(h)]
    for s in scores:
        lines.append(
            f"{s.appliance:18s}{s.n_on:>7d}{s.mae_w:>9.2f}{s.mae_on_w:>10.2f}"
            f"{s.nmae_on:>8.3f}{s.re_on_median:>8.3f}{s.bias_on_w:>+10.2f}{s.fa_w:>8.2f}"
            f"{s.fa_high_w:>12.2f}{s.transfer_w:>+8.2f}{s.f1:>7.3f}"
        )
    return "\n".join(lines)
