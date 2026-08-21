"""
Phase 1 baseline 자동 검증
===========================
특징 추출기와 GBM baseline 이 조용히 망가지는 것을 막는다.
여기 테스트 둘은 실제로 겪은 실패를 그대로 고정한 것이다:

  1. `test_near_scope_preserves_relay_phase`
     recent 범위(±60 사이클)가 핫플레이트 릴레이 주기(120 사이클)와 정확히 겹쳐
     중앙값이 통전 여부를 지워 버렸다. F1 0.521 / 편향 -216W.
  2. `test_regressor_is_trained_on_positives_only`
     타깃의 85% 가 0 이라 MAE 최적 상수가 중앙값(=0)이었고, 회귀기가 "전부 0" 으로
     수렴해 조기종료했다. RE 가 전부 1.000 으로 나왔다.
"""
import numpy as np
import pytest

from src.baseline.features import (
    BLOCK_CYCLES,
    NEAR_HALF,
    RECENT_HALF,
    extract,
    feature_names,
    sanity_check,
)
from src.baseline.train import BaselineModel, train
from src.evaluation.metrics import RE_MIN_TRUE_W, score_appliances

W = 600
TARGET = 539


def _window(p_series: np.ndarray, i1: float = 1.0, i3: float = 0.0) -> np.ndarray:
    """전력 시계열 하나로 33채널 창을 만든다."""
    x = np.zeros((33, W), dtype=np.float32)
    x[0] = i1                    # I1 실수부
    x[2] = i3                    # I3 실수부
    x[30] = p_series             # P
    x[32] = 222.0                # V
    return x


def test_feature_names_match_array():
    n, names = sanity_check()
    assert n == len(names) == 84


def test_no_nan_or_inf_on_degenerate_input():
    """전부 0 인 창에서도 NaN/Inf 가 나오면 안 된다 (0 나누기 방어)."""
    for x in (np.zeros((3, 33, W), np.float32), np.full((3, 33, W), 1e-9, np.float32)):
        f = extract(x, TARGET)
        assert np.isfinite(f).all()


def test_near_scope_preserves_relay_phase():
    """핫플레이트 릴레이 위상이 특징에 남아야 한다.

    ±60 사이클 중앙값만 쓰면 주기(120)와 정확히 겹쳐 정보가 사라진다.
    `p_target` / `p_near` 는 주기보다 짧으므로 통전/휴지를 갈라야 한다.
    """
    period, duty = 120, 0.5
    t = np.arange(W)
    on_phase = ((t % period) < period * duty).astype(np.float32) * 500.0
    off_phase = ((t + period // 2) % period < period * duty).astype(np.float32) * 500.0
    # 타깃 시점에서 하나는 통전, 하나는 휴지가 되도록 맞춘다
    assert on_phase[TARGET] != off_phase[TARGET]

    names = feature_names()
    f = extract(np.stack([_window(on_phase), _window(off_phase)]), TARGET)
    gap = lambda k: abs(float(f[0, names.index(k)] - f[1, names.index(k)]))

    assert 2 * NEAR_HALF < period, "near 범위가 릴레이 주기보다 짧아야 한다"
    assert gap("p_target") > 1.0, "타깃 샘플 전력이 통전 여부를 못 가릅니다"
    assert gap("p_near") > 1.0, "near 범위가 통전 여부를 못 가릅니다"
    # recent 는 주기 전체를 평균하므로 오히려 잘 못 가른다 - 그것이 near 를 넣은 이유다
    assert gap("p_near") > gap("p_recent")


def test_texture_features_separate_duty_from_steady():
    """0.5초 블록 전이 횟수가 주기 부하와 연속 부하를 갈라야 한다 (0.4절)."""
    t = np.arange(W)
    pulsed = ((t % 120) < 60).astype(np.float32) * 500.0
    steady = np.full(W, 500.0, np.float32)
    names = feature_names()
    f = extract(np.stack([_window(pulsed), _window(steady)]), TARGET)
    tr = names.index("blk_transitions")
    assert f[0, tr] >= 4, "주기 부하의 전이 횟수가 잡히지 않습니다"
    assert f[1, tr] <= 1, "연속 부하인데 전이가 잡힙니다"
    assert 2 * BLOCK_CYCLES < 120, "블록이 릴레이 주기보다 짧아야 앨리어싱을 피한다"


def test_harmonic_ratio_is_scale_invariant():
    """고조파비는 크기가 2배가 돼도 같아야 한다 (0.2절의 크기 무관 지문)."""
    names = feature_names()
    p = np.full(W, 300.0, np.float32)
    f = extract(np.stack([_window(p, i1=1.0, i3=0.05), _window(p, i1=2.0, i3=0.10)]), TARGET)
    for k in ("ratio_i3_recent", "ratio_i3_near"):
        j = names.index(k)
        assert abs(float(f[0, j] - f[1, j])) < 1e-3, f"{k} 가 크기에 따라 변합니다"


def test_regressor_is_trained_on_positives_only():
    """0 이 대부분인 타깃에 전체 학습을 하면 회귀기가 죽는다.

    켜진 창에서만 학습하고 분류기로 게이팅해야 한다 (2.4절과 같은 구조).
    """
    rng = np.random.default_rng(0)
    n = 3000
    F = rng.standard_normal((n, 6)).astype(np.float32)
    on = F[:, 0] > 1.0                                  # 약 16% 만 켜짐
    y = np.where(on, 500.0 + 50 * F[:, 1], 0.0)[:, None].astype(np.float32)
    m = train(F, y, on[:, None].astype(np.int8), ["dummy"],
              max_iter=60, early_stopping=False, verbose=False)
    pred, prob = m.predict(F)
    # 켜진 창에서 0 이 아닌 값을 내야 한다 (전부 0 이면 실패)
    assert pred[on].mean() > 300.0, "회귀기가 '전부 0' 으로 수렴했습니다"
    # 꺼진 창은 게이팅으로 눌려야 한다
    assert pred[~on].mean() < 20.0, "게이팅이 동작하지 않습니다"


def test_re_metric_does_not_diverge_on_transition_samples():
    """켜짐 라벨인데 참 전력이 0 근처인 샘플이 RE 를 발산시키면 안 된다.

    실제로 드라이기 RE 가 31,177 로 나왔던 버그다.
    """
    apps = ["x"]
    y = np.array([[1000.0], [1000.0], [0.001]])       # 마지막이 전이 샘플
    p = np.array([[1010.0], [990.0], [50.0]])
    on = np.array([[True], [True], [True]])
    s = score_appliances(y, p, apps, {"x": 1000.0}, on_true=on, on_pred=on)[0]
    assert s.re_on < 1.0, f"RE 가 발산했습니다: {s.re_on}"
    assert s.n_re == 2, "전이 샘플이 RE 에서 제외되지 않았습니다"
    assert s.re_on_median == pytest.approx(0.01, abs=1e-6)
    # nMAE 는 참 전력과 무관하게 정격으로 나누므로 항상 유한하다
    assert np.isfinite(s.nmae_on)


def test_baseline_model_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    F = rng.standard_normal((400, 5)).astype(np.float32)
    on = (F[:, 0] > 0)[:, None].astype(np.int8)
    y = (on * 100.0).astype(np.float32)
    m = train(F, y, on, ["a"], max_iter=20, early_stopping=False, verbose=False)
    m.save(tmp_path / "m.pkl")
    m2 = BaselineModel.load(tmp_path / "m.pkl")
    assert np.allclose(m.predict(F)[0], m2.predict(F)[0])
    assert m2.appliances == ["a"] and len(m2.feature_names) == 84
