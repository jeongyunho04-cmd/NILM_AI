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


def test_worker_count_does_not_change_the_generated_training_set():
    """`--seed` 가 같으면 워커 수와 무관하게 **같은 학습셋**이 나와야 한다.

    설계 문서 12.11절이 남긴 숙제다. 워커 번호로 시드하고 `imap_unordered` 로
    거두면 (a) 어느 RNG 스트림이 어느 청크를 만드는지와 (b) 이어붙이는 순서가
    실행마다 달라진다. 그 탓에 같은 명령을 두 번 돌린 GBM 이 MAE 0.02W /
    오븐->포트 2.2%p 씩 흔들렸다 (12.7절).

    이제 청크 **번호**로 시드하고 `imap`(순서 보장)으로 거둔다. 어느 워커가
    집어 가든 같은 창이 나오고 같은 자리에 붙는다.
    """
    from src.baseline.train import build_training_set

    kw = dict(n_windows=300, chunk=100, seed=11)
    single, _, _, _ = build_training_set(n_workers=1, **kw)
    multi, _, _, _ = build_training_set(n_workers=3, **kw)
    wider, _, _, _ = build_training_set(n_workers=5, **kw)

    assert np.array_equal(single, multi), "워커 1개와 3개가 다른 학습셋을 만들었습니다"
    assert np.array_equal(multi, wider), "워커 3개와 5개가 다른 학습셋을 만들었습니다"


def test_chunk_seed_is_reproducible_and_decorrelated():
    """청크 시드는 재현되어야 하고, 이웃한 청크가 같은 스트림을 쓰면 안 된다."""
    from src.synthesis.dataset import chunk_seed

    assert chunk_seed(0, 0) == chunk_seed(0, 0)
    seeds = [chunk_seed(3, i) for i in range(64)]
    assert len(set(seeds)) == 64, "청크 시드가 충돌합니다"
    assert chunk_seed(0, 1) != chunk_seed(1, 0), "seed_base 와 index 가 뒤섞였습니다"


def test_trunk_input_layout_is_frozen():
    """`trunk` 입력의 **연결 순서**가 조용히 바뀌면 안 된다 (설계 문서 12.21.4절).

    실제로 겪은 사고다. `fine_dropout` 을 넣으면서 창 통계 4개를 세밀 2개와 광역
    2개로 쪼개 `feats` 중간에 삽입했더니 순서가 바뀌었고, 저장된 체크포인트가
    **뒤섞인 입력**을 받았다. 형상이 맞고 오류도 안 나서 `cnn_v17` 의 test_4
    정답률이 1.2% -> 0.0% 로 떨어진 것을 성능 변화로 읽을 뻔했다.

    여기서는 각 구간이 무엇인지 고정한다. 순서를 바꿔야 할 이유가 생기면 이
    테스트가 먼저 깨지고, 그때 옛 체크포인트를 어떻게 할지 결정하게 된다.
    """
    import torch
    from src.model.inputs import FINE_CHANNELS
    from src.model.net import NILMNet

    m = NILMNet(["a", "b"], [2, 2])
    c1, c2, w2 = 64, 128, 64
    # [원본타깃, tap0, tap1, 깊은평균, 깊은최대, 깊은타깃, 광역평균, 창통계4]
    expected = [FINE_CHANNELS, c1, c1, c2, c2, c2, w2, 4]
    assert m.trunk[0].in_features == sum(expected) == 618

    # 세밀 유래 차원은 앞에서부터 연속이고, 창통계의 앞 2개가 추가로 세밀이다.
    mask = m.fine_dim_mask
    n_fine_conv = FINE_CHANNELS + c1 + c1 + c2 + c2 + c2      # 552 - 2 = 550
    assert mask[:n_fine_conv].sum() == n_fine_conv, "세밀 conv 구간이 앞에 연속이 아닙니다"
    assert mask[n_fine_conv:n_fine_conv + w2].sum() == 0, "광역 평균 자리가 어긋났습니다"
    assert list(mask[-4:]) == [1.0, 1.0, 0.0, 0.0], (
        "창통계 순서가 (fp_max, fp_min, wp_max, wp_mean) 이 아닙니다"
    )
    assert int(mask.sum()) == 552

    # 버퍼가 state_dict 에 들어가면 옛 체크포인트 로딩이 깨진다
    assert "fine_dim_mask" not in m.state_dict(), (
        "fine_dim_mask 는 persistent=False 여야 합니다 (옛 체크포인트 호환)"
    )


def test_fine_dropout_is_off_at_inference_and_masks_only_fine():
    """드롭아웃은 학습에서만 걸리고, 광역 차원은 건드리지 않아야 한다."""
    import torch
    from src.model.net import NILMNet

    m = NILMNet(["a", "b"], [2, 2], fine_dropout=1.0)   # 항상 가린다
    f, w = torch.randn(4, 38, 600), torch.randn(4, 12, 120)

    m.eval()
    with torch.no_grad():
        a, b = m(f, w)["on_logit"], m(f, w)["on_logit"]
    assert torch.allclose(a, b), "추론에서 드롭아웃이 걸리고 있습니다"

    # 학습 모드에서 세밀을 전부 가리면 세밀 conv 에 기울기가 안 가야 한다
    m.train()
    m(f, w)["power"].sum().backward()
    g_fine = sum(p.grad.abs().sum().item() for p in m.fine.parameters())
    g_wide = sum(p.grad.abs().sum().item() for p in m.wide.parameters())
    assert g_fine == 0.0, f"가려진 창에서 세밀에 기울기가 갔습니다: {g_fine}"
    assert g_wide > 0.0, "광역에 기울기가 안 갔습니다"
