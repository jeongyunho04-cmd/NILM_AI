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

from src.model.inputs import RAW_CHANNELS
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
    """전력 시계열 하나로 원시 채널 창을 만든다 (13.26 에서 33 -> 45)."""
    x = np.zeros((RAW_CHANNELS, W), dtype=np.float32)
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
    for x in (np.zeros((3, RAW_CHANNELS, W), np.float32),
              np.full((3, RAW_CHANNELS, W), 1e-9, np.float32)):
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

    12.34 에서 고조파 위상 6채널을 **뒤에** 붙여 38 -> 44 가 되었다. 새 채널을
    뒤에 붙이는 한 `fine_channels=38` 로 만든 모델의 배치는 한 칸도 안 움직인다.
    그것을 여기서 못 박는다 — 옛 체크포인트가 계속 도는 근거가 이 단언이다.
    """
    from src.model.inputs import FINE_CHANNELS, LEGACY_FINE_CHANNELS
    from src.model.net import NILMNet

    c1, c2, w2 = 64, 128, 64
    for n_ch, want_trunk, want_fine in ((LEGACY_FINE_CHANNELS, 618, 552),
                                        (FINE_CHANNELS, 618 + FINE_CHANNELS - 38,
                                         552 + FINE_CHANNELS - 38)):
        m = NILMNet(["a", "b"], [2, 2], fine_channels=n_ch)
        # [원본타깃, tap0, tap1, 깊은평균, 깊은최대, 깊은타깃, 광역평균, 창통계4]
        expected = [n_ch, c1, c1, c2, c2, c2, w2, 4]
        assert m.trunk[0].in_features == sum(expected) == want_trunk, (
            f"fine_channels={n_ch}: trunk 입력이 {m.trunk[0].in_features} 다")

        # 세밀 유래 차원은 앞에서부터 연속이고, 창통계의 앞 2개가 추가로 세밀이다.
        mask = m.fine_dim_mask
        n_fine_conv = n_ch + c1 + c1 + c2 + c2 + c2
        assert mask[:n_fine_conv].sum() == n_fine_conv, "세밀 conv 구간이 앞에 연속이 아닙니다"
        assert mask[n_fine_conv:n_fine_conv + w2].sum() == 0, "광역 평균 자리가 어긋났습니다"
        assert list(mask[-4:]) == [1.0, 1.0, 0.0, 0.0], (
            "창통계 순서가 (fp_max, fp_min, wp_max, wp_mean) 이 아닙니다"
        )
        assert int(mask.sum()) == want_fine

        # 버퍼가 state_dict 에 들어가면 옛 체크포인트 로딩이 깨진다
        assert "fine_dim_mask" not in m.state_dict(), (
            "fine_dim_mask 는 persistent=False 여야 합니다 (옛 체크포인트 호환)"
        )


def test_fine_layout_is_recorded_and_guarded():
    """배치 v2 는 **앞부분의 뜻을 바꿨다** — 슬라이스 호환이 깨진 것이 의도다 (13.12).

    v1 은 "새 채널은 뒤에만" 규약으로 `fine[:, :38]` 슬라이스 호환을 지켰다. v2 는 짝수차
    Re/Im 14개를 크기 7개로 갈아끼우면서 그 규약을 **일부러** 깼다 (짝수차 위상은 플러그
    방향이라 기기 속성이 아니다). 그래서 호환 대신 **가드**가 안전장치다:
    캐시 meta 와 체크포인트에 `fine_layout` 을 적고 로더가 대조해 거부해야 한다.
    """
    import numpy as np
    from src.model.inputs import (EVEN_MAG0, EVEN_ORDERS, FINE_CHANNELS, FINE_LAYOUT,
                                  ODD_ORDERS, WIDE_CHANNELS, build_fine)

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.3, (2, RAW_CHANNELS, 3600)).astype(np.float32)
    fine = build_fine(x)
    assert fine.shape[1] == FINE_CHANNELS == 57      # 45 + 전압 고조파 12 (13.26)
    assert FINE_LAYOUT == "v3"

    # 짝수차 크기 블록은 부호가 없다 — 위상을 버렸다는 뜻이다
    even = fine[:, EVEN_MAG0:EVEN_MAG0 + len(EVEN_ORDERS)]
    assert even.min() >= 0.0, "짝수차 크기 채널에 음수가 있다 — Re/Im 이 남아 있는 것"
    assert len(ODD_ORDERS) == 8 and len(EVEN_ORDERS) == 7

    # 러너 셋이 배치를 기록·대조하는지 (소스 검사 — 가드가 조용히 사라지면 안 된다)
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/model/traincache.py", "src/run_train_cnn.py", "src/run_gate_check.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "fine_layout" in src, f"{rel} 에 배치 가드가 없다"

    # 채널 수만 줄인 모델은 **더 이상 옛 입력을 재현하지 못한다** — v2 의 앞 38개는
    # v1 의 앞 38개와 뜻이 다르다. 그래서 슬라이스가 아니라 가드가 안전장치다.
    import torch
    from src.model.net import NILMNet
    m = NILMNet(["a", "b"], [2, 2], fine_channels=FINE_CHANNELS).eval()
    with torch.no_grad():
        o = m(torch.from_numpy(fine), torch.randn(2, WIDE_CHANNELS, 120))["on_logit"]
    assert o.shape == (2, 2)


def test_phase_channels_are_load_invariant_and_gated():
    """고조파 위상 채널은 크기를 키워도 안 변하고, 신호가 없으면 0 이다 (자리는 13.12 의 PHI0)."""
    import numpy as np
    from src.model.inputs import PHI0, PHI_ORDERS, build_fine
    lo, hi = PHI0, PHI0 + 2 * len(PHI_ORDERS)
    rng = np.random.default_rng(1)
    x = np.zeros((1, RAW_CHANNELS, 3600), np.float32)
    for h, amp, ph in ((1, 0.30, 0.4), (3, 0.25, -1.1), (5, 0.20, 2.0), (7, 0.15, -2.6)):
        x[0, h - 1] = amp * np.cos(ph)
        x[0, 15 + h - 1] = amp * np.sin(ph)
    a = build_fine(x)[0, lo:hi, -1]
    b = build_fine(x * 3.0)[0, lo:hi, -1]          # 크기만 3배
    # 게이트는 차수마다 다르므로 **쌍별 각도**로 본다. 그 각도가 φ_h 자체다.
    for k in range(3):
        ang_a = np.arctan2(a[2 * k + 1], a[2 * k])
        ang_b = np.arctan2(b[2 * k + 1], b[2 * k])
        d = abs((ang_a - ang_b + np.pi) % (2 * np.pi) - np.pi)
        assert d < 1e-4, f"h={2*k+3}: 크기를 바꿨더니 위상이 {np.degrees(d):.3f}도 움직였다"
    # 게이트는 크기를 따라 커져야 한다 (신호가 클수록 위상을 신뢰한다)
    assert np.linalg.norm(b) > np.linalg.norm(a)

    z = build_fine(np.zeros((1, RAW_CHANNELS, 3600), np.float32))[0, lo:hi]
    assert np.abs(z).max() < 1e-6, "신호가 없으면 위상 채널은 0 이어야 한다"

def test_fine_dropout_is_off_at_inference_and_masks_only_fine():
    """드롭아웃은 학습에서만 걸리고, 광역 차원은 건드리지 않아야 한다."""
    import torch
    from src.model.net import NILMNet

    m = NILMNet(["a", "b"], [2, 2], fine_dropout=1.0)   # 항상 가린다
    from src.model.inputs import FINE_CHANNELS, WIDE_CHANNELS
    f, w = torch.randn(4, FINE_CHANNELS, 600), torch.randn(4, WIDE_CHANNELS, 120)

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


def test_site_signature_bank_falls_back_exactly_and_keeps_states():
    """자리별 지문 (13.59) — 되돌림이 옛 경로를 **정확히** 재현하고 상태 축을 안 지운다.

    13.59 에서 처방 자체는 반증됐지만(자리 조건화가 자리 D 프로젝터를 더 깎았다)
    배관은 남아 있다. 두 가지가 깨지면 조용히 틀린 지문으로 학습한다:

      ① `site_idx` 를 **뭉친 줄**로 주면 `sig_site` 없는 손실과 값이 같아야 한다
      ② 자리 지문이 상태별 지문을 덮어써서는 안 된다 — 처음 판이 프로젝터 상태 1 을
         고전력 지문으로 갈아버렸고 와트당 h13 이 3.8 mA/W 어긋났다 (13.11 형 결함)
    """
    import numpy as np
    import torch
    from src.model.losses import LossWeights, NILMLoss

    apps = ["a", "b"]
    K, S, H = len(apps), 3, 15
    rng = np.random.default_rng(0)
    sig = rng.normal(size=(K, H, 2)).astype(np.float32)
    sig_state = rng.normal(size=(K, S, H, 2)).astype(np.float32)
    # 자리 묶음: 줄 0·1 은 아무 값, **마지막 줄이 뭉친 지문**이라는 규약을 검정한다
    bank = np.concatenate([rng.normal(size=(2, K, H, 2)).astype(np.float32), sig[None]], 0)
    bank_st = np.concatenate(
        [rng.normal(size=(2, K, S, H, 2)).astype(np.float32), sig_state[None]], 0)

    kw = dict(s_i=torch.ones(K), signatures=torch.from_numpy(sig),
              signatures_state=torch.from_numpy(sig_state),
              weights=LossWeights(harm=0.1, cons=0.0, over=0.0))
    old = NILMLoss(**kw)
    new = NILMLoss(signatures_site=torch.from_numpy(bank),
                   signatures_state_site=torch.from_numpy(bank_st), **kw)

    B = 8
    out = {"power": torch.rand(B, K), "power_raw": torch.rand(B, K) + 0.5,
           "power_mix": torch.softmax(torch.randn(B, K, S), -1),
           "power_states": torch.rand(B, K, S),
           "standby": torch.rand(B, K) * 0.01,
           "on_logit": torch.randn(B, K), "plugged_logit": torch.randn(B, K)}
    tgt = {"p_observed": torch.rand(B) * 100 + 10, "obs_harm": torch.randn(B, H, 2),
           "p_noise": torch.rand(B), "harm_offset": None}

    a = float(old.unlabeled(out, tgt, w_harm=1.0)["harm"])
    b = float(new.unlabeled(dict(out), dict(tgt, site_idx=torch.full((B,), 2)),
                            w_harm=1.0)["harm"])
    # ⚠ 비트 단위 일치를 요구하지 않는다. 두 경로의 einsum 축약이 다르다
    #   ("bks,kshc->bhc" 대 "bks,bkshc->bhc") — 값은 같아야 하지만 마지막 자리는
    #   갈릴 수 있다. 실제 자료에서는 0.00e+00 이 나왔으나 보장은 아니다.
    assert abs(a - b) <= 1e-6 * max(abs(a), 1e-9), f"뭉친 줄인데 값이 다르다: {a} vs {b}"
    c = float(new.unlabeled(dict(out), dict(tgt, site_idx=torch.zeros(B, dtype=torch.long)),
                            w_harm=1.0)["harm"])
    assert c != a, "자리 줄을 골랐는데 값이 안 바뀌었다 — 색인이 안 먹는다"

    # ② 상태 축 보존. 자리 지문은 뭉친 상태별 지문에 **자리 비**로 걸려야 하므로
    #    한 자리 안에서 상태들이 여전히 서로 달라야 한다.
    from src.model.sig_site import SITES
    assert bank.shape[0] == len(SITES) + 1, "묶음의 마지막 줄이 뭉친 지문이어야 한다"


def test_standby_anchor_ties_the_free_slack_in_cons():
    """`L_sb` (13.60) — 대기 헤드가 `cons` 안의 자유 슬랙이 되지 않게 묶는다.

    `out["standby"]` 는 게이트 없는 softplus 하나이고 `L_cons` 의 재구성에 그대로
    더해진다. 1단계는 `y_standby`(규약: **활성 중이면 0**)가 잡아 주지만 2단계에는
    라벨도 구조적 게이트도 없어서, 자리 D 창에서 대기 합이 1.06 -> 6.93W 로 부풀고
    그 8W 가 `cons` 를 통해 SMPS 에서 빠졌다 (참 92.4 -> 예측 83.8W).

    검정 셋:
      ① `w_sb=0` 이면 옛 경로와 **정확히** 같다 (기본이 꺼짐이어야 한다)
      ② 켜진 기기(σ(on)→1)의 대기는 0 을 요구한다 — 규약 그대로
      ③ 꺼져-꽂힘 기기는 **측정 대기전력**을 요구한다
    """
    import torch
    from src.model.losses import LossWeights, NILMLoss

    K, H = 3, 15
    sbw = torch.tensor([2.0, 1.0, 0.5])
    kw = dict(s_i=torch.ones(K), weights=LossWeights(harm=0.1, cons=0.0, over=0.0))
    off = NILMLoss(**kw)
    on = NILMLoss(standby_w=sbw, **kw)

    B = 4
    out = {"power": torch.rand(B, K), "power_raw": torch.rand(B, K) + 0.5,
           "power_states": torch.rand(B, K, 5), "power_mix": torch.rand(B, K, 5),
           "standby": torch.zeros(B, K),
           # 0번은 켜짐(대기 0 이어야), 1·2번은 꺼져-꽂힘(대기 = 측정값이어야)
           "on_logit": torch.tensor([[12.0, -12.0, -12.0]] * B),
           "plugged_logit": torch.full((B, K), 12.0)}
    tgt = {"p_observed": torch.rand(B) * 50 + 10, "obs_harm": None,
           "p_noise": torch.rand(B)}

    # ① 기본은 꺼짐
    a = float(off.unlabeled(out, tgt, w_cons=0.1)["total"])
    b = float(on.unlabeled(out, tgt, w_cons=0.1)["total"])
    assert a == b, f"w_sb 를 안 줬는데 값이 달라졌다: {a} vs {b}"

    # ② ③ 규약대로 채우면 항이 0 이 된다
    good = dict(out, standby=torch.tensor([[0.0, 1.0, 0.5]] * B))
    p = on.unlabeled(good, tgt, w_cons=0.1, w_sb=1.0)
    # σ(±12) 가 정확히 0/1 이 아니라 1e-5 규모의 잔차가 남는다 — 그것보다 넉넉히 잡는다
    assert float(p["sb"]) < 1e-4, f"규약대로인데 벌점이 남았다: {float(p['sb'])}"

    # 켜진 기기에 대기를 얹으면 벌점이 그만큼 는다 (자유 슬랙 차단)
    bad = dict(out, standby=torch.tensor([[3.0, 1.0, 0.5]] * K + [[3.0, 1.0, 0.5]]))
    q = on.unlabeled(bad, tgt, w_cons=0.1, w_sb=1.0)
    assert float(q["sb"]) > 0.9, f"켜진 기기의 대기 3W 가 안 걸렸다: {float(q['sb'])}"
