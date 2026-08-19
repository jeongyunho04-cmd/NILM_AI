"""
합성 / 증강 / 계통 시뮬레이터 / 대기전력 자동 검증 스위트
"""
import numpy as np
import pytest

from src.preprocessing.file_registry import get_low_load_appliances, is_periodic_duty
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.grid_simulator import GridSimulator, OBSERVED_VOLTAGE_CLUSTERS
from src.synthesis.augmentor import DataAugmentor
from src.synthesis.synthesizer import ApplianceSchedule, LoadSynthesizer
from src.synthesis.dataset import NILMBatchGenerator


@pytest.fixture(scope="module")
def segment_pool():
    return SegmentPool(npz_dir="processed_data/npz")


@pytest.fixture(scope="module")
def synthesizer(segment_pool):
    return LoadSynthesizer(segment_pool=segment_pool)


def test_segment_pool_loading_and_standby_profiles(segment_pool):
    app_types = segment_pool.get_appliance_types()
    assert len(app_types) >= 8
    assert "air_conditioner" in app_types
    assert "electiric_kettle" in app_types
    assert "minipc" in app_types

    ac_standby = segment_pool.get_standby_profile("air_conditioner")
    assert ac_standby.harmonics_ri.shape == (15, 2)
    assert ac_standby.power_w >= 0.0


def test_composite_eval_files_never_enter_pool(segment_pool):
    """test*.csv / nilm_*.csv 가 가전으로 둔갑하지 않아야 한다."""
    for app in segment_pool.get_appliance_types():
        assert not app.startswith("test"), f"복합 부하 파일이 가전으로 들어왔습니다: {app}"
        assert not app.startswith("nilm_"), f"복합 부하 파일이 가전으로 들어왔습니다: {app}"
        for act in segment_pool.appliance_activations[app]:
            assert not act.source_file.startswith("test")
            assert not act.source_file.startswith("nilm_")


def test_standby_power_is_physically_bounded(segment_pool):
    """대기전력 총합이 계측 보드 자체 소비를 기기 수만큼 중복 계상하지 않아야 한다.

    이전 구현은 보드 소비(1.4~2.4W)를 기기마다 더해 9대 합계 26.7W 라는,
    미니PC 아이들(9.8W)보다 큰 유령 대기전력을 만들었다.
    """
    total = sum(segment_pool.get_standby_profile(a).power_w
                for a in segment_pool.get_appliance_types())
    assert total < 15.0, f"대기전력 총합이 비현실적으로 큽니다: {total:.2f}W"

    # 기계식 스위치 기기는 꺼지면 회로가 끊겨 대기전력이 사실상 0 이어야 한다.
    for app in ["electiric_kettle", "fan", "laptop_charger"]:
        if app in segment_pool.get_appliance_types():
            p = segment_pool.get_standby_profile(app).power_w
            assert p < 0.5, f"{app} 의 대기전력이 물리적으로 과합니다: {p:.3f}W"


def test_standby_is_not_a_perfect_constant(segment_pool):
    """대기 상태가 완전한 상수면 '변하지 않는 성분 = 대기전력'이라는
    합성 데이터에만 존재하는 단서를 모델이 학습해 버린다."""
    profile = segment_pool.get_standby_profile("air_conditioner")
    if not profile.is_true_standby:
        pytest.skip("대기 회로가 없는 기기")
    _, p_series = segment_pool.sample_standby_series("air_conditioner", 600)
    assert p_series.std() > 0.0, "대기전력이 완전한 상수입니다"
    assert np.all(p_series >= 0.0), "대기전력에 음수가 있습니다"


def test_stochastic_standby_power_simulation(synthesizer):
    # 1. 전부 미플러그 - 계측계 자체 소비만 남아야 한다
    unplugged = {app: False for app in synthesizer.known_appliances}
    s_unplugged = synthesizer.synthesize_scenario(
        300, [], plugged_in_appliances=unplugged, include_noise=True, simulate_voltage_drop=False
    )
    p_unplugged = float(np.mean(s_unplugged.power_features[:, 0]))
    assert p_unplugged < 3.0, f"미플러그 상태 전력이 너무 큽니다: {p_unplugged:.2f}W"

    # 2. 에어컨/오븐만 대기 상태로 연결
    plugged = {app: False for app in synthesizer.known_appliances}
    plugged["air_conditioner"] = True
    plugged["oven"] = True
    s_standby = synthesizer.synthesize_scenario(
        300, [], plugged_in_appliances=plugged, include_noise=True, simulate_voltage_drop=False
    )
    p_standby = float(np.mean(s_standby.power_features[:, 0]))

    assert p_standby > p_unplugged
    assert np.all(s_standby.gt_is_on["air_conditioner"] == 0)
    assert np.all(s_standby.gt_is_on["oven"] == 0)
    assert np.all(s_standby.gt_target_power_w["air_conditioner"] == 0.0)
    # 대기전력은 활성전력이 아니라 대기 채널에 담겨야 한다
    assert np.all(s_standby.gt_is_plugged["air_conditioner"] == 1)
    assert np.mean(s_standby.gt_standby_power_w["air_conditioner"]) > 0.0


def test_power_decomposition_is_exact(synthesizer):
    """P_total = Σ활성 + Σ대기 + 계측계 소비 가 정확히 성립해야 한다."""
    plugged = {app: True for app in synthesizer.known_appliances}
    sample = synthesizer.synthesize_scenario(
        600,
        [ApplianceSchedule("electiric_kettle", start_cycle=60, duration_cycles=300)],
        plugged_in_appliances=plugged,
    )
    ok, max_err = sample.verify_power_decomposition(tolerance_w=0.01)
    assert ok, f"전력 분해가 맞지 않습니다. 최대 오차 {max_err:.4f}W"


def test_ground_truth_labels_are_mutually_consistent(synthesizer):
    """꺼진 시점의 정답 고조파는 0 이어야 한다.

    이전 구현은 gt_is_on=0, gt_target_power_w=0 인데 gt_harmonics_ri 에는
    대기 전류가 남아 있어 멀티태스크 학습에 모순된 지도신호를 주었다.
    """
    plugged = {app: True for app in synthesizer.known_appliances}
    sample = synthesizer.synthesize_scenario(
        600,
        [ApplianceSchedule("electiric_kettle", start_cycle=60, duration_cycles=300)],
        plugged_in_appliances=plugged,
    )
    for app in synthesizer.known_appliances:
        off = sample.gt_is_on[app] == 0
        if not off.any():
            continue
        assert np.all(sample.gt_target_power_w[app][off] == 0.0)
        assert np.abs(sample.gt_harmonics_ri[app][off]).sum() == 0.0, \
            f"{app}: 꺼진 구간에 정답 고조파가 남아 있습니다"
        # 활성 구간에는 대기전력이 따로 잡히면 안 된다 (활성 전력에 이미 포함)
        on = ~off
        assert np.all(sample.gt_standby_power_w[app][on] == 0.0)


def test_grid_simulator_voltage_drop_and_coupling():
    grid_sim = GridSimulator(
        nominal_voltage=220.0, r_grid=0.3, x_grid=0.05, voltage_variation_std=0.0,
        sag_rate_per_min=0.0,
    )

    zero_c = np.zeros((10, 15), dtype=np.complex64)
    v_bus, kappa = grid_sim.compute_voltage_drop(zero_c)
    assert np.allclose(v_bus, 220.0)
    assert np.allclose(kappa, 1.0)

    heavy_c = np.zeros((10, 15), dtype=np.complex64)
    heavy_c[:, 0] = 6.0 + 0j
    v_bus_heavy, kappa_heavy = grid_sim.compute_voltage_drop(heavy_c)
    assert np.all(v_bus_heavy < 219.0)
    assert np.all(kappa_heavy < 1.0)


def test_voltage_environment_covers_observed_range():
    """실측에서 관찰된 두 전압 무리(약 221V / 234V)를 모두 만들어야 한다."""
    grid_sim = GridSimulator()
    samples = [grid_sim.sample_environment().base_voltage_v for _ in range(600)]
    samples = np.array(samples)

    assert samples.min() > 200.0 and samples.max() < 250.0
    # 두 관측 무리 근처가 모두 나와야 한다
    for cluster in OBSERVED_VOLTAGE_CLUSTERS:
        near = np.abs(samples - cluster.mean_v) < 4.0
        assert near.sum() > 0, f"{cluster.name} 근처 전압이 생성되지 않았습니다"
    # 미측정 구간도 일부 탐색해야 일반화가 된다
    assert (samples < 215.0).sum() > 0 or (samples > 240.0).sum() > 0


def test_voltage_measurement_is_quantized_like_real_sensor(synthesizer):
    """실측 센서는 0.5초(30사이클)마다 전압을 갱신한다. 합성도 같아야 한다."""
    sample = synthesizer.synthesize_random_window(window_size_cycles=600)
    v = sample.v_bus
    # 30사이클 단위로 값이 유지되는지 확인
    for start in range(0, 600, 30):
        block = v[start:start + 30]
        assert np.allclose(block, block[0]), "전압이 프레임 안에서 변합니다"
    assert len(np.unique(v)) <= 20 + 1


def test_voltage_sag_metric_is_never_negative(synthesizer):
    """기저 전압이 220V 를 넘는 환경에서도 전압 강하 지표가 음수가 되면 안 된다."""
    for _ in range(20):
        sample = synthesizer.synthesize_random_window(window_size_cycles=300)
        assert sample.metadata["max_v_sag_v"] >= -0.01, \
            f"전압 강하 지표가 음수입니다: {sample.metadata['max_v_sag_v']}"


def test_data_augmentor_time_warping_and_scaling(segment_pool):
    augmentor = DataAugmentor()
    act = segment_pool.sample_activation("fan")
    target_len = act.duration_cycles * 2

    aug_act = augmentor.augment_activation(act, target_duration_cycles=target_len, power_scale=1.1)
    assert aug_act.duration_cycles == target_len
    assert aug_act.net_harmonics_ri.shape == (target_len, 15, 2)
    assert aug_act.net_harmonics_complex.shape == (target_len, 15)
    assert len(aug_act.target_power_w) == target_len
    assert aug_act.v_ref_v == act.v_ref_v


def test_periodic_load_preserves_duty_cycle(segment_pool):
    """오븐/핫플레이트를 2배로 늘일 때 서모스탯 주기가 늘어나면 안 된다."""
    assert is_periodic_duty("hotplate")
    augmentor = DataAugmentor()
    act = segment_pool.sample_activation("hotplate")
    assert act.periodic_duty

    target_len = act.duration_cycles * 2
    aug = augmentor.augment_activation(act, target_duration_cycles=target_len, power_scale=1.0)
    assert aug.duration_cycles == target_len

    # 순환 이어붙이기이므로 원본에 있던 값들이 그대로 재등장해야 한다
    # (리샘플링이었다면 중간값들이 새로 만들어진다)
    orig_levels = np.unique(np.round(act.target_power_w, 3))
    aug_levels = np.unique(np.round(aug.target_power_w, 3))
    assert np.isin(aug_levels, orig_levels).all(), "주기 부하가 리샘플링되었습니다"


def test_activation_onset_is_not_biased_to_window_start(synthesizer):
    """돌입 전류가 항상 윈도우 0번 인덱스에 몰리면 안 된다.

    이전에는 시작 시점을 max(0, start) 로 잘라 온셋의 52.8% 가 index 0 에 몰렸다.
    """
    onsets = []
    for _ in range(200):
        s = synthesizer.synthesize_random_window(window_size_cycles=600, max_concurrent_appliances=3)
        for app in s.active_appliances:
            on = s.gt_is_on[app]
            if on[0] == 1:
                continue  # 윈도우 시작 전부터 켜져 있던 경우
            rises = np.where(np.diff(on.astype(int)) == 1)[0]
            if len(rises):
                onsets.append(int(rises[0]) + 1)
    if len(onsets) < 20:
        pytest.skip("표본이 부족합니다")
    onsets = np.array(onsets)
    at_zero = float((onsets <= 1).mean())
    assert at_zero < 0.20, f"온셋이 윈도우 시작에 몰려 있습니다: {at_zero:.1%}"


def test_batch_generator_fast_throughput(segment_pool):
    batch_gen = NILMBatchGenerator(
        segment_pool=segment_pool,
        window_size_cycles=600,
        max_concurrent_appliances=3,
        target_mode="seq2point",
    )

    X, y_pow, y_state, y_on = batch_gen.generate_batch(batch_size=8)
    assert X.shape == (8, 33, 600)
    assert X.dtype == np.float32

    n_apps = len(batch_gen.appliance_list)
    assert y_pow.shape == (8, n_apps)
    assert y_state.shape == (8, n_apps)
    assert y_on.shape == (8, n_apps)
    assert not np.isnan(X).any()
    assert not np.isnan(y_pow).any()


def test_hard_negative_windows_are_generated(segment_pool):
    """대기전력 오탐 방지용 하드 네거티브가 실제로 섞여 나와야 한다."""
    batch_gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=300)
    d = batch_gen.generate_batch_dict(batch_size=64)

    recipes = set(d["recipe"].tolist())
    assert "standby_only" in recipes or "low_load_among_standby" in recipes

    # standby_only 윈도우는 정답이 전부 OFF/0W 인데 관측 전력은 0 이 아니어야 한다
    idx = np.where(d["recipe"] == "standby_only")[0]
    if len(idx):
        i = idx[0]
        assert d["y_on"][i].sum() == 0, "대기 전용 윈도우에 켜진 기기가 있습니다"
        assert d["y_power"][i].sum() == 0.0
        assert d["y_plugged"][i].sum() > 0, "대기 전용 윈도우인데 꽂힌 기기가 없습니다"

    assert d["y_plugged"].shape == d["y_on"].shape
    assert d["y_standby_power"].shape == d["y_power"].shape
    # 대기전력은 활성 기기에는 잡히지 않아야 한다
    assert np.all(d["y_standby_power"][d["y_on"] == 1] == 0.0)


def test_low_load_among_standby_is_the_confusable_case(synthesizer):
    """저부하 1대 + 대기전력 최대 상황이 실제로 헷갈릴 만한 크기인지 확인한다."""
    low_load = get_low_load_appliances()
    assert low_load, "저부하 가전이 등록되어 있지 않습니다"

    sample = synthesizer.synthesize_low_load_among_standby_window(600)
    active = [a for a in sample.appliance_types if sample.gt_is_on[a].any()]
    assert len(active) == 1, f"활성 기기가 1대가 아닙니다: {active}"
    assert active[0] in low_load

    standby_total = float(np.mean(sum(sample.gt_standby_power_w[a] for a in sample.appliance_types)))
    active_total = float(np.mean(sample.gt_target_power_w[active[0]]))
    # 대기전력이 존재하고, 활성 기기도 존재하는 상황이어야 학습 가치가 있다
    assert standby_total > 0.0
    assert active_total > 0.0
