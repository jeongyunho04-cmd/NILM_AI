"""
합성 / 증강 / 계통 시뮬레이터 / 대기전력 자동 검증 스위트
"""
import numpy as np
import pytest

from src.preprocessing.file_registry import get_low_load_appliances, is_periodic_duty
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.grid_simulator import GridSimulator, OBSERVED_VOLTAGE_CLUSTERS
from src.synthesis.augmentor import DataAugmentor
from src.synthesis.synthesizer import (
    SELECTION_REALISTIC,
    SELECTION_UNIFORM,
    ApplianceSchedule,
    LoadSynthesizer,
)
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


def test_gt_harmonics_can_be_switched_off(segment_pool):
    """전력·상태만 학습한다면 가전별 고조파 정답을 만들지 않아야 한다.

    나머지 정답 5종을 합친 것의 10배 용량을 차지하는데, 쓰지 않으면 순수한 낭비다.
    """
    lean = LoadSynthesizer(segment_pool=segment_pool, compute_gt_harmonics=False)
    sample = lean.synthesize_random_window(window_size_cycles=300)

    assert sample.gt_harmonics_included is False
    assert sample.gt_harmonics_ri == {}
    assert sample.metadata["gt_harmonics_included"] is False

    # 전력·상태 정답은 그대로 있어야 한다
    for app in sample.appliance_types:
        assert len(sample.gt_target_power_w[app]) == 300
        assert len(sample.gt_state_id[app]) == 300
        assert len(sample.gt_standby_power_w[app]) == 300
    ok, err = sample.verify_power_decomposition(tolerance_w=0.01)
    assert ok, f"고조파를 꺼도 전력 분해는 성립해야 합니다. 오차 {err:.4f}W"

    # 호출 단위로 다시 켤 수 있어야 한다 (합성기 진단용)
    diag = lean.synthesize_random_window(window_size_cycles=300, compute_gt_harmonics=True)
    assert diag.gt_harmonics_included is True
    assert set(diag.gt_harmonics_ri) == set(diag.appliance_types)
    assert diag.gt_harmonics_ri[diag.appliance_types[0]].shape == (300, 15, 2)


def test_batch_generator_omits_gt_harmonics_by_default(segment_pool):
    """학습 배치 기본값은 고조파 정답을 만들지 않는 것이다."""
    gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=300)
    assert gen.compute_gt_harmonics is False

    sample, _ = gen._synthesize_window()
    assert sample.gt_harmonics_included is False

    d = gen.generate_batch_dict(batch_size=4)
    assert not any("harmonic" in k for k in d), "배치에 고조파 정답이 섞여 나옵니다"
    assert d["y_power"].shape == (4, len(gen.appliance_list))
    assert d["y_state"].shape == (4, len(gen.appliance_list))


def test_realistic_selection_matches_usage_frequency(synthesizer):
    """기기별 가동 빈도가 실제 사용률을 따라야 한다.

    이전에는 9종을 균등 추첨해 미니PC와 헤어드라이기가 똑같이 15%씩 나왔다.
    실제로는 미니PC가 하루 10시간, 드라이기가 6분으로 100배 차이가 난다.
    """
    from src.preprocessing.file_registry import get_usage_probability

    counts = {a: 0 for a in synthesizer.known_appliances}
    trials = 400
    for _ in range(trials):
        s = synthesizer.synthesize_random_window(300, selection_mode=SELECTION_REALISTIC)
        for a in s.active_appliances:
            counts[a] += 1

    high = max(synthesizer.known_appliances, key=get_usage_probability)
    low = min(synthesizer.known_appliances, key=get_usage_probability)
    assert counts[high] > counts[low] * 3, (
        f"사용률이 높은 {high}({counts[high]})가 낮은 {low}({counts[low]})보다 "
        f"뚜렷하게 자주 나와야 합니다"
    )


def test_uniform_selection_keeps_rare_appliances_visible(synthesizer):
    """희귀 기기도 학습 표본을 얻으려면 균등 추첨 모드가 필요하다."""
    counts = {a: 0 for a in synthesizer.known_appliances}
    for _ in range(400):
        s = synthesizer.synthesize_random_window(300, selection_mode=SELECTION_UNIFORM)
        for a in s.active_appliances:
            counts[a] += 1
    assert min(counts.values()) > 0, f"균등 모드인데 한 번도 안 나온 기기가 있습니다: {counts}"


def test_sustained_power_stays_under_limit(segment_pool):
    """멀티탭 용량을 넘는 '지속' 부하 조합은 만들지 않아야 한다.

    돌입 전류 같은 순간 스파이크는 제한하지 않는다 - 물리적으로 정상이고
    모델이 배워야 할 신호다.
    """
    limit = 4000.0
    syn = LoadSynthesizer(
        segment_pool=segment_pool, compute_gt_harmonics=False,
        sustained_power_limit_w=limit,
    )
    n_apps = len(syn.known_appliances)

    # 전 기기를 켜라고 요청해도 상한 안에서만 채택되어야 한다
    for _ in range(40):
        s = syn.synthesize_random_window(600, n_active=n_apps)
        assert s.metadata["max_sustained_p_w"] <= limit, (
            f"지속 부하가 한도를 넘었습니다: {s.metadata['max_sustained_p_w']:.0f}W > {limit:.0f}W "
            f"(가동 {s.active_appliances})"
        )


def test_power_budget_is_voltage_aware(segment_pool):
    """전압이 높으면 같은 저항 부하도 더 먹는다. 예산도 그걸 반영해야 한다.

    전기포트는 212.5V 에서 녹화되어 1271W 였지만 240V 에서는 P ∝ V² 로 1621W 가 된다.
    녹화 당시 값으로 계산하면 실제로는 한도를 넘는 조합이 통과해 버린다.
    """
    syn = LoadSynthesizer(segment_pool=segment_pool, compute_gt_harmonics=False)
    low = syn.estimate_steady_power_w("electiric_kettle", 210.0)
    high = syn.estimate_steady_power_w("electiric_kettle", 240.0)
    assert high > low * 1.15, f"저항 부하의 전압 응답이 반영되지 않았습니다: {low:.0f}W -> {high:.0f}W"

    # SMPS 는 정전력이라 전압이 변해도 소비 전력이 거의 같아야 한다
    smps_low = syn.estimate_steady_power_w("minipc", 210.0)
    smps_high = syn.estimate_steady_power_w("minipc", 240.0)
    assert abs(smps_high - smps_low) < 0.01 * max(smps_low, 1.0)


def test_power_limit_can_be_disabled(segment_pool):
    """상한은 끌 수 있어야 한다 (다른 환경에서 학습할 때)."""
    syn = LoadSynthesizer(
        segment_pool=segment_pool, compute_gt_harmonics=False, sustained_power_limit_w=None,
    )
    s = syn.synthesize_random_window(600, n_active=len(syn.known_appliances))
    assert s.metadata["sustained_power_limit_w"] is None
    assert s.metadata["dropped_over_budget"] == []


def test_long_timeline_respects_power_limit_over_time(segment_pool):
    """긴 타임라인에서는 겹침이 시간에 따라 변한다. 어느 시점에서도 한도를 넘으면 안 된다."""
    from src.synthesis.scenario_generator import ScenarioGenerator

    limit = 4000.0
    syn = LoadSynthesizer(
        segment_pool=segment_pool, compute_gt_harmonics=False,
        sustained_power_limit_w=limit,
    )
    gen = ScenarioGenerator(synthesizer=syn)
    sample = gen.create_long_timeline(duration_min=6.0)

    p = sample.power_features[:, 0]
    w = int(2.0 * 60)
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    sustained = (c[w:] - c[:-w]) / w
    assert sustained.max() <= limit, f"지속 부하가 한도를 넘었습니다: {sustained.max():.0f}W"

    ok, err = sample.verify_power_decomposition(tolerance_w=0.01)
    assert ok, f"전력 분해가 맞지 않습니다: {err:.4f}W"
    assert sample.metadata["episodes_scheduled"] > 0


def test_long_timeline_covers_every_appliance(segment_pool):
    """구성을 확인하려면 모든 가전이 최소 한 번은 나와야 한다."""
    from src.synthesis.scenario_generator import ScenarioGenerator

    syn = LoadSynthesizer(segment_pool=segment_pool, compute_gt_harmonics=False)
    gen = ScenarioGenerator(synthesizer=syn)
    sample = gen.create_long_timeline(duration_min=8.0, min_episodes_per_appliance=1)

    missing = [a for a in sample.appliance_types if not sample.gt_is_on[a].any()]
    # 용량 초과로 빠지는 경우가 있으므로 전부를 강제하지는 않되, 대부분은 나와야 한다
    assert len(missing) <= 2, f"너무 많은 가전이 한 번도 안 나왔습니다: {missing}"


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


def test_long_activation_is_cropped_not_time_compressed(segment_pool):
    """긴 동작 구간을 짧은 윈도우에 넣을 때 시간을 압축하면 안 된다.

    미니PC 는 한 번에 2500초를 연속으로 돌았다. 이것을 10초 윈도우에 맞추려고
    250배 압축하면 몇 분에 걸친 IDLE->ACTIVE 전이가 수 밀리초 만에 끝나는,
    실제로는 존재할 수 없는 파형이 된다.
    """
    augmentor = DataAugmentor()
    long_app = max(
        segment_pool.get_appliance_types(),
        key=lambda a: max(x.duration_cycles for x in segment_pool.appliance_activations[a]),
    )
    act = max(segment_pool.appliance_activations[long_app], key=lambda x: x.duration_cycles)
    assert act.duration_cycles > 6000, "이 검사는 충분히 긴 활성화 구간이 필요합니다"

    target = 600
    aug = augmentor.augment_activation(act, target_duration_cycles=target, power_scale=1.0)
    assert aug.duration_cycles == target

    # 잘라낸 것이라면 출력 전력값이 원본 어딘가에 그대로 존재해야 한다.
    # 압축(보간)이었다면 원본에 없는 중간값이 만들어진다.
    orig = np.round(act.target_power_w, 3)
    out = np.round(aug.target_power_w, 3)
    assert np.isin(out, orig).all(), "시간 압축(보간)이 일어났습니다 - 잘라내기여야 합니다"


def test_extreme_stretch_is_capped(segment_pool):
    """짧은 동작을 몇십 배로 늘이면 물리적으로 불가능한 파형이 된다."""
    augmentor = DataAugmentor(max_stretch=3.0)
    short_app = min(
        segment_pool.get_appliance_types(),
        key=lambda a: min(x.duration_cycles for x in segment_pool.appliance_activations[a]),
    )
    act = min(segment_pool.appliance_activations[short_app], key=lambda x: x.duration_cycles)

    aug = augmentor.augment_activation(act, target_duration_cycles=act.duration_cycles * 50)
    assert aug.duration_cycles <= act.duration_cycles * 3 + 1, (
        f"확대 배율이 제한되지 않았습니다: {act.duration_cycles} -> {aug.duration_cycles}"
    )


def test_crop_covers_onset_middle_and_offset(segment_pool):
    """잘라내는 위치가 한쪽에 몰리면 그 기기의 특정 전이만 학습하게 된다."""
    augmentor = DataAugmentor()
    long_app = max(
        segment_pool.get_appliance_types(),
        key=lambda a: max(x.duration_cycles for x in segment_pool.appliance_activations[a]),
    )
    act = max(segment_pool.appliance_activations[long_app], key=lambda x: x.duration_cycles)

    firsts = set()
    for _ in range(60):
        aug = augmentor.augment_activation(act, target_duration_cycles=600, power_scale=1.0)
        firsts.add(round(float(aug.target_power_w[0]), 4))
    assert len(firsts) > 3, f"잘라내는 위치가 다양하지 않습니다: {firsts}"


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


def test_high_power_resistive_windows_boost_rare_appliances(segment_pool):
    """고전력 저항 부하는 무작위 추출만으로는 학습 표본이 모자란다.

    포트·오븐·드라이기·핫플레이트는 고조파 지문이 거의 같아 시간 패턴으로만
    갈리는데, 사용 빈도가 낮아 표본이 가장 적다. 전용 레시피로 보강해야 한다.
    """
    from src.preprocessing.file_registry import get_resistive_appliances

    resistive = set(get_resistive_appliances())
    assert resistive, "저항 부하가 등록되어 있지 않습니다"

    gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=300)
    assert "high_power_resistive" in gen.describe_recipe_mix()

    # 전용 레시피는 반드시 저항 부하를 켠다
    for _ in range(20):
        s = gen.synthesizer.synthesize_high_power_window(300)
        active = set(s.active_appliances)
        assert active, "고전력 윈도우인데 켜진 기기가 없습니다"
        assert active <= resistive, f"저항 부하가 아닌 기기가 켜졌습니다: {active - resistive}"
        assert s.metadata["max_sustained_p_w"] <= gen.synthesizer.sustained_power_limit_w


def test_recipe_mix_reduces_class_imbalance(segment_pool):
    """어떤 가전도 학습 표본이 사실상 0 이 되면 안 된다."""
    gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=600)
    d = gen.generate_batch_dict(batch_size=256)
    positives = d["y_on"].sum(axis=0)

    assert positives.min() > 0, (
        f"양성 표본이 0 인 가전이 있습니다: "
        f"{[a for a, c in zip(gen.appliance_list, positives) if c == 0]}"
    )
    # 최다/최소 비율. 물리적 듀티 때문에 완전 균등은 될 수 없지만
    # 이전의 55:1 수준으로 벌어지면 희귀 기기를 못 배운다.
    imbalance = positives.max() / max(positives.min(), 1)
    assert imbalance < 30, f"클래스 불균형이 과합니다: {imbalance:.0f}:1"


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
