"""
합성 / 증강 / 계통 시뮬레이터 / 대기전력 자동 검증 스위트
"""
import numpy as np

from src.model.inputs import RAW_CHANNELS
import pytest

from src.preprocessing.file_registry import (
    get_low_load_appliances,
    is_low_load,
    is_periodic_duty,
)
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.grid_simulator import GridSimulator, OBSERVED_VOLTAGE_CLUSTERS
from src.synthesis.augmentor import DataAugmentor
from src.synthesis.synthesizer import (
    DEFAULT_TARGET_LOOKAHEAD_CYCLES,
    SELECTION_REALISTIC,
    SELECTION_UNIFORM,
    ApplianceSchedule,
    LoadSynthesizer,
    window_target_index,
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


def test_reactive_power_sign_matches_real_meter_convention(synthesizer):
    """Q 의 부호는 imag(I1) 의 '반대'여야 한다.

    펌웨어의 두 출력은 phase_deg = -ihdeg1 관계이고(실측 전 구간에서
    |phase_deg + ihdeg1| 중앙값이 정확히 0), FeatureExtractor 는 phase_deg 부호로
    Q 를 정한다. 따라서 imag(I1) = ih1*sin(ihdeg1) 과는 부호가 뒤집힌다.
    실측 3개 파일 전 구간에서 Q 와 imag(I1) 의 부호가 같은 비율은 0.0% 였다.
    이전 구현은 같은 부호를 주어 합성 데이터의 Q 채널이 통째로 반대였다.
    """
    sample = synthesizer.synthesize_random_window(600, n_active=2)
    q = sample.power_features[:, 1]
    i_im = np.imag(sample.harmonics_complex[:, 0])

    m = (np.abs(q) > 5.0) & (np.abs(i_im) > 1e-3)
    if m.sum() < 30:
        pytest.skip("무효전력이 유의미한 표본이 부족합니다")
    same_sign = float(np.mean(np.sign(q[m]) == np.sign(i_im[m])))
    assert same_sign < 0.05, (
        f"Q 와 imag(I1) 의 부호가 같은 비율이 {same_sign:.1%} 입니다. "
        f"실측 규약에서는 0% 여야 합니다."
    )


def test_grid_resistance_matches_measured_outlets():
    """배선 저항이 실측 콘센트 값 근처여야 한다.

    실측 복합 부하에서 3가지 방법(회귀 2종 + 오븐 펄스 dV/dI 직접 측정)이 모두
    221V 콘센트 R ≈ 1.55 Ohm, 234V 콘센트 R ≈ 0.45 Ohm 으로 일치했다.
    이전 모델의 0.15~0.35 Ohm 은 실측의 1/4~1/6 이라 전압 강하가 거의 없었다.
    """
    grid = GridSimulator()
    envs = [grid.sample_environment() for _ in range(500)]
    r = np.array([e.r_grid_ohm for e in envs])

    assert np.median(r) > 0.4, f"배선 저항이 실측보다 너무 작습니다: 중앙값 {np.median(r):.3f} Ohm"
    assert r.max() < 3.0, f"배선 저항이 비현실적으로 큽니다: 최대 {r.max():.3f} Ohm"

    # 두 실측 콘센트 근방이 모두 생성되어야 한다
    assert np.any(np.abs(r - 1.55) < 0.5), "221V 콘센트(R≈1.55) 근방이 생성되지 않았습니다"
    assert np.any(np.abs(r - 0.45) < 0.3), "234V 콘센트(R≈0.45) 근방이 생성되지 않았습니다"

    # 임피던스가 큰 회선일수록 전압이 낮게 관측된다 (같은 회선의 성질이므로)
    low_v = np.array([e.r_grid_ohm for e in envs if e.source == "outlet_low_221v"])
    high_v = np.array([e.r_grid_ohm for e in envs if e.source == "outlet_high_234v"])
    if len(low_v) > 10 and len(high_v) > 10:
        assert np.median(low_v) > np.median(high_v), \
            "낮은 전압 콘센트가 더 큰 임피던스를 가져야 합니다"


def test_activation_sampling_is_duration_weighted(segment_pool):
    """짧고 특이한 활성화가 실제보다 자주 뽑히면 안 된다.

    오븐은 활성화가 2개뿐인데 2.1분(통전율 80.8%)과 32.9분(21.9%)이다.
    균등 추첨하면 절반의 확률로 예열 구간이 뽑혀 실측(22~41%)보다 과하게 통전한다.
    """
    acts = segment_pool.appliance_activations["oven"]
    if len(acts) < 2:
        pytest.skip("오븐 활성화가 1개뿐입니다")

    longest = max(acts, key=lambda a: a.duration_cycles)
    picks = [segment_pool.sample_activation("oven") for _ in range(300)]
    share = sum(1 for p in picks if p is longest) / len(picks)
    expected = longest.duration_cycles / sum(a.duration_cycles for a in acts)
    assert share > expected * 0.8, (
        f"가장 긴 활성화가 뽑힌 비율 {share:.1%} 가 길이 비중 {expected:.1%} 에 못 미칩니다"
    )


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


def test_periodic_activation_contains_off_gaps(segment_pool):
    """주기 부하의 활성화는 통전 구간과 휴지 구간을 모두 담아야 한다.

    이 조건이 깨지면 위의 '리샘플링 안 함' 검사는 통과하는데도 duty 가 사라진다.
    통전 펄스만 잘라 온 배열은 전부 같은 전력 레벨이라, 순환 이어붙이기를 해도
    원본 레벨 집합을 벗어나지 않기 때문이다. 실제로 그렇게 되어 있었고
    핫플레이트가 실측 42% 통전에서 합성 95% 통전으로 부풀었다.
    """
    for act in segment_pool.appliance_activations["hotplate"]:
        on_ratio = float(act.is_on.mean())
        assert 0.05 < on_ratio < 0.95, (
            f"핫플레이트 활성화의 통전율이 {on_ratio:.3f} 다. "
            "0 또는 1 에 붙어 있으면 휴지 구간이 활성화 밖으로 빠져나간 것이다."
        )


def test_synthesized_periodic_duty_matches_measurement(synthesizer):
    """합성 창의 핫플레이트 통전율이 실측 범위 안에 들어와야 한다.

    실측: hotplate_1 42.4% / hotplate_2 31.8%, 10초당 전이 8.3~8.6회.
    수정 전에는 통전율 95%, 전이는 타일 이음매의 1사이클 노치뿐이었다.
    """
    duties, off_runs = [], []
    for _ in range(60):
        s = synthesizer.synthesize_random_window(
            600, force_active=["hotplate"], force_plugged_all=True,
            target_biased_placement=True, compute_gt_harmonics=False,
        )
        on = s.gt_is_on["hotplate"].astype(bool)
        if on.sum() < 60:
            continue
        # 통전 런의 시작/끝. 창 앞뒤의 '아직 안 켜짐/이미 끝남' 구간은 릴레이 휴지가
        # 아니므로 세면 안 된다. 통전 런 사이의 공백만 본다.
        d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
        on_starts, on_ends = np.where(d == 1)[0], np.where(d == -1)[0]
        # 동작 구간(첫 통전 ~ 마지막 통전) 안에서의 통전율
        span = on_ends[-1] - on_starts[0]
        if span > 0:
            duties.append(float(on[on_starts[0]:on_ends[-1]].mean()))
        off_runs += list(on_starts[1:] - on_ends[:-1])

    assert duties, "핫플레이트가 든 창을 만들지 못했습니다"
    mean_duty = float(np.mean(duties))
    assert 0.20 < mean_duty < 0.75, (
        f"동작 구간 내 통전율이 {mean_duty:.3f} 다. 실측은 0.32~0.42 이고, "
        "1.0 에 가까우면 휴지 구간이 사라진 것이다."
    )
    # 휴지 구간이 1~2 사이클짜리 타일 이음매 노치가 아니라 실제 릴레이 공백(약 1초)이어야 한다
    assert off_runs, "통전 런이 하나뿐입니다 - 릴레이 휴지 구간이 만들어지지 않았습니다"
    med_off = float(np.median(off_runs))
    assert 20 < med_off < 200, (
        f"휴지 구간 중앙값 {med_off:.0f} 사이클. 실측 릴레이 공백은 약 64~75 사이클이다."
    )


def test_lookahead_constants_agree_across_modules():
    """`synthesizer` 와 `model.inputs` 의 lookahead 상수가 같아야 한다.

    두 곳에 따로 선언돼 있는데 지금까지 일치를 강제하는 검사가 없었다.
    어긋나면 **라벨을 읽는 시점과 입력을 자르는 시점이 달라져 조용히 틀린다**
    (11.2절이 세 곳을 하나로 묶은 것과 같은 종류의 결함이다).
    """
    from src.model.inputs import (FINE_CYCLES, TARGET_LOOKAHEAD,
                                  fine_target_index, target_index)

    assert TARGET_LOOKAHEAD == DEFAULT_TARGET_LOOKAHEAD_CYCLES
    assert target_index(3600) == window_target_index(3600)
    # 세밀 갈래 안에서의 타깃도 같은 lookahead 를 따라야 한다.
    # **길이를 하드코딩하지 않는다** — 12.45 에서 FINE_CYCLES 를 600 -> 780 으로
    # 올렸고, 그때 이 줄이 룩어헤드와 무관한 이유로 깨졌다.
    assert FINE_CYCLES - 1 - fine_target_index() == TARGET_LOOKAHEAD
    # 세밀 갈래가 타깃 이후를 전부 덮어야 한다 (덮지 못하면 룩어헤드가 헛돈다)
    assert fine_target_index() >= 0, "타깃이 세밀 갈래 밖으로 나갔습니다"


def test_target_index_is_causal_not_centered():
    """타깃은 선언된 lookahead 만큼만 미래를 요구해야 한다.

    2026-08-22: 지연을 1초 -> 6초로 늘렸다 (12.9.12절). 오븐+핫플 동시 발열을
    전기포트로 오인하는 실패를 고치려면 타깃 **이후**의 오븐 전이를 봐야 하는데,
    앞 1초로는 5%, 6초면 약 50% 를 잡는다.

    **그래도 중앙은 아니다.** 60초 창의 중앙 타깃이면 30초 지연이라 5.1절의
    실시간 동선이 무너진다. 운영 창(3600)에서 타깃이 뒤 20% 안에 있는지 본다.
    """
    for w in (600, 3600, 7200):
        idx = window_target_index(w)
        assert w - 1 - idx == DEFAULT_TARGET_LOOKAHEAD_CYCLES
    assert DEFAULT_TARGET_LOOKAHEAD_CYCLES <= 600, "지연 10초를 넘으면 시연이 안 된다"
    w = 3600
    assert window_target_index(w) > w * 0.8, "60초 창 타깃이 중앙 쪽으로 밀렸다"


def test_generator_and_cache_use_the_same_target_index(segment_pool, tmp_path):
    """배치 생성기와 캐시의 타깃 시점이 어긋나면 균형 보정이 조용히 틀린다."""
    from src.synthesis.cache import WindowCache, build_cache

    gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=600)
    assert gen.target_index == window_target_index(600)

    build_cache(cache_dir=tmp_path / "c", n_scenarios=3, scenario_seconds=15.0,
                window_cycles=600, stride_cycles=150, seed=1, progress_every=0)
    cache = WindowCache(tmp_path / "c")
    assert cache.target_offset == gen.target_index

    # 캐시가 돌려주는 라벨이 정말 그 시점의 값인지 원본 배열로 대조한다
    w = cache.get(0)
    s, off = int(cache.index[0, 0]), int(cache.index[0, 1])
    raw = np.load(tmp_path / "c" / "y_on.npy", mmap_mode="r")
    assert np.array_equal(w["y_on"], raw[s, :, off + cache.target_offset].astype(np.float32))


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
    assert X.shape == (8, RAW_CHANNELS, 600)
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

    # **창 길이는 운영값(3600 = 60초)을 쓴다.** 300(5초)이면 오븐이 통전 없는
    # 창을 만든다 — 12.111 이 오븐 ON 을 '히터 통전'으로 바꾸면서 서모스탯
    # 공백(25~31초)이 라벨에 드러났고, 그것이 5초 창보다 훨씬 길다.
    gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=3600)
    assert "high_power_resistive" in gen.describe_recipe_mix()

    # 전용 레시피는 저항 부하만 켠다. 다만 **듀티 부하는 창 안에서 통전이 없을 수
    # 있으므로**(오븐 통전율 25~43%) 매 창이 아니라 대부분의 창을 본다.
    n_active = 0
    for _ in range(20):
        s = gen.synthesizer.synthesize_high_power_window(3600)
        active = set(s.active_appliances)
        n_active += bool(active)
        assert active <= resistive, f"저항 부하가 아닌 기기가 켜졌습니다: {active - resistive}"
        assert s.metadata["max_sustained_p_w"] <= gen.synthesizer.sustained_power_limit_w
    assert n_active >= 18, f"20창 중 {n_active}창에서만 저항이 통전했습니다"


def test_high_low_mixed_creates_the_error_bleed_case(segment_pool):
    """고부하 오차가 저부하로 전가되는 상황을 학습 데이터가 담고 있어야 한다.

    고부하와 저부하가 같이 켜진 창에서 크기 차이가 31배라(1139W vs 37W),
    고부하 3% 오차가 저부하 전체의 93% 를 왜곡할 수 있다. 그런데 다른 레시피는
    이 조합을 만들지 않아 동시 가동 창이 3.3% 뿐이었다.
    """
    from src.preprocessing.file_registry import get_resistive_appliances

    resistive = set(get_resistive_appliances())
    syn = LoadSynthesizer(segment_pool=segment_pool, compute_gt_harmonics=False)
    low_load = {a for a in syn.known_appliances if is_low_load(a)}

    # 듀티 부하(오븐 통전율 25~43%)는 창 안에서 통전이 없을 수 있다 — 12.111.
    # 그래서 매 창이 아니라 대부분의 창에서 고부하가 잡히는지 본다.
    n_hi = 0
    for _ in range(20):
        s = syn.synthesize_high_low_mixed_window(3600)
        active = set(s.active_appliances)
        n_hi += bool(active & resistive)
        assert active & low_load, f"저부하가 없습니다: {active}"
        assert s.metadata["max_sustained_p_w"] <= syn.sustained_power_limit_w
    assert n_hi >= 17, f"20창 중 {n_hi}창에서만 고부하가 통전했습니다"


def test_recipe_mix_covers_high_low_cooccurrence(segment_pool):
    """배치 전체에서 고부하+저부하 동시 가동 창이 충분히 나와야 한다."""
    gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=600)
    assert "high_low_mixed" in gen.describe_recipe_mix()

    apps = gen.appliance_list
    hi = [apps.index(a) for a in ("electiric_kettle", "oven", "hair_dryer", "hotplate") if a in apps]
    lo = [apps.index(a) for a in ("beam_projector", "laptop_charger", "fan", "minipc") if a in apps]

    y = np.concatenate([gen.generate_batch_dict(32)["y_on"] for _ in range(12)])
    both = y[:, hi].any(1) & y[:, lo].any(1)
    assert both.mean() > 0.07, (
        f"고부하+저부하 동시 가동 창이 {100*both.mean():.1f}% 뿐입니다 "
        f"(보강 전 3.3%). 오차 전가를 배울 표본이 부족합니다."
    )


def test_smps_overlap_puts_two_or_three_smps_on_the_target(segment_pool):
    """SMPS 2~3대가 **타깃 시점에** 동시에 켜져 있어야 한다 (12.88.4 의 1번).

    12.81 이 미니PC 미검출의 조건을 "경쟁 SMPS 가 함께 켜져 있을 때" 로 좁혔는데
    (경쟁 없으면 재현율 99%, 있으면 30~67%), 학습 분포에는 그 상황이 타깃 시점
    기준 12.2% 뿐이었다. 켜 두는 것만으로는 부족하고 타깃 시점에 걸려야 한다 -
    `resistive_overlap` 이 겪은 것과 같은 함정이다.
    """
    from src.model.inputs import TARGET_LOOKAHEAD
    from src.preprocessing.file_registry import get_smps_appliances
    from src.synthesis.synthesizer import window_target_index

    smps = set(get_smps_appliances())
    syn = LoadSynthesizer(segment_pool=segment_pool, compute_gt_harmonics=False)
    ti = window_target_index(600, TARGET_LOOKAHEAD)

    n_on_target, trio = 0, 0
    for _ in range(20):
        s = syn.synthesize_smps_overlap_window(600, target_lookahead_cycles=TARGET_LOOKAHEAD)
        on = {a for a in smps if int(s.gt_is_on[a][ti]) == 1}
        n_on_target += len(on) >= 2
        trio += len(on) >= 3
    assert n_on_target >= 17, f"타깃 시점에 SMPS 2대 이상인 창이 20 중 {n_on_target} 뿐입니다"
    assert trio >= 3, f"3종 동시가 20 중 {trio} 뿐입니다 (p_trio 기본 0.5)"


def test_smps_overlap_is_off_by_default_in_the_recipe_mix(segment_pool):
    """기본 믹스에서는 지분이 0 이어야 한다 - 기존 캐시와 같은 분포를 유지한다."""
    from src.synthesis.dataset import DEFAULT_RECIPE_MIX

    assert DEFAULT_RECIPE_MIX["smps_overlap"] == 0.0
    gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=600)
    i = gen.recipe_names.index("smps_overlap")
    assert gen.recipe_probs[i] == 0.0


def test_smps_preset_raises_smps_cooccurrence(segment_pool):
    """`smps_hi` 프리셋이 SMPS 동시 가동 창을 실제로 늘려야 한다.

    12.68 이 **전체** 동시성을 올렸다가 실패했다. 늘어야 하는 것은 그것이 아니라
    SMPS 끼리의 겹침이다 (측정: 기본 믹스 타깃 12.2% -> smps_hi 33.8%).
    """
    from src.run_recipe_mix_probe import PRESETS

    def smps_pairs(mix):
        gen = NILMBatchGenerator(segment_pool=segment_pool, window_size_cycles=600,
                                 recipe_mix=mix)
        idx = [gen.appliance_list.index(a)
               for a in ("beam_projector", "laptop_charger", "minipc")
               if a in gen.appliance_list]
        y = np.concatenate([gen.generate_batch_dict(32)["y_on"] for _ in range(8)])
        return (y[:, idx].sum(1) >= 2).mean()

    base = smps_pairs(None)
    hi = smps_pairs(PRESETS["smps_hi"])
    assert hi > base + 0.10, (
        f"SMPS 동시 가동 창이 {100*base:.1f}% -> {100*hi:.1f}% 로 거의 안 늘었습니다"
    )


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


# ── 학습용 윈도우 캐시 ────────────────────────────────────────────────────────

def test_cache_build_and_read(tmp_path):
    """캐시가 생성되고 창 형태가 학습에 바로 쓸 수 있어야 한다."""
    from src.synthesis.cache import WindowCache, build_cache

    meta = build_cache(
        cache_dir=tmp_path / "c", n_scenarios=24, scenario_seconds=20.0,
        window_cycles=600, stride_cycles=120, seed=0,
    )
    assert meta["n_windows"] > 0

    cache = WindowCache(tmp_path / "c")
    assert len(cache) == meta["n_windows"]
    assert len(cache.appliances) == 9

    w = cache.get(0)
    assert w["X"].shape == (RAW_CHANNELS, 600) and w["X"].dtype == np.float32
    for key in ("y_power", "y_standby", "y_on", "y_plugged"):
        assert w[key].shape == (9,)
    assert w["y_state"].shape == (9,)
    assert not np.isnan(w["X"]).any()

    seq = cache.get_sequence(0)
    assert seq["y_power"].shape == (9, 600)


def test_cache_weights_restore_class_balance(tmp_path):
    """부창을 그냥 자르면 균형이 무너진다. 가중치가 그것을 되돌려야 한다.

    측정: 보정 없이 자르면 핫플레이트가 5.5% -> 0.4% 로 사라지고
    불균형이 2.7:1 -> 34.9:1 이 된다.
    """
    from src.synthesis.cache import WindowCache, build_cache

    build_cache(cache_dir=tmp_path / "c", n_scenarios=120, scenario_seconds=60.0,
                window_cycles=600, stride_cycles=120, seed=0)
    cache = WindowCache(tmp_path / "c")

    uniform = np.array(list(cache.class_share(weighted=False).values()))
    weighted = np.array(list(cache.class_share(weighted=True).values()))

    imb_u = uniform.max() / max(uniform.min(), 1e-9)
    imb_w = weighted.max() / max(weighted.min(), 1e-9)
    assert imb_w < imb_u, f"가중치가 균형을 개선하지 못했습니다: {imb_u:.1f} -> {imb_w:.1f}"
    assert imb_w < 5.0, f"보정 후에도 불균형이 큽니다: {imb_w:.1f}:1"

    # 실제로 뽑아도 유지되어야 한다
    idx = cache.sample_indices(3000, np.random.default_rng(0))
    got = cache.on_center[idx].mean(axis=0)
    assert got.min() > 0, "한 번도 안 뽑힌 가전이 있습니다"
    assert got.max() / got.min() < 6.0


def test_cache_reports_effective_sample_size(tmp_path):
    """가중 추출로 잃는 다양성을 ESS 로 확인할 수 있어야 한다.

    ESS = 1/Σp² 이므로 균등이면 N, 한쪽에 몰리면 1 에 가까워진다.
    과적합 판단(재사용 횟수 = 추출 횟수 / ESS)의 기준값이다.
    """
    from src.synthesis.cache import WindowCache, build_cache

    meta = build_cache(cache_dir=tmp_path / "c", n_scenarios=60, scenario_seconds=60.0,
                       window_cycles=600, stride_cycles=120, seed=0)
    cache = WindowCache(tmp_path / "c")
    ess, ratio = cache.effective_sample_size()

    assert 1.0 <= ess <= len(cache), f"ESS 가 범위를 벗어났습니다: {ess} / {len(cache)}"
    assert 0.0 < ratio <= 1.0
    assert abs(meta["effective_sample_size"] - ess) < 1.0, "meta.json 값이 계산과 다릅니다"

    # 가중치를 끄면 균등 추출이므로 ESS = N 이어야 한다
    flat = WindowCache(tmp_path / "c", use_weights=False)
    ess_u, ratio_u = flat.effective_sample_size()
    assert ess_u == len(flat) and ratio_u == 1.0
    assert ess < ess_u, "가중 추출인데 ESS 가 줄지 않았습니다"


def test_cache_weights_sum_to_one(tmp_path):
    """가중치는 확률분포여야 한다 (np.random.choice 에 그대로 넘긴다)."""
    from src.synthesis.cache import compute_balance_weights

    rng = np.random.default_rng(0)
    on = (rng.random((500, 9)) < 0.15).astype(np.int8)
    on[:80] = 0  # 전부 꺼진 창을 확실히 섞는다
    w = compute_balance_weights(on, negative_share=0.2)

    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w >= 0)

    # 무작위 생성분에도 전부 꺼진 창이 섞이므로(0.85^9 = 23%) 앞 80개가 아니라
    # 전체 음성 집합의 몫을 봐야 한다.
    negative = on.sum(axis=1) == 0
    assert negative.sum() > 80
    assert abs(w[negative].sum() - 0.2) < 1e-6, "전부 꺼진 창의 몫이 지정값과 다릅니다"


def test_duty_period_is_not_frozen_across_augmented_samples():
    """주기 부하의 **주기**가 증강 표본마다 흔들려야 한다 (설계 문서 12.16절).

    이 검사가 없을 때 핫플레이트의 릴레이 주기는 증강을 거쳐도 10~90% 구간이
    [1.97, 2.03]초로 사실상 **상수**였다. 학습 활성화가 3개뿐인데 그 셋이 모두
    같은 주기를 내면, 모델은 듀티를 볼 필요 없이 파형을 대조해 맞힐 수 있다 -
    실제로 60초 내내 연속 통전한 창에서도 검출이 1.000 이었다 (12.16.2절).

    서모스탯 부하에서 통전 길이와 주기는 기기의 성질이 아니라 설정·주위 온도·
    부하의 함수다. 실측이 그 폭을 보여 준다 (핫플 통전율 0.358~0.578,
    오븐 듀티 0.22~0.81).
    """
    import numpy as np
    from src.synthesis.augmentor import DataAugmentor
    from src.synthesis.segment_pool import SegmentPool

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    acts = pool.appliance_activations.get("hotplate", [])
    if len(acts) < 1:
        pytest.skip("핫플레이트 활성화가 없습니다")

    def spread(act, randomize: bool):
        """**같은 활성화 하나**를 여러 번 증강했을 때의 주기 폭.

        ⚠ 여러 활성화를 섞어서 재면 안 된다 (2026-09-06, 13.17). `hotplate_2` 가
        들어와 원본이 1개 -> 6개가 되자 '고정' 쪽 폭이 0 -> 53 사이클로 뛰었다 —
        그것은 증강이 만든 폭이 아니라 **원본끼리의 차이**다. 그때 옛 기준
        (`varied > 2*frozen`)이 1.998 로 아슬아슬하게 깨졌다. 재는 것은 증강의
        몫이므로 원본을 고정하고 잰다.
        """
        aug = DataAugmentor(randomize_duty=randomize)
        np.random.seed(0)
        periods = []
        for _ in range(40):
            b = aug.augment_activation(act, target_duration_cycles=3600)
            tp = np.asarray(b.target_power_w)
            hot = tp > 0.5 * np.percentile(tp, 99)
            n_tr = int(np.abs(np.diff(hot.astype(int))).sum())
            periods.append(3600 / max(n_tr, 1) * 2)
        p = np.asarray(periods)
        return float(np.percentile(p, 90) - np.percentile(p, 10))

    worst = None
    for act in acts:
        frozen, varied = spread(act, False), spread(act, True)
        assert varied > 2 * frozen + 10.0, (
            f"듀티 주기가 충분히 흔들리지 않습니다: 10~90% 폭 {varied:.1f} vs "
            f"고정 {frozen:.1f} 사이클 (원본 하나 기준)"
        )
        worst = varied if worst is None else min(worst, varied)
    assert worst > 30, f"주기 폭이 0.5초에도 못 미칩니다: {worst:.1f} 사이클"


def test_duty_retiming_never_mixes_on_and_off_waveforms():
    """구간 길이를 흔들되 통전/휴지 파형이 섞이면 안 된다.

    `_retime_duty` 는 구간 **안에서만** 순환한다. 섞이면 릴레이가 반쯤 닫힌,
    실재하지 않는 상태가 만들어진다.
    """
    import numpy as np
    from src.synthesis.augmentor import DataAugmentor
    from src.synthesis.segment_pool import SegmentPool

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    acts = pool.appliance_activations.get("hotplate", [])
    if not acts:
        pytest.skip("핫플레이트 활성화가 없습니다")

    # `_retime_duty` 를 직접 본다. `augment_activation` 은 이 뒤에 전력 스케일(±15%)
    # 과 위상 지터를 걸므로 값 범위가 원본을 벗어나는 것이 정상이다.
    aug = DataAugmentor(randomize_duty=True)
    src = acts[0]
    original = set(np.asarray(src.target_power_w).tolist())
    np.random.seed(3)
    for _ in range(20):
        _, _, state, tp, on = aug._retime_duty(
            src.net_harmonics_complex, src.net_power_features,
            src.state_id, src.target_power_w, src.is_on,
        )
        tp = np.asarray(tp)
        # 리타이밍은 원본 사이클을 **골라 쓰기만** 한다. 새 값을 만들면 안 된다.
        assert set(tp.tolist()) <= original, "원본에 없던 전력이 생겼습니다"
        # 통전/휴지 구분이 살아 있어야 한다
        hot = tp > 0.5 * np.percentile(tp, 99)
        assert 0.05 < hot.mean() < 0.95, f"통전/휴지 구분이 사라졌습니다: {hot.mean():.3f}"
        assert len(state) == len(tp) == len(on), "배열 길이가 어긋났습니다"


def test_load_stratified_crop_covers_rare_high_load():
    """긴 활성화에서 드문 고부하 구간이 시간 비율보다 자주 뽑혀야 한다 (12.34.6).

    미니PC 가 그 예다. 33.9분짜리 활성화 안에서 CPU 부하가 걸린 구간이 6.7% 뿐이라
    시간 균등으로 자르면 풀 전체의 >=20W 가 13.9% 밖에 안 된다. 실측 미니PC 단독은
    30.3W 인데 합성 60초 창의 최대가 27.9W 였다 (분포 밖).

    여기서는 90% 가 10W, 10% 가 30W 인 활성화를 만들어, 계층 추출이 고부하 창을
    시간 비율(10%)보다 확실히 자주 고르는지 본다.
    """
    n, win = 60_000, 3_600
    target_p = np.full(n, 10.0, np.float32)
    target_p[-6_000:] = 30.0                      # 마지막 10% 만 고부하
    on = np.ones(n, np.int8)
    aug_on = DataAugmentor(load_stratified=True)
    aug_off = DataAugmentor(load_stratified=False)

    def high_share(aug, trials=400):
        rng = np.random.RandomState(0)
        np.random.seed(0)
        hits = 0
        for _ in range(trials):
            s = aug._stratified_start(target_p, on, win, n - win)
            if float(np.median(target_p[s:s + win])) > 20.0:
                hits += 1
        return hits / trials

    off, onr = high_share(aug_off), high_share(aug_on)
    assert onr > off * 1.8, f"계층 추출이 고부하를 더 뽑지 못했습니다: {off:.3f} -> {onr:.3f}"
    assert onr > 0.20, f"고부하 창 비율이 여전히 낮습니다: {onr:.3f}"


def test_load_stratified_crop_is_noop_on_flat_activation():
    """전력이 평평한 기기(프로젝터)에서는 계층 추출이 무작위와 같아야 한다."""
    n, win = 20_000, 3_600
    target_p = np.full(n, 47.5, np.float32)
    on = np.ones(n, np.int8)
    aug = DataAugmentor(load_stratified=True)
    np.random.seed(0)
    starts = [aug._stratified_start(target_p, on, win, n - win) for _ in range(300)]
    # 한쪽으로 몰리면 안 된다 - 시작 위치가 전 구간에 퍼져 있어야 한다
    span = n - win
    assert min(starts) < span * 0.15 and max(starts) > span * 0.85, (
        f"평평한 활성화인데 자르는 지점이 몰렸습니다: {min(starts)}~{max(starts)} / {span}")


def test_duty_appliances_never_reach_stratified_crop():
    """듀티 부하는 `_crop_window` 에 오지 않아야 한다 (통전율이 물리다).

    오븐은 팬/조명만 도는 활성화와 히터 활성화가 섞여 있어(통전중앙 14/14/15/1160W),
    전력으로 계층화하면 히터 노출이 줄어든다. `augment_activation` 이 앞에서
    `_tile_or_crop` 으로 보내는지 확인한다.
    """
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    aug = DataAugmentor(load_stratified=True)
    called = []
    orig = aug._stratified_start
    aug._stratified_start = lambda *a, **k: (called.append(1), orig(*a, **k))[1]
    for app in ("oven", "hotplate"):
        acts = pool.appliance_activations.get(app, [])
        if not acts:
            pytest.skip(f"{app} 활성화가 없습니다")
        np.random.seed(0)
        for _ in range(20):
            aug.augment_activation(pool.sample_activation(app), target_duration_cycles=3600)
    assert not called, "듀티 부하가 전력 계층 자르기를 탔습니다"


# ─────────────────────────────────────────────────────────────────────────
# 차수별 지문 지터 (12.62절)
# ─────────────────────────────────────────────────────────────────────────

def _odd_median_spread(acts):
    """활성화 목록 -> (홀수차 진폭 상대산포 중앙값 %, 위상 원형산포 중앙값 도)."""
    from src.run_fingerprint_spread_probe import circ_spread, rel_spread, signature

    sigs = []
    for r in acts:
        c = r.net_harmonics_complex
        v = np.median(c.real, 0) + 1j * np.median(c.imag, 0)
        if abs(v[0]) > 1e-6:
            sigs.append(signature(v))
    assert len(sigs) > 30
    g = rel_spread(np.array([s[0] for s in sigs]))
    psi = circ_spread(np.array([s[1] for s in sigs]))
    odd = [k - 2 for k in range(3, 16, 2)]
    return float(np.nanmedian(g[odd])), float(np.nanmedian(psi[odd]))


def test_기존_증강은_지문을_바꾸지_못한다():
    """12.62 의 출발점. 이것이 깨지면 지터의 근거가 사라진다.

    공통 배율은 |I_k|/|I_1| 을, k차 위상 회전은 ∠I_k − k∠I_1 을 정확히
    불변으로 남긴다. 그래서 같은 활성화를 몇 번 증강해도 정규화 지문이 안 움직인다.
    """
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    if "beam_projector" not in pool.get_appliance_types():
        pytest.skip("프로젝터 활성화가 없습니다")
    aug = DataAugmentor()
    np.random.seed(0)
    a0 = pool.sample_activation("beam_projector")
    np.random.seed(1)
    g, psi = _odd_median_spread([aug.augment_activation(a0) for _ in range(120)])
    assert g < 5.0, f"진폭 지문이 {g:.1f}% 움직였습니다 - 증강이 바뀌었습니까"
    assert psi < 3.0, f"위상 지문이 {psi:.1f}도 움직였습니다"


def test_지터가_요청한_크기로_지문을_흔든다():
    """지정한 값이 **홀수차 중앙값**이 되고 차수에 비례해 커져야 한다."""
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    if "beam_projector" not in pool.get_appliance_types():
        pytest.skip("프로젝터 활성화가 없습니다")
    aug = DataAugmentor(harmonic_dither_amp=0.50, harmonic_dither_phase_deg=15.0)
    np.random.seed(0)
    a0 = pool.sample_activation("beam_projector")
    np.random.seed(1)
    outs = [aug.augment_activation(a0) for _ in range(300)]
    g, psi = _odd_median_spread(outs)
    assert 35.0 < g < 75.0, f"진폭 지터 중앙값이 {g:.1f}% 입니다 (목표 50%)"
    assert 10.0 < psi < 22.0, f"위상 지터 중앙값이 {psi:.1f}도 입니다 (목표 15도)"

    # 차수 비례 — k=15 가 k=3 보다 뚜렷하게 커야 한다
    from src.run_fingerprint_spread_probe import rel_spread, signature
    A = np.array([signature(np.median(r.net_harmonics_complex.real, 0)
                            + 1j * np.median(r.net_harmonics_complex.imag, 0))[0]
                  for r in outs])
    gk = rel_spread(A)
    assert gk[13] > 2.5 * gk[1], f"차수 비례가 아닙니다: k=3 {gk[1]:.1f}% / k=15 {gk[13]:.1f}%"


def test_지터는_기본파와_전력라벨을_건드리지_않는다():
    """k=1 과 P·Q 를 두면 전력 라벨이 정확히 유효하게 남는다."""
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    if "beam_projector" not in pool.get_appliance_types():
        pytest.skip("프로젝터 활성화가 없습니다")
    np.random.seed(0)
    a0 = pool.sample_activation("beam_projector")

    off = DataAugmentor()
    on = DataAugmentor(harmonic_dither_amp=0.50, harmonic_dither_phase_deg=15.0)
    np.random.seed(7)
    r_off = off.augment_activation(a0, duration_scale=1.0, power_scale=1.0,
                                   phase_jitter_deg=0.0)
    np.random.seed(7)
    r_on = on.augment_activation(a0, duration_scale=1.0, power_scale=1.0,
                                 phase_jitter_deg=0.0)
    np.testing.assert_allclose(r_on.net_harmonics_complex[:, 0],
                               r_off.net_harmonics_complex[:, 0], rtol=0, atol=0)
    np.testing.assert_allclose(r_on.net_power_features, r_off.net_power_features,
                               rtol=0, atol=0)
    np.testing.assert_allclose(r_on.target_power_w, r_off.target_power_w, rtol=0, atol=0)
    # 그런데 고차는 실제로 달라져 있어야 한다
    assert not np.allclose(r_on.net_harmonics_complex[:, 2],
                           r_off.net_harmonics_complex[:, 2])


def test_다단_강하_채널이_기착을_짚는다():
    """12.62 의 다단 강하 2채널. 프로젝터 팬 기착이 있고 충전기는 없다.

    긴 탭(5.5초)이 기착 1~5초 전부에서 발화하고, 기착이 없으면 0 이어야 한다.
    짧은 탭(3.0초)은 짧은 기착에서만 발화해 **기착 길이**를 준다.
    """
    from src.model.inputs import (DROP_TAPS, RIPPLE_SCALE, build_fine,
                                 fine_target_index)

    T = 3600
    tgt = T - 1 - 360                       # 창 전체의 타깃 시점
    ti = fine_target_index()
    assert DROP_TAPS[1] > DROP_TAPS[0]

    def drop_w(pedestal_s: float):
        prof = np.zeros(T, np.float32)
        prof[:tgt] = 45.0
        n = int(pedestal_s * 60)
        prof[tgt:tgt + n] = 4.0
        x = np.zeros((1, RAW_CHANNELS, T), np.float32)
        x[0, 30] = prof
        x[0, 32] = 220.0
        f = build_fine(x)
        from src.model.inputs import DROP0
        return [float(np.sinh(f[0, DROP0 + i, ti]) * RIPPLE_SCALE) for i in range(2)]

    # 충전기 — 기착 없음. 두 탭 다 0
    short, long = drop_w(0.0)
    assert abs(short) < 0.1 and abs(long) < 0.1, f"기착이 없는데 {short:.2f}/{long:.2f}W"

    # 프로젝터 — 기착 1~5초. 긴 탭이 언제나 발화한다 (12.60.1 의 실측·합성 범위)
    for ped in (1.0, 2.0, 3.0, 4.0, 5.0):
        short, long = drop_w(ped)
        assert long > 3.0, f"기착 {ped}초에서 긴 탭이 {long:.2f}W 입니다"

    # 짧은 탭은 기착이 짧을 때만 — 그래서 둘이 함께 길이를 준다
    assert drop_w(2.0)[0] > 3.0
    assert abs(drop_w(5.0)[0]) < 0.1


def test_voltage_texture_stays_in_its_site_and_balances_sessions():
    """전압 텍스처는 **그 자리에서만**, 세션은 고르게 뽑혀야 한다 (설계 13.19).

    왜 검사하나 — 텍스처 선택이 고차 전류를 크게 움직인다. 같은 자리라도 세션이 다르면
    미니PC 의 |I13|/P 가 30~45% 갈리고, 자리가 섞이면 vh3 이 0.7% 대 3.0% 로 4배 갈린다.
    그런데 옛 코드는 **기저 전압만** 보고 골랐다:

      · 236V 위 창(탐색 성분의 22%)은 ±4V 안에 후보가 없어 **라이브러리 전체 균등**으로
        떨어졌고, 그 절반이 216V 자리의 텍스처였다.
      · 후보 안에서 텍스처를 균등하게 뽑아, 녹화가 긴 세션 하나가 그 자리를 대표했다
        (E1 82% 대 E2 18%).

    둘 다 **조용히** 틀린다 — 합성은 정상으로 보이고 고차 자리 반응만 어긋난다.
    """
    import numpy as np
    from collections import Counter
    from src.synthesis.vtexture import VoltageTextureLibrary

    lib = VoltageTextureLibrary.from_npz_dir()
    if len(lib) < 20:
        pytest.skip("텍스처 라이브러리가 없습니다")
    sites = {t.site for t in lib.textures if t.site}
    if len(sites) < 2:
        pytest.skip("자리가 하나뿐입니다")

    rng = np.random.default_rng(0)
    for site, v in (("D", 216.5), ("E", 229.5)):
        got = [lib.sample(rng, vrms_target=v, site=site) for _ in range(400)]
        bad = [t.stem for t in got if t.site != site]
        assert not bad, f"{site} 창에 다른 자리의 텍스처가 섞였습니다: {sorted(set(bad))[:3]}"
        c = Counter(t.session for t in got)
        if len(c) > 1:
            lo = min(c.values()) / sum(c.values())
            assert lo > 0.25, (
                f"{site} 의 세션이 한쪽으로 쏠립니다: "
                + ", ".join(f"{k} {n/len(got)*100:.0f}%" for k, n in c.most_common()))

    # 라이브러리 전압 범위 **밖**에서도 균등 난수로 떨어지면 안 된다 — 가장 가까운 쪽이어야 한다
    top = max(t.vrms for t in lib.textures)
    far = [lib.sample(rng, vrms_target=top + 12.0) for _ in range(200)]
    assert all(t.vrms > top - 8.0 for t in far), (
        "전압 범위 밖 창이 먼 텍스처를 받았습니다 — 균등 난수로 떨어진 것입니다")


def test_circuit_model_does_not_fail_silently():
    """회로 모델이 **조용히 전부 실패**하지 않는가 (설계 13.23.5).

    `SmpsCircuit.current` 는 예외를 삼키고 `failures` 만 올린다. 그래서 회로 모델이 통째로
    죽어도 합성은 그냥 돌아가고 — 텍스처·결합 델타가 0 이 될 뿐 — 아무도 모른다.
    실제로 2026-09-07 에 그랬다: `circuit12` 를 한 번은 최상위로, 한 번은 `circuit_model.circuit12`
    로 임포트하자 `_core12` 의 numba 디스크 캐시(`@njit(cache=True)`)가 섞여
    `ModuleNotFoundError: No module named 'circuit12'` 로 모든 호출이 실패했다. 혼합검증이
    "SMPS 창이 없다" 로 나와서 겨우 알아챘다. 걸리면 `circuit_model/__pycache__` 를 지운다.
    """
    import numpy as np
    from src.synthesis.coupling import SmpsCircuit
    from src.synthesis.vtexture import VoltageTextureLibrary

    lib = VoltageTextureLibrary.from_npz_dir()
    if len(lib) == 0:
        pytest.skip("텍스처 라이브러리가 없습니다")
    tex = lib.textures[0]
    circ = SmpsCircuit()
    if not circ.models:
        pytest.skip("회로 파라미터(pkl)가 없습니다")

    for dev in sorted(circ.models):
        I = circ.current(dev, 30.0, tex.source_rel(), 220.0)
        assert I is not None and np.all(np.isfinite(I)), f"{dev} 전류가 안 나옵니다"
        assert abs(I[0]) > 1e-4, f"{dev} 기본파가 0 입니다"
    d = circ.coupling_delta({"laptop_charger": 45.0}, tex.source_rel(), tex.id, 220.0, 1.0, 100e-6)
    assert d, "단독 SMPS 결합 델타가 비었습니다 (13.22 에서 문턱을 1 로 내렸다)"
    assert circ.stats()["failures"] == 0, (
        f"회로 모델이 조용히 실패했습니다: {circ.stats()} — "
        "`circuit_model/__pycache__` 를 지우고 다시 보십시오 (13.23.5)")


def test_scoring_does_not_assume_a_fixed_test_file():
    """채점기가 **파일 이름**으로 옛 시대 자료 모양을 가정하면 안 된다 (13.24.10).

    2026-09-06 계측기 교체로 `test_1`~`test_5` 가 전부 **다른 녹화**로 갈렸는데
    이름은 그대로다. 그래서 이름에 건 가정이 셋이나 새 자료를 덮쳤다:
      · `results/seq_time_map.json` 이 새 test_4 를 옛 seq 로 납치 (13.19)
      · `run_line_impedance` 가 새 test_5 를 "장소 A" 로 (13.19)
      · `run_gate_check.oven_on_breakdown` 이 `ev["test_4"]` 하드코딩 -> KeyError (13.24.10)

    이 검사는 **자료의 모양으로 걸러야 한다**는 규약을 지킨다.
    """
    from src.evaluation.real_events import load_events
    from src.run_gate_check import oven_breakdown_ok

    ev = load_events()
    assert ev, "real_events 가 비었습니다"

    # 모든 파일이 같은 기기를 담고 있지 않다 — 채점기는 그걸 견뎌야 한다
    sets = {s: frozenset(ev[s]["appliances_present"]) for s in ev}
    assert len(set(sets.values())) > 1, (
        "모든 파일의 기기 구성이 같습니다 — 이 검사가 지키려는 상황이 아닙니다")

    for stem in ev:
        ok = oven_breakdown_ok(stem, ev)          # 던지면 안 된다
        if ok:
            iv = ev[stem]["intervals"]
            assert "_heater_pulses" in iv["oven"], f"{stem}: 통과했는데 키가 없습니다"
            assert "hotplate" in iv and "electiric_kettle" in iv

    # 없는 이름을 물어도 조용히 False 여야 한다
    assert not oven_breakdown_ok("test_does_not_exist", ev)


def test_coupling_solver_converges_at_high_impedance():
    """결합 고정점 — 무감쇠는 Z 가 크면 안 잠기고, 감쇠는 잠긴다 (13.24.13).

    그리고 `relax=1.0` 이 옛 무감쇠 반복과 **한 비트도 안 다른지** 지킨다 — 감쇠를 넣으려고
    풀개를 `_solve_vterm` 으로 뽑아냈으므로 되돌림이 깨지면 여기서 걸려야 한다.
    """
    import numpy as np
    from src.synthesis.coupling import SmpsCircuit
    from src.synthesis.vtexture import default_library

    rng = np.random.default_rng(0)
    tex = default_library().sample(rng)
    rel = tex.source_rel()
    mix = {"laptop_charger": 55.0, "minipc": 12.0, "beam_projector": 40.0}
    v1 = 220.0

    now = SmpsCircuit(n_iter=3, relax=1.0)
    ref = SmpsCircuit(models=now.models, n_iter=200, relax=0.5)
    damp = SmpsCircuit(models=now.models, n_iter=6, relax=0.5)

    def err(c, Z):
        V, Vr = c.solve_terminal(mix, rel, v1, Z, 100e-6), ref.solve_terminal(mix, rel, v1, Z, 100e-6)
        assert V is not None and Vr is not None, "풀개가 실패했습니다"
        I = sum(np.asarray(c.current(d, p, V / v1, v1)) for d, p in mix.items())
        Ir = sum(np.asarray(ref.current(d, p, Vr / v1, v1)) for d, p in mix.items())
        return float(np.max(np.abs(I - Ir) / np.maximum(np.abs(Ir), 1e-9)))

    # 감쇠는 모든 Z 에서 잠긴다
    for Z in (0.42, 1.15, 2.00):
        e = err(damp, Z)
        assert e < 0.03, f"감쇠 풀개가 Z={Z}Ω 에서 {e:.1%} 틀립니다 (3% 이내여야 합니다)"

    # 무감쇠는 Z 가 크면 못 잠근다 — 이 사실이 사라지면 기본값을 다시 봐야 한다
    assert err(now, 2.00) > 0.10, (
        "무감쇠 풀개가 Z=2Ω 에서 잠겼습니다 — 결함이 고쳐졌다면 이 검사와 "
        "SmpsCircuit 의 기본값을 같이 갱신하십시오 (13.24.13)")

    # relax=1.0 은 옛 거동 그대로여야 한다 (되돌림)
    a = SmpsCircuit(models=now.models, n_iter=3, relax=1.0).coupling_delta(
        mix, rel, 1, v1, 1.15, 100e-6)
    b = SmpsCircuit(models=now.models, n_iter=3).coupling_delta(mix, rel, 1, v1, 1.15, 100e-6)
    assert a and b
    for k in a:
        assert np.max(np.abs(a[k] - b[k])) == 0.0, f"{k}: relax=1.0 이 옛 거동과 다릅니다"


def test_state_mix_raises_rare_state_share():
    """상태 계층 표집이 좁은 상태를 시간 비율보다 자주 뽑아야 한다 (13.35).

    미니PC 가 그 예다. IDLE 은 8.8~12.0W 로 좁고 ACTIVE 는 11.5~26.7W 로 넓어서,
    전력 균등 계층화(12.34.6)가 IDLE 에 17.2% 밖에 안 준다. 그런데 실측 복합은
    미니PC 를 IDLE 로만 돌린다.

    여기서는 80% 가 ACTIVE(19W), 20% 가 IDLE(9.7W) 인 활성화를 만들어 반반 요청이
    IDLE 몫을 실제로 올리는지 본다.
    """
    n, win = 60_000, 3_600
    state = np.full(n, 2, np.int16)
    state[:12_000] = 1                                  # 앞 20% 만 IDLE
    target_p = np.where(state == 1, 9.7, 19.0).astype(np.float32)
    on = np.ones(n, np.int8)
    span = n - win

    aug = DataAugmentor(state_mix={"minipc": {1: 0.5, 2: 0.5}})
    assert aug.state_mix == {"minipc": {1: 0.5, 2: 0.5}}

    np.random.seed(0)
    hits = 0
    for _ in range(400):
        s = aug._state_start(state, on, win, span, 1)
        assert s is not None
        if float(np.median(state[s:s + win])) == 1.0:
            hits += 1
    share = hits / 400
    assert share > 0.35, f"IDLE 몫이 안 올랐습니다: {share:.3f} (시간 비율은 0.20)"


def test_state_mix_falls_back_when_state_absent():
    """그 상태가 활성화에 없으면 `None` 을 돌려 전력 계층화로 떨어져야 한다 (13.35).

    다른 상태로 메우면 IDLE 이 없는 활성화가 ACTIVE 를 두 번 내게 되어, 요청한
    비율이 조용히 어긋난다.
    """
    n, win = 20_000, 3_600
    state = np.full(n, 2, np.int16)                     # ACTIVE 뿐이다
    on = np.ones(n, np.int8)
    aug = DataAugmentor(state_mix={"minipc": {1: 1.0}})
    np.random.seed(0)
    assert all(aug._state_start(state, on, win, n - win, 1) is None
               for _ in range(20))
    assert aug._state_fill(state, on, win, 1) is None


def test_state_mix_only_touches_named_appliances():
    """이름이 없는 기기는 기존 경로 그대로여야 한다 (13.35)."""
    n, win = 20_000, 3_600
    state = np.full(n, 2, np.int16); state[:4_000] = 1
    target_p = np.where(state == 1, 9.7, 19.0).astype(np.float32)
    on = np.ones(n, np.int8)
    c = np.zeros((n, 15), np.complex64)
    pw = np.zeros((n, 6), np.float32); pw[:, 0] = target_p
    aug = DataAugmentor(state_mix={"minipc": {1: 1.0}})

    called = []
    orig = aug._state_start
    aug._state_start = lambda *a, **k: (called.append(1), orig(*a, **k))[1]
    np.random.seed(0)
    for _ in range(60):
        aug._crop_window(c, pw, state, target_p, on, target_len=win,
                         appliance_type="beam_projector")
    assert not called, "이름 없는 기기에 상태 표집이 걸렸습니다"
    for _ in range(60):
        aug._crop_window(c, pw, state, target_p, on, target_len=win,
                         appliance_type="minipc")
    assert called, "이름 있는 기기에 상태 표집이 안 걸렸습니다"


def test_state_fill_makes_a_window_full_of_one_state():
    """상태 구간이 창보다 짧아도 이어 붙여 창을 채워야 한다 (13.35).

    미니PC IDLE 은 풀 전체 최장이 51.8초라 60초 창을 혼자 못 채운다. 그래서
    캐시에 IDLE 80% 이상인 창이 0.4% 뿐인데, **실측 복합은 분 단위로 IDLE 이다.**
    """
    n, win = 60_000, 3_600
    state = np.full(n, 2, np.int16)
    for k in range(10):                                  # 30초짜리 IDLE 10토막
        state[k * 6_000:k * 6_000 + 1_800] = 1
    on = np.ones(n, np.int8)
    aug = DataAugmentor()
    np.random.seed(0)
    sel = aug._state_fill(state, on, win, 1)
    assert sel is not None and len(sel) == win
    assert (state[sel] == 1).all(), "채운 창에 다른 상태가 섞였습니다"
    assert len(np.unique(sel)) == win, "같은 사이클을 두 번 쓰면 안 됩니다"


def test_state_fill_prefers_long_runs():
    """길이 가중이므로 긴 구간이 먼저 쓰여 이음매가 적어야 한다 (13.35)."""
    n, win = 60_000, 3_600
    state = np.full(n, 2, np.int16)
    state[0:3_000] = 1                                   # 50초 한 덩어리
    for k in range(20):                                  # 1초짜리 스무 토막
        state[10_000 + k * 500:10_000 + k * 500 + 60] = 1
    on = np.ones(n, np.int8)
    aug = DataAugmentor()
    np.random.seed(0)
    seams = []
    for _ in range(40):
        sel = aug._state_fill(state, on, win, 1)
        if sel is None:
            continue
        seams.append(int((np.diff(sel) != 1).sum()))
    assert seams, "채우기가 한 번도 성공하지 못했습니다"
    assert np.median(seams) <= 8, f"이음매가 너무 많습니다: 중앙 {np.median(seams)}"


def test_state_fill_rejects_too_short_runs():
    """1초 미만 토막만 있으면 이어 붙이지 않는다 — 전이의 잔재다 (13.35)."""
    n, win = 60_000, 3_600
    state = np.full(n, 2, np.int16)
    for k in range(200):
        state[k * 250:k * 250 + 30] = 1                  # 0.5초씩
    on = np.ones(n, np.int8)
    aug = DataAugmentor()
    np.random.seed(0)
    assert aug._state_fill(state, on, win, 1) is None


def test_cache_index_plan_matches_iter_batches():
    """로더 작업자 경로는 **색인 열을 바꾸지 않는다** (13.47).

    `run_train_cnn.cache_index_plan` 이 `CachedWindows.iter_batches` 와 같은 순서로
    같은 난수를 소모해야, `--cache-workers` 를 켜도 학습이 그대로다. 둘 중 하나만
    고치면 v21 이 v19·v20 과 비교 불가가 되므로 여기서 못박는다.

    캐시가 없어도 도는 시험이다 — 색인 논리만 재현해 견준다.
    """
    import numpy as np
    from src.run_train_cnn import cache_index_plan

    def reference(n, batch_size, n_batches, rng, block_windows):
        """`CachedWindows.iter_batches` 의 색인 부분 그대로 (traincache.py)."""
        n_blocks = max(1, (n + block_windows - 1) // block_windows)
        out, made = [], 0
        while made < n_batches:
            for b in rng.permutation(n_blocks):
                lo = int(b) * block_windows
                hi = min(lo + block_windows, n)
                if hi - lo < batch_size:
                    continue
                order = lo + rng.permutation(hi - lo)
                for k in range(0, len(order) - batch_size + 1, batch_size):
                    out.append(order[k:k + batch_size])
                    made += 1
                    if made >= n_batches:
                        return out
        return out

    for n, bs, nb, bw in ((300_000, 512, 40, 24_000), (50_000, 256, 25, 8_000),
                          (9_000, 128, 30, 24_000)):
        a = cache_index_plan(n, bs, nb, np.random.default_rng(7), block_windows=bw)
        b = reference(n, bs, nb, np.random.default_rng(7), bw)
        assert len(a) == len(b) == nb, f"배치 수가 다릅니다: {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert np.array_equal(x, y), f"n={n} 배치 {i} 의 색인이 다릅니다"
        # 난수 소모량도 같아야 뒤따르는 난수가 안 어긋난다
        r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
        cache_index_plan(n, bs, nb, r1, block_windows=bw)
        reference(n, bs, nb, r2, bw)
        assert np.array_equal(r1.integers(0, 2**31, 4), r2.integers(0, 2**31, 4)), \
            f"n={n} 에서 rng 소모량이 다릅니다"
