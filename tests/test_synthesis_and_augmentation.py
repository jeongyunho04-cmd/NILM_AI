"""
Automated Pytest Suite for NILM Synthesis, Augmentation, Grid Simulator, and Standby Power
"""
import numpy as np
import pytest

from src.synthesis.segment_pool import SegmentPool
from src.synthesis.grid_simulator import GridSimulator
from src.synthesis.augmentor import DataAugmentor
from src.synthesis.synthesizer import ApplianceSchedule, LoadSynthesizer
from src.synthesis.dataset import NILMBatchGenerator


@pytest.fixture(scope="module")
def segment_pool():
    return SegmentPool(npz_dir="processed_data/npz")


def test_segment_pool_loading_and_standby_profiles(segment_pool):
    app_types = segment_pool.get_appliance_types()
    assert len(app_types) >= 8
    assert "air_conditioner" in app_types
    assert "electiric_kettle" in app_types
    assert "minipc" in app_types

    # Verify standby profile existence
    ac_standby = segment_pool.get_standby_profile("air_conditioner")
    assert ac_standby.harmonics_ri.shape == (15, 2)
    assert ac_standby.power_w >= 0.0


def test_stochastic_standby_power_simulation(segment_pool):
    synthesizer = LoadSynthesizer(segment_pool=segment_pool)

    # 1. Case A: All appliances unplugged (is_plugged = False) & none active
    unplugged_dict = {app: False for app in synthesizer.known_appliances}
    sample_unplugged = synthesizer.synthesize_scenario(
        total_duration_cycles=300,
        schedules=[],
        plugged_in_appliances=unplugged_dict,
        include_noise=True,
        simulate_voltage_drop=False,
    )
    p_unplugged = np.mean(sample_unplugged.power_features[:, 0])
    # Should be just single background noise floor (~1.4W)
    assert p_unplugged < 3.0, f"Unplugged noise floor too high: {p_unplugged:.2f}W"

    # 2. Case B: Air conditioner and Oven plugged in standby (is_plugged = True)
    plugged_dict = {app: False for app in synthesizer.known_appliances}
    plugged_dict["air_conditioner"] = True
    plugged_dict["oven"] = True
    sample_standby = synthesizer.synthesize_scenario(
        total_duration_cycles=300,
        schedules=[],
        plugged_in_appliances=plugged_dict,
        include_noise=True,
        simulate_voltage_drop=False,
    )
    p_standby = np.mean(sample_standby.power_features[:, 0])
    
    # Standby power must be strictly greater than unplugged noise
    assert p_standby > p_unplugged
    # Ground truth is_on must be 0 for all appliances
    assert np.all(sample_standby.gt_is_on["air_conditioner"] == 0)
    assert np.all(sample_standby.gt_is_on["oven"] == 0)
    assert np.all(sample_standby.gt_target_power_w["air_conditioner"] == 0.0)


def test_grid_simulator_voltage_drop_and_coupling():
    grid_sim = GridSimulator(nominal_voltage=220.0, r_grid=0.3, x_grid=0.05, voltage_variation_std=0.0)
    
    zero_c = np.zeros((10, 15), dtype=np.complex64)
    v_bus, kappa = grid_sim.compute_voltage_drop(zero_c)
    assert np.allclose(v_bus, 220.0)
    assert np.allclose(kappa, 1.0)

    heavy_c = np.zeros((10, 15), dtype=np.complex64)
    heavy_c[:, 0] = 6.0 + 0j
    v_bus_heavy, kappa_heavy = grid_sim.compute_voltage_drop(heavy_c)
    assert np.all(v_bus_heavy < 219.0)
    assert np.all(kappa_heavy < 1.0)


def test_data_augmentor_time_warping_and_scaling(segment_pool):
    augmentor = DataAugmentor()
    act = segment_pool.sample_activation("fan")
    orig_len = act.duration_cycles

    target_len = orig_len * 2
    aug_act = augmentor.augment_activation(act, target_duration_cycles=target_len, power_scale=1.1)
    assert aug_act.duration_cycles == target_len
    assert aug_act.net_harmonics_ri.shape == (target_len, 15, 2)
    assert aug_act.net_harmonics_complex.shape == (target_len, 15)
    assert len(aug_act.target_power_w) == target_len


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
