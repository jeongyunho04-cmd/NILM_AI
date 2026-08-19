"""
Automated Pytest Suite for NILM Preprocessing, Multi-Tier Labeling, and NumPy Binary (.npz) Export
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.numpy_exporter import NumpyDatasetExporter, load_nilm_npz
from src.preprocessing.pipeline import PreprocessingPipeline
from src.labeling.state_definitions import get_appliance_config
from src.labeling.state_classifier import StateClassifier
from src.labeling.annotator import DataAnnotator


@pytest.fixture
def sample_raw_dataframe():
    """Generates a synthetic raw dataframe simulating STM32 receiver output."""
    n_frames = 10
    cycles_per_frame = 30
    total_rows = n_frames * cycles_per_frame
    
    rows = []
    for seq in range(n_frames):
        for cycle in range(cycles_per_frame):
            t_s = seq * 0.5 + cycle / 60.0
            idx = seq * 30 + cycle
            if idx < 90:
                p_w = 2.0 + np.random.normal(0, 0.1)
                irms = 0.01 + np.random.normal(0, 0.001)
            elif idx < 180:
                p_w = 24.0 + np.random.normal(0, 0.3)
                irms = 0.12 + np.random.normal(0, 0.002)
            else:
                p_w = 42.0 + np.random.normal(0, 0.5)
                irms = 0.18 + np.random.normal(0, 0.003)
                
            row = {
                "host_time": "2026-08-11 12:00:00.000",
                "t_s": t_s,
                "seq": seq,
                "cycle": cycle,
                "irms": max(0.001, irms),
                "p_w": p_w,
                "phase_deg": 15.0,
                "range": 0,
                "over_count": 0,
                "over_range": 0,
                "freq_hz": 60.0,
                "vrms": 220.0,
                "thd_v": 0.015,
                "pll_locked": 1,
                "cal_applied": 1,
                "win_range": 0,
                "win_over_range_count": 0,
                "win_clip_volt_count": 0,
            }
            # Harmonics
            for h in range(1, 16):
                row[f"ih{h}"] = irms * (0.8 if h == 1 else 0.1 / h)
                row[f"ihdeg{h}"] = 10.0 * h
                row[f"vh{h}"] = 220.0 if h == 1 else 1.0 / h
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    # Introduce out-of-order frames simulating Wi-Fi retransmission
    f3 = df[(df["seq"] == 3)].copy()
    f4 = df[(df["seq"] == 4)].copy()
    df_reordered = pd.concat([df[df["seq"] < 3], f4, f3, df[df["seq"] > 4]], ignore_index=True)
    df_reordered.loc[50, "p_w"] = -150.0
    
    return df_reordered


def test_cleaner_sorting_and_glitch_filtering(sample_raw_dataframe):
    cleaner = DataCleaner(sampling_hz=60.0, noise_floor_w=1.4)
    cleaned_df, stats = cleaner.clean_dataframe(sample_raw_dataframe)
    
    seqs = cleaned_df["seq"].values
    cycles = cleaned_df["cycle"].values
    total_idx = seqs * 30 + cycles
    assert np.all(np.diff(total_idx) >= 0), "Dataframe is not sorted monotonically!"
    assert (cleaned_df["p_w"] < 0).sum() == 0, "Negative power was not clamped!"
    assert (cleaned_df["p_target_w"] < 0).sum() == 0, "Target power has negative values!"
    
    t_diff = np.diff(cleaned_df["t_rel_s"].values)
    assert np.allclose(t_diff, 1.0 / 60.0, atol=1e-5), "Timeline is not strictly 60Hz continuous!"


def test_feature_extractor_physical_consistency(sample_raw_dataframe):
    cleaner = DataCleaner(sampling_hz=60.0)
    cleaned_df, _ = cleaner.clean_dataframe(sample_raw_dataframe)
    
    extractor = FeatureExtractor(harmonics_count=15)
    feat_df = extractor.extract_features(cleaned_df)
    
    assert np.all(feat_df["s_va"] >= 0)
    assert np.all(feat_df["power_factor"] >= 0.0)
    assert np.all(feat_df["power_factor"] <= 1.0)
    assert np.all(feat_df["thd_i"] >= 0.0)
    
    for h in range(2, 16):
        assert f"ih_ratio_{h}" in feat_df.columns
        assert f"ih_re_{h}" in feat_df.columns
        assert f"ih_im_{h}" in feat_df.columns


def test_state_classifier_and_annotator(sample_raw_dataframe):
    pipeline = PreprocessingPipeline(sampling_hz=60.0)
    cleaned_df, stats = pipeline.cleaner.clean_dataframe(sample_raw_dataframe)
    feat_df = pipeline.extractor.extract_features(cleaned_df)
    
    annotator = DataAnnotator(sampling_hz=60.0)
    annotated_df, events, summary = annotator.annotate_dataframe(feat_df, appliance_type="fan")
    
    assert "is_on" in annotated_df.columns
    assert "state_id" in annotated_df.columns
    assert "state_name" in annotated_df.columns
    assert "target_power_w" in annotated_df.columns
    
    unique_states = set(annotated_df["state_id"].unique())
    assert 0 in unique_states
    assert 1 in unique_states
    assert 3 in unique_states
    assert len(events) >= 2


def test_state_ids_are_contiguous_from_zero():
    """상태 ID 에 빈 번호가 있으면 모델 출력 차원에 아무도 채우지 않는 자리가 생긴다."""
    from src.labeling.state_definitions import STATE_CONFIGURATIONS

    for name, config in STATE_CONFIGURATIONS.items():
        ids = sorted(s.state_id for s in config.states)
        assert ids == list(range(len(ids))), \
            f"{name}: 상태 ID 가 0부터 연속이 아닙니다 -> {ids}"


def test_state_power_ranges_have_no_gaps_or_overlaps():
    """상태 구간이 끊기거나 겹치면 그 사이 전력값이 엉뚱한 상태로 분류된다."""
    from src.labeling.state_definitions import STATE_CONFIGURATIONS

    for name, config in STATE_CONFIGURATIONS.items():
        states = sorted(config.states, key=lambda s: s.p_min)
        assert states[0].p_min == 0.0, f"{name}: 첫 상태가 0W 에서 시작하지 않습니다"
        for lower, upper in zip(states, states[1:]):
            assert lower.p_max == upper.p_min, \
                f"{name}: {lower.name}({lower.p_max}W) 와 {upper.name}({upper.p_min}W) 구간이 맞닿지 않습니다"


def test_on_threshold_matches_off_state_boundary():
    """on_threshold_w 와 OFF 상태 상한이 어긋나면 ON 판정 기준이 두 개가 된다."""
    from src.labeling.state_definitions import STATE_CONFIGURATIONS

    for name, config in STATE_CONFIGURATIONS.items():
        off_state = min(config.states, key=lambda s: s.p_min)
        if config.on_threshold_w >= 9999.0:
            continue  # 노이즈 기준 파일은 항상 OFF 로 둔다
        assert abs(off_state.p_max - config.on_threshold_w) < 1e-6, \
            f"{name}: on_threshold_w={config.on_threshold_w} 인데 OFF 상한은 {off_state.p_max}"


@pytest.mark.parametrize("stem", ["oven", "air_conditioner", "minipc_2", "fan_2"])
def test_every_defined_state_is_actually_observed(stem):
    """정의만 되어 있고 실측에서 한 번도 나오지 않는 상태가 있으면 안 된다.

    오븐의 MEDIUM_HEAT(단일 히터 400W)가 그런 경우였다. 관측 0분인데 클래스로
    잡혀 있어, 학습 예시가 하나도 없는 자리를 모델이 예측하도록 요구하고 있었다.
    (측정 오븐은 목표 온도와 시간만 조절하는 모델이라 애초에 그런 모드가 없다)
    """
    npz_path = Path("processed_data/npz") / f"{stem}.npz"
    if not npz_path.exists():
        pytest.skip(f"{npz_path} 없음 - 전처리를 먼저 실행하세요")

    data = load_nilm_npz(npz_path)
    distribution = data["metadata"].get("state_distribution", {})
    if not distribution:
        pytest.skip("상태 분포 메타데이터 없음")

    unobserved = [
        f"{name}(id={d['state_id']})"
        for name, d in distribution.items()
        if d["count"] == 0
    ]
    assert not unobserved, (
        f"{stem}: 실측에서 한 번도 관측되지 않은 상태 -> {', '.join(unobserved)}. "
        f"해당 모드가 기기에 없다면 state_definitions.py 에서 제거하고, "
        f"있는데 못 잡은 것이라면 그 모드를 측정하세요."
    )


def test_numpy_binary_export_and_complex_channels(sample_raw_dataframe):
    pipeline = PreprocessingPipeline(sampling_hz=60.0)
    cleaned_df, _ = pipeline.cleaner.clean_dataframe(sample_raw_dataframe)
    feat_df = pipeline.extractor.extract_features(cleaned_df)
    
    annotator = DataAnnotator(sampling_hz=60.0)
    annotated_df, events, summary = annotator.annotate_dataframe(feat_df, appliance_type="fan")
    
    exporter = NumpyDatasetExporter(harmonics_count=15)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        npz_out = Path(tmp_dir) / "test_export.npz"
        res_path = exporter.export_to_npz(
            annotated_df,
            output_path=npz_out,
            metadata={"test_key": "test_val"},
            compress=True,
        )
        assert Path(res_path).exists()
        
        # Load and verify contents
        loaded = load_nilm_npz(res_path)
        
        # Verify 2-channel Real/Imag shape: (N, 15, 2)
        n_samples = len(annotated_df)
        assert loaded["harmonics_ri"].shape == (n_samples, 15, 2)
        assert loaded["harmonics_ri"].dtype == np.float32
        
        # Verify complex64 array: (N, 15)
        assert loaded["harmonics_complex"].shape == (n_samples, 15)
        assert loaded["harmonics_complex"].dtype == np.complex64
        
        # Verify Real + j*Imag identity
        r_part = loaded["harmonics_ri"][:, :, 0]
        i_part = loaded["harmonics_ri"][:, :, 1]
        reconstructed_complex = r_part + 1j * i_part
        assert np.allclose(reconstructed_complex, loaded["harmonics_complex"], atol=1e-5)
        
        # Verify power features and labels
        assert loaded["power_features"].shape == (n_samples, 6)
        assert loaded["is_on"].shape == (n_samples,)
        assert loaded["state_id"].shape == (n_samples,)
        assert loaded["target_power_w"].shape == (n_samples,)
        assert loaded["metadata"]["test_key"] == "test_val"
