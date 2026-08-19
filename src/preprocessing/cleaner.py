"""
NILM 원본 데이터 정제 모듈 (Data Cleaner)
=========================================
STM32 및 ESP-01S에서 60Hz 주기로 수신된 원본 전력 CSV 데이터를 정제합니다.

[주요 기능]
1. 패킷 순서 정렬 및 중복 제거 (sort_and_deduplicate):
   - Wi-Fi 패킷 재전송으로 인한 프레임 역전(seq 37 -> 28 -> 38 등)을 seq와 cycle 복합 정렬로 정상화.
2. 60Hz 고정 타임라인 복원 및 결측 보간 (reconstruct_60hz_grid):
   - 패킷 유실로 인한 결측 사이클 선형 보간, 다중 세션 분절 봉합(연속 상대시간 t_rel_s 부여).
3. 물리적 글리치 및 이상치 필터링 (filter_glitches_and_spikes):
   - 스위치 접점 바운스 등으로 인한 1사이클 순간 음수 전력 클램핑(P >= 0) 및 단발성 스파이크 제거.
4. 센서 노이즈 바닥 차감 및 기기 순수 전력 보정 (calibrate_and_zero):
   - 계측 보드 자체 소비 전력 바닥값(~1.4W)을 차감하여 기기 정미 전력(p_target_w) 산출.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DataCleaner:
    """60Hz 주기 전력 계측 원본 데이터를 정밀 정제하는 클래스."""

    def __init__(
        self,
        sampling_hz: float = 60.0,
        cycles_per_frame: int = 30,
        noise_floor_w: float = 1.4,
        max_interpolation_gap_s: float = 5.0,
    ):
        self.sampling_hz = sampling_hz  # 계통 기본 주파수 (60Hz = 초당 60사이클)
        self.cycles_per_frame = cycles_per_frame  # 1개 통신 프레임당 포함된 계통 사이클 수 (0.5초 = 30사이클)
        self.noise_floor_w = noise_floor_w  # 계측 센서 무부하 바닥 노이즈 (외부전원 시 약 1.4W)
        self.max_interpolation_gap_s = max_interpolation_gap_s  # 보간을 허용할 최대 결측 시간 (초)

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        custom_noise_floor: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
        """단일 데이터프레임에 대한 전처리 및 정제 파이프라인 전체를 실행합니다.

        Returns:
            cleaned_df: 60Hz 연속 시간축과 결측치 없이 정제된 DataFrame
            stats: 정제 과정에서 처리된 통계 딕셔너리 (제거된 중복, 보간된 사이클 등)
        """
        raw_rows = len(df)
        noise_p = self.noise_floor_w if custom_noise_floor is None else custom_noise_floor

        # 1단계: seq와 cycle 기준 엄격한 정렬 및 중복 패킷 제거
        df_sorted, dup_count = self.sort_and_deduplicate(df)

        # 2단계: 다중 세션 분절 봉합 및 60Hz 고정 시간 격자망 구성 (작은 누락 구간은 선형 보간)
        df_grid, interp_stats = self.reconstruct_60hz_grid(df_sorted)

        # 3단계: 물리적으로 불가능한 순간 음수 전력 및 스위치 아크 노이즈 제거
        df_filtered, filter_stats = self.filter_glitches_and_spikes(df_grid)

        # 4단계: 센서 자체 소비 전력 바닥값 차감하여 순수 기기 소비 전력(p_target_w) 생성
        df_calibrated = self.calibrate_and_zero(df_filtered, noise_floor_w=noise_p)

        stats = {
            "raw_rows": raw_rows,
            "cleaned_rows": len(df_calibrated),
            "duration_s": round(len(df_calibrated) / self.sampling_hz, 2),
            "duplicates_removed": dup_count,
            "interpolated_cycles": interp_stats["interpolated_cycles"],
            "glitch_spikes_fixed": filter_stats["glitches_fixed"],
            "negative_power_clamped": filter_stats["negative_p_clamped"],
            "noise_floor_applied_w": noise_p,
        }

        return df_calibrated, stats

    def sort_and_deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Wi-Fi 재전송으로 인해 늦게 도착한 패킷의 순서를 복원하고 중복 패킷을 제거합니다."""
        initial_len = len(df)
        df_clean = df.drop_duplicates(subset=["seq", "cycle"]).copy()
        dup_count = initial_len - len(df_clean)

        # seq(프레임 번호) 오름차순, 그 안에서 cycle(0~29) 순서대로 정렬
        df_clean = df_clean.sort_values(by=["seq", "cycle"]).reset_index(drop=True)
        return df_clean, dup_count

    def reconstruct_60hz_grid(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """패킷 손실 구간을 감지하여 60Hz 균일 그리드로 재배치하고, 연속 상대시간(t_rel_s)을 부여합니다."""
        if len(df) == 0:
            return df.copy(), {"interpolated_cycles": 0}

        max_gap_frames = int(self.max_interpolation_gap_s * (self.sampling_hz / self.cycles_per_frame))

        seqs = df["seq"].values
        cycles = df["cycle"].values

        # 전체 절대 사이클 인덱스: seq * 30 + cycle
        total_cycle_idx = seqs.astype(np.int64) * self.cycles_per_frame + cycles.astype(np.int64)
        cycle_diff = np.diff(total_cycle_idx)

        # 세션 리셋(음수 점프) 또는 긴 시간 단절(max_gap_frames 초과)을 감지하여 분절
        max_cycle_gap = max_gap_frames * self.cycles_per_frame
        split_indices = np.where((cycle_diff <= 0) | (cycle_diff > max_cycle_gap))[0] + 1

        segments = []
        start_idx = 0
        interpolated_total = 0

        for end_idx in list(split_indices) + [len(df)]:
            seg = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            if len(seg) > 0:
                seg_grid, n_interp = self._fill_segment_gaps(seg)
                segments.append(seg_grid)
                interpolated_total += n_interp
            start_idx = end_idx

        # 분절된 모든 유효 세그먼트를 단일 연속 타임라인으로 병합
        combined_df = pd.concat(segments, ignore_index=True)

        # 60Hz 균일 샘플 인덱스(0, 1, 2...)와 상대 시간축(초) 부여
        combined_df["sample_idx"] = np.arange(len(combined_df), dtype=np.int64)
        combined_df["t_rel_s"] = combined_df["sample_idx"] / self.sampling_hz

        return combined_df, {"interpolated_cycles": interpolated_total}

    def _fill_segment_gaps(self, seg: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """연속 세그먼트 내부에서 1~3프레임 정도 비어있는 작은 누락 사이클을 선형 보간합니다."""
        total_idx = (seg["seq"].values.astype(np.int64) * self.cycles_per_frame + seg["cycle"].values.astype(np.int64))
        min_idx = total_idx[0]
        max_idx = total_idx[-1]

        expected_count = max_idx - min_idx + 1
        if expected_count == len(seg):
            return seg.copy(), 0

        # 누락 없는 완전한 사이클 배열 생성
        full_idx_array = np.arange(min_idx, max_idx + 1, dtype=np.int64)
        interpolated_count = len(full_idx_array) - len(seg)

        # 재색인 후 수치형 컬럼 선형 보간 (전력, 전류, 고조파 등)
        seg_indexed = seg.copy()
        seg_indexed["_full_cycle_idx"] = total_idx
        seg_indexed = seg_indexed.set_index("_full_cycle_idx").reindex(full_idx_array)

        numeric_cols = seg_indexed.select_dtypes(include=[np.number]).columns
        seg_indexed[numeric_cols] = seg_indexed[numeric_cols].interpolate(method="linear")

        # 플래그 및 문자열 컬럼 앞/뒤 채우기
        seg_indexed = seg_indexed.ffill().bfill().reset_index(drop=True)

        seg_indexed["seq"] = full_idx_array // self.cycles_per_frame
        seg_indexed["cycle"] = full_idx_array % self.cycles_per_frame

        return seg_indexed, interpolated_count

    def filter_glitches_and_spikes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """스위치 접점 아크 등으로 인한 비물리적 음수 전력 클램핑 및 단발성 임펄스 노이즈를 필터링합니다."""
        df_clean = df.copy()

        # 1. 소비 가전의 비물리적 음수 전력 클램핑 (P >= 0)
        neg_p_mask = df_clean["p_w"] < 0
        neg_count = int(neg_p_mask.sum())
        df_clean.loc[neg_p_mask, "p_w"] = 0.0

        # 고조파 실효값 음수 클램핑
        for col in [c for c in df_clean.columns if c.startswith("ih") and not c.startswith("ihdeg")]:
            df_clean.loc[df_clean[col] < 0, col] = 0.0
        for col in [c for c in df_clean.columns if c.startswith("vh")]:
            df_clean.loc[df_clean[col] < 0, col] = 0.0
        df_clean.loc[df_clean["irms"] < 0, "irms"] = 0.0

        # 2. 7사이클 로컬 롤링 미디언을 이용한 단 1사이클 단발성 통신 글리치 스파이크 복원
        p = df_clean["p_w"].values
        p_series = pd.Series(p)
        rolling_med = p_series.rolling(window=7, center=True, min_periods=1).median().values
        rolling_std = p_series.rolling(window=15, center=True, min_periods=1).std().fillna(0).values

        residual = np.abs(p - rolling_med)
        spike_thresh = np.maximum(25.0, 4.0 * rolling_std)
        is_spike = (residual > spike_thresh)

        spike_count = 0
        for i in np.where(is_spike)[0]:
            prev_ok = (i == 0) or (abs(p[i - 1] - rolling_med[i - 1]) <= spike_thresh[i - 1])
            next_ok = (i == len(p) - 1) or (abs(p[i + 1] - rolling_med[i + 1]) <= spike_thresh[i + 1])
            if prev_ok and next_ok:
                df_clean.at[i, "p_w"] = rolling_med[i]
                spike_count += 1

        return df_clean, {"negative_p_clamped": neg_count, "glitches_fixed": spike_count}

    def calibrate_and_zero(self, df: pd.DataFrame, noise_floor_w: float) -> pd.DataFrame:
        """계측 센서 자체 소비 전력 바닥값을 차감하여 순수 기기 유효 전력(p_target_w)을 산출합니다."""
        df_cal = df.copy()
        df_cal["p_target_w"] = np.maximum(0.0, df_cal["p_w"] - noise_floor_w)
        return df_cal
