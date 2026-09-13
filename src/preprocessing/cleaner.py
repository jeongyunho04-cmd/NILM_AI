"""
NILM 원본 데이터 정제 모듈 (Data Cleaner)
=========================================
STM32 및 ESP-01S에서 60Hz 주기로 수신된 원본 전력 CSV 데이터를 정제합니다.

[주요 기능]
1. 세션 분리 (assign_sessions):
   - 보드가 리셋되면 seq 가 0부터 다시 시작한다. 단순 정렬/중복제거만 하면
     두 번째 세션이 첫 세션과 (seq, cycle) 이 겹쳐 통째로 삭제되므로,
     정렬보다 먼저 세션 경계를 찾아 분리한다.
2. 패킷 순서 정렬 및 중복 제거 (sort_and_deduplicate):
   - Wi-Fi 재전송으로 인한 프레임 역전(seq 37 -> 28 -> 38 등)을 세션 안에서 정상화.
3. 계측 품질 게이팅 (mask_invalid_samples):
   - pll_locked=0, 측정범위 초과, 전압 클리핑 구간을 무효 표시한다.
     실측에서 laptop_charger_2.csv 의 10.27%(23.5초 연속)가 PLL 미락 상태였고
     그 구간 vrms 가 0.17V 까지 떨어져 S=V*I, PF, Q 가 전부 오염되어 있었다.
4. 60Hz 고정 타임라인 복원 및 결측 보간 (reconstruct_60hz_grid):
   - 짧은 결측/무효 구간은 선형 보간, 긴 구간은 세그먼트를 끊어 버린다.
5. 물리적 글리치 및 이상치 필터링 (filter_glitches_and_spikes).
6. 센서 노이즈 바닥 차감 및 기기 순수 전력 보정 (calibrate_and_zero).

[2026-09-06 녹화 규약 — MEASUREMENT_RULES.md 규칙 2·3]
7. seq 0 프레임 배제 (drop_startup_frames): 계측기가 측정을 시작하는 첫 0.5초는 pll 미락·전압 클립·
   vrms 2~3V 오차가 실린다. 보간하지 않고 버린다. 세션 앞머리의 무효 구간도 버린다 (앞에 기댈 값이 없다).
8. 플러그 뽑은 꼬리 검출 (detect_trailing_unplugged): 단독 녹화는 끝내기 전에 기기 플러그를 뽑고 ~10초를
   더 찍는다. 그 꼬리(계측계만 남은 구간)를 `is_unplugged=1` 로 표시하고 그 중앙값을 **이 파일의** 노이즈
   바닥으로 쓴다 — 계측계 자체 전류가 세션마다 1.3~1.6W / 6.2~7.5mA / −21~+13° 로 다르다. 꼬리가 없는
   파일(충전기_1: 충전 중 종료, 오븐_1: 대기 상태로 종료)은 등록부 상수로 돌아간다 (`noise_floor_source`).
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# 수신기(nilm_receiver.py)의 REORDER_MAX 와 맞춘 값.
# 이보다 크게 seq 가 뒤로 점프하면 순서 뒤바뀜이 아니라 보드 리셋으로 본다.
REORDER_TOLERANCE_FRAMES = 32

# 계통 전압으로 물리적으로 성립 가능한 범위. 벗어나면 계측 실패로 간주한다.
VALID_VRMS_RANGE = (150.0, 280.0)

#: 이 값보다 작은 seq 프레임은 버린다 (= seq 0). 계측기 시작 단계 (규칙 3). 세션마다 적용된다.
STARTUP_SEQ_DROP = 1

# ── 플러그 뽑은 꼬리 (규칙 2) ─────────────────────────────────────────────
#: 꼬리로 인정할 최소 길이 (사이클). 규약은 ~10초, 실측 최소 12초 — 5초면 넉넉하다.
UNPLUGGED_MIN_CYCLES = 300
#: 계측계만 남은 구간의 판정: P <= 등록부 바닥 + 이 값 (W), |I1| <= 기준 × 배수.
#: 실측 꼬리 1.33~1.60W / 6.2~7.5mA. 걸러야 할 것: 프로젝터 대기 3.0W/27mA, 미니PC 3.4W/31mA,
#: 오븐 대기 2.5W/14.4mA, 드라이기 꽂힘 1.64W/11.5mA (X-cap). 핫플·포트·선풍기의 꽂힌 OFF 는
#: 계측계와 같은 값이라 구분이 안 되지만 그 경우 어느 쪽을 써도 같다.
UNPLUGGED_P_MARGIN_W = 0.6
UNPLUGGED_I1_FACTOR = 1.3
UNPLUGGED_I1_REF_A = {"external": 0.0075, "selfpower": 0.0110}   # noise_* 파일 실측 |I1|


class DataCleaner:
    """60Hz 주기 전력 계측 원본 데이터를 정밀 정제하는 클래스."""

    def __init__(
        self,
        sampling_hz: float = 60.0,
        cycles_per_frame: int = 30,
        noise_floor_w: float = 1.4,
        max_interpolation_gap_s: float = 5.0,
        quality_gating: bool = True,
        valid_vrms_range: Tuple[float, float] = VALID_VRMS_RANGE,
    ):
        self.sampling_hz = sampling_hz  # 계통 기본 주파수 (60Hz = 초당 60사이클)
        self.cycles_per_frame = cycles_per_frame  # 1개 통신 프레임당 계통 사이클 수 (0.5초 = 30사이클)
        self.noise_floor_w = noise_floor_w  # 계측 센서 무부하 바닥 노이즈
        self.max_interpolation_gap_s = max_interpolation_gap_s  # 보간을 허용할 최대 결측 시간 (초)
        self.quality_gating = quality_gating  # 계측 품질 플래그로 무효 샘플을 걸러낼지 여부
        self.valid_vrms_range = valid_vrms_range

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        custom_noise_floor: Optional[float] = None,
        detect_unplugged_tail: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
        """단일 데이터프레임에 대한 전처리 및 정제 파이프라인 전체를 실행합니다.

        `detect_unplugged_tail`: 단독 기기 녹화만 True (규칙 2). 노이즈 파일은 파일 전체가 계측계라 끄고,
        복합 녹화는 꼬리 규약이 없다.

        Returns:
            cleaned_df: 60Hz 연속 시간축과 결측치 없이 정제된 DataFrame
            stats: 정제 과정에서 처리된 통계 딕셔너리
        """
        raw_rows = len(df)
        noise_p = self.noise_floor_w if custom_noise_floor is None else custom_noise_floor

        # 1단계: 보드 리셋으로 seq 가 되감긴 지점을 찾아 세션을 나눈다 (정렬보다 먼저!)
        df_sessioned, session_count = self.assign_sessions(df)

        # 1b단계: seq 0 프레임(계측기 시작 단계)을 버린다 (규칙 3)
        df_sessioned, startup_dropped = self.drop_startup_frames(df_sessioned)

        # 2단계: 세션 안에서 seq/cycle 기준 정렬 및 중복 패킷 제거
        df_sorted, dup_count = self.sort_and_deduplicate(df_sessioned)

        # 3단계: 계측 품질 플래그로 신뢰할 수 없는 샘플을 무효 표시
        df_gated, quality_stats = self.mask_invalid_samples(df_sorted)

        # 4단계: 60Hz 고정 시간 격자망 구성 (짧은 공백은 보간, 긴 공백은 분절)
        df_grid, interp_stats = self.reconstruct_60hz_grid(df_gated)

        # 5단계: 물리적으로 불가능한 순간 음수 전력 및 스위치 아크 노이즈 제거
        df_filtered, filter_stats = self.filter_glitches_and_spikes(df_grid)

        # 5b단계: 녹화 끝의 '플러그 뽑은 꼬리' (규칙 2). 있으면 이 파일의 바닥이 된다.
        if detect_unplugged_tail:
            df_filtered, tail = self.detect_trailing_unplugged(df_filtered, noise_floor_w=noise_p)
        else:
            df_filtered = df_filtered.copy()
            df_filtered["is_unplugged"] = np.zeros(len(df_filtered), dtype=np.int8)
            tail = None
        floor_applied = float(tail["p_w"]) if tail is not None else noise_p
        floor_source = "trailing_unplugged" if tail is not None else "registry"

        # 6단계: 센서 자체 소비 전력 바닥값 차감하여 순수 기기 소비 전력(p_target_w) 생성
        df_calibrated = self.calibrate_and_zero(df_filtered, noise_floor_w=floor_applied,
                                                source=floor_source)

        stats = {
            "raw_rows": raw_rows,
            "cleaned_rows": len(df_calibrated),
            "duration_s": round(len(df_calibrated) / self.sampling_hz, 2),
            "sessions_detected": session_count,
            "duplicates_removed": dup_count,
            "invalid_samples_flagged": quality_stats["invalid_flagged"],
            "invalid_samples_dropped": quality_stats["invalid_dropped"],
            "invalid_samples_interpolated": quality_stats["invalid_interpolated"],
            "interpolated_cycles": interp_stats["interpolated_cycles"],
            "timeline_segments": interp_stats["segments"],
            "glitch_spikes_fixed": filter_stats["glitches_fixed"],
            "negative_power_clamped": filter_stats["negative_p_clamped"],
            "startup_frames_dropped": startup_dropped,
            "noise_floor_registry_w": noise_p,
            "noise_floor_applied_w": floor_applied,
            "noise_floor_source": floor_source,
            "trailing_noise": tail,
        }

        return df_calibrated, stats

    # ── 1단계: 세션 분리 ─────────────────────────────────────────────────────
    def assign_sessions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """보드 리셋으로 seq 가 되감긴 지점을 찾아 session_id 를 부여합니다.

        수신기는 CSV 를 이어쓰기(append)로 열기 때문에, 한 파일 안에 보드가 리셋된
        두 세션이 들어 있을 수 있다. 그 경우 두 세션의 (seq, cycle) 이 겹치는데,
        세션 구분 없이 drop_duplicates 를 돌리면 두 번째 세션이 통째로 사라진다.

        순서 뒤바뀜(재전송)과 리셋을 구분하는 기준은 되감긴 폭이다.
        재전송 역전은 REORDER_TOLERANCE_FRAMES 이내이고, 리셋은 그보다 훨씬 크게 떨어진다.
        """
        if len(df) == 0:
            out = df.copy()
            out["session_id"] = np.array([], dtype=np.int64)
            return out, 0

        seqs = df["seq"].values.astype(np.int64)

        # 지금까지 본 최댓값보다 크게 뒤로 떨어지면 리셋으로 판정한다.
        running_max = np.maximum.accumulate(seqs)
        is_reset = seqs < (running_max - REORDER_TOLERANCE_FRAMES)

        # 리셋 판정이 연속으로 뜨는 구간은 하나의 경계로 묶는다.
        # (리셋 직후 몇 프레임은 계속 이전 running_max 보다 작기 때문)
        boundary = is_reset & ~np.concatenate([[False], is_reset[:-1]])

        session_id = np.cumsum(boundary.astype(np.int64))

        out = df.copy()
        out["session_id"] = session_id
        return out, int(session_id.max()) + 1

    # ── 1b단계: 시작 프레임 폐기 ───────────────────────────────────────────
    def drop_startup_frames(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """`seq < STARTUP_SEQ_DROP`(= seq 0) 행을 버린다. 리셋으로 seq 가 되감긴 세션마다 걸린다.

        실측(2026-09-06, 13파일): seq 0 은 pll_locked=0 이 30사이클, 전압 클립 창 1개, vrms 가 다음
        프레임과 2~3V 다르다. 보간으로 메우면 그 값이 첫 0.5초를 지어내므로 **버린다**.
        """
        if len(df) == 0 or "seq" not in df.columns:
            return df, 0
        m = df["seq"].values.astype(np.int64) < STARTUP_SEQ_DROP
        n = int(m.sum())
        if n == 0:
            return df, 0
        return df.loc[~m].reset_index(drop=True), n

    # ── 2단계: 정렬 및 중복 제거 ─────────────────────────────────────────────
    def sort_and_deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Wi-Fi 재전송으로 늦게 도착한 패킷의 순서를 복원하고 중복 패킷을 제거합니다."""
        if "session_id" not in df.columns:
            df = df.copy()
            df["session_id"] = 0

        initial_len = len(df)
        # 세션을 키에 포함해야 리셋 후 겹치는 seq 가 중복으로 오인되지 않는다.
        df_clean = df.drop_duplicates(subset=["session_id", "seq", "cycle"]).copy()
        dup_count = initial_len - len(df_clean)

        df_clean = df_clean.sort_values(by=["session_id", "seq", "cycle"]).reset_index(drop=True)
        return df_clean, dup_count

    # ── 3단계: 계측 품질 게이팅 ──────────────────────────────────────────────
    def mask_invalid_samples(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """계측기가 신뢰할 수 없다고 표시한 샘플을 무효 처리합니다.

        - pll_locked == 0     : 위상동기루프 미락. 전압/주파수/위상 전부 신뢰 불가.
        - over_range / over_count : ADC 측정 범위 초과(클리핑).
        - win_clip_volt_count : 전압 채널 클리핑 발생 창.
        - vrms 가 물리적 범위 밖 : 계측 실패.

        긴 무효 구간은 보간할 수 없으므로 행 자체를 버려 타임라인을 끊고,
        짧은 무효 구간은 물리량을 NaN 으로 만들어 이후 선형 보간에 맡긴다.
        """
        out = df.copy()
        n = len(out)
        if n == 0 or not self.quality_gating:
            out["is_valid"] = np.ones(n, dtype=np.int8)
            return out, {"invalid_flagged": 0, "invalid_dropped": 0, "invalid_interpolated": 0}

        invalid = np.zeros(n, dtype=bool)

        if "pll_locked" in out.columns:
            invalid |= (out["pll_locked"].values == 0)
        for col in ["over_range", "over_count", "win_over_range_count", "win_clip_volt_count"]:
            if col in out.columns:
                invalid |= (pd.to_numeric(out[col], errors="coerce").fillna(0).values > 0)
        if "vrms" in out.columns:
            v = out["vrms"].values
            invalid |= (v < self.valid_vrms_range[0]) | (v > self.valid_vrms_range[1])

        invalid_flagged = int(invalid.sum())
        if invalid_flagged == 0:
            out["is_valid"] = np.ones(n, dtype=np.int8)
            return out, {"invalid_flagged": 0, "invalid_dropped": 0, "invalid_interpolated": 0}

        # 무효 구간을 연속 런으로 묶어 길이에 따라 처리 방식을 나눈다.
        max_run = int(self.max_interpolation_gap_s * self.sampling_hz)
        idx = np.where(invalid)[0]
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)

        drop_mask = np.zeros(n, dtype=bool)
        interp_mask = np.zeros(n, dtype=bool)
        for run in runs:
            if len(run) > max_run:
                drop_mask[run] = True   # 너무 길다 - 버리고 타임라인을 끊는다
            elif run[0] == 0:
                drop_mask[run] = True   # 앞머리 - 기댈 값이 없어 보간이 아니라 날조가 된다 (규칙 3)
            else:
                interp_mask[run] = True  # 짧다 - 보간으로 메운다

        # 짧은 무효 구간의 물리량을 NaN 으로 바꿔 보간 대상으로 만든다.
        # seq/cycle/session_id 는 격자 재구성의 기준이므로 건드리지 않는다.
        structural = {"seq", "cycle", "session_id", "host_time"}
        physical_cols = [
            c for c in out.select_dtypes(include=[np.number]).columns
            if c not in structural
        ]
        if interp_mask.any():
            out.loc[interp_mask, physical_cols] = np.nan

        out["is_valid"] = (~invalid).astype(np.int8)

        # 긴 무효 구간은 행을 제거한다. 제거로 생긴 큰 공백은
        # reconstruct_60hz_grid 가 세그먼트 분절로 자동 처리한다.
        if drop_mask.any():
            out = out.loc[~drop_mask].reset_index(drop=True)

        return out, {
            "invalid_flagged": invalid_flagged,
            "invalid_dropped": int(drop_mask.sum()),
            "invalid_interpolated": int(interp_mask.sum()),
        }

    # ── 4단계: 60Hz 격자 복원 ────────────────────────────────────────────────
    def reconstruct_60hz_grid(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """패킷 손실 구간을 감지해 60Hz 균일 격자로 재배치하고 연속 상대시간(t_rel_s)을 부여합니다."""
        if len(df) == 0:
            out = df.copy()
            return out, {"interpolated_cycles": 0, "segments": 0}

        max_gap_frames = int(self.max_interpolation_gap_s * (self.sampling_hz / self.cycles_per_frame))
        max_cycle_gap = max_gap_frames * self.cycles_per_frame

        seqs = df["seq"].values.astype(np.int64)
        cycles = df["cycle"].values.astype(np.int64)
        sessions = (
            df["session_id"].values.astype(np.int64)
            if "session_id" in df.columns else np.zeros(len(df), dtype=np.int64)
        )

        # 전체 절대 사이클 인덱스: seq * 30 + cycle
        total_cycle_idx = seqs * self.cycles_per_frame + cycles
        cycle_diff = np.diff(total_cycle_idx)
        session_change = np.diff(sessions) != 0

        # 세션 전환, 시간 역행, 긴 단절(5초 초과) 지점에서 타임라인을 끊는다.
        split_indices = np.where(
            session_change | (cycle_diff <= 0) | (cycle_diff > max_cycle_gap)
        )[0] + 1

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

        combined_df = pd.concat(segments, ignore_index=True)

        # 분절된 세그먼트를 이어 붙인 자리를 표시한다. 이 경계에서 나온
        # 상태 전이는 실제 사건이 아니라 이어붙이기 때문일 수 있다.
        seam_flags = np.zeros(len(combined_df), dtype=np.int8)
        offset = 0
        for seg in segments[:-1]:
            offset += len(seg)
            if offset < len(seam_flags):
                seam_flags[offset] = 1
        combined_df["is_segment_seam"] = seam_flags

        # 60Hz 균일 샘플 인덱스와 상대 시간축(초) 부여
        combined_df["sample_idx"] = np.arange(len(combined_df), dtype=np.int64)
        combined_df["t_rel_s"] = combined_df["sample_idx"] / self.sampling_hz

        return combined_df, {
            "interpolated_cycles": interpolated_total,
            "segments": len(segments),
        }

    def _fill_segment_gaps(self, seg: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """세그먼트 안의 누락 사이클과 품질 무효(NaN) 값을 선형 보간합니다."""
        total_idx = (
            seg["seq"].values.astype(np.int64) * self.cycles_per_frame
            + seg["cycle"].values.astype(np.int64)
        )
        min_idx = total_idx[0]
        max_idx = total_idx[-1]

        full_idx_array = np.arange(min_idx, max_idx + 1, dtype=np.int64)
        missing_count = len(full_idx_array) - len(seg)

        seg_indexed = seg.copy()
        seg_indexed["_full_cycle_idx"] = total_idx
        seg_indexed = seg_indexed.set_index("_full_cycle_idx").reindex(full_idx_array)

        # 누락 사이클로 생긴 NaN 과 품질 게이팅으로 만든 NaN 을 함께 보간한다.
        numeric_cols = seg_indexed.select_dtypes(include=[np.number]).columns
        nan_before = int(seg_indexed[numeric_cols].isna().any(axis=1).sum())
        seg_indexed[numeric_cols] = seg_indexed[numeric_cols].interpolate(
            method="linear", limit_direction="both"
        )

        # 플래그 및 문자열 컬럼 앞/뒤 채우기
        seg_indexed = seg_indexed.ffill().bfill().reset_index(drop=True)

        # 보간으로 채운 자리는 계측값이 아니므로 is_valid 를 0 으로 유지한다.
        if "is_valid" in seg_indexed.columns:
            seg_indexed["is_valid"] = seg_indexed["is_valid"].fillna(0).astype(np.int8)

        seg_indexed["seq"] = full_idx_array // self.cycles_per_frame
        seg_indexed["cycle"] = full_idx_array % self.cycles_per_frame

        return seg_indexed, max(missing_count, nan_before)

    # ── 5단계: 글리치 제거 ───────────────────────────────────────────────────
    def filter_glitches_and_spikes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """비물리적 음수 전력 클램핑 및 단발성 임펄스 노이즈를 필터링합니다."""
        df_clean = df.copy()

        # 1. 소비 가전의 비물리적 음수 전력 클램핑 (P >= 0)
        neg_p_mask = df_clean["p_w"] < 0
        neg_count = int(neg_p_mask.sum())
        df_clean.loc[neg_p_mask, "p_w"] = 0.0

        # 고조파 실효값 음수 클램핑 (크기값이므로 음수는 물리적으로 불가능)
        for col in [c for c in df_clean.columns if c.startswith("ih") and not c.startswith("ihdeg")]:
            df_clean.loc[df_clean[col] < 0, col] = 0.0
        # ⚠ 4차 펌웨어(2026-09-04)부터 `vhdeg1~15` 가 있다 — 위상은 음수가 정상이므로 빼야 한다.
        for col in [c for c in df_clean.columns if c.startswith("vh") and not c.startswith("vhdeg")]:
            df_clean.loc[df_clean[col] < 0, col] = 0.0
        df_clean.loc[df_clean["irms"] < 0, "irms"] = 0.0

        # 2. 7사이클 로컬 롤링 미디언으로 단 1사이클 단발성 통신 글리치 복원
        p = df_clean["p_w"].values.astype(float)
        p_series = pd.Series(p)
        rolling_med = p_series.rolling(window=7, center=True, min_periods=1).median().values
        rolling_std = p_series.rolling(window=15, center=True, min_periods=1).std().fillna(0).values

        residual = np.abs(p - rolling_med)
        spike_thresh = np.maximum(25.0, 4.0 * rolling_std)
        is_spike = residual > spike_thresh

        # 앞뒤가 모두 정상인 '고립된' 스파이크만 복원한다.
        # 실제 기기 On/Off 는 여러 사이클이 연속으로 벗어나므로 살아남는다.
        ok = ~is_spike
        prev_ok = np.concatenate([[True], ok[:-1]])
        next_ok = np.concatenate([ok[1:], [True]])
        isolated = is_spike & prev_ok & next_ok

        spike_count = int(isolated.sum())
        if spike_count:
            df_clean.loc[isolated, "p_w"] = rolling_med[isolated]

        return df_clean, {"negative_p_clamped": neg_count, "glitches_fixed": spike_count}

    # ── 5b단계: 플러그 뽑은 꼬리 ───────────────────────────────────────────
    def detect_trailing_unplugged(
        self, df: pd.DataFrame, noise_floor_w: float,
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
        """녹화 **끝**에서 계측계만 남은 구간(기기 플러그를 뽑은 뒤)을 찾아 `is_unplugged` 로 표시한다.

        판정: 끝에서부터 `p_w <= 바닥 + UNPLUGGED_P_MARGIN_W` 이고 `ih1 <= 기준 |I1| × UNPLUGGED_I1_FACTOR`
        인 행이 이어지는 동안. 무효 표본은 조건을 깨지 않고 지나간다. 유효 표본이 UNPLUGGED_MIN_CYCLES
        미만이거나 **파일 전체**가 조건을 만족하면(노이즈 파일) 꼬리로 보지 않는다.
        반환 stats: n_cycles, start_t_rel_s, p_w(중앙), i1_ma, i1_deg(원형 평균), vrms.
        """
        out = df.copy()
        n = len(out)
        out["is_unplugged"] = np.zeros(n, dtype=np.int8)
        if n == 0 or "ih1" not in out.columns:
            return out, None
        mode = "selfpower" if noise_floor_w > 2.0 else "external"
        i1_ref = UNPLUGGED_I1_REF_A[mode]
        p = pd.to_numeric(out["p_w"], errors="coerce").fillna(np.inf).values
        i1 = pd.to_numeric(out["ih1"], errors="coerce").fillna(np.inf).values
        valid = (out["is_valid"].values == 1) if "is_valid" in out.columns else np.ones(n, dtype=bool)
        ok = (p <= noise_floor_w + UNPLUGGED_P_MARGIN_W) & (i1 <= i1_ref * UNPLUGGED_I1_FACTOR)
        k = n
        while k > 0 and (ok[k - 1] or not valid[k - 1]):
            k -= 1
        if k == 0:
            return out, None                         # 파일 전체가 계측계뿐 - 꼬리가 아니라 노이즈 파일
        tail_valid = np.arange(k, n)[valid[k:]]
        if len(tail_valid) < UNPLUGGED_MIN_CYCLES:
            return out, None
        out.loc[out.index[k:], "is_unplugged"] = np.int8(1)
        deg = np.deg2rad(pd.to_numeric(out["ihdeg1"], errors="coerce").values[tail_valid]) \
            if "ihdeg1" in out.columns else np.zeros(len(tail_valid))
        z = np.exp(1j * deg[np.isfinite(deg)])
        stats = {
            "n_cycles": int(len(tail_valid)),
            "start_t_rel_s": float(out["t_rel_s"].values[k]) if "t_rel_s" in out.columns else None,
            "p_w": float(np.median(p[tail_valid])),
            "i1_ma": float(1e3 * np.median(i1[tail_valid])),
            "i1_deg": float(np.degrees(np.angle(z.mean()))) if len(z) else 0.0,
            "vrms": float(np.median(out["vrms"].values[tail_valid])) if "vrms" in out.columns else None,
        }
        return out, stats

    # ── 6단계: 노이즈 바닥 차감 ─────────────────────────────────────────────
    def calibrate_and_zero(self, df: pd.DataFrame, noise_floor_w: float,
                           source: str = "registry") -> pd.DataFrame:
        """계측 센서 자체 소비 전력 바닥값을 차감해 순수 기기 유효 전력(p_target_w)을 산출합니다.

        `noise_floor_w` 는 그 파일의 '플러그 뽑은 꼬리' 중앙값(있으면)이거나 등록부 상수다. 출처를 열에 남긴다.
        """
        df_cal = df.copy()
        df_cal["p_target_w"] = np.maximum(0.0, df_cal["p_w"] - noise_floor_w)
        # 이 데이터에 적용된 바닥값을 남겨 두어야 합성 단계에서
        # 보드 자체 소비를 기기마다 중복으로 더하는 실수를 피할 수 있다.
        df_cal["noise_floor_w"] = float(noise_floor_w)
        df_cal["noise_floor_source"] = str(source)
        return df_cal
