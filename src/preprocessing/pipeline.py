"""
Preprocessing Pipeline for NILM AI
Executes full data cleaning and feature engineering across all raw appliance CSV files.

파일의 역할(단일 가전 / 노이즈 / 복합 부하 검증용)은 file_registry 가 단독으로 결정한다.
파이프라인은 더 이상 파일 이름을 가전 종류로 추측하지 않는다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from .cleaner import DataCleaner
from .feature_extractor import FeatureExtractor
from .file_registry import (
    FileClassification,
    FileRole,
    UnregisteredFileError,
    classify_file,
    require_known,
)


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for NILM raw files."""

    def __init__(
        self,
        sampling_hz: float = 60.0,
        noise_floor_w: float = 1.4,
        harmonics_count: int = 15,
        quality_gating: bool = True,
    ):
        self.cleaner = DataCleaner(
            sampling_hz=sampling_hz,
            noise_floor_w=noise_floor_w,
            quality_gating=quality_gating,
        )
        self.extractor = FeatureExtractor(harmonics_count=harmonics_count)

    def process_file(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        strict: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Loads, cleans, extracts features, and optionally saves the preprocessed dataset.

        Args:
            strict: True 이면 미등록 파일에 대해 UnregisteredFileError 를 발생시킨다.
                    이전처럼 파일명을 가전 종류로 추측해 조용히 넘어가지 않는다.
        """
        path = Path(file_path)
        classification = classify_file(path)
        if strict:
            require_known(classification)

        stem = classification.stem

        # Load raw CSV
        df_raw = pd.read_csv(path)
        # 보드 교정 상태가 틀린 채 녹화된 파일은 등록부가 정한 만큼 위상을 되돌린다 (12.184.3).
        # 그리고 2026-09-04 17:00 이전 파일은 LOW 교정 규약(0.44 -> 2.62°)에 맞춰 range==0 사이클을
        # −2.18°×h 돌린다 (12.184.13). 둘 다 read_raw_csv 와 같은 규칙이다.
        from src.preprocessing.file_registry import low_cal_shift_deg, phase_fix_of
        from src.preprocessing.raw_phasors import apply_phase_rotation
        _fix = phase_fix_of(stem)
        if _fix:
            df_raw = apply_phase_rotation(df_raw, _fix)
        _ht = str(df_raw["host_time"].iloc[0]) if "host_time" in df_raw.columns and len(df_raw) else None
        _shift = low_cal_shift_deg(stem, _ht)
        if _shift and "range" in df_raw.columns:
            df_raw = apply_phase_rotation(df_raw, _shift, mask=df_raw["range"].to_numpy() == 0)

        # 계측 보드 전원 방식에 따른 바닥 전력은 레지스트리가 정한다.
        df_clean, clean_stats = self.cleaner.clean_dataframe(
            df_raw, custom_noise_floor=classification.noise_floor_w
        )

        # Extract physical and harmonic features
        df_features = self.extractor.extract_features(df_clean)

        # Add metadata columns
        df_features["source_file"] = path.name
        df_features["appliance_type"] = classification.appliance_type or stem
        df_features["file_role"] = classification.role.value

        # Save to output directory if specified
        output_file = None
        if output_dir is not None:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_dir / f"{stem}_clean.csv"
            df_features.to_csv(output_file, index=False)

        # 이 측정 구간의 대표 계통 전압. 합성 시 전압 스케일링의 기준(v_ref)이 되며,
        # 하드코딩된 220V 대신 이 값을 써야 물리적으로 올바른 환산이 된다.
        valid = df_features["is_valid"] == 1 if "is_valid" in df_features.columns else slice(None)
        v_series = df_features.loc[valid, "vrms"] if "vrms" in df_features.columns else pd.Series(dtype=float)
        v_ref = float(v_series.median()) if len(v_series) else 220.0

        stats = {
            **clean_stats,
            "source_file": path.name,
            "file_role": classification.role.value,
            "appliance_type": classification.appliance_type or stem,
            "load_class": classification.load_class.value,
            "classification_reason": classification.reason,
            "output_file": str(output_file) if output_file else None,
            "feature_columns_count": len(df_features.columns),
            "v_ref_v": round(v_ref, 2),
            "p_mean": round(float(df_features["p_w"].mean()), 2),
            "p_max": round(float(df_features["p_w"].max()), 2),
            "irms_mean": round(float(df_features["irms"].mean()), 4),
        }

        return df_features, stats

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.csv",
        roles: Optional[List[FileRole]] = None,
        strict: bool = True,
    ) -> Dict[str, Dict]:
        """Processes all CSV files matching pattern in input_dir.

        Args:
            roles: 처리할 역할 목록. 기본은 단일 가전과 노이즈만 처리하고
                   복합 부하 검증 파일(test*, nilm_*)은 건너뛴다.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        allowed = set(roles or [FileRole.DEVICE, FileRole.NOISE])

        files = sorted(in_path.glob(pattern))
        all_stats = {}

        for f in files:
            classification = classify_file(f)
            if strict:
                require_known(classification)
            if classification.role not in allowed:
                continue
            _, stats = self.process_file(f, output_dir=out_path, strict=strict)
            all_stats[classification.stem] = stats

        return all_stats

    @staticmethod
    def survey_directory(input_dir: Union[str, Path], pattern: str = "*.csv") -> Dict[str, List[str]]:
        """디렉터리의 파일들을 역할별로 분류만 해서 보여준다 (처리 없음)."""
        survey: Dict[str, List[str]] = {role.value: [] for role in FileRole}
        for f in sorted(Path(input_dir).glob(pattern)):
            c = classify_file(f)
            survey[c.role.value].append(c.stem)
        return survey
