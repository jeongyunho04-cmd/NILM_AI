"""
NILM NumPy 전용 바이너리 데이터셋 변환기 (.npz / .npy)
======================================================
정제된 DataFrame을 PyTorch / 딥러닝 모델 학습에 최적화된 고속 다차원 NumPy 배열로 변환합니다.

[포함되는 텐서 구조]
1. harmonics_ri: (N, 15, 2) float32
   - 채널 0: 1~15차 고조파 전류 실수부 (R_k = I_k * cos(theta_k))
   - 채널 1: 1~15차 고조파 전류 허수부 (I_k = I_k * sin(theta_k))
   - 2D Conv 또는 1D Conv(in_channels=30) 입력에 바로 사용 가능.
2. harmonics_complex: (N, 15) complex64
   - 복소수 페이저 텐서 (Z_k = R_k + j * I_k)
   - 신호 선형 합성(Phasor addition) 시 '+' 연산 한 줄로 초고속 계산 가능.
3. power_features: (N, 6) float32 [p_w, q_var, s_va, power_factor, vrms, thd_i]
4. harmonic_ratios: (N, 14) float32 [ih2/ih1 ~ ih15/ih1]
5. 라벨 및 타깃:
   - is_on: (N,) int8 [0: OFF, 1: ON] (이진 분류용)
   - state_id: (N,) int16 [0..K 다중 상태 ID] (상태 분류용)
   - target_power_w: (N,) float32 [순수 유효전력 타깃] (Seq2Point 회귀용)
   - t_rel_s: (N,) float32 [60Hz 연속 상대 시간]
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd


class NumpyDatasetExporter:
    """정제된 NILM 시계열 데이터를 압축된 NumPy 바이너리 아카이브(.npz)로 변환하는 클래스."""

    def __init__(self, harmonics_count: int = 15):
        self.harmonics_count = harmonics_count

    def dataframe_to_numpy_dict(
        self,
        df: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """DataFrame의 각 열을 고효율 다차원 NumPy 텐서 딕셔너리로 재구성합니다."""
        n_samples = len(df)

        # 1. 2채널 복소 직교 텐서 (N, 15, 2) 및 complex64 텐서 (N, 15) 생성
        harmonics_ri = np.zeros((n_samples, self.harmonics_count, 2), dtype=np.float32)
        harmonics_complex = np.zeros((n_samples, self.harmonics_count), dtype=np.complex64)

        for k in range(1, self.harmonics_count + 1):
            idx = k - 1
            re_col = f"ih_re_{k}"
            im_col = f"ih_im_{k}"

            if re_col in df.columns and im_col in df.columns:
                r_val = df[re_col].values.astype(np.float32)
                i_val = df[im_col].values.astype(np.float32)
            else:
                mag_col = f"ih{k}"
                deg_col = f"ihdeg{k}"
                mag = df[mag_col].values.astype(np.float32)
                rad = np.radians(df[deg_col].values.astype(np.float32))
                r_val = mag * np.cos(rad)
                i_val = mag * np.sin(rad)

            harmonics_ri[:, idx, 0] = r_val  # 실수부 채널
            harmonics_ri[:, idx, 1] = i_val  # 허수부 채널
            harmonics_complex[:, idx] = r_val + 1j * i_val

        # 2. 물리 전력 특징 행렬 (N, 6)
        power_cols = ["p_w", "q_var", "s_va", "power_factor", "vrms", "thd_i"]
        power_features = np.zeros((n_samples, len(power_cols)), dtype=np.float32)
        for i, col in enumerate(power_cols):
            if col in df.columns:
                power_features[:, i] = df[col].values.astype(np.float32)

        # 3. 정규화 고조파 비율 행렬 (N, 14)
        ratio_cols = [f"ih_ratio_{k}" for k in range(2, self.harmonics_count + 1)]
        harmonic_ratios = np.zeros((n_samples, len(ratio_cols)), dtype=np.float32)
        for i, col in enumerate(ratio_cols):
            if col in df.columns:
                harmonic_ratios[:, i] = df[col].values.astype(np.float32)

        # 4. 정답 라벨 및 회귀 타깃 배열
        is_on = df["is_on"].values.astype(np.int8) if "is_on" in df.columns else np.zeros(n_samples, dtype=np.int8)
        state_id = df["state_id"].values.astype(np.int16) if "state_id" in df.columns else np.zeros(n_samples, dtype=np.int16)
        target_power_w = df["target_power_w"].values.astype(np.float32) if "target_power_w" in df.columns else df["p_target_w"].values.astype(np.float32)
        t_rel_s = df["t_rel_s"].values.astype(np.float32) if "t_rel_s" in df.columns else (np.arange(n_samples) / 60.0).astype(np.float32)

        # 5. 메타데이터 JSON 직렬화
        meta_dict = metadata or {}
        meta_dict.update({
            "samples_count": n_samples,
            "harmonics_count": self.harmonics_count,
            "power_feature_names": power_cols,
            "harmonic_ratio_names": ratio_cols,
            "harmonics_ri_shape": list(harmonics_ri.shape),
            "harmonics_ri_format": "(N, harmonics_15, [Real, Imag])",
        })

        return {
            "harmonics_ri": harmonics_ri,
            "harmonics_complex": harmonics_complex,
            "power_features": power_features,
            "harmonic_ratios": harmonic_ratios,
            "is_on": is_on,
            "state_id": state_id,
            "target_power_w": target_power_w,
            "t_rel_s": t_rel_s,
            "metadata_json": json.dumps(meta_dict, ensure_ascii=False),
        }

    def export_to_npz(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True,
    ) -> str:
        """DataFrame을 압축된 .npz 바이너리 파일로 저장합니다."""
        out_p = Path(output_path).with_suffix(".npz")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        data_dict = self.dataframe_to_numpy_dict(df, metadata=metadata)

        if compress:
            np.savez_compressed(out_p, **data_dict)
        else:
            np.savez(out_p, **data_dict)

        return str(out_p)


def load_nilm_npz(npz_path: Union[str, Path]) -> Dict[str, Any]:
    """저장된 NILM .npz 바이너리 파일을 고속으로 로드하고 메타데이터를 파싱합니다."""
    npz = np.load(npz_path, allow_pickle=True)
    res = {k: npz[k] for k in npz.files}
    if "metadata_json" in res:
        res["metadata"] = json.loads(str(res["metadata_json"]))
    return res
