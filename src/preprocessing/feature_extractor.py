"""
NILM 전기적 물리/스펙트럼 특징 추출기 (Feature Extractor)
=========================================================
정제된 60Hz 전압/전류 및 고조파 시계열로부터 AI 모델 학습 및 신호 합성에 필요한
다양한 물리적 파생 특징(P, Q, S, PF, THD_i, 고조파 비율, 복소 직교 성분)을 계산합니다.

[주요 파생 특징 수식]
1. 피상 전력 (Apparent Power): S = V_rms * I_rms (VA)
2. 무효 전력 (Reactive Power): Q = sqrt(max(0, S^2 - P^2)) * sign(phase_deg) (VAR)
3. 역률 (Power Factor): PF = P / S (0.0 ~ 1.0)
4. 정규화 고조파 비율: ih_ratio_k = ih_k / ih_1 (k = 2 ~ 15)
   - 전력 크기(Scale)와 무관한 기기 고유의 비선형 지문(Fingerprint) 추출.
5. 전류 전고조파왜율: THD_i = sqrt(sum(ih_k^2)) / ih_1
6. 복소 고조파 2채널 (Real / Imaginary):
   - R_k = ih_k * cos(radians(ihdeg_k))  [실수부: 유효 성분]
   - I_k = ih_k * sin(radians(ihdeg_k))  [허수부: 무효 성분]
   - 직교 좌표계 상에서 위상각 불연속성 없이 선형 합성(Phasor Addition) 가능.
"""
from typing import List, Optional
import numpy as np
import pandas as pd


class FeatureExtractor:
    """정제된 60Hz 계측 데이터로부터 물리적 도메인 특징을 일괄 추출하는 클래스."""

    def __init__(self, harmonics_count: int = 15, epsilon: float = 1e-6):
        self.harmonics_count = harmonics_count  # 분석 고조파 차수 (1~15차)
        self.epsilon = epsilon  # 0으로 나누기 방지용 상수

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """전기적 물리 법칙 및 푸리에 급수 기반 파생 특징을 계산하여 DataFrame에 추가합니다."""
        out = df.copy()

        # 1. 피상 전력 S (Volt-Amperes, VA)
        out["s_va"] = out["vrms"] * out["irms"]

        # 2. 무효 전력 Q (Volt-Amperes Reactive, VAR)
        # S^2 = P^2 + Q^2 관계식을 활용하며, phase_deg의 부호로 유도성(지상)/용량성(진상) 결정
        p_val = out["p_w"].values
        s_val = out["s_va"].values
        phase_deg = out["phase_deg"].values

        q_mag_sq = np.maximum(0.0, s_val**2 - p_val**2)
        q_mag = np.sqrt(q_mag_sq)
        phase_sign = np.where(phase_deg < 0, -1.0, 1.0)
        out["q_var"] = q_mag * phase_sign

        # 3. 역률 PF (Power Factor, [0.0, 1.0] 클램핑)
        pf = p_val / (s_val + self.epsilon)
        out["power_factor"] = np.clip(pf, 0.0, 1.0)

        # 4. 기본파 전류(ih1) 대비 고조파 성분비 계산
        ih1 = out["ih1"].values
        ih_higher_sq = np.zeros_like(ih1)

        for h in range(2, self.harmonics_count + 1):
            col_name = f"ih{h}"
            if col_name in out.columns:
                ih_k = out[col_name].values
                ih_higher_sq += ih_k**2
                # 고조파 비율: ih_k / ih_1 (스케일 불변 가전 고유 지문)
                out[f"ih_ratio_{h}"] = ih_k / (ih1 + self.epsilon)

        # 5. 전류 전고조파왜율 (Total Harmonic Distortion of Current, THD_i)
        out["thd_i"] = np.sqrt(ih_higher_sq) / (ih1 + self.epsilon)

        # 6. 복소수 2채널 직교 분리 (Real: R_k, Imag: I_k)
        # 신경망 2D/1D 합성곱 입력 및 선형 교류 신호 합성(Phasor Addition)에 사용
        for h in range(1, self.harmonics_count + 1):
            mag_col = f"ih{h}"
            deg_col = f"ihdeg{h}"
            if mag_col in out.columns and deg_col in out.columns:
                mag = out[mag_col].values
                rad = np.radians(out[deg_col].values)
                out[f"ih_re_{h}"] = mag * np.cos(rad)  # 실수부 (In-phase)
                out[f"ih_im_{h}"] = mag * np.sin(rad)  # 허수부 (Quadrature)

        # 7. 1초(60사이클) 롤링 미디언 평활 전력 (상태 분류기 안정화용)
        out["p_smooth_1s"] = (
            out["p_w"].rolling(window=60, center=True, min_periods=1).median().values
        )

        return out

    def get_feature_column_groups(self) -> dict:
        """AI 모델 훈련 시 피처 선택을 위한 컬럼 그룹 딕셔너리를 반환합니다."""
        return {
            "time": ["sample_idx", "t_rel_s", "seq", "cycle"],
            "power": ["p_w", "p_target_w", "q_var", "s_va", "power_factor", "p_smooth_1s"],
            "voltage": ["vrms", "freq_hz", "thd_v"],
            "current_rms": ["irms"] + [f"ih{h}" for h in range(1, self.harmonics_count + 1)],
            "harmonic_ratios": [f"ih_ratio_{h}" for h in range(2, self.harmonics_count + 1)],
            "current_thd": ["thd_i"],
            "harmonic_phases": [f"ihdeg{h}" for h in range(1, self.harmonics_count + 1)],
            "harmonic_complex": [f"ih_re_{h}" for h in range(1, self.harmonics_count + 1)]
            + [f"ih_im_{h}" for h in range(1, self.harmonics_count + 1)],
        }
