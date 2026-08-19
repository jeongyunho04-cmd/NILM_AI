"""
Grid Impedance, Baseline Voltage Range Augmentation, and Voltage Drop Simulator for NILM AI
===========================================================================================
가정별로 상이한 배전 전압 환경(215V ~ 238V)과 계통 임피던스(Z_grid)에 의한 순간 전압 강하를 시뮬레이션하고,
전압 변화에 따른 부하 유형별(저항성, SMPS, 인버터 모터) 비선형 물리 거동을 정밀하게 반영합니다.

[부하 유형별 전압 응답 물리 법칙]
1. 순수 저항성 부하 (히터, 전기포트, 드라이기, 핫플레이트, 오븐):
   - 옴의 법칙: I proportional to V (kappa)
   - 소비 전력: P proportional to V^2 (kappa^2) -> 235V에서 전력 +14% 증가, 전압 강하 시 전력 감소
2. SMPS 정전력 전자 부하 (미니PC, 노트북 충전기, 빔프로젝터):
   - 2차측 정전압 PWM 제어: 소비 전력 P = Constant
   - 입력 전류: I proportional to 1/V (1/kappa) -> 235V에서 전류 감소, 저전압 시 전류 증가 & 3차 고조파 왜율 증가
3. 인버터 / 전동기 부하 (에어컨, 선풍기):
   - 중간 슬립/토크 제어 거동: I proportional to kappa^0.7, P proportional to kappa^0.7
"""
from typing import Dict, Optional, Tuple
import numpy as np


class GridSimulator:
    """배전 전압 범위 증강 및 계통 임피던스 전압 강하(Sag) 시뮬레이터."""

    def __init__(
        self,
        nominal_voltage_range: Tuple[float, float] = (215.0, 238.0),  # 실제 가정별 인가 전압 범위
        default_ref_voltage: float = 220.0,  # 원데이터 기준 공칭 전압
        nominal_voltage: Optional[float] = None,  # 이전 버전 호환용 단일 전압
        r_grid_range: Tuple[float, float] = (0.15, 0.35),  # 옥내 배선 저항 범위 (Ohm)
        x_grid_range: Tuple[float, float] = (0.02, 0.08),  # 옥내 배선 리액턴스 범위 (Ohm)
        r_grid: Optional[float] = None,  # 단일 배선 저항 지정용
        x_grid: Optional[float] = None,  # 단일 배선 리액턴스 지정용
        voltage_variation_std: float = 0.8,  # 느린 계통 자연 전압 요동 (Volts)
    ):
        if nominal_voltage is not None:
            self.nominal_voltage_range = (nominal_voltage, nominal_voltage)
        else:
            self.nominal_voltage_range = nominal_voltage_range

        self.default_ref_voltage = default_ref_voltage
        self.r_grid_range = (r_grid, r_grid) if r_grid is not None else r_grid_range
        self.x_grid_range = (x_grid, x_grid) if x_grid is not None else x_grid_range
        self.voltage_variation_std = voltage_variation_std

    def compute_voltage_drop(
        self,
        total_current_complex: np.ndarray,  # (N, 15) complex64
        base_voltage: Optional[float] = None,
        r_grid: Optional[float] = None,
        x_grid: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """순간 단자 버스 전압(V_bus)과 기준 전압(220V) 대비 전압 스케일 비율(kappa_v)을 계산합니다.

        Args:
            total_current_complex: (N, 15) complex64 배열
            base_voltage: 기준 무부하 계통 전압 (지정하지 않으면 nominal_voltage_range에서 무작위 샘플링)

        Returns:
            v_bus: (N,) float32 단자 실효 전압 (Vrms)
            kappa_v: (N,) float32 전압 스케일 비율 V_bus(t) / 220.0
        """
        n_samples = len(total_current_complex)
        
        # 1. 가정별 기본 전압 샘플링 (예: 235.2V 또는 218.4V)
        if base_voltage is None:
            v0 = float(np.random.uniform(self.nominal_voltage_range[0], self.nominal_voltage_range[1]))
        else:
            v0 = float(base_voltage)

        # 2. 옥내 배선 임피던스 샘플링
        rg = float(np.random.uniform(self.r_grid_range[0], self.r_grid_range[1])) if r_grid is None else r_grid
        xg = float(np.random.uniform(self.x_grid_range[0], self.x_grid_range[1])) if x_grid is None else x_grid

        # 3. 자연 전압 요동(Drift) 추가
        if self.voltage_variation_std > 0:
            drift = np.cumsum(np.random.normal(0, 0.02, size=n_samples))
            drift = np.clip(drift, -self.voltage_variation_std * 3, self.voltage_variation_std * 3)
            v0_series = v0 + drift
        else:
            v0_series = np.full(n_samples, v0, dtype=np.float32)

        # 4. Fundamental Active/Reactive Current
        i1_c = total_current_complex[:, 0]
        i_re = np.real(i1_c)
        i_im = np.imag(i1_c)

        # Delta V = R_grid * I_active + X_grid * I_reactive
        delta_v = rg * i_re + xg * i_im

        # 5. 최종 단자 전압 및 220V 대비 스케일 계수 kappa_v
        v_bus = np.clip(v0_series - delta_v, 180.0, 255.0).astype(np.float32)
        kappa_v = (v_bus / self.default_ref_voltage).astype(np.float32)

        return v_bus, kappa_v

    def apply_cross_appliance_coupling(
        self,
        appliance_type: str,
        harmonics_complex: np.ndarray,  # (N, 15) complex64
        kappa_v: np.ndarray,            # (N,) float32 전압 비율 V_bus(t) / 220.0
    ) -> np.ndarray:
        """전압 변화(kappa_v)에 따른 가전별 비선형 물리 전류 및 고조파 변형을 적용합니다."""
        kappa_col = kappa_v[:, np.newaxis]  # (N, 1)

        # 1. 저항성 히터 부하 (전기포트, 드라이기, 핫플레이트, 오븐): I proportional to V
        if appliance_type in ["electiric_kettle", "hotplate", "hair_dryer", "oven"]:
            return (harmonics_complex * kappa_col).astype(np.complex64)

        # 2. SMPS 정전력 전자 부하 (미니PC, 노트북 충전기, 빔프로젝터): I proportional to 1/V
        elif appliance_type in ["minipc", "laptop_charger", "beam_projector"]:
            inv_kappa = np.clip(1.0 / np.maximum(0.7, kappa_col), 0.85, 1.30)
            mod_c = harmonics_complex * inv_kappa
            # 저전압(V < 220V, inv_kappa > 1.0) 시 정류 다이오드 도통각 변화로 3차 고조파 왜율 소폭 상승
            if mod_c.shape[1] >= 3:
                distortion_factor = 1.0 + 0.4 * (inv_kappa[:, 0] - 1.0)
                mod_c[:, 2] *= distortion_factor
            return mod_c.astype(np.complex64)

        # 3. 모터 / 인버터 부하 (에어컨, 선풍기): I proportional to V^0.7
        elif appliance_type in ["fan", "air_conditioner"]:
            motor_kappa = np.power(np.maximum(0.5, kappa_col), 0.7)
            return (harmonics_complex * motor_kappa).astype(np.complex64)

        # 기본 일반 부하
        return (harmonics_complex * kappa_col).astype(np.complex64)
