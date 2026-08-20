"""
Realistic Household Scenario Generator for NILM AI
Generates rich multi-hour appliance usage scenarios (morning, evening, office, random daily)
and exports synthetic benchmark datasets.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import os
import numpy as np

from .synthesizer import ApplianceSchedule, LoadSynthesizer, SyntheticLoadSample


class ScenarioGenerator:
    """Generates structured household consumption scenarios."""

    def __init__(self, synthesizer: LoadSynthesizer, sampling_hz: float = 60.0):
        self.synthesizer = synthesizer
        self.sampling_hz = sampling_hz

    def create_morning_routine(self, duration_min: float = 15.0) -> SyntheticLoadSample:
        """Morning routine: Electric kettle, hair dryer, laptop charger, fan."""
        n_cycles = int(duration_min * 60.0 * self.sampling_hz)
        schedules = [
            # Kettle turns on at t=1 min for ~4 min
            ApplianceSchedule("electiric_kettle", start_cycle=int(1.0 * 60 * 60), duration_cycles=int(4.0 * 60 * 60)),
            # Hair dryer turns on at t=6 min for ~3 min
            ApplianceSchedule("hair_dryer", start_cycle=int(6.0 * 60 * 60), duration_cycles=int(3.0 * 60 * 60)),
            # Laptop charger starts at t=3 min, runs throughout
            ApplianceSchedule("laptop_charger", start_cycle=int(3.0 * 60 * 60), duration_cycles=int(11.0 * 60 * 60)),
            # Fan runs on low speed from t=2 min
            ApplianceSchedule("fan", start_cycle=int(2.0 * 60 * 60), duration_cycles=int(10.0 * 60 * 60)),
        ]
        return self.synthesizer.synthesize_scenario(
            total_duration_cycles=n_cycles,
            schedules=schedules,
            include_noise=True,
            simulate_voltage_drop=True,
        )

    def create_evening_cooking_routine(self, duration_min: float = 20.0) -> SyntheticLoadSample:
        """Evening routine: Air conditioner, oven, hotplate, beam projector."""
        n_cycles = int(duration_min * 60.0 * self.sampling_hz)
        schedules = [
            # Air conditioner starts early at t=0.5 min, runs whole duration
            ApplianceSchedule("air_conditioner", start_cycle=int(0.5 * 60 * 60), duration_cycles=int(18.0 * 60 * 60)),
            # Oven starts at t=3 min, bakes for 12 min
            ApplianceSchedule("oven", start_cycle=int(3.0 * 60 * 60), duration_cycles=int(12.0 * 60 * 60)),
            # Hotplate cooks at t=5 min for 8 min
            ApplianceSchedule("hotplate", start_cycle=int(5.0 * 60 * 60), duration_cycles=int(8.0 * 60 * 60)),
            # Beam projector turns on at t=12 min
            ApplianceSchedule("beam_projector", start_cycle=int(12.0 * 60 * 60), duration_cycles=int(7.0 * 60 * 60)),
        ]
        return self.synthesizer.synthesize_scenario(
            total_duration_cycles=n_cycles,
            schedules=schedules,
            include_noise=True,
            simulate_voltage_drop=True,
        )

    def create_work_office_routine(self, duration_min: float = 25.0) -> SyntheticLoadSample:
        """Work routine: Mini PC, laptop charger, fan, electric kettle breaks."""
        n_cycles = int(duration_min * 60.0 * self.sampling_hz)
        schedules = [
            # Mini PC runs whole duration
            ApplianceSchedule("minipc", start_cycle=int(0.2 * 60 * 60), duration_cycles=int(24.0 * 60 * 60)),
            # Laptop charger runs
            ApplianceSchedule("laptop_charger", start_cycle=int(1.0 * 60 * 60), duration_cycles=int(20.0 * 60 * 60)),
            # Fan runs
            ApplianceSchedule("fan", start_cycle=int(4.0 * 60 * 60), duration_cycles=int(15.0 * 60 * 60)),
            # Coffee/tea break 1
            ApplianceSchedule("electiric_kettle", start_cycle=int(8.0 * 60 * 60), duration_cycles=int(3.5 * 60 * 60)),
            # Coffee/tea break 2
            ApplianceSchedule("electiric_kettle", start_cycle=int(18.0 * 60 * 60), duration_cycles=int(3.0 * 60 * 60)),
        ]
        return self.synthesizer.synthesize_scenario(
            total_duration_cycles=n_cycles,
            schedules=schedules,
            include_noise=True,
            simulate_voltage_drop=True,
        )

    def create_random_scenario(
        self,
        duration_min: float = 10.0,
        num_activations: int = 6,
    ) -> SyntheticLoadSample:
        """Generates a fully randomized multi-appliance scenario."""
        n_cycles = int(duration_min * 60.0 * self.sampling_hz)
        known_apps = self.synthesizer.known_appliances

        schedules = []
        for _ in range(num_activations):
            app = np.random.choice(known_apps)
            start_min = np.random.uniform(0.1, duration_min * 0.7)
            dur_min = np.random.uniform(1.0, min(10.0, duration_min - start_min))
            schedules.append(
                ApplianceSchedule(
                    appliance_type=app,
                    start_cycle=int(start_min * 60 * self.sampling_hz),
                    duration_cycles=int(dur_min * 60 * self.sampling_hz),
                )
            )

        return self.synthesizer.synthesize_scenario(
            total_duration_cycles=n_cycles,
            schedules=schedules,
            include_noise=True,
            simulate_voltage_drop=True,
        )

    @staticmethod
    def export_synthetic_sample_to_npz(
        sample: SyntheticLoadSample,
        output_path: Union[str, Path],
    ) -> str:
        """Exports a SyntheticLoadSample into an optimized .npz archive."""
        out_p = Path(output_path).with_suffix(".npz")
        out_p.parent.mkdir(parents=True, exist_ok=True)

        data_dict = {
            "harmonics_ri": sample.harmonics_ri,
            "harmonics_complex": sample.harmonics_complex,
            "power_features": sample.power_features,
            "v_bus": sample.v_bus,             # 계측 해상도로 계단화된 전압 (모델 입력)
            "v_bus_true": sample.v_bus_true,   # 연속 실제 전압 (진단용)
            "t_rel_s": sample.t_rel_s,
            # 어느 기기의 것도 아닌 계측계 자체 소비. 전력 분해 검산에 필요하다.
            "p_noise_w": sample.p_noise_w,
            "metadata_json": json.dumps(sample.metadata, ensure_ascii=False),
        }

        # Flatten per-appliance ground truths into dictionary keys
        for app in sample.appliance_types:
            data_dict[f"gt_is_on_{app}"] = sample.gt_is_on[app]
            data_dict[f"gt_state_id_{app}"] = sample.gt_state_id[app]
            data_dict[f"gt_target_power_{app}"] = sample.gt_target_power_w[app]
            # 대기전력을 활성전력과 구분해 학습시키기 위한 채널
            data_dict[f"gt_is_plugged_{app}"] = sample.gt_is_plugged[app]
            data_dict[f"gt_standby_power_{app}"] = sample.gt_standby_power_w[app]
            # 고조파 정답은 만들어졌을 때만 저장한다. 전력·상태 회귀만 학습한다면
            # 쓰이지 않으면서 파일 용량의 73%를 차지한다.
            if sample.gt_harmonics_included:
                data_dict[f"gt_harmonics_ri_{app}"] = sample.gt_harmonics_ri[app]

        # 임시 파일에 쓴 뒤 원자적으로 교체한다.
        # 기존 파일을 직접 열어 덮어쓰면 (1) 도중에 실패했을 때 반쯤 쓰인 파일이 남고,
        # (2) Windows 에서 백신이 방금 쓴 대용량 파일을 스캔하는 동안
        #     OSError(Errno 22) 로 열기가 실패하는 일이 있다.
        # 이름이 .npz 로 끝나야 한다. np.savez_compressed 는 그렇지 않으면
        # 뒤에 .npz 를 덧붙여 버려서 os.replace 가 찾을 파일이 사라진다.
        tmp_p = out_p.with_name(f"{out_p.stem}.tmp.npz")
        try:
            np.savez_compressed(tmp_p, **data_dict)
            os.replace(tmp_p, out_p)
        except BaseException:
            tmp_p.unlink(missing_ok=True)
            raise
        return str(out_p)
