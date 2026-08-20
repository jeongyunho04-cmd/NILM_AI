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

    def create_long_timeline(
        self,
        duration_min: float = 120.0,
        min_episodes_per_appliance: int = 1,
        plugged_prob: float = 0.85,
    ) -> SyntheticLoadSample:
        """몇 시간짜리 가정 소비 타임라인을 만든다.

        짧은 윈도우를 뽑는 것과 접근이 다르다. 긴 시간축에서는 각 가전이
        '몇 번, 얼마나 오래' 켜지는지가 핵심이므로, 기기별 사용률에서 총 가동
        시간을 잡고 그것을 실제 활성화 길이만큼의 에피소드로 쪼개 배치한다.

        멀티탭 용량을 넘는 겹침은 배치 단계에서 걸러낸다. 윈도우 단위 검사와 달리
        여기서는 시간에 따라 겹침이 변하므로, 에피소드를 하나씩 넣어 보며
        어느 시점에서도 한도를 넘지 않을 때만 확정한다.
        """
        from src.preprocessing.file_registry import get_usage_probability

        n_cycles = int(duration_min * 60.0 * self.sampling_hz)
        syn = self.synthesizer
        pool = syn.pool
        limit = syn.sustained_power_limit_w

        # 이 타임라인의 전압 환경을 먼저 정해야 용량 계산이 맞는다.
        env = syn.grid_sim.sample_environment()

        # 1. 기기별로 가동 에피소드 후보를 만든다
        candidates: List[Tuple[str, int, int, float]] = []  # (가전, 시작, 끝, 지속전력)
        for app in syn.known_appliances:
            acts = pool.appliance_activations[app]
            typical = float(np.median([a.duration_cycles for a in acts]))
            target_on = get_usage_probability(app) * n_cycles
            n_episodes = max(min_episodes_per_appliance, int(round(target_on / max(typical, 1.0))))
            steady = syn.estimate_steady_power_w(app, env.base_voltage_v)
            duty_period = pool.duty_period_cycles.get(app)

            if duty_period and duty_period > typical:
                # 서모스탯 부하는 낱개 펄스를 흩뿌리면 안 된다. 실제로는 한 번의
                # 조리 세션(수십 분) 안에서 일정 주기로 통전이 반복된다.
                # 필요한 펄스 수를 몇 개의 세션으로 묶고, 세션 안에서는 실측 주기로 배치한다.
                n_sessions = max(min_episodes_per_appliance,
                                 int(np.ceil(n_episodes / 400)))
                per_session = max(1, n_episodes // n_sessions)
                session_span = int(per_session * duty_period)
                for _ in range(n_sessions):
                    if session_span >= n_cycles:
                        s0 = 0
                    else:
                        s0 = int(np.random.randint(0, n_cycles - session_span))
                    for k in range(per_session):
                        s = s0 + k * duty_period
                        dur = int(np.clip(typical * np.random.uniform(0.8, 1.2), 30, duty_period))
                        if s + dur >= n_cycles:
                            break
                        candidates.append((app, s, s + dur, steady))
                continue

            for _ in range(n_episodes):
                # 실제 활성화 길이 분포에서 뽑되 증강 한도(3배) 안에서 흔든다
                dur = int(np.clip(typical * np.random.uniform(0.5, 2.0), 30, n_cycles))
                start = int(np.random.randint(0, max(1, n_cycles - dur)))
                candidates.append((app, start, start + dur, steady))

        # 2. 같은 가전끼리 겹치면 하나로 합쳐지므로, 겹치는 후보는 버린다
        occupied: Dict[str, np.ndarray] = {
            a: np.zeros(n_cycles, dtype=bool) for a in syn.known_appliances
        }
        # 3. 어느 시점에서도 멀티탭 한도를 넘지 않게 한다
        load = np.zeros(n_cycles, dtype=np.float64)

        np.random.shuffle(candidates)
        accepted: List[ApplianceSchedule] = []
        rejected_overlap = 0
        rejected_budget = 0

        for app, s, e, steady in candidates:
            if occupied[app][s:e].any():
                rejected_overlap += 1
                continue
            if limit is not None and float((load[s:e] + steady).max()) > limit:
                rejected_budget += 1
                continue
            occupied[app][s:e] = True
            load[s:e] += steady
            accepted.append(ApplianceSchedule(app, start_cycle=s, duration_cycles=e - s))

        # 시작 시각 순으로 정렬해 두면 이후 진단과 그래프가 읽기 쉬워진다
        accepted.sort(key=lambda x: x.start_cycle)

        plugged = {a: bool(np.random.rand() < plugged_prob) for a in syn.known_appliances}
        sample = syn.synthesize_scenario(
            total_duration_cycles=n_cycles,
            schedules=accepted,
            plugged_in_appliances=plugged,
            include_noise=True,
            simulate_voltage_drop=True,
            voltage_environment=env,
        )
        sample.metadata["episodes_scheduled"] = len(accepted)
        sample.metadata["episodes_rejected_overlap"] = rejected_overlap
        sample.metadata["episodes_rejected_over_budget"] = rejected_budget
        return sample

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
