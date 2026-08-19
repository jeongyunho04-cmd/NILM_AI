"""
PyTorch 호환 NILM 데이터셋 및 실시간 배치 생성기
==================================================
학습 중에 복합 부하 윈도우를 즉석에서 만들어 공급한다.

[대기전력 오탐을 막는 하드 네거티브 커리큘럼]
NILM 모델의 대표적 오답은 "여러 기기의 대기전력 합"을 "저전력 기기 1대가 켜진 것"으로
읽는 것이다. 미니PC 아이들 9.8W, 선풍기 1단 23.5W 인데 9대가 꽂혀만 있어도 8W 가 깔린다.
무작위 윈도우만 뽑으면 이 경계 상황이 드물게 나와 모델이 배우지 못하므로,
헷갈리기 쉬운 상황을 의도적으로 일정 비율 섞어 준다.

  random                  : 일반 무작위 복합 윈도우
  standby_only            : 활성 기기 0대, 대기전력만 깔림  -> 정답은 전부 OFF/0W
  low_load_among_standby  : 대기전력 최대 상태에서 저전력 기기 딱 1대만 ON
  unplugged_baseline      : 아무것도 꽂혀 있지 않음 -> 계측계 자체 소비만

[보조 학습 신호]
generate_batch_dict() 는 활성 라벨과 별도로 '콘센트 연결 여부(y_plugged)'와
'대기 전력(y_standby_power)'을 함께 준다. 이 둘을 보조 태스크로 함께 학습시키면
모델이 "전력은 있지만 켜진 건 아니다"라는 상태를 명시적으로 표현할 수 있게 된다.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .segment_pool import SegmentPool
from .synthesizer import LoadSynthesizer, SyntheticLoadSample

# 윈도우 종류별 기본 혼합 비율
DEFAULT_RECIPE_MIX: Dict[str, float] = {
    "random": 0.55,
    "standby_only": 0.18,
    "low_load_among_standby": 0.22,
    "unplugged_baseline": 0.05,
}


class NILMBatchGenerator:
    """NILM 학습용 고속 실시간 배치 생성기."""

    def __init__(
        self,
        segment_pool: SegmentPool,
        window_size_cycles: int = 600,  # 기본 10초 윈도우
        max_concurrent_appliances: int = 3,
        include_power_channels: bool = True,
        target_mode: str = "seq2point",  # "seq2point"(중앙 시점) 또는 "seq2seq"(전 구간)
        recipe_mix: Optional[Dict[str, float]] = None,
        synthesizer: Optional[LoadSynthesizer] = None,
    ):
        self.synthesizer = synthesizer or LoadSynthesizer(segment_pool=segment_pool)
        self.window_size = window_size_cycles
        self.max_concurrent = max_concurrent_appliances
        self.include_power_channels = include_power_channels
        self.target_mode = target_mode
        self.appliance_list = sorted(self.synthesizer.known_appliances)
        self.app_to_idx = {app: i for i, app in enumerate(self.appliance_list)}

        mix = dict(recipe_mix or DEFAULT_RECIPE_MIX)
        total = sum(mix.values())
        if total <= 0:
            raise ValueError("recipe_mix 의 비율 합이 0 보다 커야 합니다.")
        self.recipe_names = list(mix.keys())
        self.recipe_probs = np.array([mix[k] / total for k in self.recipe_names], dtype=np.float64)

    # ── 윈도우 1개 합성 ─────────────────────────────────────────────────────
    def _synthesize_window(self) -> Tuple[SyntheticLoadSample, str]:
        """혼합 비율에 따라 윈도우 종류를 골라 합성한다."""
        recipe = str(np.random.choice(self.recipe_names, p=self.recipe_probs))

        if recipe == "standby_only":
            sample = self.synthesizer.synthesize_standby_only_window(self.window_size)
        elif recipe == "low_load_among_standby":
            sample = self.synthesizer.synthesize_low_load_among_standby_window(self.window_size)
        elif recipe == "unplugged_baseline":
            sample = self.synthesizer.synthesize_scenario(
                total_duration_cycles=self.window_size,
                schedules=[],
                plugged_in_appliances={a: False for a in self.appliance_list},
                include_noise=True,
                simulate_voltage_drop=True,
            )
        else:
            sample = self.synthesizer.synthesize_random_window(
                window_size_cycles=self.window_size,
                max_concurrent_appliances=self.max_concurrent,
            )
        return sample, recipe

    def _format_inputs(self, sample: SyntheticLoadSample) -> np.ndarray:
        """모델 입력 텐서 (Channels, W) 를 만든다."""
        r_part = sample.harmonics_ri[:, :, 0].T  # (15, W)
        i_part = sample.harmonics_ri[:, :, 1].T  # (15, W)

        if not self.include_power_channels:
            return np.concatenate([r_part, i_part], axis=0).astype(np.float32)

        p_chan = sample.power_features[:, 0:1].T  # (1, W) 유효전력
        q_chan = sample.power_features[:, 1:2].T  # (1, W) 무효전력
        v_chan = sample.power_features[:, 4:5].T  # (1, W) 단자 전압 (계측 해상도 반영됨)
        return np.concatenate([r_part, i_part, p_chan, q_chan, v_chan], axis=0).astype(np.float32)

    def _format_targets(self, sample: SyntheticLoadSample) -> Dict[str, np.ndarray]:
        """가전별 정답을 배열로 정리한다."""
        n_apps = len(self.appliance_list)

        if self.target_mode == "seq2point":
            mid = self.window_size // 2
            out = {
                "y_power": np.zeros(n_apps, dtype=np.float32),
                "y_state": np.zeros(n_apps, dtype=np.int16),
                "y_on": np.zeros(n_apps, dtype=np.int8),
                "y_plugged": np.zeros(n_apps, dtype=np.int8),
                "y_standby_power": np.zeros(n_apps, dtype=np.float32),
            }
            for i, app in enumerate(self.appliance_list):
                out["y_power"][i] = sample.gt_target_power_w[app][mid]
                out["y_state"][i] = sample.gt_state_id[app][mid]
                out["y_on"][i] = sample.gt_is_on[app][mid]
                out["y_plugged"][i] = sample.gt_is_plugged[app][mid]
                out["y_standby_power"][i] = sample.gt_standby_power_w[app][mid]
            return out

        W = self.window_size
        out = {
            "y_power": np.zeros((n_apps, W), dtype=np.float32),
            "y_state": np.zeros((n_apps, W), dtype=np.int16),
            "y_on": np.zeros((n_apps, W), dtype=np.int8),
            "y_plugged": np.zeros((n_apps, W), dtype=np.int8),
            "y_standby_power": np.zeros((n_apps, W), dtype=np.float32),
        }
        for i, app in enumerate(self.appliance_list):
            out["y_power"][i] = sample.gt_target_power_w[app]
            out["y_state"][i] = sample.gt_state_id[app]
            out["y_on"][i] = sample.gt_is_on[app]
            out["y_plugged"][i] = sample.gt_is_plugged[app]
            out["y_standby_power"][i] = sample.gt_standby_power_w[app]
        return out

    def generate_single_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """윈도우 1개를 생성한다 (기존 4-튜플 형식 유지).

        Returns:
            X: (Channels, W) float32 - 15 Real + 15 Imag (+ P, Q, V)
            y_power, y_state, y_on
        """
        sample, _ = self._synthesize_window()
        t = self._format_targets(sample)
        return self._format_inputs(sample), t["y_power"], t["y_state"], t["y_on"]

    def generate_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """학습 배치 1개를 생성한다 (기존 4-튜플 형식 유지)."""
        d = self.generate_batch_dict(batch_size)
        return d["X"], d["y_power"], d["y_state"], d["y_on"]

    def generate_batch_dict(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """대기전력 보조 라벨까지 포함한 전체 배치를 딕셔너리로 반환한다.

        Returns:
            X               : (B, C, W) float32
            y_power         : 가전별 활성 전력 (꺼졌으면 0)
            y_state         : 가전별 동작 상태 ID
            y_on            : 가전별 활성 여부
            y_plugged       : 가전별 콘센트 연결 여부 (대기전력 존재 여부)
            y_standby_power : 가전별 대기 전력 (활성 중이면 0)
            p_noise_w       : 계측계 자체 소비 (어느 기기 것도 아님)
            recipe          : 각 윈도우가 어떤 종류로 생성되었는지
        """
        xs, recipes, noises = [], [], []
        target_keys = ["y_power", "y_state", "y_on", "y_plugged", "y_standby_power"]
        collected: Dict[str, List[np.ndarray]] = {k: [] for k in target_keys}

        for _ in range(batch_size):
            sample, recipe = self._synthesize_window()
            xs.append(self._format_inputs(sample))
            t = self._format_targets(sample)
            for k in target_keys:
                collected[k].append(t[k])
            recipes.append(recipe)
            noises.append(
                np.mean(sample.p_noise_w) if self.target_mode == "seq2point" else sample.p_noise_w
            )

        out: Dict[str, np.ndarray] = {"X": np.stack(xs, axis=0)}
        for k in target_keys:
            out[k] = np.stack(collected[k], axis=0)
        out["p_noise_w"] = np.asarray(noises, dtype=np.float32)
        out["recipe"] = np.asarray(recipes)
        return out

    # ── 진단 ────────────────────────────────────────────────────────────────
    def describe_recipe_mix(self) -> Dict[str, float]:
        """윈도우 종류별 혼합 비율."""
        return {n: float(p) for n, p in zip(self.recipe_names, self.recipe_probs)}


# ── PyTorch Dataset 래퍼 (torch 가 있을 때만) ────────────────────────────────
try:
    import torch
    from torch.utils.data import Dataset

    class NILMPyTorchDataset(Dataset):
        """NILM 부하 합성용 PyTorch Dataset 래퍼."""

        def __init__(
            self,
            segment_pool: SegmentPool,
            epoch_size: int = 5000,
            window_size_cycles: int = 600,
            target_mode: str = "seq2point",
            include_power_channels: bool = True,
            recipe_mix: Optional[Dict[str, float]] = None,
            return_standby_targets: bool = False,
        ):
            self.generator = NILMBatchGenerator(
                segment_pool=segment_pool,
                window_size_cycles=window_size_cycles,
                target_mode=target_mode,
                include_power_channels=include_power_channels,
                recipe_mix=recipe_mix,
            )
            self.epoch_size = epoch_size
            self.return_standby_targets = return_standby_targets

        def __len__(self) -> int:
            return self.epoch_size

        def __getitem__(self, idx: int):
            sample, _ = self.generator._synthesize_window()
            x = self.generator._format_inputs(sample)
            t = self.generator._format_targets(sample)

            tensors = [
                torch.from_numpy(x),
                torch.from_numpy(t["y_power"]),
                torch.from_numpy(t["y_state"].astype(np.int64)),
                torch.from_numpy(t["y_on"].astype(np.float32)),
            ]
            if self.return_standby_targets:
                # 대기전력 오탐 방지를 위한 보조 학습 신호
                tensors.append(torch.from_numpy(t["y_plugged"].astype(np.float32)))
                tensors.append(torch.from_numpy(t["y_standby_power"]))
            return tuple(tensors)

except ImportError:
    NILMPyTorchDataset = None
