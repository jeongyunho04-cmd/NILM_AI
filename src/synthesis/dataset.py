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
from .synthesizer import (
    DEFAULT_TARGET_LOOKAHEAD_CYCLES,
    SELECTION_REALISTIC,
    SELECTION_UNIFORM,
    LoadSynthesizer,
    SyntheticLoadSample,
    window_target_index,
)

# 윈도우 종류별 기본 혼합 비율
#
# random_realistic 과 random_uniform 을 나눈 이유가 있다.
# 실제 가정에서 기기별 사용 빈도는 100배까지 차이 난다(미니PC 42% vs 드라이기 0.4%).
# 그런데 그 분포 그대로만 학습시키면 드라이기 표본이 거의 없어 못 배우고,
# 반대로 균등하게만 뽑으면 사전확률이 틀려 실제 집에서 오탐이 늘어난다.
# 둘을 섞어 커버리지와 보정을 동시에 잡는다.
DEFAULT_RECIPE_MIX: Dict[str, float] = {
    "random_realistic": 0.16,        # 기기별 사용률대로 각자 독립적으로 켜짐
    "random_uniform": 0.14,          # 균등 추첨 - 희귀 기기 학습 표본 확보
    "standby_only": 0.16,            # 대기전력만 - 저부하 오탐 방지
    "low_load_among_standby": 0.20,  # 대기전력 속 저부하 1대
    "high_power_resistive": 0.10,    # 고전력 저항 부하 1~2대 - 아래 설명 참조
    "high_low_mixed": 0.14,          # 고부하 + 저부하 동시 - 오차 전가 방지, 아래 참조
    "resistive_overlap": 0.05,       # 저항 2종이 타깃 시점에 **동시 통전** - 아래 참조
    "unplugged_baseline": 0.05,
}

# resistive_overlap 를 따로 둔 이유 (2026-08-22)
# 0.2절이 "저항성끼리 겹칠 때가 진짜 시험대" 라고 했는데 그 시험을 칠 데이터가 없었다.
# 홀드아웃 8,000창에서 오븐+핫플 동시 발열이 6창(0.07%) 뿐이다.
# high_power_resistive 가 40% 확률로 2대를 켜는데도 그렇다 - 오븐 통전율 25%,
# 핫플 45% 라 둘 다 켜 두어도 타깃 시점 동시 통전은 11% 이기 때문이다.
# 그래서 이 레시피는 타깃 시점의 발열을 확인하고 아니면 다시 뽑는다.
# 실측 test_4 의 전기포트 환각(창의 4.5%, 최대 1,550W)이 이 공백에서 나온다.
#
# 비중을 0.05 로 낮춘 이유: 이 레시피는 창마다 저항 2종을 강제로 켜므로 저항 4종의
# 사전확률을 함께 밀어 올린다. 0.10 이면 각각 +5%p 라 포트가 10.2 -> 15.2% 가 되어
# 환각을 오히려 키운다 (cnn_v13). 0.05 면 +2.5%p 이고, 오븐+핫플 겹침은
# 0.05 x 1/6 x 75%(기각률) ~ 0.6% 로 v13 이 실제로 얻은 0.75% 와 비슷하다.

# high_low_mixed 를 따로 둔 이유
# 고부하와 저부하가 같이 켜진 창에서 둘의 크기 차이가 31배다 (1139W vs 37W).
# 고부하 예측이 3% 만 틀려도 그 오차가 저부하 전체의 93% 를 왜곡할 수 있어,
# 모델이 고부하 오차를 저부하 기기로 흘리는 법을 배우기 쉽다.
# 그런데 다른 레시피는 이 조합을 만들지 않는다 - high_power_resistive 는 저항 부하만,
# low_load_among_standby 는 저부하만 켠다. 무작위에 맡기면 동시 가동 창이 3.3% 뿐이다.
# 모델이 전가하지 않는 법을 배우려면 이 상황을 충분히 봐야 한다.

# high_power_resistive 를 따로 둔 이유
# 전기포트·오븐·드라이기·핫플레이트는 모두 니크롬선 부하라 고조파 지문이 거의 같다
# (포트 vs 오븐 거리 0.596%p). 서로를 가르는 단서는 시간 패턴뿐인데, 실제 사용 빈도가
# 낮아 무작위 추출에만 맡기면 2016 윈도우당 양성 라벨이 11~37개까지 떨어졌다.
# 대기전력 하드네거티브(45%)가 이 기기들의 자리를 밀어낸 영향도 있다.
# 대기전력 학습 비중은 그대로 두고 random_realistic 에서 몫을 떼어 보강한다.

# 구버전 설정 호환: "random" 하나만 주면 현실/균등 7:3 으로 나눠 준다.
_LEGACY_RANDOM_SPLIT = {"random_realistic": 0.7, "random_uniform": 0.3}


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
        compute_gt_harmonics: bool = False,
        target_lookahead_cycles: int = DEFAULT_TARGET_LOOKAHEAD_CYCLES,
    ):
        self.synthesizer = synthesizer or LoadSynthesizer(segment_pool=segment_pool)
        # 학습 배치에는 가전별 고조파 정답이 나가지 않는다. 전력·상태 회귀만 한다면
        # 쓰이지 않으면서 윈도우당 0.65MB(나머지 정답 전부의 10배)를 만들었다 버리게 되므로
        # 기본적으로 만들지 않는다. 합성기 진단이 필요할 때만 True 로 켠다.
        self.compute_gt_harmonics = compute_gt_harmonics
        self.window_size = window_size_cycles
        self.max_concurrent = max_concurrent_appliances
        self.include_power_channels = include_power_channels
        self.target_mode = target_mode
        # seq2point 타깃 시점. 창 중앙이 아니라 끝쪽이다.
        # 중앙을 쓰면 추론할 때 창 절반만큼의 미래가 필요해 실시간이 성립하지 않는다.
        self.target_lookahead_cycles = int(target_lookahead_cycles)
        self.target_index = window_target_index(window_size_cycles, target_lookahead_cycles)
        self.appliance_list = sorted(self.synthesizer.known_appliances)
        self.app_to_idx = {app: i for i, app in enumerate(self.appliance_list)}

        mix = dict(recipe_mix or DEFAULT_RECIPE_MIX)
        # 구버전 "random" 키는 현실/균등으로 갈라 준다.
        if "random" in mix:
            share = mix.pop("random")
            for name, frac in _LEGACY_RANDOM_SPLIT.items():
                mix[name] = mix.get(name, 0.0) + share * frac
        total = sum(mix.values())
        if total <= 0:
            raise ValueError("recipe_mix 의 비율 합이 0 보다 커야 합니다.")
        self.recipe_names = list(mix.keys())
        self.recipe_probs = np.array([mix[k] / total for k in self.recipe_names], dtype=np.float64)

    # ── 윈도우 1개 합성 ─────────────────────────────────────────────────────
    def _synthesize_window(self) -> Tuple[SyntheticLoadSample, str]:
        """혼합 비율에 따라 윈도우 종류를 골라 합성한다."""
        recipe = str(np.random.choice(self.recipe_names, p=self.recipe_probs))
        gt_h = self.compute_gt_harmonics

        if recipe == "standby_only":
            sample = self.synthesizer.synthesize_standby_only_window(
                self.window_size, compute_gt_harmonics=gt_h
            )
        elif recipe == "low_load_among_standby":
            sample = self.synthesizer.synthesize_low_load_among_standby_window(
                self.window_size, compute_gt_harmonics=gt_h,
                target_lookahead_cycles=self.target_lookahead_cycles,
            )
        elif recipe == "high_power_resistive":
            sample = self.synthesizer.synthesize_high_power_window(
                self.window_size, compute_gt_harmonics=gt_h,
                target_lookahead_cycles=self.target_lookahead_cycles,
            )
        elif recipe == "resistive_overlap":
            sample = self.synthesizer.synthesize_resistive_overlap_window(
                self.window_size, compute_gt_harmonics=gt_h,
                target_lookahead_cycles=self.target_lookahead_cycles,
            )
        elif recipe == "high_low_mixed":
            sample = self.synthesizer.synthesize_high_low_mixed_window(
                self.window_size, compute_gt_harmonics=gt_h,
                target_lookahead_cycles=self.target_lookahead_cycles,
            )
        elif recipe == "unplugged_baseline":
            sample = self.synthesizer.synthesize_scenario(
                total_duration_cycles=self.window_size,
                schedules=[],
                plugged_in_appliances={a: False for a in self.appliance_list},
                include_noise=True,
                simulate_voltage_drop=True,
                compute_gt_harmonics=gt_h,
            )
        else:
            sample = self.synthesizer.synthesize_random_window(
                window_size_cycles=self.window_size,
                max_concurrent_appliances=self.max_concurrent,
                compute_gt_harmonics=gt_h,
                selection_mode=(
                    SELECTION_UNIFORM if recipe == "random_uniform" else SELECTION_REALISTIC
                ),
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
            mid = self.target_index
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
