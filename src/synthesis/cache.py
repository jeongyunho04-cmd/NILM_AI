"""
학습용 윈도우 캐시 (Window Cache)
==================================
합성 시나리오를 미리 대량 생성해 디스크에 두고, 학습 중에는 잘라 쓰기만 한다.

[왜 필요한가]
배치 생성기를 학습 루프에서 그대로 호출하면 372 windows/sec 밖에 안 나온다.
모델은 RTX 5070 에서 8,000~16,000 win/s 를 처리하므로 GPU 가 3% 만 일하게 된다.

원인은 신호 합성이 아니라 창당 고정 비용이다. 합성기는 길이와 무관하게 초당
20~30만 사이클을 처리하는데(10초 3.0ms / 1200초 346ms), 10초 창 하나를 위해
레시피 추첨·전압 환경·가전 선택·증강·전압 되먹임을 매번 새로 한다.
긴 시나리오를 한 번 만들어 여러 창으로 나누면 그 비용이 분산된다.

[그런데 그냥 자르면 클래스 균형이 무너진다]
레시피는 시나리오 단위로 정해지고, center_biased_placement 는 원래 창의 중앙만
겨냥하므로 부창에는 적용되지 않는다. 실측 결과:

    핫플레이트   5.5% -> 0.4%
    불균형      2.7:1 -> 34.9:1

그래서 이 모듈은 부창마다 중앙 라벨을 미리 계산해 두고, 역빈도 가중치로
샘플링 확률을 보정한다. 가중치를 쓰지 않으면 위 불균형이 그대로 남는다.

[저장 구조]
    <cache_dir>/
        inputs.npy     (n_scenarios, 33, L) float32   15 Re + 15 Im + P, Q, V
        y_power.npy    (n_scenarios, 9, L)  float32   활성 전력
        y_standby.npy  (n_scenarios, 9, L)  float16   대기 전력
        y_state.npy    (n_scenarios, 9, L)  int8
        y_on.npy       (n_scenarios, 9, L)  int8
        y_plugged.npy  (n_scenarios, 9, L)  int8
        p_noise.npy    (n_scenarios, L)     float32   계측계 자체 소비
        meta.json      가전 순서, 레시피, 창 설정, 통계
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import json
import numpy as np

from .dataset import DEFAULT_RECIPE_MIX, NILMBatchGenerator
from .segment_pool import SegmentPool
from .synthesizer import LoadSynthesizer

# 캐시가 담는 채널 수: 15 Real + 15 Imag + P + Q + V
INPUT_CHANNELS = 33
# 모델 입력에 쓰는 36채널 중 나머지 3개(고조파 비율)는 학습 쪽에서 만든다.
# 원본에서 바로 계산되므로 디스크에 중복 저장하지 않는다.

_ARRAYS = {
    "inputs": ("inputs.npy", np.float32),
    "y_power": ("y_power.npy", np.float32),
    "y_standby": ("y_standby.npy", np.float16),
    "y_state": ("y_state.npy", np.int8),
    "y_on": ("y_on.npy", np.int8),
    "y_plugged": ("y_plugged.npy", np.int8),
    "p_noise": ("p_noise.npy", np.float32),
}


def _sample_to_arrays(sample, appliances: Sequence[str]) -> Dict[str, np.ndarray]:
    """SyntheticLoadSample 을 (채널, 시간) 배치 레이아웃으로 편다."""
    n = sample.duration_cycles
    x = np.empty((INPUT_CHANNELS, n), dtype=np.float32)
    x[0:15] = sample.harmonics_ri[:, :, 0].T     # 실수부
    x[15:30] = sample.harmonics_ri[:, :, 1].T    # 허수부
    x[30] = sample.power_features[:, 0]          # P
    x[31] = sample.power_features[:, 1]          # Q
    x[32] = sample.power_features[:, 4]          # V (계측 해상도 반영됨)

    k = len(appliances)
    out = {
        "inputs": x,
        "y_power": np.empty((k, n), np.float32),
        "y_standby": np.empty((k, n), np.float16),
        "y_state": np.empty((k, n), np.int8),
        "y_on": np.empty((k, n), np.int8),
        "y_plugged": np.empty((k, n), np.int8),
        "p_noise": sample.p_noise_w.astype(np.float32),
    }
    for i, a in enumerate(appliances):
        out["y_power"][i] = sample.gt_target_power_w[a]
        out["y_standby"][i] = sample.gt_standby_power_w[a]
        out["y_state"][i] = sample.gt_state_id[a]
        out["y_on"][i] = sample.gt_is_on[a]
        out["y_plugged"][i] = sample.gt_is_plugged[a]
    return out


def build_cache(
    cache_dir: Union[str, Path],
    npz_dir: Union[str, Path] = "processed_data/npz",
    n_scenarios: int = 4000,
    scenario_seconds: float = 60.0,
    window_cycles: int = 600,
    stride_cycles: int = 60,
    recipe_mix: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
    progress_every: int = 200,
) -> Dict:
    """시나리오를 생성해 캐시를 만든다.

    Args:
        n_scenarios: 만들 시나리오 개수. 하나당 약 0.85MB (60초 기준).
        scenario_seconds: 시나리오 1개의 길이. 길수록 창당 비용이 낮아지지만
            같은 시나리오에서 나온 창끼리 상관이 커진다. 60초가 무난하다.
        window_cycles: 학습 창 길이 (600 = 10초)
        stride_cycles: 부창 간격 (60 = 1초). 좁을수록 표본은 늘지만 중복도 커진다.
    """
    if seed is not None:
        np.random.seed(seed)

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_cycles = int(scenario_seconds * 60)
    if scenario_cycles <= window_cycles:
        raise ValueError(
            f"시나리오({scenario_cycles})가 창({window_cycles})보다 길어야 자를 수 있습니다."
        )

    pool = SegmentPool(npz_dir=npz_dir)
    # 고조파 정답은 전력·상태 학습에 쓰지 않으므로 만들지 않는다 (윈도우당 0.65MB 절약).
    synthesizer = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    gen = NILMBatchGenerator(
        segment_pool=pool,
        window_size_cycles=scenario_cycles,
        recipe_mix=recipe_mix or DEFAULT_RECIPE_MIX,
        synthesizer=synthesizer,
        compute_gt_harmonics=False,
    )
    appliances = gen.appliance_list
    k = len(appliances)

    shapes = {
        "inputs": (n_scenarios, INPUT_CHANNELS, scenario_cycles),
        "y_power": (n_scenarios, k, scenario_cycles),
        "y_standby": (n_scenarios, k, scenario_cycles),
        "y_state": (n_scenarios, k, scenario_cycles),
        "y_on": (n_scenarios, k, scenario_cycles),
        "y_plugged": (n_scenarios, k, scenario_cycles),
        "p_noise": (n_scenarios, scenario_cycles),
    }
    mm = {
        name: np.lib.format.open_memmap(
            out_dir / _ARRAYS[name][0], mode="w+", dtype=_ARRAYS[name][1], shape=shapes[name]
        )
        for name in _ARRAYS
    }

    recipes: List[str] = []
    total_bytes = sum(np.prod(s) * np.dtype(_ARRAYS[n][1]).itemsize for n, s in shapes.items())
    print(f"[cache] 시나리오 {n_scenarios:,}개 x {scenario_seconds:.0f}초 "
          f"= 신호 {n_scenarios * scenario_seconds / 3600:.1f}시간 | 예상 {total_bytes / 1e9:.2f} GB")

    for i in range(n_scenarios):
        sample, recipe = gen._synthesize_window()
        arrays = _sample_to_arrays(sample, appliances)
        for name in _ARRAYS:
            mm[name][i] = arrays[name]
        recipes.append(recipe)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1:>6,}/{n_scenarios:,}", flush=True)

    for m in mm.values():
        m.flush()

    # ── 부창 색인과 균형 보정 가중치 ────────────────────────────────────────
    offsets = np.arange(0, scenario_cycles - window_cycles + 1, stride_cycles, dtype=np.int32)
    centers = offsets + window_cycles // 2
    on_center = mm["y_on"][:, :, centers]                      # (n, k, n_off)
    on_center = np.ascontiguousarray(on_center.transpose(0, 2, 1))  # (n, n_off, k)
    flat_on = on_center.reshape(-1, k).astype(np.int8)

    weights = compute_balance_weights(flat_on)

    index = np.stack(
        np.meshgrid(np.arange(n_scenarios, dtype=np.int32), offsets, indexing="ij"), axis=-1
    ).reshape(-1, 2).astype(np.int32)

    np.save(out_dir / "window_index.npy", index)
    np.save(out_dir / "window_on.npy", flat_on)
    np.save(out_dir / "window_weights.npy", weights.astype(np.float32))

    meta = {
        "appliances": appliances,
        "n_scenarios": n_scenarios,
        "scenario_cycles": scenario_cycles,
        "window_cycles": window_cycles,
        "stride_cycles": stride_cycles,
        "n_windows": int(len(index)),
        "input_channels": INPUT_CHANNELS,
        "channel_layout": "0:15 harmonic Real, 15:30 harmonic Imag, 30 P, 31 Q, 32 V",
        "recipe_mix": gen.describe_recipe_mix(),
        "recipe_counts": {r: recipes.count(r) for r in sorted(set(recipes))},
        "signal_hours": round(n_scenarios * scenario_seconds / 3600, 2),
        "bytes": int(total_bytes),
        "class_share_uniform": _share(flat_on, None).tolist(),
        "class_share_weighted": _share(flat_on, weights).tolist(),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2, ensure_ascii=False)
    return meta


def compute_balance_weights(
    on_center: np.ndarray, negative_share: float = 0.20, eps: float = 1e-9
) -> np.ndarray:
    """부창별 샘플링 가중치. 희귀 가전이 든 창을 더 자주 뽑는다.

    다중 라벨 역빈도 방식이다. 창 하나에 여러 가전이 켜져 있을 수 있으므로
    활성 가전들의 역빈도 평균을 쓴다. 가전이 하나도 안 켜진 창(대기전력 전용)은
    별도 몫으로 고정한다 - 이 창들도 대기전력 오탐 방지에 필요하기 때문이다.

    Args:
        on_center: (n_windows, n_appliances) int8, 각 창 중앙 시점의 on/off
        negative_share: 전부 꺼진 창에 배정할 표본 비율
    """
    n, k = on_center.shape
    counts = on_center.sum(axis=0).astype(np.float64)          # 가전별 양성 창 수
    inv = 1.0 / np.maximum(counts, 1.0)

    active = on_center.sum(axis=1)
    w = np.zeros(n, dtype=np.float64)

    pos = active > 0
    if pos.any():
        # 활성 가전들의 역빈도 평균
        w[pos] = (on_center[pos] * inv).sum(axis=1) / np.maximum(active[pos], 1)
        s = w[pos].sum()
        if s > eps:
            w[pos] *= (1.0 - negative_share) / s

    neg = ~pos
    if neg.any():
        w[neg] = negative_share / neg.sum()
    elif pos.any():
        w[pos] /= w[pos].sum()

    total = w.sum()
    return w / total if total > eps else np.full(n, 1.0 / n)


def _share(on_center: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """가전별 양성 라벨 비율. 가중치를 주면 그 분포에서의 기대값."""
    if weights is None:
        return on_center.mean(axis=0)
    return (on_center * weights[:, None]).sum(axis=0)


class WindowCache:
    """캐시를 읽어 창 단위로 꺼내 준다.

    메모리맵이므로 수 GB 캐시라도 RAM 을 거의 쓰지 않는다.
    워커 프로세스마다 열어도 OS 페이지 캐시를 공유한다.
    """

    def __init__(self, cache_dir: Union[str, Path], use_weights: bool = True):
        self.dir = Path(cache_dir)
        with open(self.dir / "meta.json", encoding="utf-8") as fp:
            self.meta = json.load(fp)
        self.appliances: List[str] = self.meta["appliances"]
        self.window_cycles: int = self.meta["window_cycles"]

        self._arr = {
            name: np.load(self.dir / fname, mmap_mode="r")
            for name, (fname, _) in _ARRAYS.items()
        }
        self.index = np.load(self.dir / "window_index.npy")
        self.on_center = np.load(self.dir / "window_on.npy")
        self.weights = np.load(self.dir / "window_weights.npy") if use_weights else None

    def __len__(self) -> int:
        return len(self.index)

    def sample_indices(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """균형 보정된 분포에서 창 색인을 뽑는다."""
        rng = rng or np.random.default_rng()
        if self.weights is None:
            return rng.integers(0, len(self.index), size=n)
        p = self.weights.astype(np.float64)
        return rng.choice(len(self.index), size=n, replace=True, p=p / p.sum())

    def get(self, i: int) -> Dict[str, np.ndarray]:
        """창 하나. 입력은 (33, W), 라벨은 창 중앙 시점의 (9,) 값."""
        s, off = int(self.index[i, 0]), int(self.index[i, 1])
        sl = slice(off, off + self.window_cycles)
        mid = off + self.window_cycles // 2
        return {
            "X": np.asarray(self._arr["inputs"][s, :, sl], dtype=np.float32),
            "y_power": np.asarray(self._arr["y_power"][s, :, mid], dtype=np.float32),
            "y_standby": np.asarray(self._arr["y_standby"][s, :, mid], dtype=np.float32),
            "y_state": np.asarray(self._arr["y_state"][s, :, mid], dtype=np.int64),
            "y_on": np.asarray(self._arr["y_on"][s, :, mid], dtype=np.float32),
            "y_plugged": np.asarray(self._arr["y_plugged"][s, :, mid], dtype=np.float32),
            "p_noise_w": np.float32(self._arr["p_noise"][s, mid]),
        }

    def get_sequence(self, i: int) -> Dict[str, np.ndarray]:
        """seq2seq 용. 라벨도 창 전체 (9, W) 로 돌려준다."""
        s, off = int(self.index[i, 0]), int(self.index[i, 1])
        sl = slice(off, off + self.window_cycles)
        return {
            "X": np.asarray(self._arr["inputs"][s, :, sl], dtype=np.float32),
            "y_power": np.asarray(self._arr["y_power"][s, :, sl], dtype=np.float32),
            "y_standby": np.asarray(self._arr["y_standby"][s, :, sl], dtype=np.float32),
            "y_state": np.asarray(self._arr["y_state"][s, :, sl], dtype=np.int64),
            "y_on": np.asarray(self._arr["y_on"][s, :, sl], dtype=np.float32),
            "y_plugged": np.asarray(self._arr["y_plugged"][s, :, sl], dtype=np.float32),
            "p_noise_w": np.asarray(self._arr["p_noise"][s, sl], dtype=np.float32),
        }

    def class_share(self, weighted: bool = True) -> Dict[str, float]:
        """가전별 양성 라벨 비율. 균형이 유지되는지 확인용."""
        share = _share(self.on_center, self.weights if weighted else None)
        return {a: float(v) for a, v in zip(self.appliances, share)}
