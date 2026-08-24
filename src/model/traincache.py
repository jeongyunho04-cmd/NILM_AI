"""
학습용 창 캐시 — 독립 창을 미리 만들어 둔다
=============================================
`src/synthesis/cache.py` 의 `WindowCache` 와 **다른 물건이다.** 그쪽은 긴 시나리오
1개를 스트라이드로 잘라 부창을 만들어 인접 창이 90% 겹친다. 여기는 창 하나하나를
**독립적으로 합성**해 저장한다.

[왜 필요해졌나]
12.1절에서는 캐시를 쓰지 말라고 했다. 근거는 10초 창 기준 생성 14,860 win/s 가
모델 10,000 win/s 보다 빨라서 캐시가 이득 없이 재사용만 만든다는 것이었다.
**60초 창으로 바꾸면서 그 전제가 깨졌다** (12.8.2절):

    생성 550 win/s   vs   모델 9,439 win/s   ->  GPU 사용률 7~10%
    2M 창 학습에 61분이 걸리는데 그중 57분이 CPU 합성 대기다.

[변환 후를 저장한다 — 용량이 10배 작다]
    원시 (33, 3600) float32            475 KB/창
    변환 후 (36,600)+(12,120) float16   45 KB/창

    창 수     용량      생성(1회)   2M 학습 시 재사용
    100k     4.5 GB     3분        20배
    300k    13.5 GB     9분         6.7배
    500k    22.5 GB    15분         4배

[재사용 6.7배가 견딜 만한 이유]
**진짜 다양성 천장은 캐시가 아니라 세그먼트 풀이 정한다.** 학습 풀에 오븐 활성화가
2개, 드라이기가 13개(3.1분)뿐이다(12.3절). 30만 번째 창이 첫 창보다 새로울 여지가
애초에 크지 않다. 그래도 공짜는 아니므로 같은 조건으로 실시간 생성과 한 번
비교해 확인할 것.

[대가]
변환 후를 저장하므로 **창 구성을 바꾸면 재생성해야 한다.** 창 길이 ablation 이
그렇다. `w_cons` / `L_harm` / 게이팅 / 폭 / 시드 는 재생성이 필요 없고,
광역 갈래 유무는 wide 입력을 0 으로 만들면 된다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import time

import numpy as np

from src.model.inputs import FINE_CHANNELS, FINE_CYCLES, WIDE_CHANNELS, build_inputs

# (이름, dtype, 창당 모양)
_SPEC = {
    "fine":       (np.float16, (FINE_CHANNELS, FINE_CYCLES)),
    "wide":       (np.float16, (WIDE_CHANNELS, 0)),      # 두 번째 축은 창 길이에 따라 정해진다
    "y_power":    (np.float32, (0,)),
    "y_on":       (np.int8,    (0,)),
    "y_plugged":  (np.int8,    (0,)),
    "y_standby":  (np.float16, (0,)),
    "y_state":    (np.int8,    (0,)),
    "obs_harm":   (np.float16, (15, 2)),
    "p_noise":    (np.float32, ()),
    "p_observed": (np.float32, ()),
}

_GEN = None
_SEED_BASE = 0


def _init(npz_dir: str, window_cycles: int, time_split: str, seed: int,
          exclude_files_json: str = "", dither_amp: float = 0.0,
          dither_phase_deg: float = 0.0, recipe_mix_json: str = "") -> None:
    global _GEN, _SEED_BASE
    from src.synthesis.augmentor import DataAugmentor
    from src.synthesis.dataset import NILMBatchGenerator
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import LoadSynthesizer
    # 시드는 여기서 걸지 않는다. 워커 번호로 걸면 어느 워커가 어느 청크를 집어 가느냐에
    # 따라 결과가 달라진다 (`chunk_seed` 주석). 청크마다 `_chunk` 안에서 건다.
    _SEED_BASE = int(seed)
    # 녹화 단위 홀드아웃 (설계 문서 12.18절). JSON 문자열로 넘기는 이유는
    # `spawn` 워커에 dict 를 그대로 보내면 피클 경계에서 다루기 번거로워서다.
    excl = json.loads(exclude_files_json) if exclude_files_json else None
    # 레시피 믹스 (12.67절). 빈 문자열이면 `DEFAULT_RECIPE_MIX` 다.
    mix = json.loads(recipe_mix_json) if recipe_mix_json else None
    pool = SegmentPool(npz_dir=npz_dir, time_split=time_split,
                       exclude_activation_files=excl)
    # 차수별 지터 (12.62절). 0 이면 `DataAugmentor` 기본과 같다.
    aug = DataAugmentor(harmonic_dither_amp=float(dither_amp),
                        harmonic_dither_phase_deg=float(dither_phase_deg))
    _GEN = NILMBatchGenerator(
        segment_pool=pool, window_size_cycles=window_cycles,
        synthesizer=LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False,
                                    augmentor=aug),
        recipe_mix=mix, compute_gt_harmonics=False)


def _chunk(task: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """청크 하나를 만든다. 시드는 **청크 번호**로 건다 (`chunk_seed` 주석)."""
    from src.synthesis.dataset import chunk_seed
    index, n = task
    np.random.seed(chunk_seed(_SEED_BASE, index))
    g = _GEN
    w = g.window_size
    k = len(g.appliance_list)
    ti = g.target_index
    xs = np.empty((n, 33, w), np.float32)
    out = {
        "y_power": np.empty((n, k), np.float32), "y_on": np.empty((n, k), np.int8),
        "y_plugged": np.empty((n, k), np.int8), "y_standby": np.empty((n, k), np.float16),
        "y_state": np.empty((n, k), np.int8), "obs_harm": np.empty((n, 15, 2), np.float16),
        "p_noise": np.empty(n, np.float32), "p_observed": np.empty(n, np.float32),
    }
    for j in range(n):
        smp, _ = g._synthesize_window()
        t = g._format_targets(smp)
        xs[j] = g._format_inputs(smp)
        out["y_power"][j] = t["y_power"]; out["y_on"][j] = t["y_on"]
        out["y_plugged"][j] = t["y_plugged"]; out["y_standby"][j] = t["y_standby_power"]
        out["y_state"][j] = t["y_state"]
        out["obs_harm"][j] = smp.harmonics_ri[ti]
        out["p_noise"][j] = smp.p_noise_w[ti]
        out["p_observed"][j] = smp.power_features[ti, 0]
    f, wd = build_inputs(xs)
    out["fine"] = f.astype(np.float16)
    out["wide"] = wd.astype(np.float16)
    return out


def build_cache(
    out_dir: Union[str, Path],
    n_windows: int = 300_000,
    npz_dir: str = "processed_data/npz",
    window_cycles: int = 3600,
    time_split: str = "train",
    seed: int = 0,
    n_workers: int = 11,
    chunk: int = 250,
    exclude_activation_files: Optional[Dict[str, List[str]]] = None,
    dither_amp: float = 0.0,
    dither_phase_deg: float = 0.0,
    recipe_mix: Optional[Dict[str, float]] = None,
) -> dict:
    """독립 창 `n_windows` 개를 만들어 memmap 으로 저장한다."""
    import multiprocessing as mp

    from src.synthesis.segment_pool import SegmentPool
    excl_json = json.dumps(exclude_activation_files) if exclude_activation_files else ""
    mix_json = json.dumps(recipe_mix) if recipe_mix else ""
    apps = SegmentPool(npz_dir=npz_dir, time_split=time_split,
                       exclude_activation_files=exclude_activation_files).get_appliance_types()
    k = len(apps)
    n_wide = window_cycles // 30

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shapes = {
        "fine": (n_windows, FINE_CHANNELS, FINE_CYCLES),
        "wide": (n_windows, WIDE_CHANNELS, n_wide),
        "y_power": (n_windows, k), "y_on": (n_windows, k), "y_plugged": (n_windows, k),
        "y_standby": (n_windows, k), "y_state": (n_windows, k),
        "obs_harm": (n_windows, 15, 2), "p_noise": (n_windows,), "p_observed": (n_windows,),
    }
    mm = {name: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+",
                                          dtype=_SPEC[name][0], shape=shapes[name])
          for name in shapes}
    total = sum(int(np.prod(s, dtype=np.int64)) * np.dtype(_SPEC[n][0]).itemsize
                for n, s in shapes.items())
    print(f"[traincache] 독립 창 {n_windows:,}개 x {window_cycles/60:.0f}초 "
          f"| 예상 {total/1e9:.2f} GB | 창당 {total/n_windows/1024:.0f} KB")

    sizes = [chunk] * (n_windows // chunk)
    if n_windows % chunk:
        sizes.append(n_windows % chunk)
    tasks = list(enumerate(sizes))
    t0 = time.time(); pos = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init,
                  initargs=(npz_dir, window_cycles, time_split, seed, excl_json,
                            dither_amp, dither_phase_deg, mix_json)) as pool:
        # `imap` — 순서 보장. `imap_unordered` 는 이어붙이는 순서가 실행마다 달라져
        # 같은 시드로도 다른 캐시가 나왔다 (12.11절).
        for i, r in enumerate(pool.imap(_chunk, tasks), 1):
            m = len(r["y_power"])
            for name in shapes:
                mm[name][pos:pos + m] = r[name]
            pos += m
            if i % 40 == 0:
                el = time.time() - t0
                print(f"  {pos:>7,}/{n_windows:,}  ({pos/el:,.0f} win/s, "
                      f"남은 {max(0,(n_windows-pos)/max(pos/el,1))/60:.1f}분)", flush=True)
    for m_ in mm.values():
        m_.flush()

    meta = {"n_windows": int(pos), "window_cycles": window_cycles, "appliances": apps,
            "time_split": time_split, "seed": seed, "n_wide": n_wide,
            "exclude_activation_files": exclude_activation_files,
            "dither_amp": float(dither_amp), "dither_phase_deg": float(dither_phase_deg),
            "recipe_mix": recipe_mix,
            "fine_shape": [FINE_CHANNELS, FINE_CYCLES], "bytes": int(total),
            "build_seconds": round(time.time() - t0, 1),
            "positive_rate": {a: float((mm["y_on"][:pos, j] > 0).mean())
                              for j, a in enumerate(apps)}}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"[traincache] 완료 {pos:,}창 / {meta['build_seconds']:.0f}s "
          f"({pos/meta['build_seconds']:,.0f} win/s) -> {out.resolve()}")
    return meta


class CachedWindows:
    """memmap 캐시에서 배치를 뽑는다. 매 epoch 순서를 새로 섞는다."""

    def __init__(self, cache_dir: Union[str, Path]):
        d = Path(cache_dir)
        mp_ = d / "meta.json"
        if not mp_.exists():
            raise FileNotFoundError(
                f"학습 캐시가 없습니다: {d.resolve()}\n"
                f"  python -m src.run_build_traincache 를 먼저 실행하십시오.")
        self.meta = json.loads(mp_.read_text(encoding="utf-8"))
        # 캐시는 build_fine 의 산출물을 그대로 담는다. 채널 수가 바뀌면
        # (12.34: 38 -> 44) 옛 캐시는 못 쓴다. memmap 은 모양을 안 검사하므로
        # 여기서 막지 않으면 엉뚱한 축으로 reshape 되어 조용히 틀린다.
        want = [FINE_CHANNELS, FINE_CYCLES]
        got = list(self.meta.get("fine_shape", want))
        if got != want:
            raise ValueError(
                "학습 캐시의 세밀 채널이 다릅니다: "
                f"캐시 {got} vs 현재 코드 {want}  ({d.resolve()}).  "
                "python -m src.run_build_traincache 로 다시 만드십시오.")
        self.n = self.meta["n_windows"]
        self.appliances = self.meta["appliances"]
        self.arr = {name: np.load(d / f"{name}.npy", mmap_mode="r") for name in _SPEC}

    def __len__(self) -> int:
        return self.n

    def iter_batches(self, batch_size: int, n_batches: int,
                     rng: np.random.Generator, block_windows: int = 24_000):
        """블록 셔플로 배치를 낸다. **작업집합을 블록 크기로 묶는다.**

        매 epoch 전역 셔플로 읽으면 무작위 접근이라 캐시 13GB 전체가 작업집합에
        올라온다. 실측에서 python 작업집합이 18.3GB 까지 커져 물리 메모리 여유가
        0 이 되었다 (파일 기반이라 회수는 되지만 다른 앱이 밀려난다).

        블록을 섞어 순서를 정하고 블록 **안에서만** 무작위로 뽑으면, 동시에 손대는
        구간이 block_windows 개로 제한된다. 24,000창이면 약 1.1GB 다.
        캐시 자체가 창마다 독립 합성이라 블록 안이 이미 무작위이므로,
        전역 셔플 대비 잃는 것이 거의 없다.
        """
        n_blocks = max(1, (self.n + block_windows - 1) // block_windows)
        made = 0
        while made < n_batches:
            for b in rng.permutation(n_blocks):
                lo = int(b) * block_windows
                hi = min(lo + block_windows, self.n)
                if hi - lo < batch_size:
                    continue
                order = lo + rng.permutation(hi - lo)
                for k in range(0, len(order) - batch_size + 1, batch_size):
                    yield self.batch(order[k:k + batch_size])
                    made += 1
                    if made >= n_batches:
                        return

    def batch(self, idx: np.ndarray) -> Tuple[np.ndarray, ...]:
        i = np.sort(np.asarray(idx))          # memmap 은 정렬 접근이 훨씬 빠르다
        a = self.arr
        return (
            np.asarray(a["fine"][i], np.float32), np.asarray(a["wide"][i], np.float32),
            np.asarray(a["y_power"][i], np.float32), np.asarray(a["y_on"][i], np.float32),
            np.asarray(a["y_plugged"][i], np.float32), np.asarray(a["y_standby"][i], np.float32),
            np.asarray(a["y_state"][i], np.int64), np.asarray(a["obs_harm"][i], np.float32),
            np.asarray(a["p_noise"][i], np.float32), np.asarray(a["p_observed"][i], np.float32),
        )
