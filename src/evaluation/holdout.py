"""
고정 합성 홀드아웃 평가 셋 (Frozen Synthetic Holdout)
======================================================
**시드만 바꾼 테스트 셋은 홀드아웃이 아니다.** 같은 측정 파형을 다시 쓰기 때문이다.
오븐은 활성화가 2개뿐이라 모델이 그 파형 자체를 외울 수 있고, 그 상태로 잰
"테스트 성능" 은 아무것도 말해 주지 않는다.

여기서는 `SegmentPool(time_split="holdout")` — 각 원본 녹화의 **뒤 20%** — 로만
평가 셋을 만든다. 9종 전부에 대해 학습에서 본 적 없는 파형이 된다.

    가전            학습(앞 80%)   홀드아웃(뒤 20%)
    air_conditioner   43.1분           12.1분
    oven              27.8분            7.2분
    hair_dryer         3.1분            0.8분   ← 가장 얇다
    ...

**한 번 만들어 디스크에 얼려 두고 재사용한다.** 매번 새로 만들면 실행 간 비교가
잡음에 묻힌다. 내용 해시를 meta 에 남겨 두어 같은 셋인지 확인할 수 있다.

    python -m src.run_build_holdout

    from src.evaluation.holdout import load_holdout
    hs = load_holdout()
    hs.X            # (N, 33, 600) float32
    hs.y_power      # (N, 9)  타깃 시점 전력
    hs.recipe       # (N,)    레시피별로 잘라 볼 수 있다
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Dict, List, Optional, Union
import hashlib
import json
import numpy as np

from src.synthesis.dataset import DEFAULT_RECIPE_MIX, NILMBatchGenerator
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

DEFAULT_DIR = Path("processed_data/holdout")
DEFAULT_N_WINDOWS = 8000
DEFAULT_SEED = 20260821


@dataclass
class HoldoutSet:
    X: np.ndarray               # (N, 33, W) float32
    y_power: np.ndarray         # (N, K) float32  활성 전력
    y_standby: np.ndarray       # (N, K) float32
    y_on: np.ndarray            # (N, K) int8
    y_state: np.ndarray         # (N, K) int16
    y_plugged: np.ndarray       # (N, K) int8
    p_noise: np.ndarray         # (N,) float32
    p_observed: np.ndarray      # (N,) float32  타깃 시점 관측 총전력
    recipe: np.ndarray          # (N,) <U32
    appliances: List[str]
    meta: dict

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, mask: np.ndarray) -> "HoldoutSet":
        """레시피 등으로 잘라 본다."""
        m = np.asarray(mask, bool)
        return HoldoutSet(
            X=self.X[m], y_power=self.y_power[m], y_standby=self.y_standby[m],
            y_on=self.y_on[m], y_state=self.y_state[m], y_plugged=self.y_plugged[m],
            p_noise=self.p_noise[m], p_observed=self.p_observed[m], recipe=self.recipe[m],
            appliances=self.appliances, meta={**self.meta, "subset_n": int(m.sum())},
        )


def build_holdout(
    out_dir: Union[str, Path] = DEFAULT_DIR,
    npz_dir: Union[str, Path] = "processed_data/npz",
    n_windows: int = DEFAULT_N_WINDOWS,
    window_cycles: int = 600,
    seed: int = DEFAULT_SEED,
    holdout_frac: float = 0.2,
    recipe_mix: Optional[Dict[str, float]] = None,
    progress_every: int = 1000,
    ablate_pedestal_apps: Optional[Sequence[str]] = None,
) -> dict:
    """홀드아웃 구간에서만 평가 셋을 만들어 저장한다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    pool = SegmentPool(npz_dir=npz_dir, time_split="holdout", holdout_frac=holdout_frac,
                       ablate_pedestal_apps=ablate_pedestal_apps)
    syn = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    gen = NILMBatchGenerator(
        segment_pool=pool, window_size_cycles=window_cycles,
        recipe_mix=recipe_mix or DEFAULT_RECIPE_MIX, synthesizer=syn,
        compute_gt_harmonics=False,
    )
    apps = gen.appliance_list
    k, tgt = len(apps), gen.target_index

    # 60초 창이면 X 가 3.8GB 라 메모리에 다 못 올린다. 디스크에 바로 쓴다.
    out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32,
                                  shape=(n_windows, 33, window_cycles))
    yp = np.empty((n_windows, k), np.float32)
    ys = np.empty((n_windows, k), np.float32)
    yo = np.empty((n_windows, k), np.int8)
    yst = np.empty((n_windows, k), np.int16)
    ypl = np.empty((n_windows, k), np.int8)
    pn = np.empty(n_windows, np.float32)
    pobs = np.empty(n_windows, np.float32)
    rec: List[str] = []

    print(f"[holdout] 뒤 {holdout_frac:.0%} 구간에서 {n_windows:,}창 생성 "
          f"| 타깃 시점 {tgt}/{window_cycles}")
    for i in range(n_windows):
        smp, recipe = gen._synthesize_window()
        t = gen._format_targets(smp)
        X[i] = gen._format_inputs(smp)
        yp[i], ys[i] = t["y_power"], t["y_standby_power"]
        yo[i], yst[i], ypl[i] = t["y_on"], t["y_state"], t["y_plugged"]
        pn[i] = smp.p_noise_w[tgt]
        pobs[i] = smp.power_features[tgt, 0]
        rec.append(recipe)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1:>6,}/{n_windows:,}", flush=True)

    X.flush()
    arrays = {"y_power": yp, "y_standby": ys, "y_on": yo, "y_state": yst,
              "y_plugged": ypl, "p_noise": pn, "p_observed": pobs,
              "recipe": np.asarray(rec)}
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", arr)

    # 내용 해시 - 같은 평가 셋인지 확인용. X 는 커서 표본만 쓴다.
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X[::37]).tobytes())
    h.update(yp.tobytes()); h.update(yo.tobytes())

    meta = {
        "n_windows": n_windows, "window_cycles": window_cycles,
        "target_index": tgt, "seed": seed,
        "time_split": "holdout", "holdout_frac": holdout_frac,
        "ablate_pedestal_apps": list(ablate_pedestal_apps or []),
        "appliances": apps,
        "channel_layout": "0:15 harmonic Real, 15:30 harmonic Imag, 30 P, 31 Q, 32 V",
        "recipe_counts": {r: rec.count(r) for r in sorted(set(rec))},
        "positive_rate": {a: float(yo[:, j].mean()) for j, a in enumerate(apps)},
        "pool_holdout_minutes": {
            a: round(sum(x.duration_cycles for x in v) / 3600, 2)
            for a, v in pool.appliance_activations.items()
        },
        "content_sha256": h.hexdigest()[:16],
        "bytes": int(X.nbytes + sum(a.nbytes for a in arrays.values())),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[holdout] 저장 완료 {out.resolve()} | {meta['bytes'] / 1e9:.2f} GB "
          f"| sha {meta['content_sha256']}")
    return meta


def load_holdout(out_dir: Union[str, Path] = DEFAULT_DIR) -> HoldoutSet:
    """얼려 둔 홀드아웃 평가 셋을 읽는다."""
    d = Path(out_dir)
    mp = d / "meta.json"
    if not mp.exists():
        raise FileNotFoundError(
            f"홀드아웃 셋이 없습니다: {d.resolve()}\n  python -m src.run_build_holdout 을 먼저 실행하십시오."
        )
    meta = json.loads(mp.read_text(encoding="utf-8"))
    g = lambda n: np.load(d / f"{n}.npy", allow_pickle=False)
    # X 는 최대 3.8GB 라 메모리맵으로 연다. 슬라이스할 때만 실제로 읽힌다.
    return HoldoutSet(
        X=np.load(d / "X.npy", mmap_mode="r"), y_power=g("y_power"), y_standby=g("y_standby"), y_on=g("y_on"),
        y_state=g("y_state"), y_plugged=g("y_plugged"), p_noise=g("p_noise"),
        p_observed=g("p_observed"), recipe=g("recipe"),
        appliances=meta["appliances"], meta=meta,
    )
