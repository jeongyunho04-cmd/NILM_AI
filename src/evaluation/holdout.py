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
import multiprocessing as mp
import os as _os
import time

import numpy as np

from src.model.inputs import RAW_CHANNELS, VOLT_ORDERS
from src.synthesis.dataset import DEFAULT_RECIPE_MIX, NILMBatchGenerator
from src.synthesis.augmentor import DataAugmentor
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
    #: ★ 14.353 — (N,2) float32 선로 임피던스 `(r_grid_ohm, x_grid_ohm)`.
    #: 옛 홀드아웃에는 **없다** — 그때는 `None` 이고 쓰는 쪽이 "모름" 으로 다뤄야 한다.
    #: ⚠ 기본값이 있으니 **반드시 맨 뒤**다 (앞에 두면 dataclass 가 안 만들어진다).
    z_grid: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.X)

    def subset(self, mask: np.ndarray) -> "HoldoutSet":
        """레시피 등으로 잘라 본다."""
        m = np.asarray(mask, bool)
        return HoldoutSet(
            X=self.X[m], y_power=self.y_power[m], y_standby=self.y_standby[m],
            y_on=self.y_on[m], y_state=self.y_state[m], y_plugged=self.y_plugged[m],
            p_noise=self.p_noise[m], p_observed=self.p_observed[m], recipe=self.recipe[m],
            z_grid=None if self.z_grid is None else self.z_grid[m],
            appliances=self.appliances, meta={**self.meta, "subset_n": int(m.sum())},
        )


_WGEN = None            #: 워커의 `NILMBatchGenerator` (spawn 풀, 워커마다 하나)
_WTGT = 0               #: 그 생성기의 타깃 시점
_WSEED = 0


def _w_init(opts_json: str, seed: int) -> None:
    """워커 초기자. `spawn` 이라 **워커마다** 조립을 다시 한다 (14.33 ⓑ 의 교훈).

    ⚠ 시드는 여기서 걸지 않는다 — 워커 번호로 걸면 어느 워커가 어느 청크를 집어 가느냐에
    따라 결과가 달라진다. 청크마다 `_w_chunk` 안에서 **청크 번호**로 건다 (`chunk_seed`).
    """
    global _WGEN, _WTGT, _WSEED
    o = json.loads(opts_json)
    # JSON 은 튜플을 잃는다 — `DataAugmentor` 가 기대하는 모양으로 되돌린다.
    if o.get("level_scramble"):
        o["level_scramble"] = {k: tuple(v) for k, v in o["level_scramble"].items()}
    if o.get("state_mix"):
        o["state_mix"] = {a: {int(x): float(y) for x, y in m.items()}
                          for a, m in o["state_mix"].items()}
    o["carrier_apps"] = tuple(o["carrier_apps"]) if o.get("carrier_apps") else None
    o["ablate_pedestal_apps"] = tuple(o["ablate_pedestal_apps"] or ())
    _WSEED = int(seed)
    _, _, _WGEN = _build_generator(o, quiet=True)
    _WTGT = _WGEN.target_index


def _w_chunk(task):
    """청크 하나. 시드는 **청크 번호**로 건다 — 워커 수를 바꿔도 같은 바이트가 나온다."""
    from src.synthesis.dataset import chunk_seed
    index, n = task
    np.random.seed(chunk_seed(_WSEED, index))
    g, tgt = _WGEN, _WTGT
    k = len(g.appliance_list)
    r = {"X": np.empty((n, RAW_CHANNELS, g.window_size), np.float32),
         "y_power": np.empty((n, k), np.float32), "y_standby": np.empty((n, k), np.float32),
         "y_on": np.empty((n, k), np.int8), "y_state": np.empty((n, k), np.int16),
         "y_plugged": np.empty((n, k), np.int8),
         "p_noise": np.empty(n, np.float32), "p_observed": np.empty(n, np.float32),
         #: ★ 14.353 — 선로 임피던스 `(r_grid, x_grid)`. 캐시의 `z_grid` 와 **같은 규약**이다
         #  (`traincache` 도 `smp.metadata` 에서 그대로 읽는다). 창 하나는 배전 환경
         #  하나 위에 놓인다 (`sample_environment`). ⚠ 이걸 안 적으면 홀드아웃에서
         #  Z 주입 갈래를 **아예 못 재고** 학습과 평가가 다른 모델이 된다.
         "z_grid": np.empty((n, 2), np.float32)}
    rec = []
    for j in range(n):
        smp, recipe = g._synthesize_window()
        t = g._format_targets(smp)
        r["X"][j] = g._format_inputs(smp)
        r["y_power"][j], r["y_standby"][j] = t["y_power"], t["y_standby_power"]
        r["y_on"][j], r["y_state"][j], r["y_plugged"][j] = t["y_on"], t["y_state"], t["y_plugged"]
        r["p_noise"][j] = smp.p_noise_w[tgt]
        r["p_observed"][j] = smp.power_features[tgt, 0]
        r["z_grid"][j] = (smp.metadata.get("r_grid_ohm", np.nan),
                          smp.metadata.get("x_grid_ohm", np.nan))
        rec.append(recipe)
    r["recipe"] = np.asarray(rec)
    return r


def _build_generator(o: dict, quiet: bool = False):
    """설정 한 벌 -> `(SegmentPool, LoadSynthesizer, NILMBatchGenerator)`.

    ⚠⚠ **직렬 경로와 병렬 워커가 반드시 이 한 곳을 탄다** (14.51). 이 저장소는 같은 것을
    두 입구에서 조립하다 설정이 갈린 사고를 세 번 냈다 — 13.84.27(시퀀스 캐시에 설정 아홉이
    빠짐) · 14.33 ⓑ(창 캐시에 `vtex_step_s` 배선 없음) · 14.44(홀드아웃에 같은 것이 또 없음).
    [[pin-the-two-entry-points-against-each-other]]

    `quiet` 는 워커용이다 — 24개 워커가 같은 안내를 24번 찍지 않게.
    """
    npz_dir = o['npz_dir']
    holdout_frac = o['holdout_frac']
    ablate_pedestal_apps = o['ablate_pedestal_apps']
    carrier_apps = o['carrier_apps']
    standby_jitter_cap = o['standby_jitter_cap']
    level_scramble = o['level_scramble']
    state_mix = o['state_mix']
    sp_curves = o['sp_curves']
    sp_per_texture = o['sp_per_texture']
    float_fill = o['float_fill']
    steady_crop = o['steady_crop']
    sibling_rotate = o['sibling_rotate']
    #: 14.390 — 기기별 차수비례 위상 지터. 빈 문자열이면 **옛 경로**(비트 동일).
    phase_jitter_map = o.get('phase_jitter_map') or ''
    # 14.62 ⚠ 이 줄이 빠져 있었다 — 14.51 에서 `_build_generator` 를 함수로 뽑을 때
    #   따라오지 않았고, `harmonic_z` 를 쓰는 아래 세 줄이 **NameError** 로 죽었다
    #   (984064). `run_gate_hzwire` (1) 이 이 파일에 그 두 줄이 **있는지만** 보는
    #   글자 검사라 못 잡았다 — 이제 `o[...]` 키가 opts 에 다 있는지 AST 로 본다.
    harmonic_z = o['harmonic_z']
    vtex_step_s = o['vtex_step_s']
    vtail = o['vtail']
    background = o['background']
    couple_ext = o['couple_ext']
    vtex_seg_s = o['vtex_seg_s']
    vtex_coarse_s = o['vtex_coarse_s']          # 14.103 — 세밀 창 밖의 토막 길이
    window_cycles = o['window_cycles']
    recipe_mix = o['recipe_mix']
    smps_focus_off_p = o['smps_focus_off_p']

    pool = SegmentPool(npz_dir=npz_dir, time_split="holdout", holdout_frac=holdout_frac,
                       ablate_pedestal_apps=ablate_pedestal_apps,
                       carrier_apps=carrier_apps,
                       standby_jitter_cap_pct=(float(standby_jitter_cap) if standby_jitter_cap else None))
    # `sp_curves`/`background` 는 **학습 캐시와 반드시 같아야 한다** (12.168.4).
    # 배경 없이 만든 홀드아웃으로 배경 있는 모델을 재면, 모델이 기대하는 5.5W 를
    # 없는 데서 차감해 **최소 부하만** 무너진다 (미니PC −0.119, 선풍기 −0.093).
    # `state_mix`(13.35) 는 **일부러 학습 캐시와 다르게 둘 수 있다** — 홀드아웃은
    # 자를 시간 구간이 달라 미니PC IDLE 이 자연히 47.8% 라, 손대지 않으면 그 자체로
    # 상태가 고른 잣대가 된다. 판을 견줄 때는 **같은 홀드아웃을 그대로 쓴다.**
    aug = DataAugmentor(level_scramble=level_scramble or None,
                        # 14.390 — 빈 문자열이면 None 이라 **비트 동일**이다.
                        phase_jitter_std_map=(phase_jitter_map or None),
                        state_mix=state_mix,
                        sp_curves=bool(sp_curves),
                        sp_per_texture=bool(sp_per_texture),
                        # 13.83.23 상태 채움 + 전력 축소. None 이면 옛 경로 그대로다.
                        float_fill=float_fill,
                        steady_crop=steady_crop,
                        # 13.84.8 ② 형제 전용 차수 비례 회전. None 이면 옛 경로.
                        sibling_rotate=sibling_rotate)
    from src.synthesis.vtexture import (DEFAULT_VTAIL_NPZ, default_library,
                                        set_default_harmonic_z, set_default_step_s,
                                        set_default_vtail)
    # ⚠⚠ **배선 구멍이었다** (14.44). `run_build_traincache` 는 14.33 ⓑ 에서 `vtex_step_s`
    #   를 받게 고쳤는데 **이쪽은 안 고쳤다** — 두 입구가 또 갈려 있었다
    #   ([[pin-the-two-entry-points-against-each-other]]). 0 이면 옛 경로(60초)와 **비트 동일**이다.
    #   여기는 `spawn` 풀이 아니라 한 프로세스라 한 번만 걸면 된다.
    if vtex_step_s and vtex_step_s > 0:
        _n0 = len(default_library().textures)
        set_default_step_s(float(vtex_step_s))
        _n1 = len(default_library().textures)
        if not quiet:
            print(f"  ** 전압 텍스처 표집 간격 {vtex_step_s}s (14.12): "
                  f"텍스처 {_n0} -> {_n1}개 **")
        if _n1 <= _n0:
            raise SystemExit("[holdout] 간격을 줄였는데 텍스처가 안 늘었다 — 안 걸렸다")
    set_default_vtail(DEFAULT_VTAIL_NPZ if vtail else None)
    # 14.59 — 차수별 `Z_h`. ⚠⚠ **학습 캐시와 반드시 같아야 한다** — 다르면 홀드아웃의
    #   텍스처가 다른 Z 로 벗겨져 두 분포가 갈린다. 끄면 비트 동일.
    _hz = None
    if harmonic_z:
        from src.synthesis.grid_simulator import HARMONIC_Z_K
        _hz = HARMONIC_Z_K
    set_default_harmonic_z(_hz)
    syn = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False,
                          augmentor=aug, background=bool(background),
                          couple_ext=bool(couple_ext),
                          # 14.51 — 창 안 텍스처 교체. 학습 캐시와 짝이 맞아야 한다.
                          vtex_seg_s=float(vtex_seg_s or 0.0),
                          # 14.103 — 비균일. 0 이면 균일(옛 경로)로 비트 동일.
                          # `NILM_HO_BLIND_COARSE` 는 **관문의 음성 대조 전용**이다 —
                          # 배선을 일부러 끊어, 아래 되짚기가 정말 잡는지 본다
                          # (`run_gate_hocoarse` [6]). 평소엔 안 걸린다.
                          vtex_coarse_s=(0.0 if _os.environ.get("NILM_HO_BLIND_COARSE")
                                         else float(vtex_coarse_s or 0.0)))
    syn.grid_sim.harmonic_z_table = _hz
    if harmonic_z:
        from src.synthesis import vtexture as _vt
        if getattr(_vt, "_DEFAULT_HZ", None) is None:
            raise SystemExit("[holdout] harmonic_z 가 vtexture 에 안 걸렸다 (14.59)")
        print("  ** 차수별 Z_h 표 (14.59): 자리 D h3·h5 · E h3 **")
    if vtex_seg_s and float(vtex_seg_s) > 0:
        _got = float(getattr(syn.grid_sim, "vtex_seg_s", 0.0) or 0.0)
        if _got != float(vtex_seg_s):
            raise SystemExit("[holdout] vtex_seg_s=%s 인데 시뮬레이터는 %s 다 — 안 걸렸다"
                             % (vtex_seg_s, _got))
        if abs(float(vtex_step_s or 60.0) - float(vtex_seg_s)) > 1e-6:
            raise SystemExit("[holdout] vtex_seg_s=%s 인데 vtex_step_s=%s 다 — 텍스처는 "
                             "step_s 구간의 중앙값이라 둘이 같아야 한다 (14.51)"
                             % (vtex_seg_s, vtex_step_s or 60.0))
        if not quiet:
            print("  ** 창-안 텍스처 교체 %.2f초마다 (60초 창 -> %d장) (14.51) **"
                  % (float(vtex_seg_s), syn.grid_sim.n_texture_segments(3600)))
    # 14.103 — 비균일. **배선을 여기서 되짚는다**: `vtex_seg_s` 만 걸리고 이것이 안 걸리면
    #   조용히 균일로 구워져 학습 캐시와 홀드아웃이 갈린다 ([[verify-the-input-path-not-just-the-model]]).
    if vtex_coarse_s and float(vtex_coarse_s) > 0:
        _gotc = float(getattr(syn.grid_sim, "vtex_coarse_s", 0.0) or 0.0)
        if _gotc != float(vtex_coarse_s):
            raise SystemExit("[holdout] vtex_coarse_s=%s 인데 시뮬레이터는 %s 다 — 안 걸렸다"
                             % (vtex_coarse_s, _gotc))
        if float(vtex_coarse_s) < float(vtex_seg_s or 0.0):
            raise SystemExit("[holdout] vtex_coarse_s=%s 가 vtex_seg_s=%s 보다 짧다 — "
                             "바깥이 더 촘촘하면 비균일의 뜻이 뒤집힌다"
                             % (vtex_coarse_s, vtex_seg_s))
        if not quiet:
            _rng = __import__("numpy").random.default_rng(0)
            _k, _ = syn.grid_sim._texture_plan(3600, _rng)
            print("  ** 비균일 토막 (14.103): 세밀 창 %.2f초 · 바깥 %.0f초 -> 창당 %d장 **"
                  % (float(vtex_seg_s or 0.0), float(vtex_coarse_s), _k))
    gen = NILMBatchGenerator(
        segment_pool=pool, window_size_cycles=window_cycles,
        recipe_mix=recipe_mix or DEFAULT_RECIPE_MIX, synthesizer=syn,
        compute_gt_harmonics=False, smps_focus_off_p=smps_focus_off_p,
    )
    return pool, syn, gen


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
    level_scramble: Optional[Dict[str, tuple]] = None,
    state_mix: Optional[Dict[str, Dict[int, float]]] = None,
    carrier_apps: Optional[Sequence[str]] = None,
    sp_curves: bool = False,
    sp_per_texture: bool = False,
    #: ★ 14.390 — "measured" 면 기기별 위상 지터. 빈 문자열이면 옛 경로.
    phase_jitter_map: str = "",
    vtail: bool = False,
    background: bool = False,
    couple_ext: bool = False,
    smps_focus_off_p: Optional[float] = None,
    float_fill: Optional[Dict[str, dict]] = None,
    steady_crop: Optional[Dict[str, dict]] = None,
    standby_jitter_cap: float = 0.0,
    sibling_rotate: Optional[Dict[str, dict]] = None,
    vtex_step_s: float = 0.0,
    #: 텍스처 한 장이 덮는 합성 시간 (초). 0 이면 창 전체 하나 = 옛 경로 (14.51).
    #: ⚠ **학습 캐시와 같은 값이어야 한다** — 다르면 홀드아웃 창의 창-안 전압 변동이
    #:   학습 창과 달라져 전압 블록에 대한 민감도를 잘못 잰다.
    vtex_seg_s: float = 0.0,
    #: **세밀 창 밖**에서 텍스처 한 장이 덮는 시간 (초) (14.103). 0 이면 창 전체가
    #: `vtex_seg_s` 로 균일 = 옛 경로와 비트 동일. 세밀 창(뒤 600사이클)만 촘촘히
    #: 하고 그 앞은 성기게 두어 토막 수를 20 -> 12 로 줄인다.
    #: ⚠ **학습 캐시와 같은 값이어야 한다** (`train60_v32hs3` 은 25).
    vtex_coarse_s: float = 0.0,
    #: 차수별 `Z_h` 표 (14.59). ⚠⚠ **학습 캐시와 같아야 한다.** 끄면 비트 동일.
    harmonic_z: bool = False,
    #: 병렬 워커 수 (14.51). **0 이면 옛 직렬 경로와 비트 동일**이다.
    #: 1 이상이면 창을 `chunk_windows` 짜리 청크로 잘라 `spawn` 풀에 던진다 — 시드를
    #: **청크 번호**로 걸고 `imap`(순서 보장)으로 받으므로 **워커 수를 바꿔도 같은
    #: 바이트**가 나온다 (`traincache` 가 14.13 항등 검정으로 확인한 그 설계).
    #: ⚠ 그래도 `workers=0` 과 `workers>=1` 은 **서로 다른 홀드아웃**이다 — 난수를
    #:   자르는 방식이 다르다. 옛 체크포인트 점수와 견주려면 그때 쓴 홀드아웃을 그대로 써라.
    workers: int = 0,
    #: 청크 하나가 만드는 창 수. ⚠ 이 값이 바뀌면 **내용도 바뀐다** (청크 경계가 난수를 가른다).
    #: ⚠⚠ **워커 수에 따라 바꾸지 마라** — 그러면 "워커 수가 바뀌어도 같은 바이트" 가 깨진다.
    #: 100 인 까닭: 8,000창이면 80청크라 워커 24개에도 꼬리 불균형이 작고, 청크 하나가
    #: 돌려주는 X 가 100x45x3600x4B = **65MB** 라 워커 24개가 동시에 던져도 1.5GB 다
    #: (250이면 3.9GB).
    chunk_windows: int = 100,
) -> dict:
    """홀드아웃 구간에서만 평가 셋을 만들어 저장한다.

    `smps_focus_off_p` (13.83.4): `smps_overlap` 에서 미니PC 를 끄고 형제 SMPS 만
    켤 확률. **학습 캐시와 같은 값을 줘야 한다** — 다르면 홀드아웃이 학습과 다른
    동시성 구조를 재게 된다. None/0.0 이면 옛 경로다.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # 설정 한 벌. 직렬 경로와 워커가 **같은 조립기**(`_build_generator`)를 탄다.
    opts = {n: v for n, v in (
        ('npz_dir', npz_dir),
        ('holdout_frac', holdout_frac),
        ('ablate_pedestal_apps', ablate_pedestal_apps),
        ('carrier_apps', carrier_apps),
        ('standby_jitter_cap', standby_jitter_cap),
        ('level_scramble', level_scramble),
        ('state_mix', state_mix),
        ('sp_curves', sp_curves),
        ('sp_per_texture', sp_per_texture),
        ('float_fill', float_fill),
        ('steady_crop', steady_crop),
        ('sibling_rotate', sibling_rotate),
        ('phase_jitter_map', phase_jitter_map),
        ('vtex_step_s', vtex_step_s),
        ('vtail', vtail),
        ('background', background),
        ('couple_ext', couple_ext),
        ('vtex_seg_s', vtex_seg_s),
        ('vtex_coarse_s', vtex_coarse_s),
        ('harmonic_z', harmonic_z),
        ('window_cycles', window_cycles),
        ('recipe_mix', recipe_mix),
        ('smps_focus_off_p', smps_focus_off_p),
    )}
    pool, syn, gen = _build_generator(opts)
    apps = gen.appliance_list
    k, tgt = len(apps), gen.target_index

    # 60초 창이면 X 가 3.8GB 라 메모리에 다 못 올린다. 디스크에 바로 쓴다.
    out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32,
                                  shape=(n_windows, RAW_CHANNELS, window_cycles))
    yp = np.empty((n_windows, k), np.float32)
    ys = np.empty((n_windows, k), np.float32)
    yo = np.empty((n_windows, k), np.int8)
    yst = np.empty((n_windows, k), np.int16)
    ypl = np.empty((n_windows, k), np.int8)
    pn = np.empty(n_windows, np.float32)
    pobs = np.empty(n_windows, np.float32)
    zg = np.empty((n_windows, 2), np.float32)          # ★ 14.353 (r_grid, x_grid)
    rec: List[str] = []

    nw = max(0, int(workers or 0))
    print(f"[holdout] 뒤 {holdout_frac:.0%} 구간에서 {n_windows:,}창 생성 "
          f"| 타깃 시점 {tgt}/{window_cycles}"
          + ("  | 직렬" if nw == 0 else f"  | 워커 {nw}개 · 청크 {chunk_windows}창"))
    t0 = time.time()
    if nw == 0:
        for i in range(n_windows):
            smp, recipe = gen._synthesize_window()
            t = gen._format_targets(smp)
            X[i] = gen._format_inputs(smp)
            yp[i], ys[i] = t["y_power"], t["y_standby_power"]
            yo[i], yst[i], ypl[i] = t["y_on"], t["y_state"], t["y_plugged"]
            pn[i] = smp.p_noise_w[tgt]
            pobs[i] = smp.power_features[tgt, 0]
            zg[i] = (smp.metadata.get("r_grid_ohm", np.nan),
                     smp.metadata.get("x_grid_ohm", np.nan))
            rec.append(recipe)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"  {i + 1:>6,}/{n_windows:,}", flush=True)
    else:
        # 청크 번호로 시드하고 `imap`(순서 보장)으로 받는다 — 워커 수와 무관하게 같은 바이트다.
        # ⚠ `imap_unordered` 를 쓰면 안 된다. 이어붙이는 순서가 실행마다 달라진다 (12.11).
        cw = max(1, int(chunk_windows))
        tasks = [(i, min(cw, n_windows - i * cw)) for i in range((n_windows + cw - 1) // cw)]
        dst = {"y_power": yp, "y_standby": ys, "y_on": yo, "y_state": yst,
               "y_plugged": ypl, "p_noise": pn, "p_observed": pobs, "z_grid": zg}
        ctx = mp.get_context("spawn")
        pos = 0
        with ctx.Pool(nw, initializer=_w_init,
                      initargs=(json.dumps(opts, ensure_ascii=False), int(seed))) as wp:
            for i, r in enumerate(wp.imap(_w_chunk, tasks), 1):
                m = len(r["y_power"])
                X[pos:pos + m] = r["X"]
                for name, arr in dst.items():
                    arr[pos:pos + m] = r[name]
                rec.extend(r["recipe"].tolist())
                pos += m
                if progress_every and (pos % max(progress_every, 1) < cw):
                    el = max(time.time() - t0, 1e-9)
                    print(f"  {pos:>7,}/{n_windows:,}  ({pos/el:,.0f} win/s, "
                          f"남은 {max(0, (n_windows-pos)/max(pos/el, 1))/60:.1f}분)", flush=True)
        if pos != n_windows:
            raise SystemExit(f"[holdout] 청크가 {pos}창만 냈다 (원한 것 {n_windows})")

    X.flush()
    print(f"[holdout] 생성 {time.time() - t0:.0f}초 "
          f"({n_windows / max(time.time() - t0, 1e-9):,.0f} win/s)")
    arrays = {"y_power": yp, "y_standby": ys, "y_on": yo, "y_state": yst,
              "y_plugged": ypl, "p_noise": pn, "p_observed": pobs, "z_grid": zg,
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
        "recipe_mix": recipe_mix,
        "level_scramble": {k: list(v) for k, v in (level_scramble or {}).items()},
        "state_mix": state_mix,
        "carrier_apps": list(carrier_apps or []),
        # 13.45: 학습 캐시의 `couple_ext` 와 반드시 같아야 한다.
        "couple_ext": bool(couple_ext),
        "smps_focus_off_p": (None if smps_focus_off_p is None else float(smps_focus_off_p)),
        "float_fill": float_fill,          # 13.83.23 (None 이면 옛 경로)
        "steady_crop": steady_crop,        # 13.83.26
        "standby_jitter_cap": float(standby_jitter_cap or 0.0),   # 13.83.26
        "sibling_rotate": sibling_rotate,  # 13.84.8 ②
        # 학습 캐시와 짝이 맞아야 하는 설정 (12.168.4)
        "sp_curves": bool(sp_curves),
        # 14.44 — 0 이면 옛 경로(기본 60초)다. 학습 캐시와 **같아야** 한다.
        "vtex_step_s": float(vtex_step_s or 0.0),
        "vtex_seg_s": float(vtex_seg_s or 0.0),          # 14.51
        "vtex_coarse_s": float(vtex_coarse_s or 0.0),    # 14.103
        "harmonic_z": bool(harmonic_z),                  # 14.59
        #: 14.333 — **전압 고조파 차수**. `traincache` 와 체크포인트가 적는 것과 같은 규약이다
        #  (14.331 의 막이 ⓒ). ⚠ 이걸 안 적으면 v32h 와 v49 홀드아웃의 meta 가 **한 글자도
        #  안 달라진다** — 채널 수도 shape 도 meta 에 없어서, 짝 대조 관문이 **다른 배치를
        #  통과시킨다.** 옛 홀드아웃에는 이 키가 없고 그건 '6차수' 를 뜻한다.
        "volt_orders": [int(h) for h in VOLT_ORDERS],
        # 14.51 — 병렬로 구웠는가. **0 과 1 이상은 서로 다른 홀드아웃이다** (난수를 자르는
        # 방식이 다르다). 1 이상끼리는 워커 수와 무관하게 같은 바이트다.
        "workers_chunked": bool(workers and int(workers) > 0),
        "chunk_windows": (int(chunk_windows) if (workers and int(workers) > 0) else 0),
        "sp_per_texture": bool(sp_per_texture),
        "phase_jitter_map": str(phase_jitter_map or ""),
        "vtail": bool(vtail),
        "background": bool(background),
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
        z_grid=(np.load(d / "z_grid.npy") if (d / "z_grid.npy").exists() else None),
        appliances=meta["appliances"], meta=meta,
    )
