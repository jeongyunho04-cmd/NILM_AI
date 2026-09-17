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
    원시 (45, 3600) float32            648 KB/창   (33 -> 45: 전압 고조파 12채널, 13.26)
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
from typing import Dict, List, Optional, Tuple, Union, Sequence
import json
import time

import numpy as np

from src.model.inputs import (ZERO_EVEN_HARMONICS, FINE_CHANNELS, FINE_CYCLES, FINE_LAYOUT,
                             RAW_CHANNELS, VOLT_ORDERS, WIDE_CHANNELS, build_inputs)

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
    #: ★ 14.413 (사용자 지적) — **기기별 참 고조파** (k, 15, 2).
    #  합성기는 `SyntheticLoadSample.gt_harmonics_ri` 로 이것을 **정확히** 알고 있는데
    #  여태 캐시가 `obs_harm`(총합)만 담고 버렸다. 그래서 손실이 전력->고조파를 갈 때
    #  **반드시 상수 사전 `sig` 를 거쳐야** 했고, 그 사전 오차가 곧 "못 줄이는 95%" 다
    #  (14.411). §51 이 잰 대로 **합성에서조차** 사전이 틀려 있다 — 충전기 h11 0.592.
    #  ⚠ 2단계(실측)에는 이 정답이 없다. 1단계 전용 재료다.
    #  ⚠ **`batch()` 튜플에는 안 넣는다** — 원소를 끼우면 모든 소비자가 밀린다
    #    (14.103). `realdata` 가 `human()`·`reactive()` 를 따로 뺀 것과 같은 규약으로
    #    `harm_truth()` 메서드로 낸다.
    #  크기: 300,000 x 9 x 15 x 2 x 2B = **162 MB** (24GB 캐시의 0.7%).
    "y_harm":     (np.float16, (0, 15, 2)),
    "p_noise":    (np.float32, ()),
    "p_observed": (np.float32, ()),
    # 13.55 — 이 창의 **선로 임피던스** [r_grid, x_grid] Ω. 모델 입력이 아니라
    # 보조 감독 목표다. 13.54 가 잰 것: 입력에서 log Z 를 R² 0.935 로 뽑을 수
    # 있는데 몸통 z 에서는 0.661 로 흐려진다 — 아무도 보존하라고 안 해서다.
    "z_grid":     (np.float32, (2,)),
}

#: 옛 캐시에는 없는 배열. 없으면 NaN 으로 채워 내보낸다 (배치 길이는 항상 같다).
_OPTIONAL = ("z_grid", "y_harm")

_GEN = None
_SEED_BASE = 0


def _init(npz_dir: str, window_cycles: int, time_split: str, seed: int,
          exclude_files_json: str = "", dither_amp: float = 0.0,
          dither_phase_deg: float = 0.0, recipe_mix_json: str = "",
          dither_even_amp: float = 0.0, dither_even_phase_deg: float = 0.0,
          power_scale_std_json: str = "",
          sp_curves: bool = False,
          sp_per_texture: bool = False, vtail: bool = False,
          vtex_step_s: float = 0.0, vtex_seg_s: float = 0.0,
          vtex_coarse_s: float = 0.0,
          harmonic_z: bool = False,
          background: bool = False,
          level_scramble: Optional[Dict[str, tuple]] = None,
          state_mix_json: str = "",
          carrier_apps: Optional[Sequence[str]] = None,
          dither_min_order: int = 2,
          couple_ext: bool = False,
          smps_focus_off_p: Optional[float] = None,
          float_fill_json: str = "",
          steady_crop_json: str = "",
          standby_jitter_cap: float = 0.0,
          sibling_rotate_json: str = "",
          #: ★ 14.389 — 기기별 차수비례 위상 지터. 빈 문자열이면 **옛 경로**
          #: (일괄 `phase_jitter_max_deg`). "measured" 면 잰 표를 쓴다.
          #: ⚠ **맨 뒤에 붙였다** — `initargs` 가 위치 인자라 중간에 끼우면
          #:   뒤엣것이 밀려 **조용히 다른 캐시를 굽는다** (위 14.103 경고).
          phase_jitter_map: str = "") -> None:
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
                       exclude_activation_files=excl,
                       carrier_apps=carrier_apps,
                       # 13.83.26 대기 잔차 상한. 0 이면 옛 경로.
                       standby_jitter_cap_pct=(float(standby_jitter_cap) if standby_jitter_cap else None))
    # 차수별 지터 (12.62절). 0 이면 `DataAugmentor` 기본과 같다.
    # 기기별 전력 증강 폭 (12.118). 빈 문자열이면 일괄 `power_scale_std` 다.
    pss = json.loads(power_scale_std_json) if power_scale_std_json else None
    # 상태 계층 표집 (13.35). JSON 키는 문자열이므로 int 로 되돌린다.
    smx = ({a: {int(s): float(v) for s, v in m.items()}
            for a, m in json.loads(state_mix_json).items()} if state_mix_json else None)
    aug = DataAugmentor(harmonic_dither_amp=float(dither_amp),
                        harmonic_dither_phase_deg=float(dither_phase_deg),
                        harmonic_dither_even_amp=float(dither_even_amp),
                        harmonic_dither_even_phase_deg=float(dither_even_phase_deg),
                        harmonic_dither_min_order=int(dither_min_order),
                        power_scale_std_map=pss,
                        level_scramble=level_scramble or None,
                        state_mix=smx,
                        sp_curves=bool(sp_curves),
                        sp_per_texture=bool(sp_per_texture),
                        # 13.83.23 상태 채움 + 전력 축소. 빈 문자열이면 옛 경로 그대로다.
                        float_fill=(json.loads(float_fill_json) if float_fill_json else None),
                        # 13.83.26 정상 구간 자르기. 빈 문자열이면 옛 경로.
                        steady_crop=(json.loads(steady_crop_json) if steady_crop_json else None),
                        # 13.84.8 ② 형제 전용 차수 비례 회전. 빈 문자열이면 옛 경로.
                        sibling_rotate=(json.loads(sibling_rotate_json) if sibling_rotate_json else None),
                        # 14.389 — 빈 문자열이면 None 이라 **비트 동일**이다.
                        phase_jitter_std_map=(phase_jitter_map or None))
    # 13.78: 전압 꼬리(h17~h31)를 켠다. 기본은 꺼짐이라 안 부르면 옛 거동 그대로다.
    from src.synthesis.vtexture import (DEFAULT_VTAIL_NPZ, set_default_harmonic_z,
                                        set_default_step_s, set_default_vtail)
    # ⚠ 풀이 `spawn` 이라 **워커마다** 다시 걸어야 한다 — 부모 프로세스에서만
    #   부르면 워커에는 안 물려진다 (14.33 ⓑ). `set_default_vtail` 과 같은 자리다.
    if vtex_step_s and vtex_step_s > 0:
        set_default_step_s(float(vtex_step_s))
    set_default_vtail(DEFAULT_VTAIL_NPZ if vtail else None)
    # 14.59 — 차수별 `Z_h` 표. ⚠⚠ **여기서 안 걸면 워커의 텍스처가 옛 Z 로 지어진다**
    #   (`set_default_step_s` 와 정확히 같은 함정, 14.33 ⓑ). 끄면 비트 동일.
    _hz = None
    if harmonic_z:
        from src.synthesis.grid_simulator import HARMONIC_Z_K
        _hz = HARMONIC_Z_K
    set_default_harmonic_z(_hz)
    _GEN = NILMBatchGenerator(
        segment_pool=pool, window_size_cycles=window_cycles,
        #: ★ 14.413 — **기기별 참 고조파를 켠다** (`y_harm`). 여태 False 였고, 그래서
        #  `gt_harmonics_ri` 가 **빈 딕셔너리**로 와서 배열이 조용히 전부 0 이 됐다
        #  (첫 판에서 실제로 그렇게 구웠다 — 작은 판으로 굽고 확인해서 잡았다).
        #  ⚠ 창당 0.65MB 가 더 든다는 옛 주석은 float32 x 9종 x (N,15,2) 기준이고,
        #    우리는 **타깃 한 점만** float16 으로 담아 창당 **540B** 다 (30만창 = 162MB).
        synthesizer=LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=True,
                                    augmentor=aug, background=bool(background),
                                    couple_ext=bool(couple_ext),
                                    # 14.51 — 창 안에서 텍스처 갈아 끼우기. 0 이면 옛 경로.
                                    vtex_seg_s=float(vtex_seg_s or 0.0),
                                    vtex_coarse_s=float(vtex_coarse_s or 0.0)),
        recipe_mix=mix, compute_gt_harmonics=True,
        smps_focus_off_p=smps_focus_off_p)
    # 14.62 ⚠⚠ **굽기 경로가 반쪽이었다.** 위의 `set_default_harmonic_z` 는 텍스처를 **새 Z 로
    #   벗기게** 만드는데, 단자 전압을 **다시 입히는** 쪽(`_terminal_voltage_harmonics` ->
    #   `grid_simulator.harmonic_z`)은 시뮬레이터의 표를 본다. 여기서 안 걸면 벗긴 Z 와 입힌 Z 가
    #   달라 **서로 안 지워지고 (Z_새 − Z_옛)·I 잔차가 남는다** (자리 D h3 은 Z/Z_1 이 6.71 대 1.02).
    #   `holdout._w_init` 과 `genopts.build_synthesizer` 엔 있었고 **여기에만 없었다** —
    #   그리고 굽기 전 관문은 `genopts` 로 따로 지은 물건을 재서 **통과했다**.
    #   [[pin-the-two-entry-points-against-each-other]] · `run_gate_hzwire.py` 가 이 줄을 지킨다.
    _GEN.synthesizer.grid_sim.harmonic_z_table = _hz


def _chunk(task: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """청크 하나를 만든다. 시드는 **청크 번호**로 건다 (`chunk_seed` 주석)."""
    from src.synthesis.dataset import chunk_seed
    index, n = task
    np.random.seed(chunk_seed(_SEED_BASE, index))
    g = _GEN
    w = g.window_size
    k = len(g.appliance_list)
    ti = g.target_index
    xs = np.empty((n, RAW_CHANNELS, w), np.float32)
    out = {
        "y_power": np.empty((n, k), np.float32), "y_on": np.empty((n, k), np.int8),
        "y_plugged": np.empty((n, k), np.int8), "y_standby": np.empty((n, k), np.float16),
        "y_state": np.empty((n, k), np.int8), "obs_harm": np.empty((n, 15, 2), np.float16),
        "y_harm": np.empty((n, k, 15, 2), np.float16),
        "p_noise": np.empty(n, np.float32), "p_observed": np.empty(n, np.float32),
        "z_grid": np.empty((n, 2), np.float32),
    }
    for j in range(n):
        smp, _ = g._synthesize_window()
        t = g._format_targets(smp)
        xs[j] = g._format_inputs(smp)
        out["y_power"][j] = t["y_power"]; out["y_on"][j] = t["y_on"]
        out["y_plugged"][j] = t["y_plugged"]; out["y_standby"][j] = t["y_standby_power"]
        out["y_state"][j] = t["y_state"]
        out["obs_harm"][j] = smp.harmonics_ri[ti]
        #: ★ 14.413 — 기기별 참 고조파. `appliance_list` **순서 그대로** 채운다 (y_power 와 같은 축).
        #  꺼진 기기는 합성기가 이미 0 을 넣어 둔다 (synthesizer.py 머리말: 활성 성분만 담는다).
        for _kk, _nm in enumerate(g.appliance_list):
            _gh = smp.gt_harmonics_ri.get(_nm)
            out["y_harm"][j, _kk] = _gh[ti] if _gh is not None else 0.0
        out["p_noise"][j] = smp.p_noise_w[ti]
        out["p_observed"][j] = smp.power_features[ti, 0]
        # 한 창은 배전 환경 하나 위에 놓인다 (`sample_environment`). metadata 가
        # 이미 담고 있으므로 새로 계산하지 않는다 — 소수 4자리는 0.3~2.0Ω 에서 무해하다.
        out["z_grid"][j] = (smp.metadata.get("r_grid_ohm", np.nan),
                            smp.metadata.get("x_grid_ohm", np.nan))
    #: ⚠⚠ 14.413 — **조용한 실패를 막는다.** `compute_gt_harmonics` 가 꺼져 있으면
    #  `gt_harmonics_ri` 가 빈 딕셔너리라 위 줄이 **전부 0** 을 쓴다. 그러면 나중에
    #  "지도가 걸린 줄 알았는데 안 걸린" 판이 나온다. 켜진 기기가 하나라도 있는데
    #  `y_harm` 이 통째로 0 이면 여기서 멈춘다.
    if out["y_on"].any() and not np.any(out["y_harm"]):
        raise SystemExit("y_harm 이 전부 0 이다 — `compute_gt_harmonics` 가 꺼져 있다 "
                         "(LoadSynthesizer 와 NILMBatchGenerator **둘 다** 켜야 한다)")
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
    dither_even_amp: float = 0.0,
    dither_even_phase_deg: float = 0.0,
    power_scale_std_map: Optional[Dict[str, float]] = None,
    level_scramble: Optional[Dict[str, tuple]] = None,
    state_mix: Optional[Dict[str, Dict[int, float]]] = None,
    carrier_apps: Optional[Sequence[str]] = None,
    sp_curves: bool = False,
    sp_per_texture: bool = False,
    #: ★ 14.389 — "measured" 면 기기별 위상 지터를 쓴다. 빈 문자열이면 옛 경로.
    phase_jitter_map: str = "",
    vtail: bool = False,
    #: 전압 텍스처 표집 간격 (초). 0 이면 `vtexture.DEFAULT_STEP_S`(60) — 옛 경로다.
    #: 14.12 가 **표류의 모자란 몫이 이 상수였다**고 쟀다: 텍스처는 `step_s` 구간의
    #: 중앙값이라 그보다 짧은 전압 변동이 뭉개진다. 20 으로 줄이면 파일 안 전압 산포가
    #: 원시의 0.73 -> 0.95 가 되고 합성 표류가 실측의 0.75 -> **0.80** 배가 된다
    #: (이론값 √0.654 = 0.809 와 일치). 프리셋 `genopts.V32T` 가 이 값 20.0 하나다.
    #: ⚠ **여태 창 캐시에는 배선이 없었다** — `genopts.build_synthesizer` 는 걸지만
    #:   `build_cache` 는 `set_default_vtail` 만 불렀다 (14.33 ⓑ). 두 입구가 갈렸던 자리다.
    vtex_step_s: float = 0.0,
    #: 텍스처 **한 장이 덮는 합성 시간** (초). 0 이면 창 전체에 한 장 — 옛 경로다 (14.51).
    #: 14.50 이 남긴 구멍: `_terminal_voltage_harmonics` 가 `rel` 을 창 전체에 정적으로
    #: 곱해 **창-안 `V_h/|V_1|` 변동이 정확히 0** 이었다. 합성이 내던 13~35% 는 전부
    #: `−Z·I` 항이었고 텍스처가 소유한 축은 비어 있었다. 창을 토막 내 그 녹화의 연속
    #: 텍스처를 얹으면 실측 대비 회수율이 `2토막 35% · 3토막 55% · 6토막 78%` 다
    #: (`run_diag_vtexseg.py`). 프리셋 `genopts.V32S` 가 10.0/10.0 이다.
    #: ⚠ **`vtex_step_s` 와 같은 값**이어야 한다 — 텍스처는 그 구간의 중앙값이라
    #:   20초 중앙값을 10초씩 틀면 변화를 2배로 빨리 감는 것이 된다.
    vtex_seg_s: float = 0.0,
    #: 14.103 — **세밀 창 바깥**(창 끝 `FINE_CYCLES` 를 뺀 앞부분)의 토막 길이 (초).
    #: 0 이면 창 전체를 `vtex_seg_s` 로 균일하게 — **옛 경로와 비트 동일**이다.
    #: 세밀 갈래는 창 3600사이클 중 **마지막 600(10초)** 만 본다. 균일 3초면 거기 토막이
    #: 3개뿐이고 나머지 17개는 광역만 보는데, 광역은 저항 판정 기여가 0.0~0.2% 다.
    #: `seg_s 1.25 + coarse 25` 면 총 12토막으로 **비용 1.61배 절감 · 세밀 해상도 3배**다.
    #: ⚠ 여기는 `vtex_step_s` 와 같을 필요가 **없다** — 짧은 중앙값으로 긴 구간을 덮는 것은
    #:   스냅샷 대표라 변화를 빨리 감지 않는다 (14.51 이 막는 것은 반대 방향이다).
    vtex_coarse_s: float = 0.0,
    #: 차수별 `Z_h` 표 (14.59). `False` 면 `r + j·h·x` — 옛 경로와 **비트 동일**.
    #: ⚠⚠ 켜면 **텍스처 자신이 달라진다** (`rel_open` de-embed) — 홀드아웃도 같이 켜야 한다.
    harmonic_z: bool = False,
    background: bool = False,
    dither_min_order: int = 2,
    couple_ext: bool = False,
    smps_focus_off_p: Optional[float] = None,
    float_fill: Optional[Dict[str, dict]] = None,
    steady_crop: Optional[Dict[str, dict]] = None,
    standby_jitter_cap: float = 0.0,
    sibling_rotate: Optional[Dict[str, dict]] = None,
    shard: Optional[Tuple[int, int]] = None,
) -> dict:
    """독립 창 `n_windows` 개를 만들어 memmap 으로 저장한다.

    `shard=(k, n)`: 청크 목록의 **k/n 번째 토막만** 만든다 (14.13). 노드 여러 대로 나눠 굽고
    `run_merge_traincache` 로 번호순으로 이어붙이면 단일 노드 결과와 **비트 동일**하다 —
    `_chunk` 가 **청크 번호**로 시드하고(`chunk_seed`) `imap` 이 순서를 보장하기 때문이다.
    시드·청크 크기·`n_windows` 는 전체 기준 그대로 줘야 한다. 그래야 번호가 안 밀린다.

    `smps_focus_off_p`: `smps_overlap` 에서 미니PC 를 끄고 형제 SMPS 만 켤 확률
    (13.83). None/0.0 이면 옛 경로 그대로다. **`recipe_mix` 재조정과 같이** 써야
    동시성 phi 가 0 에 간다 — 어느 한쪽만으로는 +0.16 / +0.14 에서 멈춘다.
    """
    import multiprocessing as mp

    from src.synthesis.segment_pool import SegmentPool
    excl_json = json.dumps(exclude_activation_files) if exclude_activation_files else ""
    mix_json = json.dumps(recipe_mix) if recipe_mix else ""
    pss_json = json.dumps(power_scale_std_map) if power_scale_std_map else ""
    smx_json = json.dumps(state_mix) if state_mix else ""
    apps = SegmentPool(npz_dir=npz_dir, time_split=time_split,
                       exclude_activation_files=exclude_activation_files).get_appliance_types()
    k = len(apps)
    n_wide = window_cycles // 30

    # ── 14.13 토막 나누기. memmap 크기를 정하기 **전에** 몫을 확정한다 ──────────
    sizes_all = [chunk] * (n_windows // chunk)
    if n_windows % chunk:
        sizes_all.append(n_windows % chunk)
    tasks_all = list(enumerate(sizes_all))
    if shard is None:
        tasks = tasks_all
    else:
        sk, sn = int(shard[0]), int(shard[1])
        if not (0 <= sk < sn):
            raise SystemExit("[traincache] shard=%s 가 잘못됐다 (0 <= k < n)" % (shard,))
        lo = (len(tasks_all) * sk) // sn
        hi = (len(tasks_all) * (sk + 1)) // sn
        tasks = tasks_all[lo:hi]
        if not tasks:
            raise SystemExit("[traincache] 토막 %d/%d 에 청크가 없다 (청크 %d개)"
                             % (sk, sn, len(tasks_all)))
    n_out = int(sum(sz for _, sz in tasks))

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    shapes = {
        "fine": (n_out, FINE_CHANNELS, FINE_CYCLES),
        "wide": (n_out, WIDE_CHANNELS, n_wide),
        "y_power": (n_out, k), "y_on": (n_out, k), "y_plugged": (n_out, k),
        "y_standby": (n_out, k), "y_state": (n_out, k),
        "obs_harm": (n_out, 15, 2), "p_noise": (n_out,), "p_observed": (n_out,),
        "y_harm": (n_out, k, 15, 2),
        "z_grid": (n_out, 2),
    }
    mm = {name: np.lib.format.open_memmap(out / f"{name}.npy", mode="w+",
                                          dtype=_SPEC[name][0], shape=shapes[name])
          for name in shapes}
    total = sum(int(np.prod(s, dtype=np.int64)) * np.dtype(_SPEC[n][0]).itemsize
                for n, s in shapes.items())
    _tag = "" if shard is None else (" [토막 %d/%d · 청크 %d~%d]"
                                     % (shard[0], shard[1], tasks[0][0], tasks[-1][0]))
    print(f"[traincache] 독립 창 {n_out:,}개 x {window_cycles/60:.0f}초{_tag} "
          f"| 예상 {total/1e9:.2f} GB | 창당 {total/n_out/1024:.0f} KB")
    t0 = time.time(); pos = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init,
                  initargs=(npz_dir, window_cycles, time_split, seed, excl_json,
                            dither_amp, dither_phase_deg, mix_json,
                            dither_even_amp, dither_even_phase_deg, pss_json,
                            bool(sp_curves), bool(sp_per_texture), bool(vtail),
                            float(vtex_step_s or 0.0), float(vtex_seg_s or 0.0),
                            # ⚠⚠ 14.103 — `initargs` 는 **위치 인자**다. `_init` 의 서명에
                            #   인자를 끼워 넣으면 여기에도 **같은 자리**에 넣어야 한다.
                            #   안 넣으면 뒤엣것이 한 칸씩 밀려 `harmonic_z` 가
                            #   `vtex_coarse_s` 자리로 들어간다 — 문법 오류가 안 나고
                            #   **조용히 다른 캐시를 굽는다** (14.62 와 같은 부류).
                            float(vtex_coarse_s or 0.0),
                            bool(harmonic_z),
                            bool(background), level_scramble,
                            smx_json, tuple(carrier_apps or ()),
                            int(dither_min_order), bool(couple_ext),
                            smps_focus_off_p,
                            json.dumps(float_fill) if float_fill else "",
                            json.dumps(steady_crop) if steady_crop else "",
                            float(standby_jitter_cap or 0.0),
                            json.dumps(sibling_rotate) if sibling_rotate else "",
                            str(phase_jitter_map or ""))) as pool:
        # `imap` — 순서 보장. `imap_unordered` 는 이어붙이는 순서가 실행마다 달라져
        # 같은 시드로도 다른 캐시가 나왔다 (12.11절).
        for i, r in enumerate(pool.imap(_chunk, tasks), 1):
            m = len(r["y_power"])
            for name in shapes:
                mm[name][pos:pos + m] = r[name]
            pos += m
            if i % 40 == 0:
                el = time.time() - t0
                print(f"  {pos:>7,}/{n_out:,}  ({pos/el:,.0f} win/s, "
                      f"남은 {max(0,(n_out-pos)/max(pos/el,1))/60:.1f}분)", flush=True)
    for m_ in mm.values():
        m_.flush()

    meta = {"vtex_step_s": float(vtex_step_s or 0.0),   # 14.33 ⓑ (0 = 기본 60초)
            "float_fill": float_fill,          # 13.83.23 상태 채움 + 전력 축소 (None 이면 옛 경로)
            "steady_crop": steady_crop,        # 13.83.26 정상 구간 자르기 (None 이면 옛 경로)
            "standby_jitter_cap": float(standby_jitter_cap or 0.0),   # 13.83.26 대기 잔차 상한 백분위 (0 = 옛 경로)
            "sibling_rotate": sibling_rotate,  # 13.84.8 ② 형제 전용 차수 비례 회전 (None 이면 옛 경로)
            "n_windows": int(pos), "window_cycles": window_cycles, "appliances": apps,
            # 14.13 — 토막이면 전체 기준값과 맡은 청크 범위를 남긴다. 이어붙일 때 관문이 읽는다.
            "shard": (None if shard is None else [int(shard[0]), int(shard[1])]),
            "shard_chunks": (None if shard is None else [tasks[0][0], tasks[-1][0]]),
            "n_windows_total": int(n_windows), "chunk": int(chunk),
            "time_split": time_split, "seed": seed, "n_wide": n_wide,
            # 14.93 — 세밀은 `fine_shape` 로 막혀 있었는데 광역은 아무것도 없었다.
            "wide_shape": [int(WIDE_CHANNELS), int(n_wide)],
            #: 14.331 — **입력 배치의 규약**. 없으면 6차수(1,3,5,7,9,11) 판이다.
            #  채널 수만으로도 걸리지만(`run_gate_cacheshape`) 어느 차수인지는
            #  여기에만 남는다 — 나중에 짝수차를 넣을 수도 있으므로 값을 적는다.
            "volt_orders": [int(h) for h in VOLT_ORDERS],
            "exclude_activation_files": exclude_activation_files,
            "dither_amp": float(dither_amp), "dither_phase_deg": float(dither_phase_deg),
            "dither_min_order": int(dither_min_order),
            "recipe_mix": recipe_mix,
            "dither_even_amp": float(dither_even_amp),
            "dither_even_phase_deg": float(dither_even_phase_deg),
            "power_scale_std_map": power_scale_std_map,
            "level_scramble": level_scramble,
            "state_mix": state_mix,
            # 13.83: smps_overlap 에서 미니PC 를 끄고 형제 SMPS 만 켠 비율.
            # None/0.0 이면 옛 경로다. 동시성 phi 를 읽으려면 recipe_mix 와 함께 봐야 한다.
            "smps_focus_off_p": (None if smps_focus_off_p is None
                                 else float(smps_focus_off_p)),
            "carrier_apps": list(carrier_apps or []),
            # 13.45: 결합 델타의 Σ 에 비SMPS 전류를 넣었는가. 홀드아웃과 짝이 맞아야 한다.
            "couple_ext": bool(couple_ext),
            # 부하 의존 서명 / 상시 배경 (12.166). 학습·손실 쪽이
            # 이 값을 읽어 짝을 맞춘다.
            "sp_curves": bool(sp_curves),
            "sp_per_texture": bool(sp_per_texture),
            # 14.389 — 캐시가 어떤 위상 지터로 구워졌는지 **적어 둔다**.
            # 이게 없으면 나중에 두 캐시를 구별할 길이 없다.
            "phase_jitter_map": str(phase_jitter_map or ""),
            "vtail": bool(vtail),
            # 14.51 — 텍스처 **한 장이 덮는 시간**. (`vtex_step_s` 는 위에 이미 있다.)
            "vtex_seg_s": float(vtex_seg_s or 0.0),
            # 14.103 — 세밀 창 **바깥**의 토막 길이. 0 이면 균일(옛 경로).
            "vtex_coarse_s": float(vtex_coarse_s or 0.0),
            "harmonic_z": bool(harmonic_z),          # 14.59
            "background": bool(background),
            "fine_shape": [FINE_CHANNELS, FINE_CYCLES], "bytes": int(total),
            "zero_even_harmonics": bool(ZERO_EVEN_HARMONICS),
            # 세밀 채널 **배치**. 채널 수가 같아도 뜻이 다를 수 있다 (13.12).
            "fine_layout": str(FINE_LAYOUT),
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
        self.arr = {name: np.load(d / f"{name}.npy", mmap_mode="r")
                    for name in _SPEC if name not in _OPTIONAL}
        # 옛 캐시(v22 이전)에는 `z_grid.npy` 가 없다. 배치 길이를 바꾸지 않으려고
        # NaN 을 내보낸다 — 손실 쪽이 유한값만 골라 쓴다.
        for name in _OPTIONAL:
            p = d / f"{name}.npy"
            self.arr[name] = np.load(p, mmap_mode="r") if p.exists() else None
        self.has_z = self.arr["z_grid"] is not None
        #: ★ 14.348 — 조합 머리(`--comb-tau`)가 쓰는 창별 `Ĝ_sum` (mS).
        #  `run_build_ghat` 이 **나중에** 얹으므로 `_SPEC` 이 아니라 여기서 따로 읽는다
        #  (굽기·병합 경로를 안 건드린다). 없으면 **NaN** 을 내보내 배치 길이를 지킨다 —
        #  `z_grid` 와 같은 규약이다. `--comb-tau` 를 켰는데 NaN 이면 트레이너가 멈춘다.
        _gp = d / "g_hat.npy"
        self.arr["g_hat"] = np.load(_gp, mmap_mode="r") if _gp.exists() else None
        self.has_ghat = self.arr["g_hat"] is not None
        # ── 14.93 배열의 **실제 모양**을 본다 ────────────────────────────────
        # 위 `fine_shape` 검사는 **meta 를 믿는다**. 광역은 meta 에 모양 키가
        # 아예 없었고(`n_wide` 만 있다) 검사도 없었다 — `WIDE_CHANNELS` 를 바꾸면
        # memmap 이 **조용히 엉뚱한 축으로 reshape** 된다. 세밀 쪽 주석이 12.34 에서
        # 그 사고를 이미 적어 뒀는데 광역만 안 막혀 있었다.
        # `np.load(mmap_mode="r")` 은 `.npy` 헤더의 모양을 그대로 주므로, 이 검사는
        # **`wide_shape` 가 없는 옛 캐시에도 통한다.**
        _nw = int(self.meta.get("n_wide", 0))
        for _nm, _want in (("fine", (int(FINE_CHANNELS), int(FINE_CYCLES))),
                           ("wide", (int(WIDE_CHANNELS), _nw) if _nw else None)):
            if _want is None:
                continue
            _got = tuple(int(x) for x in self.arr[_nm].shape[1:])
            if _got != _want:
                raise ValueError(
                    "학습 캐시의 %s 배열 모양이 다릅니다: 캐시 %s vs 기대 %s  (%s). "
                    "채널 수가 바뀌었으면 python -m src.run_build_traincache 로 "
                    "다시 만드십시오 (memmap 은 모양을 안 검사해 그냥 두면 "
                    "**조용히 틀립니다**)."
                    % (_nm, _got, _want, d.resolve()))

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

    def harm_truth(self, idx: np.ndarray):
        """★ 14.413 — 기기별 **참 고조파** (n, K, 15, 2) 또는 없으면 `None`.

        **`batch()` 의 자리수를 안 바꾼다** — `realdata.human()`·`reactive()` 와 같은
        규약이다 (14.103: 튜플에 끼우면 모든 소비자가 조용히 밀린다).

        ⚠ 옛 캐시(v52 이하)에는 없다. `None` 이면 부르는 쪽이 **멈춰야 한다** —
        NaN 으로 흘려보내면 "지도가 걸린 줄 알았는데 안 걸린" 판이 나온다.
        """
        a = self.arr.get("y_harm")
        if a is None:
            return None
        return np.asarray(a[np.sort(np.asarray(idx))], np.float32)

    def batch(self, idx: np.ndarray) -> Tuple[np.ndarray, ...]:
        i = np.sort(np.asarray(idx))          # memmap 은 정렬 접근이 훨씬 빠르다
        a = self.arr
        return (
            np.asarray(a["fine"][i], np.float32), np.asarray(a["wide"][i], np.float32),
            np.asarray(a["y_power"][i], np.float32), np.asarray(a["y_on"][i], np.float32),
            np.asarray(a["y_plugged"][i], np.float32), np.asarray(a["y_standby"][i], np.float32),
            np.asarray(a["y_state"][i], np.int64), np.asarray(a["obs_harm"][i], np.float32),
            np.asarray(a["p_noise"][i], np.float32), np.asarray(a["p_observed"][i], np.float32),
            (np.asarray(a["z_grid"][i], np.float32) if a["z_grid"] is not None
             else np.full((len(i), 2), np.nan, np.float32)),
            (np.asarray(a["g_hat"][i], np.float32) if a["g_hat"] is not None
             else np.full(len(i), np.nan, np.float32)),
        )
