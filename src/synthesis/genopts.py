# -*- coding: utf-8 -*-
"""창 캐시와 시퀀스 캐시가 **같은 생성기**를 쓰게 한다 (13.84.27).

`run_build_seqraw` 는 `LoadSynthesizer(segment_pool=pool)` 로 생성기를 맨몸으로 만들었다.
그래서 `cache_v36.sbatch` 가 창 캐시에 건 설정 열셋 중 **아홉이 빠져 있었다** — 13.29~13.84.16 에서
측정으로 얻은 물리·증강이 통째로 없는 합성이다. v37 몸통을 그 위에서 이어 학습하면 몸통이
**더 약한 생성기 쪽으로 되돌아간다**.

여기 `V36` 이 `cache_v36.sbatch` 와 **같은 값**이다. 바뀌면 두 캐시를 같이 바꿔야 한다
([[match-the-scoring-convention-before-comparing]]).

⚠ `--recipe-mix steady2` 와 `--smps-focus-off-p 0.4` 는 **창을 자르는 층**(`NILMBatchGenerator`)의
설정이라 여기 없다. 시퀀스 쪽에서 그 자리를 맡는 것은 `sequence.SEQ_MIX` 다.
"""
import json
from typing import Any, Dict, Optional, Sequence

#: `cache_v36.sbatch` 의 생성기 설정 그대로 (13.84.16). v37 몸통이 본 합성이다.
V36: Dict[str, Any] = dict(
    power_scale_std="measured",          # 12.118
    state_mix="minipc_balanced",         # 13.35  미니PC IDLE 노출
    carrier_apps=("oven",),              # 13.40
    couple_ext=True,                     # 13.45  결합 델타에 비SMPS 전류
    sp_curves=True,                      # 12.166 부하 의존 서명
    sp_per_texture=True,                 # 13.74  그 녹화의 텍스처에서
    vtail=True,                          # 13.78  전압 꼬리 h17~h31
    float_fill="charger_float",          # 13.83.23 test_1 부동 충전기
    steady_crop="smps_steady",           # 13.83.26
    standby_jitter_cap=95.0,             # 13.83.26
    sibling_rotate="smps_dev1",          # 13.84.16 형제 편차 한꺼번에
)

#: `seqraw_v1` 을 그대로 다시 만드는 설정 (대조군용).
LEGACY: Dict[str, Any] = dict(carrier_apps=("oven",))

PRESETS: Dict[str, Dict[str, Any]] = {"v36": V36, "legacy": LEGACY}


def resolve(spec: str) -> Dict[str, Any]:
    """프리셋 이름이나 JSON 을 설정 딕셔너리로."""
    if not spec:
        return dict(LEGACY)
    if spec in PRESETS:
        return dict(PRESETS[spec])
    return json.loads(spec)


def build_synthesizer(opts: Dict[str, Any], npz_dir: str, time_split: str,
                      compute_gt_harmonics: bool = False):
    """`opts` 대로 `LoadSynthesizer` 를 만든다 — `traincache.build_cache` 와 같은 조립 순서."""
    from src.synthesis.augmentor import (DataAugmentor, FLOAT_FILL_PRESETS,
                                         POWER_SCALE_STD_PRESETS, SIBLING_ROTATE_PRESETS,
                                         STATE_MIX_PRESETS, STEADY_CROP_PRESETS)
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import LoadSynthesizer
    from src.synthesis.vtexture import DEFAULT_VTAIL_NPZ, set_default_vtail

    def _p(table, v):
        if not v:
            return None
        d = table[v] if isinstance(v, str) and v in table else (json.loads(v) if isinstance(v, str) else v)
        # ⚠ 키를 깎지 않는다 — 13.84.16 의 뒤섞기·기울기 항이 조용히 사라진다.
        return {k: (dict(x) if isinstance(x, dict) else x) for k, x in d.items()}

    smx = _p(STATE_MIX_PRESETS, opts.get("state_mix"))
    if smx:
        smx = {k: {int(s): float(x) for s, x in m.items()} for k, m in smx.items()}
    cap = float(opts.get("standby_jitter_cap") or 0.0)
    pool = SegmentPool(npz_dir=npz_dir, time_split=time_split,
                       carrier_apps=tuple(opts.get("carrier_apps") or ()) or None,
                       standby_jitter_cap_pct=(cap if cap else None))
    aug = DataAugmentor(
        power_scale_std_map=_p(POWER_SCALE_STD_PRESETS, opts.get("power_scale_std")),
        state_mix=smx,
        sp_curves=bool(opts.get("sp_curves")),
        sp_per_texture=bool(opts.get("sp_per_texture")),
        float_fill=_p(FLOAT_FILL_PRESETS, opts.get("float_fill")),
        steady_crop=_p(STEADY_CROP_PRESETS, opts.get("steady_crop")),
        sibling_rotate=_p(SIBLING_ROTATE_PRESETS, opts.get("sibling_rotate")))
    set_default_vtail(DEFAULT_VTAIL_NPZ if opts.get("vtail") else None)
    return LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=compute_gt_harmonics,
                           augmentor=aug, background=bool(opts.get("background")),
                           couple_ext=bool(opts.get("couple_ext")))


def describe(opts: Dict[str, Any]) -> str:
    return " · ".join(f"{k}={v}" for k, v in sorted(opts.items()) if v)


def check(opts: Dict[str, Any], gen) -> Sequence[str]:
    """조립이 실제로 걸렸는지 되짚는다 — 빈 목록이면 통과 ([[verify-the-gate-runs-that-path]])."""
    bad = []
    a = gen.augmentor
    if opts.get("sibling_rotate") and not getattr(a, "sibling_rotate", None):
        bad.append("sibling_rotate 가 안 걸렸다")
    if opts.get("sp_curves") and not getattr(a, "_sp", None):
        bad.append("sp_curves 가 안 걸렸다 (processed_data/sp_curves.npz)")
    if opts.get("sp_per_texture") and not getattr(a, "_sp_tex", None):
        bad.append("sp_per_texture 가 안 걸렸다 (processed_data/sp_curves_tex.npz)")
    if opts.get("state_mix") and not getattr(a, "state_mix", None):
        bad.append("state_mix 가 안 걸렸다")
    if opts.get("float_fill") and not getattr(a, "float_fill", None):
        bad.append("float_fill 이 안 걸렸다")
    if opts.get("steady_crop") and not getattr(a, "steady_crop", None):
        bad.append("steady_crop 이 안 걸렸다")
    if opts.get("power_scale_std") and not getattr(a, "power_scale_std_map", None):
        bad.append("power_scale_std 가 안 걸렸다")
    if opts.get("couple_ext") and not getattr(gen.grid_sim, "couple_ext", False):
        bad.append("couple_ext 가 안 걸렸다")
    if opts.get("vtail"):
        from src.synthesis import vtexture
        if getattr(vtexture, "_DEFAULT_VTAIL", None) is None:
            bad.append("vtail 이 안 걸렸다 (processed_data/vtail.npz)")
    return bad
