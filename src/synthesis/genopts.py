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

#: **사슬 이전 생성기** (14.4, 2026-09-13). 사용자 결정으로 본선을 창별로 되돌리면서 생성기도 같이.
#:
#: 사슬 커밋(`0cbf7ff`, 09-12 01:06) 직전의 캐시 sbatch 와 대 보면 v32 와 v36 이 다른 것은
#: **`--sibling-rotate smps_dev1` 하나뿐**이고 나머지 열넷(carrier-on·couple-ext·float-fill·
#: power-scale-std·recipe-mix·smps-focus-off-p·sp-curves·sp-per-texture·standby-jitter-cap·
#: state-mix·steady-crop·vtail·window-cycles·windows)은 전부 같다. 그래서 되돌림은 한 플래그다.
#:
#: ⚠ **이득은 무승부에 가깝다.** 가림까지 맞춘 짝(`cnn_v35` 끔 대 `cnn_v37` 켬)을 같은 채점기
#: (`run_diag_rollback.py`)로 재면 SMPS 신원 +0.008 · on/off **−0.023** · 잔차 −0.3W 다.
#: 판정 줄(on/off)로는 오히려 손해다. 그래도 되돌리는 것은 **사용자 결정**이다 (14.4).
#: ⚠ 되돌리면 13.83.25 가 잡은 **합성 전용 단서**가 되살아난다 — `sibling_rotate` 는 그것을
#: 원천에서 없애려고 넣은 것이었다 (13.84.16). 실패 ③(미니PC 미탐)이 다시 나오면 여기를 의심하라.
V32: Dict[str, Any] = {k: v for k, v in V36.items() if k != "sibling_rotate"}

#: V36 + **녹화별 대기 지문을 기록당 하나씩** (13.84.40).
#: 실측 전부-OFF 배경의 파일 간 편차에서 꽂힌 기기 대기로 설명되는 몫을 빼면
#: h9~h15 에 4.7~7.0mA 가 남는데 V36 은 1.3~1.5mA 밖에 안 준다 (계측계 잡음 참조 3개).
#: 녹화별로 뽑으면 2.2~3.6mA 가 되어 `--bg-head` 가 맡을 미모형 세션 배경이 생긴다.
#: ⚠ V36 은 **그대로 둔다** — 항등 검정과 cnn_v30~v37 비교선이 거기 걸려 있다.
V36R: Dict[str, Any] = dict(V36, standby_per_record=True)

#: V36 + **형제 위상 폭을 실측에 맞춘 것** (13.84.49).
#: `smps_dev1` 은 회전 10°/h + h>=9 뒤섞기로 상대위상 산포를 실측의 12~42배로 만든다.
#: `smps_dev2` 는 0.9°/h · 뒤섞기 없음 — 실측의 0.7~1.2배다 (`run_gate_phase.py`).
#: 이 캐시에서만 v35 가림 해제(`--no-mask`)가 뜻을 갖는다.
V36P: Dict[str, Any] = dict(V36, sibling_rotate="smps_dev2")

#: 13.84.61 — 임의 모수 대신 **실측 표류**를 넣는다. `smps_drift` 는 폭이 실측 그대로고
#: `smps_drift47` 은 차수별 상한(h15 에서 3σ/미니PC=1) 아래로 낮춘 판이다.
V36D: Dict[str, Any] = dict(V36, sibling_rotate="smps_drift")
V36D47: Dict[str, Any] = dict(V36, sibling_rotate="smps_drift47")
V36DP: Dict[str, Any] = dict(V36, sibling_rotate="smps_driftp")

#: `seqraw_v1` 을 그대로 다시 만드는 설정 (대조군용).
LEGACY: Dict[str, Any] = dict(carrier_apps=("oven",))

PRESETS: Dict[str, Dict[str, Any]] = {"v32": V32, "v36": V36, "v36r": V36R, "v36p": V36P,
                                      "v36d": V36D, "v36d47": V36D47, "v36dp": V36DP,
                                      "legacy": LEGACY}


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
                       standby_jitter_cap_pct=(cap if cap else None),
                       standby_per_record=bool(opts.get("standby_per_record")))
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
    # ⚠ **끈 것도 확인한다** (14.4). v32 로 되돌릴 때 어딘가에서 형제 편차가 살아 있으면
    #   되돌린 줄 알고 안 되돌린 캐시를 굽는다 ([[verify-the-gate-runs-that-path]] 의 반대 방향).
    if not opts.get("sibling_rotate") and getattr(a, "sibling_rotate", None):
        bad.append("sibling_rotate 를 안 걸었는데 증강기에 남아 있다")
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
    if opts.get("standby_per_record"):
        # 플래그만 서 있고 녹화가 기기당 하나뿐이면 **아무것도 안 바뀐다** — 그것도 잡는다
        # ([[verify-the-gate-runs-that-path]]).
        if not getattr(gen.pool, "standby_per_record", False):
            bad.append("standby_per_record 가 안 걸렸다")
        else:
            alt = getattr(gen.pool, "standby_profiles_all", {})
            n = sum(1 for v in alt.values() if len(v) >= 2)
            if n < 2:
                bad.append("녹화가 2개 이상인 기기가 %d개뿐이다 — 뽑을 것이 없다" % n)
    if opts.get("vtail"):
        from src.synthesis import vtexture
        if getattr(vtexture, "_DEFAULT_VTAIL", None) is None:
            bad.append("vtail 이 안 걸렸다 (processed_data/vtail.npz)")
    return bad
