# -*- coding: utf-8 -*-
"""`NILMLoss` 를 세우는 준비를 한 곳에 모은다 (13.84.26).

지문·척도 준비가 `run_train_cnn` 안에 흩어져 있어 시퀀스 학습기가 같은 손실을 못 세웠다.
**두 학습기가 같은 순방향 모형을 쓰게** 하려고 함수로 뺀다 — 한쪽만 바뀌면 두 판을 못 견준다
([[match-the-scoring-convention-before-comparing]]).

`run_train_cnn` 의 조립과 **같은 순서·같은 값**이어야 한다.

⚠⚠ **그 약속이 한 번 깨졌다 (14.171).** 1단계가 `NILMLoss` 에 넘기는 인자가 34개인데
여기는 19개뿐이었다 — 18개가 어긋나 있었고 그중 둘은 **지금 조리법이 켜는 것**이었다:
`harm_sig_vnorm`(argparse 기본 True 대 NILMLoss 기본 False) 와 `cons_deadzone`.
`run_train_seq`(2단계)가 이것으로 경사를 받으므로 **1단계가 배운 물리와 다른 물리로
미세조정**하고 있었다. 이제 `src/run_gate_lossparity.py` 가 AST 로 두 호출의 인자
집합을 대조하고, 어긋나면 **sbatch 가 멈춘다**. 옮길 때 뺀 것 없음:
지문 · 대기 지문(동작 중 휴지 포함) · 잡음 지문(상시 배경 포함) · 차수 척도 · 상태별 지문 · 상태별 척도.
"""
from typing import Optional, Sequence

import numpy as np
import torch

from src.model.losses import LossWeights, NILMLoss, PHASE_COHERENT_EVEN, build_state_scales
from src.model.net import (harmonic_scales, harmonic_signatures, harmonic_signatures_by_power,
                           harmonic_signatures_by_state, noise_signature, standby_signatures)
from src.run_baseline import S_I


#: `stage1_physics` 가 낼 수 있는 `NILMLoss` 인자 이름 (14.171).
#: ⚠ **관문(`run_gate_lossparity`)이 이 상수를 읽는다.** `**stage1_physics(...)` 로 뚫은
#: 자리는 AST 로 안 보이므로, 이 목록이 그 자리의 "명시 인자" 구실을 한다.
#: 여기에 이름을 더하면 아래 함수도 같이 고쳐라 — 관문이 둘의 일치를 확인한다.
STAGE1_PHYSICS_KEYS = (
    "harm_sig_vnorm", "harm_vnorm_exp", "harm_vnorm_frac",
    "cons_deadzone",
    "res_ohm", "res_ohm_half", "res_cond_state", "swap_tol",
    "on_detach_gate", "on_power_praw", "off_detach_praw",
)
#: 아직 **못 잇는** 것 — 체크포인트가 이 값을 안 적는다. 관문이 빚으로 찍는다.
#:   harm_vnorm_vref · harm_vnorm_vref_state : 풀에서 계산하는 텐서 (체크포인트에 없다)
#:   harm_vhrel_rec · _frac · _on             : 위와 같음. 지금은 frac 0 이라 무력
#:   swap_slack · swap_tiebreak · swap_tb_orders : `--w-swap` 이 0 이라 무력


def stage1_physics(ck, apps, skip=()):
    """1단계 체크포인트가 적어 둔 **손실 물리**를 `NILMLoss` 인자로 바꾼다 (14.171).

    왜 이것이 필요한가 — 손실을 짓는 자리가 **셋**이고 셋이 다 달랐다:

    ```
      1단계 run_train_cnn  34개 · 공용 lossbuild 19개 · 2단계 run_adapt 29개
      합집합 49개 중 **셋 다 가진 것은 17개뿐**
    ```

    그래서 2단계가 1단계와 **다른 순방향 모형**으로 경사를 받고 있었다. 손잡이를
    자리마다 따로 받으면 또 갈라지므로, **체크포인트가 유일한 기준**이다.

    ⚠ 키가 없는 **옛 체크포인트**는 그 항목을 안 낸다 — 없는 값을 지어내면 지난 판과
    비교가 끊긴다. 부르는 쪽은 기본값(= 옛 동작, 비트 동일)을 그대로 쓰게 된다.
    """
    skip = set(skip)
    out = {}

    def put(k, v):
        if k not in skip:
            out[k] = v

    if ck.get("harm_sig_vnorm") is not None and ck.get("harm_vnorm_anchor"):
        from src.run_train_cnn import _vnorm_exp
        put("harm_sig_vnorm", bool(ck["harm_sig_vnorm"]))
        put("harm_vnorm_exp", _vnorm_exp(apps, ck.get("harm_vnorm_classes", "")))
        put("harm_vnorm_frac", float(ck.get("harm_vnorm_frac", 1.0)))
    if ck.get("cons_deadzone") is not None:
        put("cons_deadzone", float(ck["cons_deadzone"]))
    if ck.get("res_apps") is not None:
        from src.run_train_cnn import _res_cond, _res_ohm
        put("res_ohm", _res_ohm(apps, ck["res_apps"], half=False))
        put("res_ohm_half", _res_ohm(apps, ck["res_apps"], half=True))
        put("res_cond_state", _res_cond(apps, ck.get("res_cond_state", "")))
        put("swap_tol", float(ck.get("swap_tol", 0.02)))
    for k in ("on_detach_gate", "on_power_praw", "off_detach_praw"):
        if ck.get(k) is not None:
            put(k, bool(ck[k]))
    return out


def hcond_cols_of(head_conductance, hcond_classes, apps):
    """14.299 — `--hcond-classes`(사람이 읽는 이름) -> 열 번호. **두 입구가 같이 쓴다.**

    `--harm-vnorm-classes` 와 같은 규약이다 — 모르는 이름은 죽는다 (오타 하나가 조용히
    전 기기를 빼면 A/B 를 착각한다). 끄면 `None` 이라 `NILMLoss` 가 전 기기를 쓴다.

    ⚠⚠ **14.315 — 이 변환이 `build_loss` 안에만 있었다.** 그래서 `run_train_cnn` 이
      `hcond_classes` 를 **그대로** `NILMLoss` 로 넘겨 `TypeError` 가 났고, `0c5d5f0`
      이후 **모든 학습이 시작조차 못 했다** (985841 의 여섯 판이 1분 만에 죽어서 잡혔다).
      관문 목록에 내가 적어 둔 *"`build_loss` 가 소비하므로 `NILMLoss` 까지 안 간다"* 는
      **`build_loss` 의 사정이지 1단계의 사정이 아니었다** — 그 한 줄이 1단계를 면제했다.
      ⇒ 갈래가 둘이면 변환도 **한 곳에** 둔다 ([[count-every-path-before-claiming-you-cut-one]]).
    """
    if not (head_conductance and str(hcond_classes).strip()):
        return None
    from src.preprocessing.file_registry import get_load_class
    from src.synthesis.grid_simulator import GridSimulator
    _known = {c.name.upper() for c in GridSimulator()._LOAD_EXPONENTS}
    _want = {x.strip().upper() for x in str(hcond_classes).split(",") if x.strip()}
    _bad = _want - _known
    if _bad:
        raise SystemExit("--hcond-classes: 모르는 분류 %s — 있는 것: %s"
                         % (sorted(_bad), sorted(_known)))
    cols = [i for i, a in enumerate(apps)
            if get_load_class(a).name.upper() in _want]
    if not cols:
        raise SystemExit("--hcond-classes %r 가 기기를 하나도 안 고른다" % (hcond_classes,))
    return cols


from src.model.postproc import SMPS_GROUP as _SG375  # noqa: E402


def build_loss(apps: Sequence[str], dev: str, *,
               npz_dir: str = "processed_data/npz",
               time_split: str = "train",
               standby_operating: str = "session",
               background: bool = True,
               state_signatures: bool = True,
               #: ★ 14.395 (§52 (가)) — 사전을 굽기 전에 **녹화별 위상 회전**을 맞춘다.
               #: 기본 False = **비트 동일**. `src/model/sigalign.py` 머리말 참조.
               align_recordings: bool = False,
               power_signatures: bool = False,
               #: * 14.367 — **상태 안** 전력대 보정 (`--pow-sig-instate`).
               #: `power_signatures` 와 달리 `state_signatures` 와 **같이 쓴다**
               #: (14.172 의 축 충돌을 상태로 먼저 갈라 피한다). 기본 False = 비트 동일.
               power_signatures_instate: bool = False,
               power_rel_floor: float = 0.05,
               #: 미리 지은 표를 그대로 받는 길 (1단계가 이렇게 넘긴다). `None` 이면
               #: `power_signatures_instate` 로 여기서 짓는다.
               power_gain_state=None,
               power_edges_state=None,
               #: * 14.375 — SMPS 전용 고조파 항. 0 이면 비트 동일.
               harm_smps: float = 0.0,
               harm_smps_min_order: int = 3,
               smps_sel=None,
               power_bands: int = 3,
               power_tau: float = 0.15,
               harm_even_magnitude: bool = True,
               harm_odd_only: bool = False,
               harm_even_by_class: bool = False,
               head_conductance: bool = False,   # 14.284 — 기본 False 라 **비트 동일**
               hcond_scale: str = "log",         # 14.299 — log | watt
               hcond_on_w: float = 5.0,          # 14.299 — 이 W 위만 전도도 목표
               hcond_classes: str = "",          # 14.299 — 이 부하분류에만 (빈 값 = 전부)
               off_detach_praw: bool = False,
               gate_smooth: float = 0.0,
               gate_focal: float = 0.0,
               harm_grad_balance: str = "off",
               per_state_scale: bool = True,
               drift_basis: str = "",
               drift_k: int = 1,
               weights: Optional[LossWeights] = None,
               #: ── 14.171 — 1단계(`run_train_cnn`)가 `NILMLoss` 에 넘기던 손잡이들.
               #  여기 없어서 **2단계와 진단이 다른 순방향 모형을 쓰고 있었다.**
               #  기본값은 전부 **지금 동작과 비트 동일**이다 — 값은 부르는 쪽이 넣는다.
               harm_sig_vnorm: bool = False,
               harm_vnorm_exp=None,
               harm_vnorm_frac: float = 1.0,
               harm_vnorm_vref=None,
               harm_vnorm_vref_state=None,
               harm_vhrel_rec=None,
               harm_vhrel_frac: float = 0.0,
               harm_vhrel_on=None,
               cons_deadzone: float = -1.0,
               res_ohm=None,
               res_ohm_half=None,
               res_cond_state=None,
               swap_tol: float = 0.02,
               swap_slack: float = 0.0,
               swap_tiebreak: float = 0.0,
               swap_tb_orders: str = "",
               on_detach_gate: bool = False,
               on_power_praw: bool = False,
               verbose: bool = True) -> NILMLoss:
    """`run_train_cnn` 과 같은 `NILMLoss` 를 만든다."""

    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir=npz_dir, time_split=time_split)
    #: 14.395 — 회전표는 **기기마다 한 번**만 낸다. `sig` 와 `sig_state` 가 같은 표를
    #: 써야 두 입구의 위상 기준이 안 갈린다 ([[pin-the-two-entry-points-against-each-other]]).
    rots = None
    if align_recordings:
        from src.model.sigalign import recording_rotations
        rots = recording_rotations(pool, apps)
        if verbose:
            print("  ** 녹화 위상 정렬 (14.395): 기기 %d종 · %d녹화 **"
                  % (len(rots), sum(len(v) for v in rots.values())))
    sig = harmonic_signatures(pool, apps, rotations=rots)
    sb_sig = standby_signatures(pool, apps)
    if standby_operating != "off":
        from src.model.companion import standby_operating_signatures
        from src.synthesis.synthesizer import SESSION_PLUGGED_APPS
        only = SESSION_PLUGGED_APPS if standby_operating == "session" else None
        sb_op, sb_pw, sb_used = standby_operating_signatures(pool, apps, only=only)
        for x in sb_used:
            j = apps.index(x)
            if verbose:
                o = float(np.hypot(sb_sig[j, 0, 0], sb_sig[j, 0, 1])) * 1000
                n = float(np.hypot(sb_op[j, 0, 0], sb_op[j, 0, 1])) * 1000
                print("  ** 동작 중 휴지 지문 (%s) %s: |I1| %.2f -> %.2f mA, 전력 %.2fW **"
                      % (standby_operating, x, o, n, sb_pw[j]))
            sb_sig[j] = sb_op[j]
    nz_sig = noise_signature(pool)
    if background:
        from src.synthesis.sp_curves import background_power, background_signature
        nz_sig = nz_sig + background_signature()
        if verbose:
            print("  ** 상시 배경 (12.166): +%.2fW **" % background_power())
    h_scale = harmonic_scales(pool, apps)
    dproj = None
    if drift_basis and drift_k > 0:
        dproj = build_drift_proj(drift_basis, drift_k, len(h_scale), verbose=verbose)
    pow_gain = pow_edges = None
    import numpy as _np
    pow_gain_s = (None if power_gain_state is None else _np.asarray(power_gain_state))
    pow_edges_s = (None if power_edges_state is None else _np.asarray(power_edges_state))
    if power_signatures_instate and pow_gain_s is None:
        from src.model.net import harmonic_signatures_by_power_instate
        pow_gain_s, pow_edges_s, _usi = harmonic_signatures_by_power_instate(
            pool, apps, n_bands=power_bands, rel_floor=power_rel_floor)
    if power_signatures:
        pow_gain, pow_edges, pow_used = harmonic_signatures_by_power(
            pool, apps, n_bands=power_bands)
        if verbose:
            print("  ** 전력 의존 지문 (13.84.38): %d/%d 칸을 따로 맞췄다 (나머지는 보정비 1) **"
                  % (int(pow_used.sum()), pow_used.size))
    sig_state = None
    if state_signatures:
        sig_state, used = harmonic_signatures_by_state(pool, apps, rotations=rots)
        if verbose:
            print("  ** 상태별 지문 (13.11): %d개 상태를 따로 맞췄다 **" % int(used.sum()))
            _src = getattr(harmonic_signatures_by_state, "last_source", None)
            if _src is not None and (_src == 2).any():
                _who = ["%s s%d" % (apps[k], s) for k, s in zip(*np.nonzero(_src == 2))]
                print("     ** 그중 %d칸은 **실측 전력**을 분모로 맞췄다 (14.167): %s **"
                      % (len(_who), " · ".join(_who)))
    del pool

    #: 14.315 — 변환은 **공유 헬퍼 한 곳**에 있다 (`hcond_cols_of` 독스트링 참조).
    _hcols = hcond_cols_of(head_conductance, hcond_classes, apps)

    return NILMLoss(
        head_conductance=head_conductance,            # 14.284
        hcond_scale=hcond_scale,                      # 14.299
        hcond_on_w=hcond_on_w,                        # 14.299
        hcond_cols=_hcols,                            # 14.299
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz_sig),
        harm_scale=torch.from_numpy(h_scale),
        harm_odd_only=harm_odd_only,
        drift_proj=dproj,
        off_detach_praw=off_detach_praw,
        signatures_state=(torch.from_numpy(sig_state) if state_signatures else None),
        power_gain=(torch.from_numpy(pow_gain) if pow_gain is not None else None),
        power_edges=(torch.from_numpy(pow_edges) if pow_edges is not None else None),
        harm_smps=float(harm_smps),
        harm_smps_min_order=int(harm_smps_min_order),
        smps_sel=(smps_sel if smps_sel is not None else
                  (torch.tensor([1.0 if x in _SG375 else 0.0 for x in apps])
                   if harm_smps > 0 else None)),
        power_gain_state=(torch.from_numpy(pow_gain_s) if pow_gain_s is not None else None),
        power_edges_state=(torch.from_numpy(pow_edges_s) if pow_edges_s is not None else None),
        power_tau=power_tau,
        harm_even_magnitude=harm_even_magnitude,
        even_coherent=(torch.tensor([1.0 if x in PHASE_COHERENT_EVEN else 0.0 for x in apps],
                                    dtype=torch.float32) if harm_even_by_class else None),
        gate_smooth=gate_smooth, gate_focal=gate_focal,
        harm_grad_balance=harm_grad_balance,
        smps_group=[apps.index(x) for x in
                    ("beam_projector", "laptop_charger", "minipc") if x in apps],
        weights=weights or LossWeights(),
        s_state=(build_state_scales(apps, [S_I[x] for x in apps]) if per_state_scale else None),
        #: 14.171 — 아래는 전부 **기본값이면 옛 동작과 비트 동일**이다.
        harm_sig_vnorm=harm_sig_vnorm,
        harm_vnorm_exp=harm_vnorm_exp,
        harm_vnorm_frac=harm_vnorm_frac,
        harm_vnorm_vref=harm_vnorm_vref,
        harm_vnorm_vref_state=harm_vnorm_vref_state,
        harm_vhrel_rec=harm_vhrel_rec,
        harm_vhrel_frac=harm_vhrel_frac,
        harm_vhrel_on=harm_vhrel_on,
        cons_deadzone=cons_deadzone,
        res_ohm=res_ohm,
        res_ohm_half=res_ohm_half,
        res_cond_state=res_cond_state,
        swap_tol=swap_tol,
        swap_slack=swap_slack,
        swap_tiebreak=swap_tiebreak,
        swap_tb_orders=swap_tb_orders,
        on_detach_gate=on_detach_gate,
        on_power_praw=on_power_praw,
    ).to(dev)


def build_drift_proj(path: str, k: int, n_ord: int, verbose: bool = True) -> torch.Tensor:
    """`run_build_drift.py` 가 굳힌 기저 -> `L_harm` 용 (2H,2H) 사영 `I − VᵀV` (13.84.60).

    기저는 홀수차 일부(보통 h1~h15 의 8개)에만 있다. 나머지 차수는 **항등**으로 둔다 —
    그래야 사영을 꺼도 켜도 그 차수들이 비트 단위로 같다.

    레이아웃은 `run_diag_driftdim.vec()` 과 같아야 한다: 앞 Ho 개가 Re, 뒤 Ho 개가 Im,
    차수 순서는 npz 의 `orders` 다. 여기서 (2H,2H) 로 펼 때 그 자리를 정확히 맞춘다.
    """
    B = np.load(path, allow_pickle=True)
    V = np.asarray(B["V"], np.float64)[:k]                       # (k, 2Ho)
    ordr = [int(x) for x in B["orders"]]
    ho = len(ordr)
    if V.shape[1] != 2 * ho:
        raise ValueError("기저 모양 %s 가 차수 %d개와 안 맞는다" % (V.shape, ho))
    # 정규직교인지 확인한다 — 아니면 I−VᵀV 가 사영이 아니다
    g = V @ V.T
    if not np.allclose(g, np.eye(k), atol=1e-6):
        raise ValueError("기저가 정규직교가 아니다 (VVᵀ 최대 이탈 %.2e)"
                         % np.abs(g - np.eye(k)).max())
    idx = [o - 1 for o in ordr]
    if max(idx) >= n_ord:
        raise ValueError("기저 차수 %s 가 손실의 %d차를 넘는다" % (ordr, n_ord))
    full = np.zeros((k, 2 * n_ord))
    full[:, idx] = V[:, :ho]                                     # Re
    full[:, [i + n_ord for i in idx]] = V[:, ho:]                # Im
    P = np.eye(2 * n_ord) - full.T @ full
    if verbose:
        print("  ** 고정 표류 사영 (13.84.60): %s k=%d · 차수 %s · 버리는 차원 %d/%d **"
              % (path, k, ordr, k, 2 * n_ord))
    return torch.from_numpy(P.astype(np.float32))
