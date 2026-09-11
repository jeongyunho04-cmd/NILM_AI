# -*- coding: utf-8 -*-
"""`NILMLoss` 를 세우는 준비를 한 곳에 모은다 (13.84.26).

지문·척도 준비가 `run_train_cnn` 안에 흩어져 있어 시퀀스 학습기가 같은 손실을 못 세웠다.
**두 학습기가 같은 순방향 모형을 쓰게** 하려고 함수로 뺀다 — 한쪽만 바뀌면 두 판을 못 견준다
([[match-the-scoring-convention-before-comparing]]).

`run_train_cnn` 의 조립과 **같은 순서·같은 값**이어야 한다. 옮길 때 뺀 것 없음:
지문 · 대기 지문(동작 중 휴지 포함) · 잡음 지문(상시 배경 포함) · 차수 척도 · 상태별 지문 · 상태별 척도.
"""
from typing import Optional, Sequence

import numpy as np
import torch

from src.model.losses import LossWeights, NILMLoss, PHASE_COHERENT_EVEN, build_state_scales
from src.model.net import (harmonic_scales, harmonic_signatures, harmonic_signatures_by_power,
                           harmonic_signatures_by_state, noise_signature, standby_signatures)
from src.run_baseline import S_I


def build_loss(apps: Sequence[str], dev: str, *,
               npz_dir: str = "processed_data/npz",
               time_split: str = "train",
               standby_operating: str = "session",
               background: bool = True,
               state_signatures: bool = True,
               power_signatures: bool = False,
               power_bands: int = 3,
               harm_even_magnitude: bool = True,
               harm_odd_only: bool = False,
               harm_even_by_class: bool = False,
               off_detach_praw: bool = False,
               gate_smooth: float = 0.0,
               gate_focal: float = 0.0,
               harm_grad_balance: str = "off",
               per_state_scale: bool = True,
               weights: Optional[LossWeights] = None,
               verbose: bool = True) -> NILMLoss:
    """`run_train_cnn` 과 같은 `NILMLoss` 를 만든다."""
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir=npz_dir, time_split=time_split)
    sig = harmonic_signatures(pool, apps)
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
    pow_gain = pow_edges = None
    if power_signatures:
        pow_gain, pow_edges, pow_used = harmonic_signatures_by_power(
            pool, apps, n_bands=power_bands)
        if verbose:
            print("  ** 전력 의존 지문 (13.84.38): %d/%d 칸을 따로 맞췄다 (나머지는 보정비 1) **"
                  % (int(pow_used.sum()), pow_used.size))
    sig_state = None
    if state_signatures:
        sig_state, used = harmonic_signatures_by_state(pool, apps)
        if verbose:
            print("  ** 상태별 지문 (13.11): %d개 상태를 따로 맞췄다 **" % int(used.sum()))
    del pool

    return NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz_sig),
        harm_scale=torch.from_numpy(h_scale),
        harm_odd_only=harm_odd_only,
        off_detach_praw=off_detach_praw,
        signatures_state=(torch.from_numpy(sig_state) if state_signatures else None),
        power_gain=(torch.from_numpy(pow_gain) if pow_gain is not None else None),
        power_edges=(torch.from_numpy(pow_edges) if pow_edges is not None else None),
        harm_even_magnitude=harm_even_magnitude,
        even_coherent=(torch.tensor([1.0 if x in PHASE_COHERENT_EVEN else 0.0 for x in apps],
                                    dtype=torch.float32) if harm_even_by_class else None),
        gate_smooth=gate_smooth, gate_focal=gate_focal,
        harm_grad_balance=harm_grad_balance,
        smps_group=[apps.index(x) for x in
                    ("beam_projector", "laptop_charger", "minipc") if x in apps],
        weights=weights or LossWeights(),
        s_state=(build_state_scales(apps, [S_I[x] for x in apps]) if per_state_scale else None),
    ).to(dev)
