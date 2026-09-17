# -*- coding: utf-8 -*-
"""체크포인트 -> **모델 한 대**. 짓는 곳을 하나로 모은다 (14.378).

왜 이게 있나 — 이 인자 묶음은 `run_gate_check.load_model` 에만 있었고, 배포
묶음의 `predictor.py` 는 **자기가 따로** 다섯 개만 넘겨 모델을 지었다:
```
  배포판   width · wide_summary · periodicity · fine_dropout · prior_kappa · fine_channels
  연구판   위 + dilation · 탭 · p_state_cap · head_layout · **comb_tau · comb_over** · …
  ⇒ 같은 체크포인트인데 **다른 모델**이 선다. 조합 머리가 통째로 빠진다
```
`run_gate_lossparity` 가 **학습기 둘**에서 똑같은 병을 잡았다 (§44.7). 추론 쪽은
관문이 없어서 15일을 몰랐다. 이제 둘 다 여기를 부른다
([[verify-the-input-path-not-just-the-model]] · [[the-gate-must-build-the-real-object]]).

⚠ 이 파일은 **배포 묶음에 그대로 비춰진다** (`deploy/sync_runtime.py`). `src.evaluation`
  이나 `src.synthesis` 를 끌어오지 마라 — 배포에는 없다.
"""
from typing import Optional, Tuple
import os

import numpy as np
import torch

from src.model.inputs import LEGACY_FINE_CHANNELS
from src.model.net import NILMNet, appliance_state_counts


def comb_over_of(pk) -> float:
    """★ 14.353 — 초과 주장 꺾기의 **추론 운영점**을 정한다.

    `comb_over` 를 적은 판(14.352 이후)은 **그 값을 그대로** 따른다. 그 키가 아예
    없는 옛 조합 판은 학습 뒤에 나온 항이므로 운영점 `COMB_OVER_OP` 를 얹는다 —
    합 15 -> **8** · C포트오탐 9 -> **2** (14.352). **조용히 하지 않고 한 줄 찍는다.**
    """
    if float(pk.get("comb_tau", 0.0) or 0.0) <= 0:
        return 0.0
    #: ⚠⚠ 14.377 — **체크포인트의 `comb_over` 는 "학습에 쓴 값" 이지 "추론 설정" 이
    #  아니다.** 둘을 섞어서 판정을 하나 망쳤다: `cnn_comb` 은 키가 없어 운영점 0.05 가
    #  얹혔는데 `cnn_powsig` 는 `--comb-over 0` 으로 구웠다고 키가 0.0 이라 **추론에서도
    #  꺼졌다**. 그 상태로 견줘 합 **8 대 19** 를 내고 "기각" 이라고 적었다 —
    #  공정하게 재니 **8 대 9** 다 (§44 정정).
    #  ⇒ **추론은 항상 운영점을 쓴다.** 학습 때 넣었든 아니든 추론에서 켜는 건
    #    14.352 가 따로 잰 결정이다 (합 15 -> 8 · C포트오탐 9 -> 2).
    #    일부러 끄려면 `NILM_COMB_OVER=0` 으로 명시해라 — 조용히 꺼지지 않게.
    import os
    from src.model.gbudget import COMB_OVER_OP
    _env = os.environ.get("NILM_COMB_OVER")
    _v = float(_env) if _env not in (None, "") else float(COMB_OVER_OP)
    print("  ** 초과 주장 꺾기 comb_over=%.3f (%s) · 체크포인트 학습값 %s **"
          % (_v, "환경변수" if _env else "추론 운영점", pk.get("comb_over", "없음")))
    return _v


def check_input_convention(ck: dict, ckpt_path: str) -> None:
    """입력 배치 규약(전압 차수·짝수차 중앙값)이 체크포인트와 맞나. 어긋나면 **멈춘다**."""
    #: 14.160 — 입력 파이프라인 설정은 **모델이 아니라 자의 규약**이다. 어긋나면
    #: 채점이 학습과 다른 입력을 보게 되므로 크게 경고한다
    #: ([[match-the-scoring-convention-before-comparing]]).
    from src.model import inputs as _I
    #: 14.331 — **전압 고조파 차수**도 같은 종류의 규약이다. 어긋나면 `net.py` 가
    #  `fine[:, :fine_channels]` 로 잘라서 **다른 뜻의 채널**을 먹는다 — 죽지 않고
    #  그럴듯한 틀린 수를 낸다. `even_median` 과 같은 이유로 **멈춘다**.
    _vo = ck.get("volt_orders")
    if _vo is not None and tuple(int(x) for x in _vo) != tuple(_I.VOLT_ORDERS):
        raise SystemExit(
            "\u2716 \uc804\uc555 \uace0\uc870\ud30c \ucc28\uc218\uac00 \uc5b4\uae4b\ub09c\ub2e4 \u2014 \uccb4\ud06c\ud3ec\uc778\ud2b8 %s\ub300 \uc9c0\uae08 %s\n"
            "    %s\n"
            "  \uc785\ub825 \ubc30\uce58\uac00 \ub2ec\ub77c \uc7ac\ud559\uc2b5\ud574\uc57c \ud55c\ub2e4 (FINE %d \ub300 %d)."
            % (tuple(int(x) for x in _vo), tuple(_I.VOLT_ORDERS), ckpt_path,
               45 + 2 * len(_vo), _I.FINE_CHANNELS))
    if _vo is None:
        print("  \u26a0 \uccb4\ud06c\ud3ec\uc778\ud2b8\uc5d0 `volt_orders` \uac00 \uc5c6\ub2e4 (14.331 \uc774\uc804 \ud310) \u2014 "
              "\uc9c0\uae08 %s \ub85c \uc77d\ub294\ub2e4" % (tuple(_I.VOLT_ORDERS),))
    _em = int(ck.get("even_median", 0) or 0)
    if max(_em, 1) != max(int(_I.EVEN_MEDIAN), 1):
        #: 14.168 — **경고가 아니라 멈춘다.** 이 어긋남의 실패 방식은 죽는 것이 아니라
        #  *그럴듯한 틀린 수를 찍는 것*이라, 소리 없이 지나가면 판정이 통째로 무효가 된다.
        #  오늘 하루에 세 번 잡혔다 (`run_diag_spike` · `run_diag_rollback` · `run_diag_vblock`).
        #  분포 밖 시험을 **일부러** 하려면 `NILM_ALLOW_EVEN_MISMATCH=1` 로 명시하라.
        import os as _os
        _msg = ("✖ 짝수차 이동중앙값이 어긋난다 — 체크포인트 k=%d 인데 지금 입력은 k=%d 다@N@"
                "    %s@N@"
                "  창을 짓기 **전에** 규약을 맞춰라:@N@"
                "      from src.run_gate_check import sync_even_median@N@"
                "      sync_even_median(a.ckpt)        # 목록도 홑 문자열도 받는다@N@"
                "  규약이 갈리는 판을 한 표에 놓아야 하면 무리를 나눠 창을 다시 지어라@N@"
                "  (`run_diag_rollback` · `run_diag_ghost` 가 그렇게 한다).@N@"
                "  일부러 분포 밖을 보려면  NILM_ALLOW_EVEN_MISMATCH=1"
                ).replace("@N@", chr(10)) % (_em, int(_I.EVEN_MEDIAN), ckpt_path)
        if _os.environ.get("NILM_ALLOW_EVEN_MISMATCH", "") not in ("1", "true", "True"):
            raise SystemExit(_msg)
        print("⚠⚠ (NILM_ALLOW_EVEN_MISMATCH) " + _msg.splitlines()[0])


def model_kwargs(ck: dict, pk: Optional[dict] = None,
                 state_power_init: bool = True) -> dict:
    """체크포인트 -> `NILMNet` 인자. **여기가 정본이다.**"""
    if pk is None:
        pk = ck
    nkw = dict(width=ck.get("width", 1.0),
                    wide_summary=ck.get("wide_summary", False),
                    wide_target=ck.get("wide_target", False),
                    periodicity=ck.get("periodicity", False),
                    fine_dropout=ck.get("fine_dropout", 0.0),
                    prior_kappa=ck.get("prior_kappa", 0.0),
                    hard_gate=ck.get("hard_gate", 0.0),
                    gate_free_power=ck.get("gate_free_power", False),
                    prior_beta=ck.get("prior_beta", 0.5),
                    aux_z=ck.get("aux_z", False),
                    # 13.84.68/70 — 상태 전력 슬롯을 잰 값에서 출발시킨다. `weights=True`
                    # 면 곧바로 덮어써지므로 **물려받는 경로에는 영향이 없다**.
                    # `weights=False`(처음부터 배우는 판)에서만 실제로 듣는다.
                    state_power_init=state_power_init,
                    # 합 정합성 사영 (계획 A, 14.3). 옛 체크포인트에는 키가 없어 0 이고,
                    # 0 이면 `_project` 를 아예 안 부른다 -> **완전한 하위호환**이다.
                    proj=pk.get("proj", 0.0),
                    proj_cap=pk.get("proj_cap", 0.5),
                    proj_floor=pk.get("proj_floor", 5.0),
                    proj_resp=pk.get("proj_resp", "power"),
                    # 전압 지수 (14.7). 옛 체크포인트에는 키가 없어 False 이고,
                    # False 면 지수가 전부 0 이라 `vrel**0 = 1` -> **완전한 하위호환**이다.
                    vexp=bool(pk.get("vexp", False)),
                    # 기기 축 어텐션 (13.93). 없으면 0 이고 키 자체가 안 생긴다.
                    appl_attn=pk.get("appl_attn", 0),
                    appl_attn_heads=pk.get("appl_attn_heads", 4),
                    # 구간별 풀링 (14.46). 옛 체크포인트에는 키가 없어 0 이고,
                    # 0 이면 구간이 창 전체 하나라 **비트 동일**이다.
                    seg_pool=pk.get("seg_pool", 0),
                    wide_seg_pool=pk.get("wide_seg_pool", 0),
                    wide_extra_dilations=pk.get("wide_extra_dilations", None),
                    # 세밀 dilation (14.78). 옛 체크포인트에는 키가 없어 None 이고,
                    # None 이면 (1,2,4,8,16) 이라 **비트 동일**이다.
                    fine_dilations=pk.get("fine_dilations", None),
                    # 세밀 전역 풀링 (14.79). 없으면 "both" 라 비트 동일.
                    fine_pool=pk.get("fine_pool", "both"),
                    # 덧붙인 블록·탭 (14.80). 없으면 옛 기본이라 비트 동일.
                    fine_extra_dilations=pk.get("fine_extra_dilations", None),
                    tap_layers=pk.get("tap_layers", None),
                    # 세밀 몸통 시간 분할 (14.116). 없으면 False 라 **비트 동일**이다.
                    # ⚠ 14.116 이전에 구운 tsp 체크포인트에는 이 키가 없다 — 아래에서
                    #   `trunk.0` 모양으로 유추하되 **맞는지 확인하고** 쓴다.
                    fine_time_split=bool(pk.get("fine_time_split", False)),
                    # 상태 전력 상한 (14.121). 없으면 0 이라 **비트 동일**이다.
                    p_state_cap=float(pk.get("p_state_cap", 0.0) or 0.0),
                    # 머리에서 뺀 덩이 (14.128). 없으면 빈 값이라 **비트 동일**이다.
                    head_drop=str(pk.get("head_drop", "") or ""),
                    # 머리 배치 (14.130). 없으면 v1 이라 **비트 동일**이다.
                    head_layout=str(pk.get("head_layout", "v1") or "v1"),
                    # 세밀 패딩 (14.131). 없으면 zeros 라 **비트 동일**이다.
                    #: 14.183 — 없으면 옛 동작(비트 동일)이라 옛 체크포인트가 그대로 실린다.
                    state_power_src=str(pk.get("state_power_src", "table") or "table"),
                    fine_norm=str(pk.get("fine_norm", "window") or "window"),
                    fine_conv=str(pk.get("fine_conv", "sym") or "sym"),
                    fine_tpool=str(pk.get("fine_tpool", "whole") or "whole"),
                    fine_derive=str(pk.get("fine_derive", "window") or "window"),
                    wide_dg=bool(pk.get("wide_dg", False)),
        comb_tau=float(pk.get("comb_tau", 0.0) or 0.0),
                    z_input=bool(pk.get("z_input", False)),
                    comb_over=comb_over_of(pk),
                    comb_over_margin=float(pk.get("comb_over_margin", 2.0) or 2.0),
                    fine_dc=str(pk.get("fine_dc", "keep") or "keep"),
                    fine_pad=str(pk.get("fine_pad", "zeros") or "zeros"),
                    # 미래 토막 수 (14.122). 없으면 1 이라 **비트 동일**이다.
                    fine_future_segs=int(pk.get("fine_future_segs", 1) or 1),
                    fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS))
    return nkw


def build_model(ck: dict, dev: str, weights: bool = True, mask: bool = True,
                state_power_init: bool = True, proj_from: Optional[dict] = None,
                ckpt_path: str = "<dict>") -> Tuple[torch.nn.Module, list]:
    """체크포인트 dict -> `(model, apps)`. 가림 훅까지 건다."""
    apps = ck["appliances"]
    pk = dict(ck) if proj_from is None else {**ck, **proj_from}
    nkw = model_kwargs(ck, pk, state_power_init=state_power_init)
    model = NILMNet(apps, appliance_state_counts(apps), **nkw).to(dev)
    if weights:
        # ⚠ 14.116 직후에 구운 `cnn_tsp_*` 는 `fine_time_split` 키를 **안 적었다**
        #   (학습은 됐는데 채점 경로가 모델을 못 지어 `trunk.0` 이 1021 대 765 로
        #   어긋난다). 키가 없을 때만, **모양이 맞는지 확인하고** 켬으로 다시 짓는다.
        #   유추가 틀리면 아래 `load_state_dict` 가 그대로 터진다 — 조용히 안 넘어간다.
        want = ck["model"].get("trunk.0.weight")
        if (want is not None and "fine_time_split" not in ck
                and tuple(model.trunk[0].weight.shape) != tuple(want.shape)):
            m2 = NILMNet(apps, appliance_state_counts(apps),
                         **{**nkw, "fine_time_split": True}).to(dev)
            if tuple(m2.trunk[0].weight.shape) == tuple(want.shape):
                print("  ⚠ %s 에 `fine_time_split` 키가 없다 — `trunk.0` %s 가 맞아 "
                      "**켬**으로 지었다 (끔이면 %s)."
                      % (ckpt_path, tuple(want.shape),
                         tuple(model.trunk[0].weight.shape)))
                model = m2
        model.load_state_dict(ck["model"])
    model.eval()
    # 학습 때 0 으로 가린 채널은 **추론에서도** 0 이어야 한다 (13.80.10 · 13.84.12).
    # 값이 있으면 본 적 없는 입력이 된다 — v35 를 가리지 않고 채점하면 홀드아웃
    # F1 0.929 가 0.856 으로 나온다. 채점 경로가 여럿이라 모델에 forward pre-hook
    # 으로 붙여 어느 경로든 자동으로 가린다.
    # ⚠ `mask=False` 는 **처음부터 배우는 판에서만** 쓴다 (13.84.27).
    zf = [int(x) for x in str(ck.get("zero_channels", "") or "").split(",") if x.strip()] if mask else []
    zw = [int(x) for x in str(ck.get("zero_wide_channels", "") or "").split(",") if x.strip()] if mask else []
    if zf or zw:
        def _zero_inputs(m, args):
            f, w = args[0], args[1]
            if zf:
                f = f.clone(); f[:, [c for c in zf if c < f.shape[1]]] = 0.0
            if zw:
                w = w.clone(); w[:, [c for c in zw if c < w.shape[1]]] = 0.0
            return (f, w) + tuple(args[2:])
        model.register_forward_pre_hook(_zero_inputs)
    model.zero_channels_applied = (zf, zw)
    return model, apps
