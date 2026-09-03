"""
손실 (설계 문서 3절)
=====================
    L_power    Huber(P̂/s_i, P/s_i, δ=0.1)                가중 1.0   3.1절
    L_state    CrossEntropy (기기별 유효 클래스 마스킹)      가중 0.3   3.2절
    L_on       BCE                                        가중 0.3
    L_plugged  BCE                                        가중 0.1
    L_standby  Huber(δ=1W)                                가중 0.1
    L_harm     || Σ P̂_i·sig_i + h_noise − 관측 고조파 ||    가중 0.1   3.4절
    L_cons     | Σ P̂ + Σ Ŝ + P_noise − P_관측 |            **1단계 0.0**  3.3절

[1단계에서 L_cons 를 끄는 이유 — 3.3절]
합성에서 `P_관측 ≡ Σ라벨` 은 항등식(0.5절)이라 보존 손실은 정보를 더하지 않는다.
반면 그래디언트는 개별 전력 감독보다 35~3,000배 강해서, 배분이 미결정인 채로
"합만 맞추는 해" 로 끌고 간다. 2단계 실측 적응에서만 켠다.

[L_harm 이 필수인 이유 — 3.3/3.4절]
잔차 헤드를 뺐으므로 `L_cons` 는 "합이 얼마나 모자란지" 만 알고 어느 기기인지 모른다.
전력은 스칼라 하나라 배분이 근본적으로 미결정이다. 고조파는 30차원이라 배분을
실제로 결정한다. 12.5절에서 총전력 잔차가 전가를 전혀 못 본다는 것을 실증했다.

[Q 는 손실에 쓰지 않는다 — 3.6절]
PF≈1 구간에서 `Q = √(S²−P²)` 는 조건수가 나쁘다. 입력에는 넣되 손실에는 안 쓴다.

[L_over — 물리 상한 힌지 (12.10절)]
    L_over = mean( relu(Σ P̂ + Σ Ŝ + P_noise − P_관측) / max(P_관측, 10W) )

**보존 손실(L_cons)과 결정적으로 다르다: 한쪽 방향만 벌한다.**
실측에서 w_cons=0.05 를 켜자 모델이 붕괴했다 (포트 편향 -879W, F1 0.937 -> 0.643).
양방향 제약이라 "합만 맞추고 배분은 포기" 하는 해로 끌려간 것이다.
여기는 **넘칠 때만** 벌하므로 그 방향으로 끌 수 없다 — 과소 예측에는 기울기가 0 이다.

물리적으로도 엄밀하다. `P_관측 = Σ활성 + Σ대기 + 계측계` 이고 모든 항이 음수가
아니므로 `Σ P̂ <= P_관측` 은 **반드시 참인 부등식**이다. 근사가 아니다.

관측 전력으로 나누는 이유: 30W 짜리 창에서 393W 를 예측하는 것(비 12.1)과
1300W 창에서 100W 넘치는 것(비 0.077)은 전혀 다른 잘못이다. 절대 W 로 재면
둘이 비슷해 보이고, 정작 고쳐야 할 저전력 창의 환각이 묻힌다.

고치려는 실패: 핫플레이트가 창 전체 최대 전력이 100W 미만인 창에서 393W 로
예측되던 것 (오탐 277건 중 198건, 전부 '꽂혀 있는' 창). 12.10절 참조.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import torch
import torch.nn.functional as F

# ── 상태별 전력 척도 (12.9.9절) ──────────────────────────────────────────────
# 3.1절의 `s_i` 는 **기기 단위** p90 이다. 그런데 한 기기 안에서 상태가 2자릿수
# 차이 나면(오븐 팬/조명 15W ↔ 히터 1150W) 저전력 상태가 손실에서 사라진다.
#
#   L_power = Huber(P̂/s_i, P/s_i, δ=0.1)
#   오븐 팬/조명을 129% 틀려도 벌점 0.000124  <- 12% 틀린 선풍기(0.006773)의 1/55
#
# 실측(v9 홀드아웃)에서 `s_i/참값` 이 2.3 이하인 상태 15개는 전부 오차 12% 이내였고,
# 9.6 이상인 상태 4개는 전부 66~234% 로 실패했다. 경계가 완벽하게 갈린다.
#
# 아래는 타깃 시점 상태별 p90 (12,000창 측정, 2026-08-22). state_id 0(OFF_STANDBY)은
# 참값이 0 이라 척도가 정의되지 않으므로 기기 척도 `s_i` 를 그대로 쓴다.
#
# **하한 10W 를 둔다.** 프로젝터 예열은 p90 이 4.5W 인데, 계측계 바닥 노이즈가
# 1.4~2.4W(file_registry) 라 그 아래로 척도를 내리면 잡음을 학습시키게 된다.
MIN_STATE_SCALE_W = 10.0

S_STATE: Dict[str, Dict[int, float]] = {
    "air_conditioner": {1: 16.4, 2: 263.6, 3: 549.6, 4: 794.0},
    "beam_projector": {1: MIN_STATE_SCALE_W, 2: 50.6},   # 1 은 p90 4.5 -> 하한
    "electiric_kettle": {1: 1534.5},
    "fan": {1: 23.2, 2: 31.1, 3: 39.7},
    "hair_dryer": {1: 529.2, 2: 1022.5},
    "hotplate": {1: 549.6},
    "laptop_charger": {1: 36.4, 2: 68.0},
    "minipc": {1: 10.7, 2: 25.0},
    "oven": {1: 16.8, 2: 1357.1},
}


def build_state_scales(appliances: Sequence[str], s_i: Sequence[float],
                       max_states: int = 5) -> torch.Tensor:
    """(K, max_states) 상태별 척도. 미정의 상태는 기기 척도로 채운다."""
    out = torch.zeros(len(appliances), max_states, dtype=torch.float32)
    for i, a in enumerate(appliances):
        out[i] = float(s_i[i])                       # 기본값 = 기기 척도
        for sid, w in S_STATE.get(a, {}).items():
            if 0 <= sid < max_states:
                out[i, sid] = max(float(w), MIN_STATE_SCALE_W)
    return out


#: 정답 배분에서도 남는 차수별 잔차의 중앙값 (손실 단위, 2026-09-01 측정).
#: 사람 라벨 5파일의 60초 창 55개에서 `min_{P>=0} ‖y − A_정답·P‖` 의 잔차다.
#: **이것이 순방향 모델의 오차이고, `L_harm` 이 벌하면 안 되는 양이다.**
HARM_DEADZONE_PROFILE = [0.191, 0.843, 0.303, 0.851, 0.320, 0.772, 0.270,
                         0.879, 0.265, 1.138, 0.298, 1.068, 0.378, 1.545, 0.798]


@dataclass
class LossWeights:
    power: float = 1.0
    state: float = 0.3
    # 상태별 전력 출력을 그 상태의 실제 전력에 직접 묶는다 (12.35).
    # **기본 0 — 반증됐다.** 항 자체는 의도대로 동작하지만(충전기 상태별 출력이
    # 참값과 일치) 목적을 달성하지 못한다. 부하 상태는 독립적인 정보원이 아니라
    # 모델이 스스로 추론해야 하는 값이라, 귀속(그 W 가 누구 것인가)에는 안 듣는다.
    # 측정: test_7 전이 8/13 -> 7/13, 유령 42.1W -> 86.9W (12.35.3).
    state_power: float = 0.0
    on: float = 0.3
    plugged: float = 0.1
    standby: float = 0.1
    harm: float = 0.1
    cons: float = 0.0        # 1단계 0. 2단계에서 0.4 로 올린다 (3.3절)
    over: float = 0.1        # 물리 상한 힌지 (아래 참조)


def _huber(pred: torch.Tensor, tgt: torch.Tensor, delta: float) -> torch.Tensor:
    d = pred - tgt
    a = d.abs()
    return torch.where(a <= delta, 0.5 * d * d, delta * (a - 0.5 * delta))


class NILMLoss(torch.nn.Module):
    def __init__(
        self,
        s_i: torch.Tensor,                     # (K,) 기기별 정격 스케일 (W)
        signatures: Optional[torch.Tensor] = None,   # (K, H, 2) 와트당 고조파 페이저
        standby_sig: Optional[torch.Tensor] = None,  # (K, H, 2) 대기 상태 페이저
        noise_sig: Optional[torch.Tensor] = None,    # (H, 2) 계측계 페이저
        harm_scale: Optional[torch.Tensor] = None,   # (H,) 차수별 정규화
        harm_odd_only: bool = False,                # 짝수차를 L_harm 에서 뺀다 (12.75, 실행 기록은 12.78)
        harm_max_order: int = 0,                    # 이 차수 위를 L_harm 에서 뺀다 (12.171.4 의 B)
        weights: Optional[LossWeights] = None,
        power_delta: float = 0.1,
        standby_delta: float = 1.0,
        s_state: Optional[torch.Tensor] = None,      # (K, MAX_STATES) 상태별 척도
        harm_grad_balance: str = "off",             # off | smps | all  (12.120)
        smps_group: Optional[Sequence[int]] = None,  # SMPS 열 인덱스
        harm_deadzone: float = 0.0,                 # L_harm 불감대 배수 (12.122.16)
        harm_weight: str = "off",                   # 차수별 신뢰도 가중 (12.135)
        reactive_qp: Optional[torch.Tensor] = None,  # (K,) 기기별 Q/P (12.133)
        noise_q: float = 0.0,                        # 계측계 무효전력 (VAR)
        power_ref: Optional[torch.Tensor] = None,    # (K,) 참값 전력, 0 = 모름 (12.145)
        sig_real: Optional[torch.Tensor] = None,     # (K,H,2) **실측 갈래 전용** 지문 (12.151.1)
        companion_sig: Optional[torch.Tensor] = None,   # (K,H,2) 동반 부하 페이저 (12.156)
        companion_w: Optional[torch.Tensor] = None,     # (K,) 동반 부하 전력 (W)
        res_ohm: Optional[torch.Tensor] = None,         # (K,) 등가저항 Ω, 0 = 안 건다 (12.156)
        res_ohm_half: Optional[torch.Tensor] = None,    # (K,) 반파 상태의 등가저항 (12.157)
    ):
        super().__init__()
        self.register_buffer("s_i", s_i.clamp(min=1e-3))
        self.use_state_scale = s_state is not None
        self.register_buffer("s_state", (s_state.clamp(min=1e-3) if s_state is not None
                                         else s_i[:, None].clamp(min=1e-3).repeat(1, 5)))
        h = signatures.shape[1] if signatures is not None else 15
        self.register_buffer("sig", signatures if signatures is not None
                             else torch.zeros(len(s_i), h, 2))
        # ── 실측 갈래 전용 지문 (12.151.1) ────────────────────────────────
        # `sig` 를 h1 녹화전압으로 정규화하면 **합성 갈래에는 틀린 값**이 된다 —
        # 캐시의 `obs_harm` 은 원래 지문으로 합성한 것이라 그 전방모형이 원래
        # 지문이다. 정규화한 것을 거기 쓰면 h1 에 5% 계통오차를 새로 넣는다.
        # 그래서 **실측 갈래(`unlabeled`)만** 이 버퍼를 쓴다. 없으면 `sig` 다.
        self.register_buffer("sig_real", sig_real if sig_real is not None
                             else torch.zeros(0))
        self.register_buffer("standby_sig", standby_sig if standby_sig is not None
                             else torch.zeros(len(s_i), h, 2))
        self.register_buffer("noise_sig", noise_sig if noise_sig is not None
                             else torch.zeros(h, 2))
        self.register_buffer("harm_scale", harm_scale if harm_scale is not None
                             else torch.ones(h))
        # ── 전력 사전 (12.145) — 격리 녹화에서 통전 전력이 좁은 기기만 ──────
        # `power_ref.REFERENCE_W` 의 값이고 **사람 라벨이 아니다** (기기를 한 번
        # 따로 녹화한 상수). 모르는 기기는 0 이라 항에서 빠진다.
        self.register_buffer("power_ref", power_ref if power_ref is not None
                             else torch.zeros(len(s_i)))
        # ── 동반 부하 (2026-09-03, 12.156) ────────────────────────────────
        # **오븐만 가진 것을 오븐에게 준다.** 오븐은 `OFF_STANDBY / FAN_LIGHT /
        # HEATING` 세 상태인데 라벨이 FAN_LIGHT 를 `is_on=0, target_power_w=0`
        # 으로 적는다 — 즉 조명·컨벡션 팬의 14.2W 는 **어느 기기 몫도 아니게**
        # 배경으로 흘러간다. 그래서 `standby_profile` 이 OFF_STANDBY 의 0.40W 고,
        # 오븐이 존재하는 시간의 52~73% 를 차지하는 상태가 순방향 모형에 없다.
        #
        # 왜 이것이 배분을 고치는가: 포트와 오븐은 `L_harm` 에서 **전력 크기를
        # 빼면 축퇴**다. 판별 기여 18.27 중 h1 이 17.38(97.6%)이고 모양(h2~h8)은
        # 다 합쳐 0.886(4.9%)뿐이다 (와트당 지문이 h1 1.3%, h5 4.2%, h7 3.9% 차).
        # 그래서 압력이 어디서 오든 이 축으로 미끄러진다 — 12.155.6 의 반파 채널이
        # 포트 1,209W 를 **장소 B 에 없는 오븐**에게 넘긴 것이 그것이다.
        # 동반 부하는 크기가 아니라 **신원**을 요구한다: 오븐이 켜졌다면 어딘가에
        # 64mA/|u3| 0.058 의 SMPS 전류가 같이 있어야 한다. 포트에는 그런 상태가 없다.
        #
        # **`(1−σ(on))` 을 곱하지 않는다.** `standby` 항과 다른 점이 이것이다.
        # 팬·조명은 히터와 **동시에** 돈다 — 격리에서 HEATING 의 |I3| 중 27~58%
        # 가 팬/조명 몫이고, 빼고 남은 잔차의 |u3| 가 0.0010~0.0028 로 순수
        # 니크롬이다 (등가저항 40.6Ω -> 히터만 41.2Ω). 그래서 `σ(plugged)` 로만
        # 건다. 히터가 꺼진 주기에도, 켜진 주기에도 같은 값이 흐른다.
        #
        # 상수의 재현성 (규칙 14): 녹화 3개에서 P 14.48/14.20/14.21W,
        # |I1| 64.49/63.86/64.26mA — **폭/중앙 0.010**. `REFERENCE_W` 의 채택
        # 문턱 0.10 을 열 배 여유로 통과한다.
        self.register_buffer("companion_sig", companion_sig if companion_sig is not None
                             else torch.zeros(len(s_i), h, 2))
        self.register_buffer("companion_w", companion_w if companion_w is not None
                             else torch.zeros(len(s_i)))
        # ── 저항 컨덕턴스 정합 `L_res` (2026-09-03, 12.156) ────────────────
        # `resistive_match`(12.112) 가 후처리에서 푸는 것을 **손실로 옮긴다.**
        # 니크롬선이라 `P = V²/R` 이고 `R` 이 기기 고유값이다 (같은 기기 다른
        # 녹화에서 0.1~1.3%). 포트 35.8Ω 과 오븐 40.6Ω 은 13% 차이라 222V 에서
        # **163W** 벌어진다 — h1 이 못 가르는 그 크기를 정확히 가른다.
        #
        # 12.155.6 이 잃은 창에서 재 보면 16조합 중 최적이 포트를 93/88/93% 로
        # 지목하고 **오븐은 상위 4위 안에 한 번도 안 든다.** 정보는 깨끗한데
        # 모델에게 준 적이 없어서 후처리가 뒤늦게 고치고 있었다 (그것도 절반만 —
        # 포트 F1 0.403 -> 0.767).
        #
        # ⚠ **포트·오븐에만 건다.** 핫플은 장소 B 에서 230~240W 로 도는데 참조가
        #   460W 고(12.155 의 남은 것 [3]), 드라이기는 강 54.3Ω / 약 108.6Ω 로
        #   상태마다 다르다. 그 둘은 모델 예측을 그대로 쓴다 — 안 그러면 이 항이
        #   틀린 값을 강요한다 (규칙 14).
        self.register_buffer("res_ohm", res_ohm if res_ohm is not None
                             else torch.zeros(len(s_i)))
        # ── 상태 의존 저항 (2026-09-03, 12.157) ────────────────────────────
        # 드라이기는 **하나의 R 로 못 적는다** — 강풍은 전파 54.3Ω, 약풍은 반파라
        # 겉보기 108.6Ω 이다 (12.109.2). 그래서 12.156 은 아예 뺐다.
        #
        # **그런데 반파 채널이 바로 그 상태를 알려준다.** 12.156.7 이 실측 계단으로
        # 다시 재서 확인했다 — 반파로 가르면 강 0.997(n=27), 약 0.984(n=7) 로 둘 다
        # 맞고, 약풍 쪽 폭/중앙이 **0.008 로 다섯 기기 중 가장 안정적**이다.
        # 안 가르면 R 이 1.975 로 터진다 (두 상태가 섞여서).
        #
        # 이 버퍼가 0 이 아닌 기기는 창마다 `res_ohm` 과 이 값 중 하나를 쓴다.
        # 관문은 `postproc.HALFWAVE_ABS_MIN` 의 **절대량** 판이다 — 비율은
        # 복합에서 분모가 터져 죽는다 (12.114.2 가 반증한 형태).
        self.register_buffer("res_ohm_half", res_ohm_half if res_ohm_half is not None
                             else torch.zeros(len(s_i)))
        # `L_swap`(12.158) 이 셀 조합. 저항 열의 on/off 전수다 (4종이면 16개).
        nres = int((self.res_ohm > 0).sum()) if res_ohm is not None else 0
        if nres > 0:
            import itertools as _it
            _c = torch.tensor(list(_it.product([0.0, 1.0], repeat=nres)),
                              dtype=torch.float32)
        else:
            _c = torch.zeros(0, 0)
        self.register_buffer("_swap_combos", _c)
        #: 저항 몫이 이보다 작은 창은 안 건드린다 (`resistive_match` 의 `min_w`).
        self.swap_min_w = 150.0
        # ── 무효전력 보존 (12.133) ────────────────────────────────────────
        # 저항은 등가저항 `R = V²/P` 가 기기 고유값이라 `resistive_match` 가 조합을
        # 역산할 수 있는데, SMPS 에는 그런 둘째 판별자가 없어서 배분이 `L_harm`
        # 하나에 걸려 있었다 — 그리고 12.122.2 가 그 항의 최소는 오답 쪽이라고
        # 확정했다. `Q/P` 가 그 자리를 채운다 (SMPS 쌍 d′ 2.31~4.64 vs 고조파 0.91~1.85).
        self.register_buffer("reactive_qp", reactive_qp if reactive_qp is not None
                             else torch.zeros(len(s_i)))
        self.noise_q = float(noise_q)

        # ── L_harm 기울기 균등화 (2026-08-31, 12.120절) ─────────────────────
        # `∂L_harm/∂P̂_i = sign(err)·sig_i/harm_scale` 이라 **기울기 크기가
        # `‖sig_i‖` 에 비례한다.** 그래서 와트당 지문이 작은 기기가 잔여를 흡수하는
        # 가장 싼 자리가 된다:
        #
        #     프로젝터 0.1219/W   충전기 0.1479 (1.21배)   미니PC 0.1697 (1.39배)
        #
        # 관측된 편향의 순서와 정확히 같다 — 프로젝터 재현율 1.000·정밀도 0.47,
        # 예측 전력이 창의 87%에서 상한에 붙어 있고, 충전기는 재현율 0.62 로
        # 놓친다 (12.120.1).
        #
        # 12.87.3 은 이것을 **미결정**이라 했는데 절반만 맞다. 해가 여러 개인 것은
        # 맞지만 모델은 무작위로 고르지 않는다 — **`L_harm` 이 순서를 매긴다.**
        #
        # **값이 아니라 기울기만 고친다.** `p_eff = P·w + (P·(1−w)).detach()` 는
        # 값이 정확히 `P` 라 재구성 `Σ P̂·sig` 가 안 바뀐다 (물리 보존).
        # 바뀌는 것은 최적화가 어느 기기를 움직이기 쉬운가뿐이다.
        #
        # **2단계(`unlabeled`)에만 건다.** 1단계에는 `L_power` 가 기기별로
        # 붙잡아 주므로 비대칭이 상쇄되지만 2단계에는 그 항이 없다 (4.2절).
        sn = (self.sig[:, :, 0] ** 2 + self.sig[:, :, 1] ** 2).sqrt()
        sn = (sn / self.harm_scale.clamp(min=1e-9)[None]).norm(dim=1).clamp(min=1e-9)
        w = torch.ones_like(sn)
        if harm_grad_balance == "all":
            w = sn.mean() / sn
        elif harm_grad_balance == "smps" and smps_group:
            idx = torch.as_tensor(list(smps_group), dtype=torch.long)
            w[idx] = sn[idx].mean() / sn[idx]
        self.register_buffer("harm_gw", w)
        self.harm_grad_balance = str(harm_grad_balance)
        # 짝수차 제외 마스크 (12.75절. **그 절은 계획만 있고 비어 있었다** —
        # 유일한 실행 기록은 12.78 이고 넷을 한꺼번에 바꾼 판이라 단일 변수가
        # 아니다. 12.75.5 가 2단계만으로 다시 잰다).
        #
        # [왜] 12.72 가 짝수차를 **계측 인공물**로 확정했다 — 두 증폭 경로의 DC
        # 오프셋이 개별 보정되지 않아 레인지 전환마다 단차가 생기고, 그 1/h
        # 스펙트럼에 `s_rc_gain[h] ∝ h` 보상이 곱해져 모든 차수에서 평평한 바닥이
        # 된다. 부하의 물리량이 아니고 **같은 기기의 녹화 사이에서 1.3~1.8배씩
        # 흔들린다** (12.72.4).
        #
        # [왜 그냥 두면 안 되나] `harm_scale` 의 2차가 3.56 mA 로 작아서 **정규화
        # 뒤 2차가 가장 큰 항이 된다** (12.70.3). 손실이 가장 큰 가중을 인공물에
        # 걸고 있었다. 12.74 가 **입력** 짝수차를 0 으로 만들어 전이 귀속을
        # 21->24/41 로 올렸는데, 손실의 **타깃**(`obs_harm`)과 **지문**(`sig`)에는
        # 짝수차가 그대로 남아 있다. 여기서 나머지 절반을 막는다.
        #
        # `unlabeled()`(2단계 적응)도 같은 버퍼를 쓴다. 2단계가 실측에서 도는
        # 것이므로 오히려 그쪽이 더 중요하다.
        # ── L_harm 불감대 (2026-09-01, 12.122.16절) ────────────────────────
        # **줄일 수 없는 잔차를 줄이라고 밀면 배분이 밀린다.**
        #
        # 사람 라벨로 정답 배분을 넣고 전력만 자유롭게 풀어도 고조파 잔차의
        # **70% 가 남는다** (12.122.16: 정답 0.206 vs 자유 0.145). 순방향
        # 모델이 그만큼 틀려 있다. 그런데 `L_harm` 은 그 70% 도 줄이라고
        # 밀고, 줄일 방법이 배분뿐이라 **배분이 밀린다.** 그것이 12.120.3 의
        # '가장 싼 기기' 로 몰리는 기제이고, 12.122.12/14 에서 지문을 고칠
        # 때마다 유령이 옮겨 다닌 이유다.
        #
        # 그래서 **순방향 모델 오차만큼은 벌하지 않는다:**
        #
        #     err = relu(|pred − obs|/harm_scale − tau_h)
        #
        # `tau_h` 는 정답 배분에서 남는 차수별 잔차의 중앙값이다 (아래 프로파일).
        # 짝수차가 0.77~1.55 로 큰 것은 12.72 가 계측 인공물로 확정한 그 자리라
        # 아무것도 설명 못 하는 것이 맞다 — 불감대가 자연스럽게 용서한다.
        #
        # ⚠ **너무 키우면 `L_harm` 이 죽는다.** 12.12.2 가 `L_cons` 만 남으면
        # "합만 맞추는 해" 로 무너진다고 쟀다. 배수를 쓸어 보고 정해야 한다.
        # 기본 0 = 끔이라 이전과 글자 그대로 같다.
        dz = torch.as_tensor(HARM_DEADZONE_PROFILE[:h], dtype=torch.float32)
        if len(dz) < h:
            dz = torch.cat([dz, dz.new_full((h - len(dz),), float(dz[-1]))])
        self.register_buffer("harm_dz", dz * float(harm_deadzone))
        self.harm_deadzone = float(harm_deadzone)

        mask = torch.ones(h)
        if harm_odd_only:
            mask[1::2] = 0.0          # 0-based 라 인덱스 1,3,5.. 가 2,4,6..차다
        # ── 고차 절단 (12.171.4 의 B) ──────────────────────────────────────
        # 12.171.3 이 잰 것: 실측 창에서 `L_harm` 값의 **56%가 h11~h15** 이고,
        # 그 차수들의 예측은 관측의 1/4~2/3 다 (h15 21.6 vs 92.4 mA). 기기
        # 배분과 무관한 모델오차인데 `harm_scale` 이 작아(0.019~0.030) 정규화
        # 후에는 가장 큰 항이 된다. 12.135 가 "높은 차수는 신호가 아니라
        # 모델오차" 라고 이미 쟀고 `1/h²` 가중으로 **줄였다** — 여기서는 **끊는다.**
        # 0 이면 끔 (이전과 글자 그대로 같다).
        if harm_max_order and harm_max_order < h:
            mask[int(harm_max_order):] = 0.0
        # ── 차수별 신뢰도 가중 (12.135) ────────────────────────────────────
        # `harm_scale` 은 "판별 정보는 높은 차수에 있다"(0.2절)를 전제로 15차수를
        # **균등화**한다. 그런데 실측에서는 그 전제가 뒤집힌다 — 높은 차수는
        # 신호가 아니라 모델오차다. 차수별 정답잔차 tau 가 h1 0.079 vs h14 1.696 로
        # 21배 갈리고(12.123.1), 차수 부분집합별 모델오차/판별신호가 h1 만 1.08 에서
        # 고차만 3.68 까지 **단조**다 (12.133).
        #
        #     균등(현행) 2.23  ->  1/h 1.74  ->  **1/h² 1.57**  ->  h1,h3 만 1.41
        #
        # 1.41 이 바닥이고 거기서는 유효차원이 4 라 창당 켜진 기기 수와 맞먹는다 —
        # 더 낮추면 식별성이 죽는다 (규칙 31). `1/h²` 이 유효차원 4.6 을 남기면서
        # 비를 1.57 로 내리는 자리이고, **튜닝 상수가 없다** (차수의 역제곱뿐).
        #
        # ⚠ 이것은 결함을 **줄이지 없애지 못한다.** 어떤 가중으로도 최소는 오답
        #   쪽에 남는다 (비가 1 아래로 안 간다).
        if harm_weight != "off":
            hh = torch.arange(1, h + 1, dtype=torch.float32)
            if harm_weight == "inv_h":
                w_h = 1.0 / hh
            elif harm_weight == "inv_h2":
                w_h = 1.0 / (hh * hh)
            elif harm_weight == "inv_tau":
                t = torch.as_tensor(HARM_DEADZONE_PROFILE[:h], dtype=torch.float32)
                if len(t) < h:
                    t = torch.cat([t, t.new_full((h - len(t),), float(t[-1]))])
                w_h = 1.0 / t.clamp(min=1e-6)
            else:
                raise ValueError(f"모르는 harm_weight: {harm_weight}")
            mask = mask * (w_h / w_h.max())
        self.register_buffer("harm_mask", mask)
        # h1 만 1 인 (H,) 마스크. 12.151 의 전압 보정이 h1 에만 걸리는 이유는
        # 항등식 `Re(I1)/P = 1/V1` 이 h1 에서만 성립해서다. 고차는 `V_h/R` 로
        # 예측하면 최대 2배 틀리고 위상이 기기마다 달랐다 (12.151 의 자).
        h1 = torch.zeros_like(mask); h1[0] = 1.0
        self.register_buffer("h1_only", h1)
        self.harm_odd_only = bool(harm_odd_only)
        self.harm_max_order = int(harm_max_order)
        self.harm_weight = str(harm_weight)
        self.w = weights or LossWeights()
        self.power_delta = power_delta
        self.standby_delta = standby_delta

    def forward(self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 12.9.9절 — 정답 상태에 맞는 척도를 쓴다. 손실은 학습 라벨을 봐도 되고,
        # 이렇게 해야 '고전력 기기의 저전력 상태' 가 손실에서 지워지지 않는다.
        if self.use_state_scale and tgt.get("y_state") is not None:
            idx = tgt["y_state"].long().clamp(0, self.s_state.shape[1] - 1)   # (B,K)
            s = torch.gather(self.s_state[None].expand(idx.shape[0], -1, -1),
                             2, idx[..., None]).squeeze(-1)                   # (B,K)
        else:
            s = self.s_i[None]
        parts: Dict[str, torch.Tensor] = {}

        # 3.1절 — 스케일 정규화 전력 회귀. 절대 W 를 쓰면 오븐 60W 와 프로젝터 60W 가
        # 같은 벌점이 되고, 0.7절의 오차 전가 보호막(87배)이 사라진다.
        parts["power"] = _huber(out["power"] / s, tgt["y_power"] / s, self.power_delta).mean()

        parts["on"] = F.binary_cross_entropy_with_logits(out["on_logit"], tgt["y_on"])
        parts["plugged"] = F.binary_cross_entropy_with_logits(out["plugged_logit"], tgt["y_plugged"])
        parts["standby"] = _huber(out["standby"], tgt["y_standby"], self.standby_delta).mean()

        b, k, c = out["state"].shape
        parts["state"] = F.cross_entropy(out["state"].reshape(b * k, c),
                                         tgt["y_state"].reshape(b * k).long())

        # 3.1절의 전력 손실은 **섞인 뒤**의 out["power"] 에만 걸린다. 그래서 상태별
        # 전력 출력이 서로 달라야 할 이유가 손실에 없다. 참 전력비가 큰 기기는
        # 평균 하나로 못 맞히니 어쩔 수 없이 분화하는데, 비가 작은 기기는 붕괴한다.
        # cnn_ph1 측정 (홀드아웃, 상태별 전력 출력의 비 / 참 비):
        #     오븐 71.9 / 75.7      에어컨 35.1 / 31.0     드라이기 2.04 / 1.95
        #     미니PC 1.16 / 1.67    **충전기 1.02 / 1.37**  <- 34.00W vs 33.25W
        # 붕괴하면 p_raw = Σ mix·p_states 가 상태와 무관해지고, 저부하 SMPS 를
        # 가르는 데 쓸 수 있는 유일한 축(부하 상태)이 학습된 모델 안에서 사라진다.
        # 12.35 가 잰 대로 SMPS 3종은 **부하를 고정하면** 갈린다 (분리비 1.1~1.4
        # -> 2.2~4.2). 그 조건을 모델이 쓰게 하려면 이 항이 필요하다.
        if self.w.state_power > 0 and out.get("power_states") is not None:
            ps_all = out["power_states"]
            idx2 = tgt["y_state"].long().clamp(0, ps_all.shape[-1] - 1)
            ps = torch.gather(ps_all, 2, idx2[..., None]).squeeze(-1)      # (B,K)
            # 켜져 있을 때만. 꺼진 창의 상태별 출력은 감독할 대상이 아니다
            # (게이트가 어차피 0 으로 곱한다).
            m = tgt["y_on"] > 0.5
            d = _huber(ps / s, tgt["y_power"] / s, self.power_delta) * m
            parts["state_power"] = d.sum() / m.sum().clamp(min=1.0)
        else:
            parts["state_power"] = out["power"].sum() * 0.0

        if self.w.harm > 0 and tgt.get("obs_harm") is not None:
            # 관측 고조파 = Σ(활성 기기) + Σ(꽂힌 채 꺼진 기기의 대기 전류) + 계측계 전류.
            # 활성 항만 더하면 체계적 오프셋이 남고, 그 오프셋은 저부하 대역에서
            # 상대적으로 크다 (3.4절 경고).
            pred = torch.einsum("bk,khc->bhc", out["power"], self.sig)
            idle = torch.sigmoid(out["plugged_logit"]) * (1.0 - torch.sigmoid(out["on_logit"]))
            pred = pred + torch.einsum("bk,khc->bhc", idle, self.standby_sig)
            pred = pred + self.noise_sig[None]
            # 차수별로 같은 무게를 준다. 정규화하지 않으면 I1 이 전부 지배해
            # 고조파 제약이 전력 제약과 같아진다.
            err = (pred - tgt["obs_harm"]).abs() / self.harm_scale[None, :, None]
            # 마스크를 걸어도 손실 규모가 유지되도록 마스크 평균으로 나눈다.
            # 그래야 `w_harm=0.1` 이 이전과 같은 뜻을 갖는다.
            parts["harm"] = ((err * self.harm_mask[None, :, None]).mean()
                             / self.harm_mask.mean().clamp(min=1e-6))
        else:
            parts["harm"] = out["power"].sum() * 0.0

        if self.w.over > 0:
            recon = out["power"].sum(1) + out["standby"].sum(1) + tgt["p_noise"]
            excess = torch.relu(recon - tgt["p_observed"])
            parts["over"] = (excess / tgt["p_observed"].clamp(min=10.0)).mean()
        else:
            parts["over"] = out["power"].sum() * 0.0

        if self.w.cons > 0:
            recon = out["power"].sum(1) + out["standby"].sum(1) + tgt["p_noise"]
            parts["cons"] = (recon - tgt["p_observed"]).abs().mean()
        else:
            parts["cons"] = out["power"].sum() * 0.0

        total = sum(getattr(self.w, n) * v for n, v in parts.items())
        parts["total"] = total
        return parts

    def unlabeled(self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                  w_cons: float = 0.4, w_harm: float = 0.1,
                  w_over: float = 0.0, w_hedge: float = 0.0,
                  sample_w: Optional[torch.Tensor] = None,
                  w_real_on: float = 0.0,
                  w_consq: float = 0.0,
                  w_pref: float = 0.0,
                  w_res: float = 0.0,
                  w_swap: float = 0.0,
                  swap_tol: float = 0.02,
                  swap_slack: int = 0,
                  swap_tiebreak: str = "off",
                  swap_tb_orders: Sequence[int] = (3, 5, 7),
                  w_impl: float = 0.0,
                  impl_side: str = "both",
                  companion: bool = False) -> Dict[str, torch.Tensor]:
        """**기기별 라벨이 없는 실측 창**용 손실 (4.2절 2단계).

        실측 복합 부하에는 기기별 정답이 없다. 라벨이 필요 없는 항만 쓴다.

            L_adapt(실측) = w_cons · |Σ P̂ + Σ Ŝ + P_noise − P_관측|
                          + w_harm · ‖Σ P̂·sig + 대기 + 계측계 − 관측 고조파‖

        **`L_harm` 을 빼면 안 된다.** `L_cons` 는 합이 얼마나 어긋났는지만 알고
        어느 기기 몫인지는 모른다 (3.3절). 실측에는 기기별 라벨이 아예 없으므로,
        `L_harm` 이 없으면 배분이 미결정인 채 아무 기기에나 붙는다.
        **이 두 항이 2단계의 분해를 통째로 떠받친다** (3.4절, 4.2절).

        `w_cons` 기본 0.4 는 4.2절 값이다. 1단계의 ~0 에서 올린다 — 1단계에서는
        보존 항이 개별 전력 감독보다 35~3,000배 강해 "합만 맞추는 해" 로 빠지지만
        (3.3절, 12.9절 v4/v5 붕괴), 2단계에는 경쟁할 개별 전력 감독이 없다.
        """
        # ── 창별 가중 (12.94절) ─────────────────────────────────────────────
        # `sample_w` 는 창마다 다른 가중이다. 기본(None)은 균등이라 이전과 같다.
        # **왜 필요한가**: 창 하나당 기울기 노름이 고부하 창에서 25배 크다
        # (12.94.1 측정). 그래서 SMPS 만 있는 창이 개수로 58% 인데 기울기로는
        # 10% 미만이다. `|·|` 의 기울기는 부호뿐이라 손실 크기 탓이 아니라
        # **기기별 전력 스케일** 탓이다 — 오븐 헤드는 1100W, 미니PC 는 10W 규모다.
        def wmean(x: torch.Tensor) -> torch.Tensor:
            if sample_w is None:
                return x.mean()
            w = sample_w.to(x.dtype)
            while w.dim() < x.dim():
                w = w.unsqueeze(-1)
            return (x * w).sum() / (w.expand_as(x).sum().clamp(min=1e-6))

        parts: Dict[str, torch.Tensor] = {}
        # 동반 부하가 꽂혀 있으면 그 전력도 관측에 들어 있다 (12.156). `L_cons`
        # 가 세지 않으면 이 항이 낸 14.2W 가 잔차로 남아 다른 기기로 간다.
        comp = (torch.sigmoid(out["plugged_logit"]) if companion
                else torch.zeros_like(out["power"]))
        recon = out["power"].sum(1) + out["standby"].sum(1) + tgt["p_noise"]
        if companion:
            recon = recon + (comp * self.companion_w[None]).sum(1)
        parts["cons"] = wmean((recon - tgt["p_observed"]).abs())

        # ── 무효전력 보존 `L_cons^Q` (12.133) ────────────────────────────────
        #     |Σ qp_i·P̂_i + Σ qp_i·Ŝ_i + Q_noise − Q관측|
        #
        # `L_cons` 와 같은 꼴인데 **P 대신 Q** 를 맞춘다. 왜 이것이 배분을 고치는가:
        # `L_cons` 는 합이 얼마나 어긋났는지만 알고 누구 몫인지는 모른다. 그래서
        # 배분은 `L_harm` 하나가 정했는데, 12.122.2 가 실측에서 그 항의 최소는
        # **오답 쪽에 있다**고 확정했다. `Q/P` 는 SMPS 를 고조파보다 2.2~2.5배 잘
        # 가르므로(12.133) 두 번째 판별자가 된다 — 저항의 `resistive_match` 에 해당한다.
        #
        # ⚠ `qp` 중 검증된 것은 `reactive_signatures` 가 `usable` 로 표시한 것뿐이다
        #   (프로젝터·충전기·미니PC·오븐). 나머지는 중앙값을 그대로 쓴다 —
        #   |Q/P| 가 작아 기여가 적고(저항 ≤0.07), 실측 11파일에서 에어컨·선풍기는
        #   한 번도 안 켜진다. **에어컨이 있는 환경에서는 마스크가 필요하다.**
        if tgt.get("q_observed") is not None:
            recon_q = ((out["power"] * self.reactive_qp[None]).sum(1)
                       + (out["standby"] * self.reactive_qp[None]).sum(1)
                       + self.noise_q)
            parts["consq"] = wmean((recon_q - tgt["q_observed"]).abs())
        else:
            parts["consq"] = out["power"].sum() * 0.0

        if tgt.get("obs_harm") is not None:
            # 12.120 — 값은 그대로, 기울기만 기기별로 균등화한다 (`harm_gw` 주석).
            p_h = out["power"]
            if self.harm_grad_balance != "off":
                gw = self.harm_gw[None]
                p_h = p_h * gw + (p_h * (1.0 - gw)).detach()
            sg = self.sig_real if self.sig_real.numel() else self.sig
            pred = torch.einsum("bk,khc->bhc", p_h, sg)
            # ── h1 지문의 전압 보정 (12.151) ──────────────────────────────────
            # 유효전력의 정의에서 **항등식**이 나온다. P = V1·I1·cos(phi1) 이므로
            #
            #     Re(I1)/P = 1/V1     <-  기기와 무관하다
            #
            # 즉 와트당 h1 전류는 상수가 아니라 **그 창의 전압에 반비례**한다.
            # `sig` 는 상수로 두므로 부하가 커져 전압이 222 -> 209V 로 떨어지면
            # 예측 h1 전류를 6% 적게 낸다. 그 부족분을 모델은 **전력을 더 얹어서**
            # 메우고, 가장 싼 기기(프로젝터)로 간다 (12.87.3 의 기전).
            #
            # `harm_offset`(12.148) 과 다르다. 저쪽은 `Re(Z·I1)` 에 대해 **선형**인
            # 더하기 항이고, 이쪽은 `(dV/V)·pred_h1` 이라 전류에 대해 **2차**다.
            # 규칙 40 — 기존 항이 못 담는 모양임을 먼저 확인했다.
            if tgt.get("vscale") is not None:
                pred = pred * (1.0 + (tgt["vscale"] - 1.0)[:, None, None]
                               * self.h1_only[None, :, None])
            idle = torch.sigmoid(out["plugged_logit"]) * (1.0 - torch.sigmoid(out["on_logit"]))
            pred = pred + torch.einsum("bk,khc->bhc", idle, self.standby_sig)
            # 동반 부하 (12.156). **`(1−σ(on))` 이 없다** — 팬·조명은 히터와
            # 동시에 돈다. 이 항이 오븐의 신원을 요구한다: 켜졌다면 64mA 의
            # SMPS 전류가 같이 있어야 하고, 포트에는 그런 상태가 없다.
            if companion:
                pred = pred + torch.einsum("bk,khc->bhc", comp, self.companion_sig)
            pred = pred + self.noise_sig[None]
            # ── 교차주파수 어드미턴스 보정 (12.148) ────────────────────────────
            # `sig` 는 **fixed current injection** 모형이라 기기 전류가 계통 조건과
            # 무관하다고 본다. 문헌이 그 실패를 오래 전에 적었고(attenuation &
            # diversity), 표준 처방이 Norton 등가 `I_h = I_source,h − Y_h·V_h` 다 —
            # 그리고 **여러 차수의 전압**이 한 차수의 전류에 든다 (교차주파수 결합).
            # 12.148 이 실측에서 적합했고 배분 오차가 파일 홀드아웃에서
            # 39.4 -> 14.3W 로 준다. 창마다 다른 **상수**라 여기서는 더하기만 한다
            # (`realdata.harmonic_offset` 이 전압 고조파에서 미리 계산한다).
            # 짝수차는 0 이다 — 12.72(전류 인공물) + 12.147(전압 짝수차 미결).
            if tgt.get("harm_offset") is not None:
                pred = pred + tgt["harm_offset"]
            err = (pred - tgt["obs_harm"]).abs() / self.harm_scale[None, :, None]
            # 불감대 — 순방향 모델 오차만큼은 벌하지 않는다 (`harm_dz` 주석).
            # **2단계에만 건다.** 1단계는 합성이라 순방향이 정확하다.
            if self.harm_deadzone > 0:
                err = (err - self.harm_dz[None, :, None]).clamp(min=0.0)
            # 마스크를 걸어도 손실 규모가 유지되도록 마스크 평균으로 나눈다.
            # 그래야 `w_harm=0.1` 이 이전과 같은 뜻을 갖는다.
            parts["harm"] = (wmean(err * self.harm_mask[None, :, None])
                             / self.harm_mask.mean().clamp(min=1e-6))
        else:
            parts["harm"] = out["power"].sum() * 0.0

        excess = torch.relu(recon - tgt["p_observed"])
        parts["over"] = wmean(excess / tgt["p_observed"].clamp(min=10.0))

        # ── 헤지 벌점 (12.9.13절) ─────────────────────────────────────────
        # `P̂ = σ(on)·p` 라 게이트가 중간에 머물면 **물리적으로 불가능한 중간 전력**이
        # 나온다. 실측에서 오븐+핫플이 겹칠 때 모델이 정확히 그렇게 한다:
        #
        #   포트  p_raw 1233W (정격) x σ(on) 0.381 -> 469W
        #   핫플  p_raw  397W (정격) x σ(on) 0.004 ->   2W
        #                                      합 471W ~ 참값 468W
        #
        # 크기는 둘 다 맞게 알면서 **어느 쪽인지 결정을 못 해** 헤지하고, 합이 맞으니
        # 보존 손실도 만족된다. 1단계에는 라벨이 있어 BCE 가 확신을 강제하지만
        # (합성 저항3종 1.000), 실측에는 라벨이 없어 이 압력이 없다.
        #
        # 그래서 라벨 없이 확신을 요구하는 항을 둔다 — 이진 엔트로피다.
        # p=0.5 에서 최대, p in {0,1} 에서 0.
        q = torch.sigmoid(out["on_logit"]).clamp(1e-6, 1 - 1e-6)
        parts["hedge"] = wmean(-(q * q.log() + (1 - q) * (1 - q).log()))

        # ── 사람 스위칭 로그 지도 (2026-08-31, SMPS_PLAN 4.5절) ────────────
        # 바로 위 헤지 항의 주석이 *"실측에는 라벨이 없어 이 압력이 없다"* 로
        # 끝난다. **다섯 파일에는 있다** — 스위치를 누른 사람이 그 자리에서 적은
        # on/off 다 (`realdata.HUMAN_ON_DEFAULT_STEMS`, 3,499초).
        #
        # 헤지는 "아무거나 확실하게 정해라" 이고 이 항은 "이것으로 정해라" 다.
        # 둘이 겹치는 구간에서는 이쪽이 방향을 준다.
        #
        # **전력은 감독하지 않는다.** 로그는 on/off 만 주므로 `on_logit` 에만 건다.
        # `P̂ = σ(on)·p_raw` 라 게이트가 맞으면 전력도 따라오지만, 그 경로는
        # `L_cons`/`L_harm` 이 정하게 둔다 — 크기 정답을 지어내지 않는다.
        #
        # 마스크는 (창 x 기기) 단위다. `uncertain` 구간과 지도 범위 밖 열은 0 이라
        # **양쪽 어느 쪽으로도 안 센다** (`build_on_off_truth` 의 규칙 그대로).
        # 마스크 합으로 나누므로 라벨 있는 창의 비율이 바뀌어도 뜻이 안 변한다.
        hm = tgt.get("human_mask")
        if w_real_on > 0 and hm is not None and float(hm.sum()) > 0:
            bce = F.binary_cross_entropy_with_logits(
                out["on_logit"], tgt["human_on"], reduction="none")
            parts["real_on"] = (bce * hm).sum() / hm.sum().clamp(min=1.0)
        else:
            parts["real_on"] = out["power"].sum() * 0.0

        # ── 전력 사전 `L_pref` (12.145) ───────────────────────────────────
        # 12.144.2 가 잰 것: 저항 없는 창에서 모델이 프로젝터를 82.8W 로 낸다
        # (참값 46.9). 그런데 **실측 손실만 보면 최적이 60.0W** 이고, 그마저
        # 참값보다 13.1W 높다. 최소점 자체가 틀린 자리에 있다.
        #
        # 원인은 구조다 — SMPS 3종 지문이 11.9도 안에 몰려 있고(cos 0.979) 그
        # 안에서 프로젝터가 **와트당 전류가 가장 작다**(8.038 mA/W). 그래서
        # ① 같은 전류에 가장 많은 와트가 들고 ② 와트당 벌금이 가장 싸다.
        # 둘 다 프로젝터로 몰아넣는다.
        #
        # 규칙 35 — **기울기를 만져서는 최적점이 안 옮겨진다. 값을 만져야 한다.**
        # `--harm-grad-balance` 가 12.122.8 에서 안 움직인 이유이고, `--sig-insitu`
        # 가 움직인 이유다 (지문 h1 을 12.5% 키워 눈금을 비틀었다).
        #
        # 이 항은 그것을 **비틀지 않고** 한다 — 참값을 아는 기기의 전력에 직접
        # 건다. 12.144.2 의 최적점 훑기가 `w_pref >= 0.02` 면 최적이 47.5W 로
        # 옮겨지고 그 위로는 포화한다고 쟀다 (`1/h²` 에서는 0.002 로도 된다).
        #
        # **게이트를 지렛대로 못 쓰게 한다.** `P̂ = σ(on)·p_raw` 라 `|P̂ − ref|`
        # 에 걸면 게이트를 낮춰 벌금을 피할 수 있고 검출이 죽는다. 그래서
        # `p_raw` 에 걸고 게이트는 **기울기를 끊어** 마스크로만 쓴다.
        if w_pref > 0 and float(self.power_ref.abs().sum()) > 0:
            m = (self.power_ref > 0).to(out["power_raw"].dtype)[None]      # (1,K)
            # ── 창별 참값 마스크 (12.159) ──────────────────────────────────
            # 어떤 기기는 **특정 파일에서만** 참값을 안다. 미니PC 가 그렇다 —
            # `test_14`~`test_18` 은 마우스 고장으로 IDLE 전용이라 9.90W 이고
            # (폭/중앙 0.065), 장소 A 의 `test_13` 에는 ACTIVE(+27.3W 계단)가
            # 섞인다. 전역으로 걸면 그 창에 틀린 값을 강요한다 (규칙 14).
            if tgt.get("pref_mask") is not None:
                m = m * tgt["pref_mask"].to(m.dtype)                        # (B,K)
            g = torch.sigmoid(out["on_logit"]).detach()                    # 마스크로만
            w8 = (out["power_raw"] - self.power_ref[None]).abs() * m * g
            denom = (m.sum(-1).mean() if m.dim() > 1 else m.sum()).clamp(min=1.0)
            parts["pref"] = wmean(w8.sum(1, keepdim=True)) / denom
        else:
            parts["pref"] = out["power"].sum() * 0.0

        # ── 저항 컨덕턴스 정합 `L_res` (2026-09-03, 12.156) ────────────────
        # 니크롬선은 `P = V²/R` 이고 `R` 이 기기 고유값이다. 그래서 **저항 몫을
        # 컨덕턴스로 옮기면 조합을 셀 수 있다** — `resistive_match`(12.112) 가
        # 후처리에서 하는 그 계산이다. 여기서는 그것을 손실로 옮긴다.
        #
        #     G_예측 = Σ_{포트,오븐} σ(on_k)/R_k  +  (핫플·드라이기 예측)/V²
        #     G_관측 = (P_관측 − 비저항 예측 − 대기 − 동반 − 계측) / V²
        #     L_res  = |G_예측 − G_관측| · V²        (와트라 읽을 수 있다)
        #
        # **포트 35.8Ω, 오븐 40.6Ω 은 222V 에서 163W 벌어진다.** `L_harm` 의
        # h1 이 못 가르는 그 크기다. 12.155.6 이 잃은 창에서 재면 16조합 중
        # 최적이 포트를 93/88/93% 로 지목하고 오븐은 상위 4위에 한 번도 없다.
        #
        # 핫플·드라이기는 `res_ohm` 이 0 이라 이 항이 저항을 강요하지 않고
        # **모델 예측을 그대로 컨덕턴스로 환산해 넣는다** (위 ⚠ 주석 참조).
        # 그래서 이 항은 "저항 총량은 맞추되 포트·오븐의 **신원**만 못 박는다".
        #
        # `p_raw` 가 아니라 게이트에 건다 — 겨냥이 크기가 아니라 누구인지다.
        if w_res > 0 and float(self.res_ohm.abs().sum()) > 0 and tgt.get("v_rms") is not None:
            v2 = tgt["v_rms"].clamp(min=1.0) ** 2                          # (B,)
            fixed = (self.res_ohm > 0).to(out["power"].dtype)[None]        # (1,K)
            gcond = torch.reciprocal(self.res_ohm.clamp(min=1e-6))[None] * fixed
            # ── 상태 의존 저항 (12.157) ──────────────────────────────────
            # 드라이기 약풍은 반파라 겉보기 저항이 2배다. 관문을 **관측 고조파**로
            # 건다 — 모델 출력이 아니라 자료에서 오므로 순환이 없다.
            if float(self.res_ohm_half.abs().sum()) > 0 and tgt.get("obs_harm") is not None:
                from src.model.postproc import HALFWAVE_ABS_MIN
                h = tgt["obs_harm"]
                i2 = torch.linalg.vector_norm(h[:, 1], dim=-1)
                i4 = torch.linalg.vector_norm(h[:, 3], dim=-1)
                half = ((i2 - i4) > HALFWAVE_ABS_MIN).to(gcond.dtype)[:, None]  # (B,1)
                alt = (torch.reciprocal(self.res_ohm_half.clamp(min=1e-6))[None]
                       * (self.res_ohm_half > 0).to(gcond.dtype)[None])
                sw = (self.res_ohm_half > 0).to(gcond.dtype)[None]              # (1,K)
                gcond = gcond * (1.0 - sw * half) + alt * (sw * half)
            sg = torch.sigmoid(out["on_logit"])
            p_fixed = (sg * gcond).sum(1) * v2                             # 못 박은 기기
            p_free = (out["power"] * (1.0 - fixed)).sum(1)                 # 나머지
            recon_r = p_fixed + p_free + out["standby"].sum(1) + tgt["p_noise"]
            if companion:
                recon_r = recon_r + (comp * self.companion_w[None]).sum(1)
            parts["res"] = wmean((recon_r - tgt["p_observed"]).abs())
        else:
            parts["res"] = out["power"].sum() * 0.0

        # ── 저항 조합 맞바꿈 `L_swap` (2026-09-03, 12.158) ──────────────────
        # 12.157.4b 가 확정했다: `σ` 를 곱해 거는 항은 게이트가 바닥이면 안 닿는다.
        # 같은 `L_res` 가 포트(σ중앙 0.0344)에서는 Δ +0.563 인데 핫플(0.0033)은
        # +0.009, 드라이기 강풍(0.0001)은 −0.011 이다 — **완전한 단조**이고
        # 유일한 차이가 게이트다. `dσ/dlogit = σ(1−σ)` 가 330배 갈린다.
        #
        # 그래서 `σ` 를 안 거치고 **로짓에 직접** 건다. 무엇을 가르칠지는
        # `resistive_match`(12.112) 가 후처리에서 푸는 그 계산으로 정한다 —
        # 컨덕턴스는 병렬로 더해지므로 조합을 셀 수 있고, **라벨이 필요 없다**
        # (관측 P·V 와 기기 고유 R 만 쓴다).
        #
        # 12.112 의 두 제한을 그대로 가져온다:
        #   ① **개수는 안 바꾸고 맞바꿈만.** 제한 없이 돌리면 정합기가 없는 기기를
        #      발명한다 (test_9 유령 3.94 -> 86.98W). 겨냥인 "드라이기 강풍이
        #      꺼지고 오븐이 켜진 것" 은 개수가 같은 맞바꿈이라 이 제한으로도 닿는다.
        #   ② **tol 밖은 안 건드린다.** 설명 못 하는 창을 억지로 가르치지 않는다.
        #
        # 그리고 하나를 더 건다: **`best == 현재` 인 창은 감독하지 않는다.**
        # 거기서 BCE 는 "지금 결정을 더 확신해라" 인데, 후처리와 달리 손실은
        # 1,000 스텝을 밀므로 틀린 결정이 굳는다 (12.122.2 의 *"최소가 오답 쪽에
        # 있다"*). 맞바꿈이 필요한 창에만 힘을 준다.
        if (w_swap > 0 and float(self.res_ohm.abs().sum()) > 0
                and tgt.get("v_rms") is not None):
            cols = torch.nonzero(self.res_ohm > 0, as_tuple=False).flatten()
            with torch.no_grad():
                v2 = tgt["v_rms"].clamp(min=1.0) ** 2
                fx = (self.res_ohm > 0).to(out["power"].dtype)[None]
                gc = torch.reciprocal(self.res_ohm.clamp(min=1e-6))[None] * fx
                if (float(self.res_ohm_half.abs().sum()) > 0
                        and tgt.get("obs_harm") is not None):
                    from src.model.postproc import HALFWAVE_ABS_MIN
                    h = tgt["obs_harm"]
                    i2 = torch.linalg.vector_norm(h[:, 1], dim=-1)
                    i4 = torch.linalg.vector_norm(h[:, 3], dim=-1)
                    hf = ((i2 - i4) > HALFWAVE_ABS_MIN).to(gc.dtype)[:, None]
                    alt = (torch.reciprocal(self.res_ohm_half.clamp(min=1e-6))[None]
                           * (self.res_ohm_half > 0).to(gc.dtype)[None])
                    sw = (self.res_ohm_half > 0).to(gc.dtype)[None]
                    gc = gc * (1.0 - sw * hf) + alt * (sw * hf)
                gcr = gc.expand(out["power"].shape[0], -1)[:, cols]       # (B,R)
                # 저항 몫 = 관측 − (안 박은 기기 예측 + 대기 + 동반 + 계측)
                p_other = (out["power"] * (1.0 - fx)).sum(1)
                p_res = (tgt["p_observed"] - p_other
                         - out["standby"].sum(1) - tgt["p_noise"])
                if companion:
                    p_res = p_res - (comp * self.companion_w[None]).sum(1)
                g_need = p_res / v2                                       # (B,)
                cb = self._swap_combos.to(gcr.dtype)                      # (C,R)
                cg = cb @ gcr.T                                           # (C,B)
                cur = (out["on_logit"][:, cols] > 0.0).to(cb.dtype)       # (B,R) σ>0.5
                k = cur.sum(1)                                            # (B,)
                # ── 개수 여유 (12.158.2) ──────────────────────────────
                # 12.112 의 "개수는 안 바꾸고 맞바꿈만" 은 **후처리의 규율**이다.
                # 거기서는 겨냥이 맞바꿈이었고, 제한을 풀자 정합기가 없는 기기를
                # 발명했다 (test_9 유령 3.94 -> 86.98W).
                #
                # **여기서는 겨냥이 다르다.** 12.158.1 이 잰 것: 드라이기 강풍
                # 정답 ON 1,255창 중 **44%(548창)는 저항이 하나도 안 켜져 있다.**
                # 개수를 고정하면 빈 조합만 후보라 `best==cur` 이 되어 감독에서
                # 통째로 빠진다. 그 창을 고치려면 개수를 늘려야 한다.
                #
                # 그래서 ±`swap_slack` 을 허용한다. 0 이면 12.112 와 같다.
                # 무한정 풀지 않는 이유는 그 유령 폭주가 실재하기 때문이다 —
                # 손실은 `L_cons`/`L_harm` 과 경쟁하므로 후처리보다 덜하겠지만
                # 안 재 봤다.
                dk = (cb.sum(1)[:, None] - k[None, :]).abs()
                same_k = (dk <= float(swap_slack))                        # (C,B)
                err = (cg - g_need[None]).abs()
                err = err.masked_fill(~same_k, float("inf"))
                # 모든 후보의 상대오차. tol 안에 든 것이 **여럿**일 수 있다.
                rel_all = err * v2[None] / p_res.abs().clamp(min=1.0)[None]  # (C,B)
                feas = rel_all <= swap_tol                                # (C,B)

                # ── 고조파 동점깨기 (12.165.6) ────────────────────────────
                # 컨덕턴스가 같으면 **전력도 같다.** 12.165.5 가 쟀다:
                #   포트 1377W  ↔  드라이기강+핫플 1392W   **15W 차이(1.1%)**
                # 오븐 FAN_LIGHT(14.2W)과 같은 크기다. `L_cons`/`L_res` 처럼
                # 전력만 보는 항은 이 자리에서 **정보가 0** 이고, 컨덕턴스
                # argmin 도 마찬가지다 — 둘 다 tol 안이라 어느 쪽이 이길지가
                # 반올림에 달린다. 실측에서 그것이 포트 오탐으로 나왔다
                # (test_15 정밀도 1.00 -> 0.84).
                #
                # **모양은 갈린다.** 같은 두 조합의 `h3/h1` 이 0.37% vs 2.14% 로
                # 5.8배다. 그래서 tol 안의 후보들 중 **관측 고조파에 가장 가까운
                # 것**을 고른다. h1 은 안 쓴다 — 12.156 이 `L_harm` 판별의
                # 97.6%가 h1 이라고 쟀고, 여기서 축퇴인 축이 바로 그 h1 이다.
                #
                # 남은 축퇴는 2쌍뿐이고 둘 다 이 방식으로 갈린다. 나머지 6쌍은
                # `ch50`(반파)이 이미 `gc` 단계에서 가른다 (12.165.5).
                if (swap_tiebreak in ("h3", "mag") and tgt.get("obs_harm") is not None
                        and float(self.sig.abs().sum()) > 0):
                    od = [h - 1 for h in swap_tb_orders if 1 <= h <= self.sig.shape[1]]
                    sig_r = self.sig[cols][:, od]                          # (R,O,2)
                    # 조합별 저항 예측 고조파: 각 기기를 V²/R 로 켠다
                    pw = gcr * v2[:, None]                                 # (B,R) W
                    hc = torch.einsum("cr,br,rox->cbox", cb, pw, sig_r)    # (C,B,O,2)
                    # 관측에서 **저항이 아닌 것**을 뺀 잔차
                    other = torch.einsum("bk,kox->box",
                                         out["power"] * (1.0 - fx), self.sig[:, od])
                    idle_h = (torch.sigmoid(out["plugged_logit"])
                              * (1.0 - torch.sigmoid(out["on_logit"])))
                    other = other + torch.einsum("bk,kox->box", idle_h,
                                                 self.standby_sig[:, od])
                    other = other + self.noise_sig[od][None]
                    h_res = tgt["obs_harm"][:, od] - other                 # (B,O,2)
                    # ── 크기로 볼 것인가 복소로 볼 것인가 (12.165.7) ──────
                    # `h3`(복소)은 **반증됐다.** `h_res` 에는 `harm_offset`
                    # (12.148 의 Norton 보정, 창마다 다른 복소 오프셋)이 안 빠져
                    # 있어 위상이 돌아가 있다. 실측 test_16 h3 에서:
                    #     |h_res| 0.105  |포트| 0.024  |드+핫| 0.131
                    #     크기로는 드+핫이 맞는데 **복소 거리는 |A−h| 0.098 <
                    #     |B−h| 0.220** 로 포트가 이긴다 (B 의 위상이 거의 반대).
                    # 결과: 장소B 포트 0.935 -> 0.853, 핫플 0.901 -> 0.701.
                    #
                    # `mag` 는 차수별 **크기**만 비교한다 — 공통 위상 회전에
                    # 면역이다. 그리고 차수는 **h3 하나만** 쓰는 것이 맞다:
                    #     h3  |Δ| 포트 0.081 vs 드+핫 0.026   -> 드+핫 (옳다)
                    #     h5           0.121     0.113        -> 약하게 드+핫
                    #     h7           0.082     0.139        -> **포트 (틀리다)**
                    # 12.135 가 높은 차수는 신호가 아니라 모델오차라고 쟀다.
                    sc = self.harm_scale[od].clamp(min=1e-6)
                    if swap_tiebreak == "mag":
                        mc = torch.linalg.vector_norm(hc, dim=-1)          # (C,B,O)
                        mr = torch.linalg.vector_norm(h_res, dim=-1)       # (B,O)
                        hsc = ((mc - mr[None]).abs() / sc[None, None]).mean(2)
                    else:
                        hsc = ((hc - h_res[None]).abs()
                               / sc[None, None, :, None]).mean((2, 3))     # (C,B)
                    # tol 밖 후보는 후보가 아니다. 아무것도 없으면 컨덕턴스로 간다.
                    hsc = hsc.masked_fill(~feas, float("inf"))
                    bi = torch.where(feas.any(0), hsc.argmin(0), err.argmin(0))
                else:
                    bi = err.argmin(0)                                     # (B,)
                best = cb[bi]                                             # (B,R)
                # tol: 상대오차. 저항 몫이 작은 창은 아예 안 건드린다.
                rel = rel_all.gather(0, bi[None]).squeeze(0)
                m = ((rel <= swap_tol) & (p_res > self.swap_min_w)
                     & (best != cur).any(1)).to(out["on_logit"].dtype)     # (B,)
                # 동점이 실제로 몇 번 생기는지 — 항이 하는 일의 크기다
                parts["swap_ties"] = (feas.sum(0).float() * m).sum() / m.sum().clamp(min=1.0)
            bce = F.binary_cross_entropy_with_logits(
                out["on_logit"][:, cols], best, reduction="none")          # (B,R)
            parts["swap"] = (bce.mean(1) * m).sum() / m.sum().clamp(min=1.0)
            parts["swap_frac"] = m.mean().detach()
        else:
            parts["swap"] = out["power"].sum() * 0.0
            parts["swap_frac"] = out["power"].sum().detach() * 0.0
            parts["swap_ties"] = out["power"].sum().detach() * 0.0

        # ── 함의 제약 `on ⊂ plugged` (12.164.9) ────────────────────────────
        # 꽂히지 않은 기기가 켜질 수는 없다. 합성 30만 창에서 `on=1 & plugged=0`
        # 은 9종 전부 **0건**이다 — 라벨이 이미 이 포함관계를 담고 있다.
        #
        # 그런데 두 머리는 서로 독립인 시그모이드이고 2단계에는 라벨이 없어서,
        # **그 모순을 벌하는 항이 하나도 없었다.** 12.164 가 `gt_plugged` 를
        # "동작 세션 중" 으로 재정의하자 몸통에 "오븐이 없다" 는 특징이 생겼는데,
        # 지킬 의무가 없으니 시드 2/3 이 `σ(on)>0.5 & σ(plugged)≈0.05` 로 가서
        # 드라이기 강풍을 오븐+포트로 맞바꿨다 (장소B 유령 1.1 -> 134W).
        #
        # **로짓에 직접 건다.** `σ` 를 곱해 걸면 포화된 게이트에 안 닿는다
        # (규칙 51). `L_swap` 이 같은 이유로 로짓 BCE 를 쓴다.
        # 힌지라 `plugged_logit >= on_logit` 인 창에서는 정확히 0 이다 —
        # 옳게 하고 있는 기기·창은 건드리지 않는다.
        #
        # `impl_side` 가 **어느 쪽이 양보하는가**를 정한다. 제약은 두 가지로
        # 만족될 수 있고, 12.164.10 에서 `both` 는 틀린 쪽을 골랐다 —
        # 장소 B 에서 `on` 을 내리는 대신 `σ(plugged)` 를 0.02 -> 0.96 으로
        # 올려 버렸다 (오븐이 없는 장소인데도). 유령이 129W 로 그대로 남았다.
        # `on` 은 `plugged_logit` 을 detach 해 **`on` 쪽만** 민다.
        if w_impl > 0:
            pl = out["plugged_logit"]
            if impl_side == "on":
                pl = pl.detach()
            parts["impl"] = F.relu(out["on_logit"] - pl).mean()
            parts["impl_frac"] = (
                (out["on_logit"] > out["plugged_logit"]).to(out["power"].dtype)
                .mean().detach())
        else:
            parts["impl"] = out["power"].sum() * 0.0
            parts["impl_frac"] = out["power"].sum().detach() * 0.0

        parts["total"] = (w_cons * parts["cons"] + w_harm * parts["harm"]
                          + w_over * parts["over"] + w_hedge * parts["hedge"]
                          + w_real_on * parts["real_on"]
                          + w_consq * parts["consq"] + w_pref * parts["pref"]
                          + w_res * parts["res"] + w_swap * parts["swap"]
                          + w_impl * parts["impl"])
        return parts
