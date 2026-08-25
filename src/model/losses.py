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
        harm_odd_only: bool = False,                # 짝수차를 L_harm 에서 뺀다 (12.75)
        weights: Optional[LossWeights] = None,
        power_delta: float = 0.1,
        standby_delta: float = 1.0,
        s_state: Optional[torch.Tensor] = None,      # (K, MAX_STATES) 상태별 척도
    ):
        super().__init__()
        self.register_buffer("s_i", s_i.clamp(min=1e-3))
        self.use_state_scale = s_state is not None
        self.register_buffer("s_state", (s_state.clamp(min=1e-3) if s_state is not None
                                         else s_i[:, None].clamp(min=1e-3).repeat(1, 5)))
        h = signatures.shape[1] if signatures is not None else 15
        self.register_buffer("sig", signatures if signatures is not None
                             else torch.zeros(len(s_i), h, 2))
        self.register_buffer("standby_sig", standby_sig if standby_sig is not None
                             else torch.zeros(len(s_i), h, 2))
        self.register_buffer("noise_sig", noise_sig if noise_sig is not None
                             else torch.zeros(h, 2))
        self.register_buffer("harm_scale", harm_scale if harm_scale is not None
                             else torch.ones(h))
        # 짝수차 제외 마스크 (12.75절).
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
        mask = torch.ones(h)
        if harm_odd_only:
            mask[1::2] = 0.0          # 0-based 라 인덱스 1,3,5.. 가 2,4,6..차다
        self.register_buffer("harm_mask", mask)
        self.harm_odd_only = bool(harm_odd_only)
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
                  sample_w: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
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
        recon = out["power"].sum(1) + out["standby"].sum(1) + tgt["p_noise"]
        parts["cons"] = wmean((recon - tgt["p_observed"]).abs())

        if tgt.get("obs_harm") is not None:
            pred = torch.einsum("bk,khc->bhc", out["power"], self.sig)
            idle = torch.sigmoid(out["plugged_logit"]) * (1.0 - torch.sigmoid(out["on_logit"]))
            pred = pred + torch.einsum("bk,khc->bhc", idle, self.standby_sig)
            pred = pred + self.noise_sig[None]
            err = (pred - tgt["obs_harm"]).abs() / self.harm_scale[None, :, None]
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

        parts["total"] = (w_cons * parts["cons"] + w_harm * parts["harm"]
                          + w_over * parts["over"] + w_hedge * parts["hedge"])
        return parts
