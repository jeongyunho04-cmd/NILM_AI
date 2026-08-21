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
from typing import Dict, Optional
import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    power: float = 1.0
    state: float = 0.3
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
        weights: Optional[LossWeights] = None,
        power_delta: float = 0.1,
        standby_delta: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("s_i", s_i.clamp(min=1e-3))
        h = signatures.shape[1] if signatures is not None else 15
        self.register_buffer("sig", signatures if signatures is not None
                             else torch.zeros(len(s_i), h, 2))
        self.register_buffer("standby_sig", standby_sig if standby_sig is not None
                             else torch.zeros(len(s_i), h, 2))
        self.register_buffer("noise_sig", noise_sig if noise_sig is not None
                             else torch.zeros(h, 2))
        self.register_buffer("harm_scale", harm_scale if harm_scale is not None
                             else torch.ones(h))
        self.w = weights or LossWeights()
        self.power_delta = power_delta
        self.standby_delta = standby_delta

    def forward(self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
            parts["harm"] = ((pred - tgt["obs_harm"]).abs()
                             / self.harm_scale[None, :, None]).mean()
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
