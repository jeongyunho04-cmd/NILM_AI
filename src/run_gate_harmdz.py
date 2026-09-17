# -*- coding: utf-8 -*-
"""`--harm-deadzone` 을 **1단계에** 뚫은 것의 관문 (14.408).

⚠⚠ 이 손잡이의 병은 "값이 틀렸다" 가 아니라 **"배선이 아예 안 닿아 있었다"** 다.
버퍼(`harm_dz`)는 `__init__` 에서 채워지는데 그것을 쓰는 줄이 `unlabeled()`(2단계)
**한 곳뿐**이었다. 그래서 1단계는 배수를 아무리 줘도 **비트 동일**이었다.
주석이 이유를 적어 뒀다 — *"1단계는 합성이라 순방향이 정확하다"*. **그 전제가 틀렸다**:
§51 이 합성 풀에서 직접 쟀고 **참 배분에서도** 충전기 h11 **0.592** 가 남는다.

```
  [1] 배수 0 이면 **비트 동일**
  [2] ★★ 1단계 `forward` 가 실제로 **깎나** (여기가 안 걸려 있었다)
  [3] 2단계 `unlabeled` 도 여전히 깎나 (되돌리지 않았다)
  [4] 프로파일이 **정답 배분 잔차**와 같은 규모인가 (0.661 대 도달 harm 0.767)
  [5] ⚠ 1.0 은 **너무 세다** — 배수별로 얼마나 깎이는지 찍는다 (12.12.2)
  [6] 체크포인트에 적히나 · `build_loss` 서명이 받나 (1·2단계 동기)
```

    python -X utf8 -m src.run_gate_harmdz
"""
from typing import List
import inspect

import torch

from src import env_guard  # noqa: F401

from src.model.losses import HARM_DEADZONE_PROFILE as PROF  # noqa: E402
from src.model.losses import NILMLoss  # noqa: E402

K, H, B, S = 9, 15, 24, 5
OK: List[bool] = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-50s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def mk(dz):
    torch.manual_seed(0)
    sg = torch.randn(K, H, 2) * 0.004
    return NILMLoss(torch.ones(K), signatures=sg, harm_deadzone=dz)


def batch():
    torch.manual_seed(1)
    out = {"power": torch.rand(B, K) * 40, "power_raw": torch.rand(B, K) * 40 + 1,
           "on_logit": torch.zeros(B, K), "plugged_logit": torch.zeros(B, K),
           "standby": torch.rand(B, K) * 0.1, "state": torch.zeros(B, K, S)}
    tgt = {"obs_harm": torch.randn(B, H, 2) * 0.3, "y_on": torch.rand(B, K).round(),
           "y_plugged": torch.rand(B, K).round(), "y_standby": torch.rand(B, K) * 0.1,
           "y_power": torch.rand(B, K) * 40,
           "y_state": torch.zeros(B, K, dtype=torch.long),
           "p_noise": torch.rand(B) * 2, "p_observed": torch.rand(B) * 200 + 50}
    return out, tgt


def main() -> int:
    out, tgt = batch()
    v = {d: float(mk(d).forward(out, tgt)["harm"]) for d in (0.0, 0.3, 0.6, 1.0, 1.5)}

    chk(1, "배수 0 이면 **비트 동일**인가",
        float(mk(0.0).forward(out, tgt)["harm"]) == v[0.0],
        "두 번 지어 같은 값 %.6f — 0 은 분기라 곱이 아니다" % v[0.0])

    chk(2, "★★ **1단계 `forward` 가 실제로 깎나** (여기가 안 걸려 있었다)",
        v[0.3] < v[0.0] * 0.95,
        "배수 0 **%.4f** -> 0.3 **%.4f** (%+.1f%%) · 0.6 %.4f (%+.1f%%)"
        % (v[0.0], v[0.3], 100 * (v[0.3] / v[0.0] - 1),
           v[0.6], 100 * (v[0.6] / v[0.0] - 1)))

    #: 2단계도 그대로인가 — 되돌리지 않았다
    src = inspect.getsource(NILMLoss.unlabeled)
    chk(3, "2단계 `unlabeled` 도 여전히 깎나 (안 되돌렸다)",
        "self.harm_dz[None, :, None]" in src,
        "`unlabeled` 안에 불감대 줄이 그대로 있다 — 1단계에 **더한** 것이지 옮긴 게 아니다")

    m = sum(PROF[:15]) / 15.0
    chk(4, "프로파일이 **정답 배분 잔차** 규모인가", 0.3 < m < 1.2,
        "전 차수 평균 **%.3f** · 홀수 %.3f · 짝수 %.3f — 학습이 도달하는 harm 이 "
        "**0.767** 이니 **%.0f%%가 원리상 못 줄이는 몫**이다 (12.122.16)"
        % (m, sum(PROF[0:15:2]) / 8, sum(PROF[1:15:2]) / 7, 100 * m / 0.767))

    chk(5, "⚠ 배수가 **너무 세지 않나** (12.12.2: L_harm 이 죽으면 무너진다)",
        v[1.0] < v[0.6] < v[0.3] < v[0.0] and v[0.3] > v[0.0] * 0.2,
        "0.3 %+.0f%% · 0.6 %+.0f%% · 1.0 %+.0f%% · 1.5 %+.0f%%  ⇒ **0.3·0.6 을 쓴다** "
        "(1.0 이면 L_harm 이 거의 사라진다 — 14.411 의 새 프로파일에서 더 세졌다)"
        % tuple(100 * (v[d] / v[0.0] - 1) for d in (0.3, 0.6, 1.0, 1.5)))

    from src.model.lossbuild import build_loss
    from src import run_train_cnn as RT
    ok6 = ("harm_deadzone" in inspect.signature(build_loss).parameters
           and "harm_deadzone" in inspect.getsource(RT)
           and '"harm_deadzone": float(a.harm_deadzone)' in inspect.getsource(RT))
    chk(6, "체크포인트에 적히고 `build_loss` 서명이 받나 (1·2단계 동기)", ok6,
        "`build_loss(harm_deadzone=)` %s · 체크포인트 키 %s — 2단계가 이걸 읽어야 "
        "같은 물리로 미세조정한다 (14.171)"
        % ("harm_deadzone" in inspect.signature(build_loss).parameters,
           '"harm_deadzone": float(a.harm_deadzone)' in inspect.getsource(RT)))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
