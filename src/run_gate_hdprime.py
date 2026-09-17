# -*- coding: utf-8 -*-
"""**`--harm-dprime` 관문** (14.411).

주장: *"`L_harm` 의 분모가 차수별 **신호 크기**(`harm_scale`)라서 못 가르는 차수에 제일
큰 무게가 걸려 있다. 분모를 **판별력**으로 다시 나누면 기울기가 판별 차수로 옮겨간다."*

**손실 값이 아니라 기울기로 잰다** ([[gate-what-the-consumer-reads]]).

⚠⚠ 이 파일의 첫 판은 다른 처방(`--harm-resid-scale`, 잔차로 백색화)을 검사했고
**그 처방을 반증하고 죽였다**. 남긴 기록:

```
  den = harm_scale · profile^w   ->  짝수차 기울기 몫 90.2% -> **90.7%** (늘었다)
                                     h2 하나가 0.104 -> **0.204**
  까닭: `profile` 은 이미 `harm_scale` 로 나눈 **상대** 단위다. h2 의 상대 바닥이
        표에서 제일 작아(0.376) z-점수가 h2 에 h1 의 **80배** 를 준다.
        조용한 차수에 상을 주는데 h2 는 조용한 동시에 쓸모없다 (12.72 계측 인공물).
  그리고 같은 판에서 *"불감대는 기울기를 못 다시 나눈다"* 는 내 주장도 반증됐다 —
  문턱 아래를 **통째로 끄는** 것도 몫을 바꾼다 (총이동 0.316 대 0.281).
```

    python -X utf8 -m src.run_gate_hdprime
"""
from typing import List
import inspect
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.losses import (HARM_DEADZONE_PROFILE as PROF,  # noqa: E402
                              HARM_ORDER_DPRIME as DP)
from src.model.lossbuild import build_loss  # noqa: E402

APPS = ("air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven")
#: §41.2 가 잰 SMPS 판별자 — 미니PC↔충전기 사이각을 하나씩 빼서 순위를 냈다
#: (h11 +2.54° · h9 +2.51° · h13 +1.47° · h7 +1.05°). 540~660Hz 가 핵심이다.
DISCRIM = (3, 5, 7, 9, 11)
OK: List[bool] = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-52s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def mk(**kw):
    return build_loss(list(APPS), "cpu", harm_even_magnitude=True,
                      state_signatures=True, standby_operating="session", **kw)


def order_grad(loss):
    """차수별 **기울기 몫**.

    ⚠ 시험 분포를 **실제 수렴 지점**에 놓는다 — `|Δ|/harm_scale ≈ profile` 이다
    (도달 harm 0.767 대 바닥 0.732). `1.5·harm_scale` 로 뿌리면 불감대가 대부분을
    물어 자가 엉뚱한 자리에 선다 ([[the-gate-must-build-the-real-object]]).
    """
    torch.manual_seed(0)
    H = loss.harm_scale.numel()
    pf = torch.as_tensor(PROF[:H], dtype=torch.float32)
    obs = torch.randn(64, H, 2) * loss.harm_scale[None, :, None]
    pred = (obs + torch.randn(64, H, 2) * (loss.harm_scale * pf)[None, :, None]).detach()
    pred.requires_grad_(True)
    err = loss._harm_err(pred, obs)
    if loss.harm_deadzone > 0:
        err = (err - loss.harm_dz[None, :, None]).clamp(min=0.0)
    ((err * loss.harm_mask[None, :, None]).mean()).backward()
    g = pred.grad.abs().sum(dim=(0, 2)).numpy()
    return g / max(g.sum(), 1e-12)


def main() -> int:
    l0, l5, l1 = mk(), mk(harm_dprime=0.5), mk(harm_dprime=1.0)

    chk(1, "★ w=0 이면 분모가 `harm_scale` 과 **비트 동일**",
        torch.equal(l0.harm_den, l0.harm_scale),
        "지수 0 이라 `(d_ref/d')^0 == 1.0` 이 정확하다 — 옛 판과 계속 비교된다")

    d = np.asarray(DP[:l1.harm_scale.numel()], float)
    ref = float(np.exp(np.mean(np.log(d))))
    r = (l1.harm_den / l1.harm_scale).numpy()
    #: ⚠ 규모 보존 상수 `c` 가 곱해져 있으므로 **같음**이 아니라 **비례 + 정규화**를 건다.
    #:   느슨해진 것이 아니라 검사가 하나 늘었다 ([[dont-loosen-a-gate-to-make-it-pass]]).
    want = ref / d
    prop = np.allclose(r / r.mean(), want / want.mean(), atol=1e-5)
    norm = abs(float((l1.harm_scale / l1.harm_den).mean()) - 1.0) < 1e-5
    chk(2, "w=1 분모비가 `d_ref/d'_h` 에 **비례**하고 규모가 보존되나", prop and norm,
        "비례 %s · `mean(harm_scale/den)` = **%.6f** (1 이어야 한다) · c=%.4f · 분모비 %s"
        % (prop, float((l1.harm_scale / l1.harm_den).mean()),
           float(r[0] / want[0]),
           " ".join("h%d %.2f" % (h + 1, r[h]) for h in (0, 2, 4, 10, 13))))

    g0, g5, g1 = order_grad(l0), order_grad(l5), order_grad(l1)
    ev = np.array([(h + 1) % 2 == 0 for h in range(len(g0))])
    di = np.array([(h + 1) in DISCRIM for h in range(len(g0))])

    # [3] ★★ **짝수차 쏠림이 실제로 줄어드나** — 이 손잡이의 존재 이유
    chk(3, "★★ 짝수차 기울기 몫이 **줄어드나**", g1[ev].sum() < g0[ev].sum() - 0.01,
        "짝수 몫 **%.1f%% -> %.1f%%** (w=0.5 에서 %.1f%%) · h2 하나 %.3f -> **%.3f**"
        % (100*g0[ev].sum(), 100*g1[ev].sum(), 100*g5[ev].sum(), g0[1], g1[1]))

    # [4] ★★ **판별 차수로 옮겨가나** (§41.2: 540~660Hz 가 SMPS 를 가른다)
    chk(4, "★★ 판별 차수(h3~h11 홀수) 몫이 **늘어나나**",
        g1[di].sum() > g0[di].sum() + 0.01,
        "h3·h5·h7·h9·h11 합 **%.1f%% -> %.1f%%** (w=0.5 에서 %.1f%%) · h1 %.3f -> %.3f "
        "(h1 은 총전력이라 `L_power`/`L_cons` 가 맡는다)"
        % (100*g0[di].sum(), 100*g1[di].sum(), 100*g5[di].sum(), g0[0], g1[0]))

    # [5] 단조인가 — 용량을 올리면 같은 방향으로 더 간다
    chk(5, "용량이 **단조**인가 (0 -> 0.5 -> 1)",
        g0[ev].sum() > g5[ev].sum() > g1[ev].sum()
        and g0[di].sum() < g5[di].sum() < g1[di].sum(),
        "짝수 %.3f > %.3f > %.3f · 판별 %.3f < %.3f < %.3f — 쓸기가 뜻을 갖는다"
        % (g0[ev].sum(), g5[ev].sum(), g1[ev].sum(),
           g0[di].sum(), g5[di].sum(), g1[di].sum()))

    # [6] 배선이 **양쪽 학습기**에 닿나 (14.171 의 규율)
    import src.run_adapt as A
    import src.run_train_cnn as T
    sg = inspect.signature(build_loss).parameters
    s1 = "harm_dprime=a.harm_dprime" in inspect.getsource(T)
    s2 = "harm_dprime=a.harm_dprime" in inspect.getsource(A)
    ck = '"harm_dprime": float(a.harm_dprime)' in inspect.getsource(T)
    chk(6, "⚠ 1단계·2단계·체크포인트 셋 다 닿나",
        "harm_dprime" in sg and s1 and s2 and ck,
        "`build_loss` %s · `run_train_cnn` %s · `run_adapt` %s · 체크포인트 키 %s "
        "— 2단계가 다른 분모로 미세조정하면 **두 단계가 다른 순방향**이다"
        % ("harm_dprime" in sg, s1, s2, ck))

    # [7] 상수가 **지금 계측기** 것인가
    chk(7, "⚠ 표 둘 다 14.411 재측정판인가",
        abs(PROF[0] - 0.191) > 0.1 and len(DP) == 15
        and np.median(d[~ev[:15]]) > 2 * np.median(d[ev[:15]]),
        "profile h1 **%.3f** (옛 표 0.191 = 삭제된 옛 계측기 자료) · "
        "d' 홀수 중앙 **%.3f** 대 짝수 **%.3f** (%.1f배) — 짝수차가 못 가른다는 "
        "13.45(SMPS 짝수 위상 뭉침 R 0.12~0.43 = 난수)와 같은 말이다"
        % (PROF[0], np.median(d[~ev[:15]]), np.median(d[ev[:15]]),
           np.median(d[~ev[:15]]) / max(np.median(d[ev[:15]]), 1e-9)))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
