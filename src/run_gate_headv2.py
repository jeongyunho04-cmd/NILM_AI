# -*- coding: utf-8 -*-
"""관문 — 머리 배치 **v2** 가 규칙대로 지어졌나 (14.130).

사용자: *"지금 너무 많은 요소가 있어서 너도 나도 모델 구조를 완벽히 파악을 못하고
있잖아. 기존 구조 백업해 두고 한번 깔끔하게 처음부터 해 보자."*

**규칙 하나로 머리를 설명한다 — "모든 요약은 타깃 기준 2.0초 격자로 낸다."**

```
  v1 (765칸 · 아홉 가지)                 v2 (1475칸 · 다섯 가지)
  원시타깃 57 · 탭0 64 · 탭1 64          [1] 타깃 순간  원시 57 · 얕은탭 64 · 깊은탭 128
  탭4 128                                [2] 세밀 5토막 얕은탭(RF 19) mean+amax  = 640
  h.mean 128   <- **순서 불변**          [3] 깊은 요약  과거 mean+amax · 미래 mean+amax = 512
  h.amax 128   <- **순서 불변**          [4] 광역 전역 평균 64  <- 14.94 로 닫힌 축 (예외)
  깊은탭 128                             [5] 원시 5토막 P 채널 max/min = 10
  hw.mean 64   <- **순서 불변**
  원시창통계 4 <- **순서 불변**
```
v1 의 **순서 불변 덩이 셋이 격자로** 바뀐다. 남는 전역은 [3](깊은 과거/미래)과
[4](광역)뿐이다 — 깊은 쪽은 수용영역 **763** 이라 토막내도 구별이 안 되고, 광역은
14.94 가 닫은 축이라 용량을 주면 해만 갈라진다.

★★ **핵심은 풀링이 아니라 정규화다.** `_block` 이 `Conv1d -> GroupNorm(C, T 전체)` 라
수용영역과 무관하게 통계가 창 전체에서 나온다. 그래서 **풀링만 토막내면 안 갈린다** —
첫 판에서 한 토막을 흔들자 다섯 토막이 다 움직였다 (대각 1.435 대 비대각 0.830,
**1.73배**뿐). `--seg-pool`(14.72)·`--fine-future-segs`(14.122) 가 떨어진 진짜 이유이고,
`--fine-time-split` 만 통한 이유이기도 하다 (몸통을 따로 태워 GroupNorm 도 따로 돈다).
⇒ v2 는 **얕은 스택을 토막마다 따로 태운다.** 재검: 대각 0.253 대 비대각 **0.0000**.

얕은 탭은 `tap_layers` 의 두 번째(층 1, **RF 19**)다. 14.127 의 선형 탐침에서 파일 간
일반화가 가장 나았다 — 최악 파일 AUC 원시 0.346 · 깊은 hf 0.496 · **탭1 0.613**.

무엇을 확인하나
---------------
```
  [1] v1 이 기본이고 **비트 동일**
  [2] v2 차원이 셈과 정확히 맞는다 (249 + 640 + 512 + 256 + 10 = 1667)
  [3] ★ **격자 경계가 타깃과 맞는다** — 600 = 5x120 이고 240 이 경계다. K=4·6 은 아니다
  [4] ★ **세밀 토막이 실제로 구별된다** — 5x5 행렬의 대각이 서야 한다
  [5] 광역은 **토막내지 않는다** (14.94 가 닫은 축 — 규칙의 명시적 예외)
  [6] `fine_dim_mask` 가 맞고 광역 유래만 0 이다
  [7] `head_drop` · `fine_future_segs` 와 같이 쓰면 **거부**
  [8] autocast(bfloat16) · 체크포인트 왕복
```

    python -X utf8 src/run_gate_headv2.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.inputs import FINE_CYCLES, TARGET_LOOKAHEAD  # noqa: E402
from src.model.net import NILMNet, appliance_state_counts, wide_target_index  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
#: ⚠ 14.138 — `prior_kappa` 를 **여기서 켠다**. `NILMNet` 기본은 0.0 인데
#:   트레이너 기본은 **8.0** 이라, 안 켜면 관문이 물리 프라이어 블록을
#:   **한 번도 안 태운다**. 984924 가 그래서 통과하고 학습에서 48초에 죽었다.
BASE = dict(fine_extra_dilations=(32, 64), tap_layers=(0, 1, 4), prior_kappa=8.0)
OK, NG = "OK", "** 실패 **"
FAIL = []


def ck(name, good, note=""):
    print("  %s %s%s" % (OK if good else NG, name, ("   " + note) if note else ""))
    if not good:
        FAIL.append(name)


def _net(seed=0, **kw):
    torch.manual_seed(seed)
    m = NILMNet(APPS, appliance_state_counts(APPS), **dict(BASE, **kw))
    m.eval()
    return m


def head_in(m, f, w):
    box = []
    h = m.trunk[0].register_forward_pre_hook(lambda mod, inp: box.append(inp[0].clone()))
    with torch.no_grad():
        m(f, w)
    h.remove()
    return box[0]


def main() -> int:
    print("머리 배치 **v2** — 모든 요약을 타깃 기준 2.0초 격자로 (14.130)\n")
    f, w = torch.randn(4, 57, 600), torch.randn(4, 47, 120)

    v1, v1b = _net(0), _net(0, head_layout="v1")
    with torch.no_grad():
        o0, o1 = v1(f, w), v1b(f, w)
    ck("[1] v1 이 기본이고 **비트 동일**",
       all(torch.equal(o0[k], o1[k]) for k in o0 if torch.is_tensor(o0[k])),
       "머리 입력 %d" % v1.trunk[0].in_features)

    m = _net(0, head_layout="v2")
    CSH, C2, W2, F = m.h2_csh, v1.trunk[0].in_features and 128, 64, m.h2_fseg
    want = (57 + CSH + C2) + 2 * F * CSH + 4 * C2 + W2 + 2 * F
    ck("[2] v2 차원이 셈과 맞는다", m.trunk[0].in_features == want,
       "%d + %d + %d + %d + %d = **%d** (얕은탭 %dch · 깊은 %dch)"
       % (57 + CSH + C2, 2 * F * CSH, 4 * C2, W2, 2 * F,
          m.trunk[0].in_features, CSH, C2))

    t = FINE_CYCLES - 1 - TARGET_LOOKAHEAD
    grid = [FINE_CYCLES * k // F for k in range(F)] + [FINE_CYCLES]
    ck("[3] ★ 격자 경계가 **타깃과 맞는다**", (t + 1) in grid and F == 5,
       "600 = %dx%d · 경계 %s · 타깃+1=%d ∈ 경계 · (K=4 는 150 · K=6 은 100 이라 안 맞는다)"
       % (F, FINE_CYCLES // F, grid, t + 1))

    # ── [4] ★ 토막이 실제로 구별되나 ────────────────────────────────────────
    OFF = 57 + CSH + C2                       # [2] 세밀 토막 시작
    fa, fb = f.clone(), f.clone()
    torch.manual_seed(7)
    fa[:, :, 0:120] += 5.0 * torch.randn_like(fa[:, :, 0:120])      # 토막 0
    fb[:, :, 480:600] += 5.0 * torch.randn_like(fb[:, :, 480:600])  # 토막 4
    base = head_in(m, f, w)

    def seg_d(fx):
        d = (head_in(m, fx, w) - base).abs().mean(0)
        return [float(d[OFF + 2 * CSH * k: OFF + 2 * CSH * (k + 1)].mean()) for k in range(F)]

    #: ★ 토막 하나씩 흔들어 **5x5 행렬**을 만든다. 대각이 서야 진짜로 갈린 것이다.
    #: 첫 판(풀링만 토막)은 대각 1.435 대 비대각 0.830 = **1.73배** 뿐이었다 —
    #: GroupNorm 이 시간축 전체를 묶고 있어서다. 따로 태우면 비대각이 0 이어야 한다.
    R = []
    for k in range(F):
        a, b = 120 * k, 120 * (k + 1)
        fx = f.clone()
        torch.manual_seed(7 + k)
        fx[:, :, a:b] += 5.0 * torch.randn_like(fx[:, :, a:b])
        R.append(seg_d(fx))
    R = np.asarray(R)
    diag, offd = float(np.diag(R).mean()), float(R[~np.eye(F, dtype=bool)].mean())
    hit = int(sum(int(np.argmax(R[k])) == k for k in range(F)))
    ck("[4] ★ 토막을 **따로 태워서** 진짜로 갈린다 (5x5 대각)",
       hit == F and diag > 3.0 * offd,
       "argmax %d/%d · 대각 **%.3f** 대 비대각 **%.4f** = **%.0f배** "
       "(첫 판은 1.73배였다)" % (hit, F, diag, offd, diag / max(offd, 1e-9)))

    ck("[5] 광역은 **토막내지 않는다** (14.94 가 닫은 축)", m.h2_wseg == 1,
       "전역 평균 %d칸 — 광역을 통째로 지워도 저항 넷 게이트는 0.0~0.2%% 만 바뀐다" % W2)

    ck("[6] `fine_dim_mask` 가 맞고 **광역 유래만 0**",
       int(m.fine_dim_mask.numel()) == m.trunk[0].in_features
       and int((m.fine_dim_mask == 0).sum()) == W2,
       "%d칸 · 0 인 칸 %d (= 광역 %d)"
       % (int(m.fine_dim_mask.numel()), int((m.fine_dim_mask == 0).sum()), W2))

    bad = 0
    for kw, why in ((dict(head_drop="rawstat"), "head_drop"),
                    (dict(fine_time_split=True, fine_future_segs=3), "fine_future_segs")):
        try:
            _net(0, head_layout="v2", **kw)
        except ValueError:
            bad += 1
    ck("[7] `head_drop`·`fine_future_segs` 와 같이 쓰면 **거부**", bad == 2, "%d/2" % bad)

    err = ""
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16), torch.no_grad():
            v = float(m(f, w)["on_logit"].float().abs().max())
        g = bool(np.isfinite(v))
    except Exception as e:                               # noqa: BLE001
        g, err = False, "%s: %s" % (type(e).__name__, e)
    ck("[8a] autocast(bfloat16) 에서 돈다", g, err or "OK")

    import tempfile
    from src.run_gate_check import load_model
    p = Path("results/cnn_pcap_s0.pt")
    if not p.exists():
        ck("[8b] 체크포인트 왕복", False, "%s 가 없다" % p)
    else:
        with tempfile.TemporaryDirectory() as td:
            q = Path(td) / "t.pt"
            c = torch.load(str(p), map_location="cpu", weights_only=False)
            c["head_layout"] = "v1"
            torch.save(c, q)
            mm = load_model(str(q), "cpu")[0]
            ck("[8b] 체크포인트 왕복 — v1 이면 옛 가중치가 그대로 실린다",
               mm.head_layout == "v1" and mm.trunk[0].in_features == 765,
               "head_layout=%r · 머리 %d" % (mm.head_layout, mm.trunk[0].in_features))

    print("")
    print("  ⚠ 파라미터는 **늘어난다** (888k -> 1,070k). v2 는 '작게' 가 아니라")
    print("    **'한 규칙으로'** 가 목표다 — 아홉 가지가 다섯 가지가 되고 순서 불변")
    print("    덩이 넷이 격자로 바뀐다. 성능이 나아진다는 예측은 **하지 않는다**.")
    print("")
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
