# -*- coding: utf-8 -*-
"""세밀 갈래 **수용영역** 관문 (14.78).

왜
--
창은 `타깃 앞 3.98초 | 타깃 | 뒤 6.00초` 인데 깊은 탭의 수용영역이
`1 + 6*(1+2+4+8+16) = 187` 사이클 = **+-1.56초** 뿐이다. 그래서 창의 44%
(타깃 뒤 1.56~6.00초)는 **위치를 모르는** `h.mean`/`h.amax` 로만 머리에 닿는다.
`mean`/`amax` 는 순서 불변이라 *"계단이 미래에 있다"* 를 표현할 수 없다.

14.47 의 반사실이 이 통로를 지목한다 — 미래 6초를 **다** 지우면 mix 0.137 -> 0.939 인데
**수용영역 밖만** 지워도 **0.941** 이다. 해로운 것은 '미래' 가 아니라 '수용영역 밖' 이다.

⚠ **산수로 재지 않는다.** 입력의 한 위치를 흔들어 깊은 타깃 슬라이스가 **실제로**
움직이는지 본다 ([[the-gate-must-build-the-real-object]]).

```
(1) 끄면 비트 동일 — 파라미터 수·출력이 옛 경로와 같다
(2) 기본 dilation 의 실측 수용영역이 +-1.56초다 (산수와 맞나)
(3) 기본에서 타깃 **+4초**를 흔들면 깊은 탭이 **안 움직인다** (음성 대조 — 이것이 병이다)
(4) 1,3,9,27,81 이면 **움직인다** (+-6.06초로 창을 덮는다)
(5) 파라미터가 **안 는다** (블록 추가와 다른 점)
```

    python -X utf8 src/run_gate_finerf.py
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.model.inputs import FINE_CHANNELS, WIDE_CHANNELS, fine_target_index  # noqa: E402
from src.model.net import NILMNet  # noqa: E402

OK, NG = "OK", "NG"
FAIL = []
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]


def ck(name, ok, note=""):
    print("  [" + (OK if ok else NG) + "] " + name + (("   " + note) if note else ""))
    if not ok:
        FAIL.append(name)


def _net(dil):
    torch.manual_seed(0)
    n = NILMNet(APPS, [3] * len(APPS), fine_dilations=dil)
    n.eval()
    return n


def _deep_tap(n, x):
    """깊은 층의 **타깃 슬라이스** `h[:, :, t]`. forward 의 그 값과 같은 경로다."""
    h = x
    for blk in n.fine:
        h = blk(h)
    return h[:, :, fine_target_index()]


def _reach(n, offsets):
    """각 오프셋(사이클)에서 입력을 흔들었을 때 깊은 타깃 슬라이스가 움직이는가."""
    torch.manual_seed(1)
    x = torch.randn(1, FINE_CHANNELS, 600)
    t = fine_target_index()
    with torch.no_grad():
        base = _deep_tap(n, x).clone()
        out = []
        for o in offsets:
            j = t + o
            if not (0 <= j < 600):
                out.append(None)
                continue
            y = x.clone()
            y[0, :, j] += 10.0
            d = float((_deep_tap(n, y) - base).abs().max())
            out.append(d)
    return out


def main() -> int:
    print("세밀 갈래 **수용영역** 관문 (14.78)")
    print()
    t = fine_target_index()
    print("  창 600사이클 · 타깃 %d -> 앞 %.2f초 | 뒤 %.2f초"
          % (t, t / 60.0, (599 - t) / 60.0))
    print()

    a_net = _net(None)
    b_net = _net((1, 3, 9, 27, 81))
    pa = sum(x.numel() for x in a_net.parameters())
    pb = sum(x.numel() for x in b_net.parameters())
    ck("(1) 기본이 (1,2,4,8,16) 이다", a_net.fine_dilations == (1, 2, 4, 8, 16),
       str(a_net.fine_dilations))
    ck("(5) 파라미터가 **안 는다** (블록 추가면 +37%%였다)", pa == pb,
       "%d 대 %d" % (pa, pb))

    # 같은 씨앗으로 지었으니 가중치가 같아야 하고, dilation 만 다르다
    sa = {k: v for k, v in a_net.state_dict().items()}
    sb = {k: v for k, v in b_net.state_dict().items()}
    same = all(torch.equal(sa[k], sb[k]) for k in sa if k in sb)
    ck("(1) 두 판의 **가중치가 같다** (dilation 만 다르다 — 변수가 하나다)", same)

    offs = [-300, -200, -120, -90, -60, -30, 0, 30, 60, 90, 120, 180, 240, 300, 359]
    ra, rb = _reach(a_net, offs), _reach(b_net, offs)
    print()
    print("  깊은 타깃 슬라이스가 그 오프셋의 입력에 반응하는가 (|변화| 최대)")
    print("  %9s %10s %14s %14s" % ("오프셋", "초", "기본", "1,3,9,27,81"))
    for o, x, y in zip(offs, ra, rb):
        if x is None:
            continue
        print("  %+9d %10.2f %14s %14s"
              % (o, o / 60.0, ("%.3e" % x) if x > 1e-9 else "**0**",
                 ("%.3e" % y) if y > 1e-9 else "**0**"))

    # ── ⚠ 산수가 아니라 **위치를 구별하는가**를 본다 ────────────────────
    #   `_blk` 은 `Conv1d -> GroupNorm -> GELU` 인데 `GroupNorm` 은 **시간축 전체**로
    #   정규화한다. 그래서 창 어디를 건드려도 평균·분산을 통해 모든 위치에 번진다 —
    #   수용영역 밖이 **끊긴 게 아니라** 위치 불변 스칼라 둘로만 닿는다.
    #   (처음에 "밖은 0 이어야 한다" 로 짰다가 이 관문이 나를 잡았다.)
    #   ⇒ 진짜 물음: **미래 +2초의 계단과 +4초의 계단을 구별하는가.**
    def _step_resp(n, o):
        torch.manual_seed(1)
        x = torch.randn(1, FINE_CHANNELS, 600)
        y = x.clone()
        y[0, :, t + o:] += 5.0                      # 그 지점부터의 **계단**
        with torch.no_grad():
            return (_deep_tap(n, y) - _deep_tap(n, x)).squeeze(0)

    print()
    print("  미래 계단의 **위치**를 구별하는가 (깊은 타깃 슬라이스의 응답)")
    print("  %-16s %12s %12s %12s" % ("", "+2초 대 +4초", "각 응답 크기", "구별율"))
    for nm, n in (("기본 1,2,4,8,16", a_net), ("1,3,9,27,81", b_net)):
        r2, r4 = _step_resp(n, 120), _step_resp(n, 240)
        diff = float((r2 - r4).norm())
        mag = float(0.5 * (r2.norm() + r4.norm()))
        print("  %-16s %12.4f %12.4f %11.1f%%" % (nm, diff, mag, 100.0 * diff / max(mag, 1e-9)))
        if nm.startswith("기본"):
            d0, m0 = diff, mag
        else:
            d1, m1 = diff, mag
    ck("(2) ⚠ 기본은 미래 계단의 **위치를 거의 구별 못 한다**", 100.0 * d0 / max(m0, 1e-9) < 40,
       "구별율 %.1f%%" % (100.0 * d0 / max(m0, 1e-9)))
    ck("(3) 1,3,9,27,81 은 **훨씬 잘 구별한다**", d1 / max(d0, 1e-9) > 1.5,
       "구별율 %.1f%% (기본의 %.1f배)"
       % (100.0 * d1 / max(m1, 1e-9), (d1 / max(m1, 1e-9)) / max(d0 / max(m0, 1e-9), 1e-9)))
    ck("(4) 기본의 먼 쪽 응답은 **오프셋에 평평하다** (전역 통계의 지문)",
       float(np.std([ra[offs.index(o)] for o in (120, 180, 240, 300)])
             / np.mean([ra[offs.index(o)] for o in (120, 180, 240, 300)])) < 0.15,
       "변동계수 %.3f" % float(np.std([ra[offs.index(o)] for o in (120, 180, 240, 300)])
                            / np.mean([ra[offs.index(o)] for o in (120, 180, 240, 300)])))

    # ── (5) `fine_pool` (14.79) ─────────────────────────────────────────
    #   `h.mean` 은 수용영역이 창을 덮으면 **중복**이다 — 위치를 아는 conv 가 어떤
    #   가중평균이든 만든다. 남기면 위치 불변 지름길만 준다. `h.amax` 는 max 라
    #   conv 가 표현 못 하고 듀티 기기에 필요해서 **남긴다**.
    f_ = torch.randn(2, FINE_CHANNELS, 600)
    w_ = torch.randn(2, WIDE_CHANNELS, 120)
    torch.manual_seed(0)
    p0 = NILMNet(APPS, [3] * len(APPS)); p0.eval()
    torch.manual_seed(0)
    p1 = NILMNet(APPS, [3] * len(APPS), fine_pool="both"); p1.eval()
    with torch.no_grad():
        o0, o1 = p0(f_, w_), p1(f_, w_)
    ck("(5) `fine_pool=both` 가 기본과 **비트 동일**",
       all(torch.equal(o0[k], o1[k]) for k in o0 if torch.is_tensor(o0[k])))
    torch.manual_seed(0)
    p2 = NILMNet(APPS, [3] * len(APPS), fine_pool="amax"); p2.eval()
    with torch.no_grad():
        o2 = p2(f_, w_)
    ck("(5) `amax` 는 `h.mean` 만 뺀다 (trunk 입력이 c2*ns 만큼 준다)",
       p0.trunk[0].in_features - p2.trunk[0].in_features == 128,
       "%d -> %d" % (p0.trunk[0].in_features, p2.trunk[0].in_features))
    ck("(5) 마스크 길이가 trunk 입력과 맞는다 (갈래 드롭아웃이 안 어긋난다)",
       int(p2.fine_dim_mask.numel()) == p2.trunk[0].in_features)
    ck("(5) 원시 `fp.amax/amin` 은 그대로 남는다 (12.9.8)",
       "fp[:, _a:_b].amax(-1)" in Path("src/model/net.py").read_text(encoding="utf-8"))

    print()
    if FAIL:
        print("관문 실패 %d건: %s" % (len(FAIL), " / ".join(FAIL)))
        return 1
    print("관문 전부 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
