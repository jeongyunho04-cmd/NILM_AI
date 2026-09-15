# -*- coding: utf-8 -*-
"""`--wide-seg-pool` 관문 (14.88) — 꺼지면 **비트 동일**, 켜지면 **광역만** 바뀐다.

왜 이 관문인가
--------------
14.78 이 이 자리에서 실패했다 — 수용영역을 넓히려고 dilation 을 **교체**했더니 국소 탭이
사라져 저항 신원이 0.9815 -> 0.9321 로 무너졌다. 파라미터가 같다고 공짜가 아니었다
([[dont-replace-a-feature-that-had-another-job]]). 그래서 새 손잡이는 **더하기**여야 하고,
그것을 **주장이 아니라 관문으로** 못 박는다.

  [1] `wide_seg_pool=0` 이면 옛 경로와 **출력이 비트 동일**이다 (같은 씨앗·같은 입력)
  [2] `wide_seg_pool=1` 도 비트 동일 (구간 하나 = 창 전체)
  [3] 켜면 `trunk_in` 이 **정확히 `(w2+2) x (N-1)`** 만 는다 — 세밀 쪽은 한 칸도 안 는다
      (구간마다 광역 `hw` 평균 `w2` 개 **그리고 원시 광역 전력 통계 2개**. 관문을 처음
       만들 때 `w2 x (N-1)` 로 잡았다가 6·10·14 씩 어긋나서 알았다 — 허용오차가 아니라
       **구현에서** 나와야 한다, [[dont-loosen-a-gate-to-make-it-pass]])
  [4] 켜도 **세밀 유래 차원(`fine_flags==1`)의 개수가 그대로**다 — 광역만 늘었다는 증거
  [5] 켜면 출력이 **실제로 달라진다** (안 달라지면 배선이 안 닿은 것이다)
  [6] `seg_pool` 과 **독립**이다 — `seg_pool=2, wide_seg_pool=0` 은 옛 `seg_pool=2` 와 같다

    python -X utf8 src/run_gate_wsegpool.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.net import NILMNet  # noqa: E402
from src.model.inputs import FINE_CYCLES, WIDE_CHANNELS  # noqa: E402

WIDE_LEN = 120          # 광역 블록 수 (60초 @ 2Hz)

APPS = ("air_conditioner", "beam_projector", "electiric_kettle", "fan",
        "hair_dryer", "hotplate", "laptop_charger", "minipc", "oven")
STATES = (4, 2, 1, 3, 2, 2, 2, 2, 3)


def build(seed=0, **kw):
    torch.manual_seed(seed)
    m = NILMNet(list(APPS), list(STATES), width=1.0, **kw)
    m.eval()
    return m


def fwd(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(3, m.fine_channels, FINE_CYCLES, generator=g)
    w = torch.randn(3, WIDE_CHANNELS, WIDE_LEN, generator=g)
    with torch.no_grad():
        o = m(f, w)
    return torch.cat([o["on_logit"].reshape(-1), o["power"].reshape(-1),
                      o["state"].reshape(-1)])


def trunk_in(m):
    return m.trunk[0].in_features


def nfine(m):
    """세밀 유래 차원의 개수. 버퍼 이름은 `fine_dim_mask` 다 (`fine_flags` 가 아니다 —
    처음에 그렇게 썼다가 `hasattr` 이 False 로 빠져 **검사가 −1 대 −1 로 헛돌았다**)."""
    return int(m.fine_dim_mask.sum())


def main() -> int:
    print("`--wide-seg-pool` 관문 (14.88)")
    print("")
    ok = True

    base = build()
    y0 = fwd(base)
    w2 = base.wide[-1][0].out_channels
    print("  광역 마지막 채널 w2 = %d · trunk_in(기본) = %d" % (w2, trunk_in(base)))

    for n in (0, 1):
        m = build(wide_seg_pool=n)
        same = torch.equal(fwd(m), y0) and trunk_in(m) == trunk_in(base)
        ok &= same
        print("  [%d] wide_seg_pool=%d -> 비트 동일   %s"
              % (1 if n == 0 else 2, n, "OK" if same else "** 다르다 **"))

    for n in (4, 6, 8):
        m = build(wide_seg_pool=n)
        want = trunk_in(base) + (w2 + 2) * (n - 1)
        got = trunk_in(m)
        good = got == want
        ok &= good
        print("  [3] wide_seg_pool=%d -> trunk_in %d (기대 %d = 기본 + (w2+2)x%d)   %s"
              % (n, got, want, n - 1, "OK" if good else "** 틀리다 **"))
        fsame = nfine(m) == nfine(base)
        ok &= fsame
        print("      [4] 세밀 유래 차원 %d (기본 %d)   %s"
              % (nfine(m), nfine(base), "OK — 광역만 늘었다" if fsame else "** 세밀도 변했다 **"))
        diff = not torch.equal(fwd(m), y0)
        ok &= diff
        print("      [5] 출력이 실제로 달라진다   %s"
              % ("OK" if diff else "** 안 달라진다 — 배선이 안 닿았다 **"))

    a = build(seg_pool=2)
    b = build(seg_pool=2, wide_seg_pool=0)
    same = torch.equal(fwd(a), fwd(b)) and trunk_in(a) == trunk_in(b)
    ok &= same
    print("  [6] seg_pool=2 에서 wide_seg_pool=0 은 옛 경로와 같다   %s"
          % ("OK" if same else "** 다르다 **"))

    c = build(seg_pool=2, wide_seg_pool=8)
    okc = trunk_in(c) == trunk_in(a) + (w2 + 2) * 6 and nfine(c) == nfine(a)
    ok &= okc
    print("      seg_pool=2 + wide_seg_pool=8 -> trunk_in %d (기대 %d) · 세밀 %d (기대 %d)  %s"
          % (trunk_in(c), trunk_in(a) + (w2 + 2) * 6, nfine(c), nfine(a),
             "OK" if okc else "** 틀리다 **"))

    print("")
    print("관문 전부 통과" if ok else "** 관문 실패 **")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
