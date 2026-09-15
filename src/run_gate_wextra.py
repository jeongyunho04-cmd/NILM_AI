# -*- coding: utf-8 -*-
"""`--wide-extra-dilations` 관문 (14.91) — 꺼지면 **비트 동일**, 켜면 **창 전체를 덮는다**.

왜 이 손잡이인가
----------------
14.42 ① 이 **갈래 합치는 자리**의 병을 이미 짚어 뒀다:

```
  깊은 탭의 수용영역 밖에 있는 증거가 머리에 닿는 길은 **전역 평균·최대뿐**이고
  그것은 과거·미래를 못 가른다.
     타깃 이후 전부 지움   오븐 혼합 0.939      수용영역 **안**만 0.126 (효과 없음)
     수용영역 **밖**만     **0.941**           **광역만 덮기 0.203**
```
처방은 둘이었다.
```
  ⓐ 전역 풀링을 쪼갠다      -> `--seg-pool`(14.46) · `--wide-seg-pool`(14.88)
                              14.74 에서 2, 14.89 에서 6 — **둘 다 못 읽었다**
  ⓑ 수용영역을 창 전체로 넓혀 **위치를 아는 탭**으로 받게 한다
                           -> 세밀에는 했다 (`--fine-extra-dilations`, 14.80)
                              **그리고 그것이 유일하게 들었다** (14.90)
                              광역에는 **한 번도 안 했다**  <- 여기
```
`_blk(k=5)` 세 개(d=1,2,4)의 전폭은 `1+4x7 = 29블록`, 타깃에서 **±7초**뿐인데 창은 ±30초다
(**24%**). `(8,16)` 을 더하면 `1+4x31 = 125블록 > 창 120` 이라 창 전체를 덮는다.

  [1] 비우면 **출력 비트 동일** · 파라미터 동일
  [2] 켜면 광역 블록이 **len(extra) 만큼만** 늘고 `trunk_in` 은 **안 변한다** (w2 그대로)
  [3] 켜면 출력이 실제로 달라진다
  [4] ★ **경험적 수용영역이 실제로 넓어진다** — 산술이 아니라 재서 확인한다
      ⚠ GroupNorm 이 시간축 전체를 정규화하므로 '밖은 0' 이 아니라 **평평**하다.
        문턱이 아니라 **바닥 대비 비**로 읽는다 (14.79 에서 이걸 틀렸다).

    python -X utf8 src/run_gate_wextra.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.net import NILMNet, wide_target_index  # noqa: E402
from src.model.inputs import FINE_CYCLES, WIDE_CHANNELS  # noqa: E402

WIDE_LEN = 120
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


def reach(mod, frac=0.25):
    """타깃 블록의 응답이 **바닥의 `frac` 위**로 남아 있는 과거 도달거리 (블록)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, WIDE_CHANNELS, WIDE_LEN, generator=g)
    wt = wide_target_index(WIDE_LEN)
    with torch.no_grad():
        y0 = mod(x)[0, :, wt]
        r = []
        for p in range(wt + 1):
            xp = x.clone()
            xp[0, :, p] += 3.0
            r.append(float((mod(xp)[0, :, wt] - y0).norm()))
    r = np.array(r)
    floor = np.median(r[:max(wt // 3, 1)])          # 먼 과거 = 바닥
    thr = floor + frac * (r.max() - floor)
    above = np.nonzero(r >= thr)[0]
    return wt - int(above.min()), r, floor


def main() -> int:
    print("`--wide-extra-dilations` 관문 (14.91)")
    print("")
    ok = True
    base = build()
    y0 = fwd(base)
    n0 = sum(p.numel() for p in base.parameters())
    ti0 = base.trunk[0].in_features

    for v in (None, (), []):
        m = build(wide_extra_dilations=v)
        same = (torch.equal(fwd(m), y0)
                and sum(p.numel() for p in m.parameters()) == n0)
        ok &= same
        print("  [1] wide_extra_dilations=%-6r -> 비트 동일   %s"
              % (v, "OK" if same else "** 다르다 **"))

    for ex in ((8,), (8, 16)):
        m = build(wide_extra_dilations=ex)
        nb = len(m.wide) == len(base.wide) + len(ex)
        ti = m.trunk[0].in_features == ti0
        ok &= nb and ti
        print("  [2] extra=%-8s -> 광역 블록 %d (기본 %d, +%d) · trunk_in %d (안 변해야) %s"
              % (str(ex), len(m.wide), len(base.wide), len(ex),
                 m.trunk[0].in_features, "OK" if (nb and ti) else "** 틀리다 **"))
        diff = not torch.equal(fwd(m), y0)
        ok &= diff
        print("      [3] 출력이 실제로 달라진다   %s"
              % ("OK" if diff else "** 배선이 안 닿았다 **"))

    print("")
    print("  [4] ★ 경험적 수용영역 (타깃 블록 응답이 바닥의 25% 위로 남는 과거 거리)")
    r0, prof0, f0 = reach(base.wide)
    print("      %-18s **%3d블록 = %4.1f초**  (산술 전폭 29블록 -> 반폭 ±7.0초)"
          % ("지금 (1,2,4)", r0, r0 * 0.5))
    good = True
    for ex, want in (((8,), 15), ((8, 16), 25)):
        m = build(wide_extra_dilations=ex)
        r1, _, _ = reach(m.wide)
        g_ = r1 >= want
        good &= g_
        print("      %-18s **%3d블록 = %4.1f초**  (기대 %d블록 이상)   %s"
              % ("+ %s" % (",".join(map(str, ex))), r1, r1 * 0.5, want,
                 "OK" if g_ else "** 안 넓어졌다 **"))
    ok &= good
    print("      -> 창은 %d블록 = 60초. 지금은 그 **%.0f%%** 만 본다."
          % (WIDE_LEN, 100.0 * r0 / (WIDE_LEN * 0.9)))

    m = build(wide_extra_dilations=(8, 16))
    n2 = sum(p.numel() for p in m.parameters())
    print("")
    print("  파라미터 %s -> %s  (+%s, **+%.1f%%**)"
          % (format(n0, ","), format(n2, ","), format(n2 - n0, ","), 100.0 * (n2 - n0) / n0))
    print("")
    print("관문 전부 통과" if ok else "** 관문 실패 **")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
