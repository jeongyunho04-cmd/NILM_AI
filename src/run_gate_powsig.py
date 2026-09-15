# -*- coding: utf-8 -*-
"""관문 — **전력대 지문 `--pow-sig`** (14.172). 여덟 줄.

13.84.38 이 만든 표가 `run_train_seq --pow-sig`(2단계)에만 달려 있었다. 1단계는 손잡이가
아예 없어서 **한 번도 안 켜졌다.** 뚫기 전에 이것부터 잰다 — 특히 **④**.

```
  [1] 끄면 `_apply_pow_gain` 이 **항등** (비트 동일)
  [2] 표가 실제로 차나 (칸 수 · 경계가 오름차순)
  [3] 보정 크기를 **mA 로** — %로만 보면 고차의 잡음을 크게 본다
  [4] ★ **실측 창이 대역을 정말 쓰나** — 한 대역에만 몰리면 표가 죽는다
  [5] 부드러운 섞기가 **합이 1**이고 경계에서 안 끊긴다
  [6] 진짜 `NILMLoss` 를 지어서 순방향이 실제로 바뀌는지
  [7] ★★ **상태 지문과 이중 계산이 되나** — 대역 경계가 곧 상태 전력이다
  [8] 상태 **안에서** 전력 의존이 남나 (남아야 고친 판이 값어치가 있다)
```

    python -X utf8 -m src.run_gate_powsig
"""
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.losses import S_STATE  # noqa: E402
from src.model.net import (harmonic_signatures, harmonic_signatures_by_power)  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
SH = {"oven": "오븐", "electiric_kettle": "포트", "hotplate": "핫플", "hair_dryer": "드라이",
      "beam_projector": "빔", "laptop_charger": "충전", "minipc": "미니PC",
      "fan": "선풍", "air_conditioner": "에어컨"}


def main() -> int:
    ok = True
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, APPS)
    g, edges, used = harmonic_signatures_by_power(pool, APPS, n_bands=3)

    # [1] 끄면 항등
    from src.model.lossbuild import build_loss
    L0 = build_loss(APPS, "cpu", verbose=False)
    x = torch.randn(4, len(APPS), 15, 2)
    p = torch.rand(4, len(APPS)) * 1000
    same = torch.equal(L0._apply_pow_gain(x, p), x)
    print("[1] `--pow-sig` 끄면 `_apply_pow_gain` 이 **항등**  %s" % ("OK" if same else "FAIL"))
    ok &= same

    # [2] 표가 차나
    mono = bool((np.diff(edges, axis=1) >= 0).all())
    full = int(used.sum())
    print("[2] 표본이 찬 칸 **%d/%d** · 경계가 오름차순 %s  %s"
          % (full, used.size, mono, "OK" if (full >= used.size * 0.8 and mono) else "FAIL"))
    ok &= (full >= used.size * 0.8 and mono)

    # [3] 보정 크기 — mA
    print("\n[3] 보정 크기 (대표 전력에서 차수 합, mA). 저항은 h1(크기)이 대부분이다")
    print("    %-10s %8s %8s %10s" % ("기기", "대표W", "합 mA", "그중 h1"))
    size = {}
    for k, v in enumerate(APPS):
        P = max(S_STATE.get(v, {2: 100.0}).values())
        s = sig[k, :, 0] + 1j * sig[k, :, 1]
        tot = h1 = 0.0
        for h in range(15):
            m = max((abs(s[h] * (g[k, b, h, 0] + 1j * g[k, b, h, 1]) - s[h]) * P * 1000)
                    for b in range(g.shape[1]) if used[k, b])
            tot += m
            if h == 0:
                h1 = m
        size[v] = tot
        print("    %-10s %8.0f %8.1f %10.1f" % (SH.get(v, v), P, tot, h1))

    # [4] ★ 실측 창이 대역을 쓰나
    print("\n[4] ★ **실측에서 각 기기가 어느 대역에 사나** — 한 곳에 몰리면 표가 죽는다")
    from src.run_train_seq import real_windows
    cache = real_windows(APPS, 2.0, "cpu")
    print("    %-10s %10s | %s" % ("기기", "켜진 창", "대역별 비율 (낮음/중간/높음)"))
    dead = []
    for k, v in enumerate(APPS):
        pw = []
        for stem, d in cache.items():
            y = d["y"][:, k].astype(bool)
            if y.any():
                pw.append(np.asarray(d["p_obs"], float)[y])
        if not pw:
            print("    %-10s %10s | (실측에 없음)" % (SH.get(v, v), "-"))
            continue
        pw = np.concatenate(pw)
        e = edges[k]
        b = np.digitize(pw, e)
        frac = np.bincount(b, minlength=3)[:3] / max(len(b), 1)
        tag = ""
        if frac.max() > 0.95 and size[v] > 50:
            tag = "  <- ⚠ **한 대역에 몰린다** (보정 %.0f mA 가 죽는다)" % size[v]
            dead.append(v)
        print("    %-10s %10d | %s%s"
              % (SH.get(v, v), len(pw), " ".join("%5.1f%%" % (f * 100) for f in frac), tag))
    print("    ⚠ 실측 총전력을 대역 경계에 넣은 것이라 **상한**이다 (그 기기 몫이 아니다).")

    # [5] 부드러운 섞기 — 합이 1
    L1 = build_loss(APPS, "cpu", power_signatures=True, power_tau=0.15, verbose=False)
    e = L1.pow_edges
    pp = torch.linspace(0, 2000, 64)[:, None].expand(-1, len(APPS))[..., None]
    u = torch.sigmoid((pp - e[None]) / (L1.pow_tau * e[None].clamp(min=1e-3)))
    one = torch.ones_like(u[..., :1])
    w = torch.cat([one, u], -1) - torch.cat([u, torch.zeros_like(u[..., :1])], -1)
    wsum = float((w.sum(-1) - 1.0).abs().max())
    print("\n[5] 대역 가중의 합이 1 인가 — 최대 이탈 %.3e  %s"
          % (wsum, "OK" if wsum < 1e-5 else "FAIL"))
    ok &= wsum < 1e-5

    # [6] 진짜 손실로 순방향이 바뀌나
    torch.manual_seed(0)
    per_k = torch.randn(8, len(APPS), 15, 2)
    p8 = torch.rand(8, len(APPS)) * 1500
    d0 = L0._apply_pow_gain(per_k, p8)
    d1 = L1._apply_pow_gain(per_k, p8)
    rel = float((d1 - d0).abs().sum() / d0.abs().sum() * 100)
    print("[6] 진짜 `NILMLoss` 에서 순방향이 바뀐다 — |Δ|/|기준| **%.2f%%**  %s"
          % (rel, "OK" if rel > 0.5 else "FAIL (안 걸렸다)"))
    ok &= rel > 0.5

    # [7] ★★ 이중 계산
    from src.model.net import harmonic_signatures_by_state
    gs, us = harmonic_signatures_by_state(pool, APPS)
    print("\n[7] ★★ **상태 지문과 이중 계산** — 손실은 `sig_state x pow_gain` 으로 곱한다")
    print("    %-12s %-4s %10s %10s %8s" % ("기기", "상태", "참 h2/h1", "합성 뒤", "배수"))
    worst = 1.0
    for k, v in enumerate(APPS):
        for st in (1, 2):
            if not us[k, st]:
                continue
            P = S_STATE.get(v, {}).get(st, 0.0)
            bd = int(np.digitize([P], edges[k])[0])
            if not used[k, bd]:
                continue
            cs = gs[k, st, :, 0] + 1j * gs[k, st, :, 1]
            cg = g[k, bd, :, 0] + 1j * g[k, bd, :, 1]
            a_ = abs(cs[1]) / max(abs(cs[0]), 1e-18)
            b_ = abs((cs * cg)[1]) / max(abs((cs * cg)[0]), 1e-18)
            mult = b_ / max(a_, 1e-18)
            if a_ > 1e-5:
                worst = max(worst, max(mult, 1.0 / max(mult, 1e-18)))
            print("    %-12s s%-3d %10.5f %10.5f %8.3f%s"
                  % (SH.get(v, v), st, a_, b_, mult,
                     "  <- ⚠ **이중 계산**" if (a_ > 1e-5 and abs(mult - 1) > 0.15) else ""))
    print("    최악 배수 **x%.1f** — 1 에서 멀수록 같은 물리를 두 번 곱한 것" % worst)
    print("    ⇒ 그래서 `run_train_cnn` 이 `--pow-sig` + `--state-signatures` 를 **막는다**")

    if dead:
        print("\n⚠ **한 대역에만 사는 기기**: %s — 그 기기에서는 이 표가 일을 안 한다"
              % " · ".join(SH.get(v, v) for v in dead))
    print("\n%s" % ("전부 통과" if ok else "**실패한 줄이 있다**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
