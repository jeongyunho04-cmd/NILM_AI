# -*- coding: utf-8 -*-
"""**기기별 차수비례 위상 지터** `--phase-jitter-map measured` 의 관문 (14.389).

사용자 지적에서 나왔다: *"근데 캐시에도 이게 적용되는지도 재야하는거 아니야?"*.
재 보니 구멍이 있었다:
```
  14.386 이 잰 물리   SMPS 의 정류 도통 창이 부하로 움직여 `k = a·ln(P)` 로 돈다
                      미니PC a=3.71 R² 0.730 · 충전기 a=9.21 R² 0.755
  캐시               `s(p)` 가 복소라 그 법칙을 **제대로 옮긴다** (기울기 93~99% 보존)
  ⚠⚠ 그런데         `DataAugmentor.phase_jitter_max_deg = 4.0` 이 `exp(1j·h·θ)`,
                      θ~U(−4,4) 를 **무작위로** 더한다 — **같은 함수 모양**이다
                      ⇒ 캐시에서 잰 법칙 R² 가 0.726 -> **0.308** 로 무너진다
                      ⇒ `traincache.py` 는 이 인자를 안 넘겨 **기본값이 걸려 있었다**
```
고침은 끄는 것이 아니라 **크기를 잰 잔차로 맞추는 것**이다 (법칙은 조각에 이미 있다 —
더하면 이중계산이다).

    python -X utf8 -m src.run_gate_phaselaw
"""
from typing import List
import inspect

import numpy as np

from src import env_guard  # noqa: F401

from src.synthesis.augmentor import (DataAugmentor,  # noqa: E402
                                     PHASE_JITTER_DEG_MEASURED)
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

HS = np.array([3, 5, 7, 9, 11, 13], float)
SMPS = ("minipc", "laptop_charger", "beam_projector")
QUIET = ("electiric_kettle", "hair_dryer", "hotplate", "oven", "fan", "air_conditioner")
#: 14.388 이 캐시에서 잰 값 — 관문이 이 수를 되찾아야 한다
WANT_R2 = {"minipc": (0.308, 0.672), "laptop_charger": (0.518, 0.635)}
OK: List[bool] = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-50s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def k_of(c):
    ph = np.unwrap(np.array([np.angle(np.exp(1j * np.angle(c[:, int(h) - 1])).mean())
                             for h in HS]))
    return float(np.degrees((HS @ ph) / (HS @ HS)))


def law_r2(pool, dev, jit):
    """증강을 걸고 `k = a·ln P` 를 적합해 (a, R²) 를 돌려준다."""
    acts = pool.appliance_activations.get(dev, [])
    aug = DataAugmentor(phase_jitter_std_map=("measured" if jit == "m" else None),
                        phase_jitter_max_deg=(4.0 if jit != "off" else 0.0))
    xs, ys = [], []
    for rep in range(40):
        a = acts[rep % len(acts)]
        np.random.seed(1000 + rep)
        g = aug.augment_activation(a)
        c = np.asarray(g.net_harmonics_complex)
        p = np.asarray(g.target_power_w, float)
        m = p > 2.0
        if m.sum() < 300:
            continue
        c, p = c[m], p[m]
        qs = np.percentile(p, np.linspace(0, 100, 5))
        for i in range(4):
            s = (p >= qs[i]) & (p <= qs[i + 1])
            if s.sum() < 100:
                continue
            xs.append(np.log(np.median(p[s])))
            ys.append(k_of(c[s]))
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 8:
        return float("nan"), float("nan")
    A = np.polyfit(xs, ys, 1)
    r = ys - np.polyval(A, xs)
    return float(A[0]), float(1 - (r ** 2).sum() / max(((ys - ys.mean()) ** 2).sum(), 1e-12))


def main() -> int:
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    print("기기별 위상 지터 관문 (14.389) — 표 %s\n" % PHASE_JITTER_DEG_MEASURED)

    # [1] 표가 **SMPS 만** 담고 있나 (저항은 잡음이라 넣으면 안 된다)
    bad = [a for a in PHASE_JITTER_DEG_MEASURED if a not in SMPS]
    chk(1, "⚠⚠ 표가 **SMPS 셋만** 담나 (저항은 넣으면 안 된다)", not bad,
        "표 = %s · 표에 없는 기기 %d종은 옛 일괄값을 쓴다 — 저항의 h3 이상은 크기가 "
        "거의 0 이라 잰 σ(오븐 4.96 · 핫플 4.66)가 **신호가 아니라 잡음**이다"
        % (list(PHASE_JITTER_DEG_MEASURED), len(QUIET)))

    # [2] 끄면 **비트 동일**
    a0 = DataAugmentor()
    a1 = DataAugmentor(phase_jitter_std_map=None)
    same = True
    for dev in ("minipc", "oven"):
        acts = pool.appliance_activations.get(dev, [])
        if not acts:
            continue
        for rep in range(3):
            np.random.seed(7 + rep)
            g0 = a0.augment_activation(acts[rep % len(acts)])
            np.random.seed(7 + rep)
            g1 = a1.augment_activation(acts[rep % len(acts)])
            same &= np.array_equal(g0.net_harmonics_ri, g1.net_harmonics_ri)
    chk(2, "★ 안 주면 **비트 동일**인가", same,
        "기본 map = %s (비어 있어야 옛 경로) · 미니PC·오븐 3판씩 바이트 비교" % a0.phase_jitter_std_map)

    # [3] 켜면 **SMPS 만** 움직이나
    am = DataAugmentor(phase_jitter_std_map="measured")
    moved, still = [], []
    for dev in list(SMPS) + list(QUIET):
        acts = pool.appliance_activations.get(dev, [])
        if not acts:
            continue
        d = 0.0
        for rep in range(3):
            np.random.seed(11 + rep)
            g0 = a0.augment_activation(acts[rep % len(acts)])
            np.random.seed(11 + rep)
            gm = am.augment_activation(acts[rep % len(acts)])
            d = max(d, float(np.abs(g0.net_harmonics_ri - gm.net_harmonics_ri).max()))
        (moved if dev in SMPS else still).append((dev, d))
    chk(3, "★ 켜면 **SMPS 만** 달라지나", all(x[1] == 0.0 for x in still),
        "SMPS %s · 조용한 무리 %s"
        % ([(a[:7], "%.2e" % b) for a, b in moved], [(a[:7], "%.1e" % b) for a, b in still]))

    # [4] ★★ 캐시에서 **법칙이 되살아나나** (14.388 이 잰 그 수)
    rows, ok4 = [], True
    for dev, (was, want) in WANT_R2.items():
        _a, r_now = law_r2(pool, dev, "cur")
        _a2, r_new = law_r2(pool, dev, "m")
        rows.append("%s %.3f -> **%.3f** (14.388 예측 %.3f -> %.3f)"
                    % (dev[:7], r_now, r_new, was, want))
        ok4 &= (r_new > r_now + 0.05) and abs(r_new - want) < 0.12
    chk(4, "★★ 캐시에서 **법칙 R² 가 되살아나나**", ok4,
        " · ".join(rows) + "  — 예측과 0.12 안에 들어야 한다 "
        "([[dont-loosen-a-gate-to-make-it-pass]])")

    # [5] ⚠ 배선이 **끝까지** 닿나 — `initargs` 가 위치 인자다
    from src.model import traincache as TC
    si = list(inspect.signature(TC._init).parameters)
    sb = list(inspect.signature(TC.build_cache).parameters)
    src = inspect.getsource(TC._init)
    chk(5, "⚠ 배선이 `build_cache` -> `_init` -> 증강기까지 닿나",
        si[-1] == "phase_jitter_map" and "phase_jitter_map" in sb
        and "phase_jitter_std_map=(phase_jitter_map or None)" in src,
        "`_init` 의 마지막 인자 = **%s** (맨 뒤여야 한다 — `initargs` 가 위치 인자라 "
        "중간에 끼우면 뒤엣것이 밀려 **조용히 다른 캐시를 굽는다**, 14.103 경고) · "
        "`build_cache` 노출 %s · 증강기 전달 %s"
        % (si[-1], "phase_jitter_map" in sb,
           "phase_jitter_std_map=(phase_jitter_map or None)" in src))

    # [6] 캐시 메타에 **적히나** (안 적으면 두 캐시를 구별 못 한다)
    src2 = inspect.getsource(TC.build_cache)
    chk(6, "캐시 메타에 **적히나**", '"phase_jitter_map"' in src2,
        "`meta.json` 에 `phase_jitter_map` 을 쓴다 — 없으면 나중에 어느 지터로 구운 "
        "캐시인지 알 길이 없다 ([[verify-the-input-path-not-just-the-model]])")

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
