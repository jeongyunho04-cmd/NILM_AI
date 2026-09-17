# -*- coding: utf-8 -*-
"""`L_harm` 의 **상태 사전이 못 담는 몫**을 잰다 (14.393, 사용자 질문).

*"우리가 캐시 생성을 개선했는데 `l_harm` 은 안 바꿨는데 제대로 학습이 되는지"*

v50 캐시는 부하 의존 위상 법칙 `k = a·ln P` 를 **깨끗이 담는다** (§48.3b, R² 0.673/0.686).
그런데 `L_harm` 의 사전은 `sig_state[k, s, h]` — **기기 x 상태마다 상수 페이저 하나**다.
상태는 SMPS 셋 다 **통전이 하나**(minipc s2 · charger s2 · beam s2)라, 그 상태 **안**의
전력 변화는 사전에 **표현할 자리가 없다**.

여기서 재는 것 (전부 학습이 쓰는 그 함수 · 그 분할로):
```
  coh    |중앙 페이저| / 중앙 |페이저|     <- 1 미만이면 사전이 **크기를 잃는다**
  irr    중앙 |per_w − sig| / 중앙 |per_w| <- 참 배분에서도 **못 줄이는** 상대 잔차
  rot    ∠(sig_상위반 / sig_하위반) [도]    <- 상태 안 **부하**가 내는 회전 (체계적)
```
`rot` 을 v50 이 주입하는 지터(`PHASE_JITTER_DEG_MEASURED`)와 견준다.

    python -X utf8 -m src.run_diag_sigstate

⚠ `run_diag_sigfloor` 와 헷갈리지 마라 — 그쪽은 **녹화 간** 재현성(13.84.31),
  여기는 **상태 안** 부하 의존성이다. (오늘 같은 이름으로 한 번 덮어썬다.)
"""
from typing import List, Tuple

import numpy as np

from src import env_guard  # noqa: F401

from src.model.net import state_power_w  # noqa: E402
from src.synthesis.augmentor import PHASE_JITTER_DEG_MEASURED  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

HS = (1, 3, 5, 7, 9, 11, 13, 15)
#: SMPS 셋 + **대조군**. 저항이 같이 안 나빠져야 이 수가 부하-회전 탓이라고 말할 수 있다
WHO = (("minipc", 2), ("laptop_charger", 2), ("beam_projector", 2),
       ("oven", 2), ("electiric_kettle", 2), ("hair_dryer", 2), ("fan", 1))


def gather(pool, app: str, sid: int) -> Tuple[np.ndarray, np.ndarray]:
    """`harmonic_signatures_by_state` 와 **같은 규약**으로 사이클을 모은다."""
    cs: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    for a in pool.appliance_activations.get(app, []):
        st = getattr(a, "state_id", None)
        if st is None:
            continue
        pw = state_power_w(a, False)
        m = (pw > 1.0) & (np.asarray(st) == sid)
        if m.any():
            cs.append(np.asarray(a.net_harmonics_complex)[m])
            ps.append(pw[m])
    if not cs:
        return np.zeros((0, 15), complex), np.zeros(0)
    return np.concatenate(cs), np.concatenate(ps)


def med_sig(per_w: np.ndarray) -> np.ndarray:
    """손실이 쓰는 그 식 — 실·허를 **따로** 중앙값 (`harmonic_signatures_by_state`)."""
    return np.median(per_w.real, 0) + 1j * np.median(per_w.imag, 0)


def main() -> int:
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    print("`L_harm` 사전의 바닥 (14.393) — 학습이 쓰는 `sig_state` 그대로\n")
    print("  주입 지터 표 %s\n" % PHASE_JITTER_DEG_MEASURED)

    for app, sid in WHO:
        c, p = gather(pool, app, sid)
        if len(c) < 200:
            print("  %-18s s%d  사이클 %d — **사전이 기기 전체로 되돌아간다** (min 200)\n"
                  % (app, sid, len(c)))
            continue
        per_w = c / np.maximum(p, 1e-6)[:, None]
        sig = med_sig(per_w)
        lo = p <= np.median(p)
        s_lo, s_hi = med_sig(per_w[lo]), med_sig(per_w[~lo])

        print("  %s s%d — 사이클 %d · 전력 %.1f~%.1fW (중앙 %.1f · **폭 x%.2f**)"
              % (app, sid, len(c), p.min(), p.max(), np.median(p),
                 np.median(p[~lo]) / max(np.median(p[lo]), 1e-9)))
        print("    차수 |  coh    irr   rot(도)  |sig|(mA/W)")
        rots = []
        for h in HS:
            i = h - 1
            mag = np.median(np.abs(per_w[:, i]))
            if mag <= 0:
                continue
            coh = abs(sig[i]) / mag
            irr = np.median(np.abs(per_w[:, i] - sig[i])) / mag
            rot = np.degrees(np.angle(s_hi[i] / s_lo[i])) if abs(s_lo[i]) > 0 else np.nan
            rots.append((h, rot))
            print("    h%-3d | %5.3f  %5.3f  %+7.1f   %7.3f"
                  % (h, coh, irr, rot, abs(sig[i]) * 1e3))
        # 회전이 **차수에 비례**하나 (지연이면 그렇다). 절편 없이 적합한다.
        hh = np.array([h for h, r in rots if np.isfinite(r)], float)
        rr = np.array([r for h, r in rots if np.isfinite(r)], float)
        if len(hh) >= 4:
            k = float((hh @ rr) / (hh @ hh))
            ss = float(1 - ((rr - k * hh) ** 2).sum()
                       / max(((rr - rr.mean()) ** 2).sum(), 1e-12))
            jit = PHASE_JITTER_DEG_MEASURED.get(app)
            print("    ⇒ 상태 안 부하 회전 **k = %+.2f 도/차수** (h비례 R² %.3f)%s"
                  % (k, ss, ("  · 주입 지터 진폭 %.2f -> **%.1f배**"
                             % (jit, abs(k) / jit)) if jit else ""))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
