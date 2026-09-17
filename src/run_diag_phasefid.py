# -*- coding: utf-8 -*-
"""**캐시의 위상이 실측에서도 맞나** — 사용자 질문 (14.382).

`run_gate_smpssep` [6] 이 *"SMPS 판별력의 95%가 위상에 있다"* 를 쟀다. 그러면 곧바로
따라오는 질문이 이것이다 — **그 위상이 합성(캐시)과 실측에서 같은가?** 안 같으면
모델은 합성에서 위상을 배우고 실측에서 **틀린 각도**를 믿게 된다.

무엇을 견주나:
```
  캐시   = 격리 녹화 조각의 중첩 (`ApplianceActivation.net_harmonics_ri`)
         ⇒ 기기별 위상은 **풀의 위상 그대로**다 (구조상)
  실측   = composite_eval 5파일
  ⇒ 그래서 묻는 것은 "풀(격리)의 위상이 실측(복합)으로 **전이되나**" 다
```
어떻게:
```
  사람 라벨로 **SMPS 한 대만 켜진** 창을 고른다 (전력을 몰라도 된다 —
  위상은 크기에 1차로 무관하다). 그 창의 관측 ∠I_h 를 풀 지문의 ∠sig_h 와 견준다.
  ⚠ 저항이 같이 켜져 있으면 안 된다 — 저항 h1 이 SMPS 를 100:1 로 덮는다 (§45.2).
  ⚠ 자를 두 개 쓴다: 위상차 중앙(편향)과 **뭉침 R**(흩어짐). R 이 낮으면 각도 자체가
    난수라 편향을 읽으면 안 된다 ([[fix-the-phase-reference-before-comparing-phasors]]).
```

    python -X utf8 -m src.run_diag_phasefid
"""
from typing import List
import numpy as np

from src import env_guard  # noqa: F401

from src.evaluation.real_events import build_on_off_truth, load_events  # noqa: E402
from src.model import inputs as _I  # noqa: E402
from src.model.net import harmonic_signatures_by_state  # noqa: E402
from src.model.postproc import SMPS_GROUP  # noqa: E402
from src.model.realdata import RealWindows  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
KO = {"beam_projector": "빔", "laptop_charger": "충전기", "minipc": "미니PC"}
STEMS = ["test_1", "test_2", "test_3", "test_4", "test_5"]
ODD = [3, 5, 7, 9, 11, 13, 15]


def circ(d: np.ndarray):
    """(중앙 위상차 도, 뭉침 R). R 은 0(난수)~1(완전 뭉침)."""
    z = np.exp(1j * d)
    return float(np.degrees(np.angle(z.mean()))), float(abs(z.mean()))


def main() -> int:
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig, us = harmonic_signatures_by_state(pool, APPS)
    sc = sig[..., 0] + 1j * sig[..., 1]
    ev = load_events()
    SM = [a for a in APPS if a in SMPS_GROUP]
    RES = [a for a in APPS if a not in SMPS_GROUP]

    print("캐시(격리) 위상이 실측(복합)으로 전이되나 — 14.382\n")
    print("%-8s %-8s %6s | %s" % ("파일", "기기", "사이클",
                                  "차수별 Δ∠ [도]  (괄호는 뭉침 R · R<0.5 면 난수)"))
    tot = {a: {h: [] for h in ODD} for a in SM}
    for st in STEMS:
        z = np.load("processed_data/composite_eval/%s.npz" % st, allow_pickle=True)
        x = RealWindows._to_33ch(z).astype(np.float64)
        n = x.shape[1]
        on, _sc_mask = build_on_off_truth(st, APPS, int(ev[st]["cycles"]), ev)
        on = on[:n]
        obs = (x[0:15, :n] + 1j * x[15:30, :n]).T                  # (n,15)
        for a in SM:
            j = APPS.index(a)
            #: 그 SMPS **혼자** 켜진 사이클 — 저항이 하나라도 켜지면 뺀다
            solo = on[:, j].astype(bool).copy()
            for b in APPS:
                if b != a:
                    solo &= ~on[:, APPS.index(b)].astype(bool)
            if solo.sum() < 200:
                continue
            s_ = max([s for s in range(sc.shape[1]) if us[j, s]],
                     key=lambda s: float(np.abs(sc[j, s]).sum()), default=None)
            if s_ is None:
                continue
            cells = []
            for h in ODD:
                ref = sc[j, s_, h - 1]
                if abs(ref) <= 0:
                    cells.append("  h%-2d    -    " % h)
                    continue
                d = np.angle(obs[solo, h - 1]) - np.angle(ref)
                m, r = circ(d)
                tot[a][h].append((m, r, int(solo.sum())))
                cells.append("h%-2d %+6.1f(%.2f)" % (h, m, r))
            print("%-8s %-8s %6d | %s" % (st, KO[a], int(solo.sum()), " ".join(cells)))

    print("\n%s" % ("=" * 96))
    print("파일 통합 — 같은 기기가 여러 파일에서 **같은 각도 어긋남**을 내나\n")
    print("%-8s | %s" % ("기기", " ".join("%12s" % ("h%d" % h) for h in ODD)))
    for a in SM:
        cells = []
        for h in ODD:
            v = tot[a][h]
            if not v:
                cells.append("%12s" % "-")
                continue
            ms = [x[0] for x in v]
            rs = [x[1] for x in v]
            #: **파일 사이 일관성**이 핵심이다 — 어긋남이 파일마다 다르면 자리 탓이고,
            #  같으면 사전(격리 지문) 탓이다. 고칠 곳이 갈린다.
            cells.append("%+6.1f±%4.1f" % (float(np.mean(ms)), float(np.std(ms))))
        print("%-8s | %s" % (KO[a], " ".join(cells)))
        print("%-8s | %s  <- 뭉침 R" % ("",
              " ".join("%12s" % ("%.2f" % float(np.mean([x[1] for x in tot[a][h]]))
                                 if tot[a][h] else "-") for h in ODD)))
    print("\n읽는 법")
    print("  · Δ∠ 가 0 근처 + R 높음  -> 위상이 **전이된다.** 모델이 믿어도 된다")
    print("  · Δ∠ 가 크고 파일 사이 σ 작음 -> **사전(격리 지문)이 틀렸다.** 고칠 수 있다")
    print("  · Δ∠ 의 파일 사이 σ 큼   -> **자리마다 다르다.** 상수로는 못 담는다")
    print("  · R 낮음(<0.5)          -> 각도가 난수다. 그 차수는 위상을 쓰면 안 된다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
