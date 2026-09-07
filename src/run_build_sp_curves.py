# -*- coding: utf-8 -*-
"""s(p) 곡선을 **회로모델에서** 만든다 — 전력 증강이 고조파 모양까지 옮기게 (13.30, 2026-09-07).

왜 필요한가
----------
증강이 전력을 `a` 배 할 때 지금은 전류를 **선형으로 곱한다**. 그것은 "모양은 부하와 무관"
이라는 고정 서명 가정이고 캡 입력 SMPS 에서는 틀리다. 회로모델을 참으로 놓고 재면:

```
충전기 63->40W (0.63배)   h3 12.2%  h5 20.6%  h7 30.3%  h13 79.2%  틀린다
미니PC 18.5->9.9W (0.54)  **h1 23.3%**  h3 11.1%  h13 55.3%
프로젝터 44.9->38W (0.85)  h3  4.6%  h7 11.6%  h13 32.7%
```

이 크기는 우리가 13.28 에서 고친 결합 오차(h11~h15 에서 12~15%)보다 **크다.** 곡선 없이
전력 지터를 넓히면 더 큰 오차를 새로 넣는 셈이다.

왜 지금 필요해졌나 (13.29)
------------------------
복합 시험이 쓰는 동작점이 단독 녹화와 어긋난다 — 프로젝터 0.0% · 충전기 21.8% · 미니PC 7.3%.
메우려면 전력을 크게(0.3~0.9배) 옮겨야 하는데, 그러려면 모양이 따라와야 한다.

옛 파일은 왜 못 쓰나
-----------------
`processed_data/sp_curves.npz` 는 옛 계측기(차동 ADC) 시대 산물이고 파일이 없다.
여기서는 새 계측기 원시로 맞춘 `circuit_model/circ12_*.pkl` 에서 다시 만든다.

쓰는 법
------
    python -X utf8 -m src.run_build_sp_curves            # processed_data/sp_curves.npz
    python -X utf8 -m src.run_build_sp_curves --verify   # 만든 뒤 선형 곱과 견준다
"""
from typing import Dict, List
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.synthesis.coupling import SMPS_DEVICES, SmpsCircuit

#: 곡선을 뽑을 전력 범위 — **복합이 쓰는 대역과 단독 녹화를 둘 다 덮어야 한다** (13.29).
#:   기기        복합 계단 p10~p90   단독 동작 중앙
#:   프로젝터     36.2~41.1W         44.9W
#:   충전기       24.1~58.6W         62.5W
#:   미니PC        5.6~ 9.9W         18.5W
#: 아래는 그 둘을 덮고 양쪽으로 조금 넓힌 것이다. 하한은 회로모델이 무너지는 곳 위로 둔다.
#: `LEVEL_SCRAMBLE_SMPS` 의 배율을 녹화 전력에 걸었을 때 나오는 범위를 덮어야 한다 —
#: 밖으로 나가면 `rescale_to_power` 가 선형 곱으로 되돌아가 모양이 안 따라온다.
P_RANGE: Dict[str, tuple] = {
    "laptop_charger": (9.0, 70.0),      # 녹화 25~63W x 0.38~1.05
    "minipc": (2.5, 28.0),              # 녹화 9~20W  x 0.30~1.05
    "beam_projector": (28.0, 52.0),     # 녹화 45W    x 0.78~1.05 (+ 예열 여유)
}
N_POINTS = 28
V_REF = 224.0


def build(devices: List[str], n: int = N_POINTS, v_ref: float = V_REF) -> dict:
    circ = SmpsCircuit()
    rel = np.zeros(15, complex)
    rel[0] = 1.0 + 0j                     # 깨끗한 정현파 기준 — 텍스처 효과는 델타가 따로 맡는다
    out: Dict[str, np.ndarray] = {}
    names: List[str] = []
    for d in devices:
        if not circ.has(d):
            print(f"  {d}: 회로모델 없음 — 건너뛴다")
            continue
        lo, hi = P_RANGE.get(d, (5.0, 70.0))
        P, S, i1, ph1, k = [], [], [], [], []
        for p in np.linspace(lo, hi, n):
            I = circ.current(d, float(p), rel, v_ref)
            if I is None or not np.all(np.isfinite(I)) or abs(I[0]) < 1e-9:
                continue
            I = np.asarray(I, complex)
            s = I / I[0]                  # h1 정규화 모양, s[0] == 1
            s[0] = 1.0 + 0j
            P.append(float(p)); S.append(s)
            i1.append(float(abs(I[0])))
            ph1.append(float(np.angle(I[0])))
            k.append(float(abs(I[0]) * v_ref / p))       # |I1| = k·p/V 의 역산
        if len(P) < 4:
            print(f"  {d}: 유효 점이 {len(P)}개뿐 — 건너뛴다")
            continue
        names.append(d)
        out[f"{d}__P"] = np.asarray(P, float)
        out[f"{d}__S"] = np.asarray(S, complex)
        out[f"{d}__i1"] = np.asarray(i1, float)
        out[f"{d}__ph1_rad"] = np.asarray(ph1, float)
        out[f"{d}__k"] = np.asarray(k, float)
        out[f"{d}__V_ref"] = np.float64(v_ref)
        out[f"{d}__discrete"] = np.bool_(False)
        print(f"  {d}: {len(P)}점 {P[0]:.1f}~{P[-1]:.1f}W  "
              f"k {min(k):.3f}~{max(k):.3f}  |I1| {i1[0]*1e3:.0f}~{i1[-1]*1e3:.0f}mA")
    out["device_names"] = np.asarray(names)
    return out


def verify(path: str) -> None:
    """만든 곡선이 선형 곱보다 실제로 나은지 — 회로모델을 참으로 놓고 견준다."""
    from src.synthesis.sp_curves import load_curves, rescale_to_power
    circ = SmpsCircuit()
    cur = load_curves(path)
    rel = np.zeros(15, complex); rel[0] = 1.0
    print(f"\n{'기기':16s}{'기준W':>7s}{'목표W':>7s} | {'선형 곱':>22s}{'s(p) 곡선':>22s}")
    for d, (p0, targets) in (("laptop_charger", (63.0, [48, 40, 30])),
                             ("minipc", (18.5, [12, 9.9, 6])),
                             ("beam_projector", (44.9, [41, 38, 36]))):
        if d not in cur:
            continue
        I0 = np.asarray(circ.current(d, p0, rel, V_REF), complex)[None, :]
        for p in targets:
            a = p / p0
            It = np.asarray(circ.current(d, float(p), rel, V_REF), complex)
            lin = (I0 * a)[0]
            sp = rescale_to_power(I0, np.array([p0]), a, cur[d])[0]
            e = lambda x: np.median(np.abs(x - It)[[2, 4, 6, 8]] /
                                    np.maximum(np.abs(It)[[2, 4, 6, 8]], 1e-9))
            print(f"{d:16s}{p0:7.1f}{p:7.1f} | h3~h9 중앙 {e(lin):9.1%}"
                  f"{'':11s}{e(sp):9.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="processed_data/sp_curves.npz")
    ap.add_argument("--points", type=int, default=N_POINTS)
    ap.add_argument("--v-ref", type=float, default=V_REF)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print(f"s(p) 곡선을 회로모델에서 만든다 — {a.points}점 · V_ref {a.v_ref:g}V")
    print("=" * 78)
    d = build(list(SMPS_DEVICES), a.points, a.v_ref)
    if len(d.get("device_names", [])) == 0:
        print("만들 수 있는 기기가 없다")
        return 1
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **d)
    print(f"\n저장: {a.out}  ({len(d['device_names'])}기기)")
    if a.verify:
        verify(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
