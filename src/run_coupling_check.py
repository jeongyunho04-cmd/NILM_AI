# -*- coding: utf-8 -*-
"""결합 풀개 검사 — **학습 자료 분포에서** 고정점이 얼마나 틀리나 (13.24.20, 2026-09-07).

왜 따로 있나
-----------
`run_mixval12` 의 `[C]` 는 **복합 녹화의 평가 창**에서 풀개를 잰다. 그런데 그 창들은 SMPS 가
한 대뿐인 경우가 많아(test_1 은 미니PC 단독 25창·충전기 단독 17창) 오차가 0.03% 로 나온다.
캐시는 다르다 — 레시피가 SMPS 를 겹쳐 만들고(`smps_overlap` 0.26) Z 를 0.30~2.00Ω 에서 뽑는다.
**틀리는 곳은 거기다.** 그래서 생성기가 실제로 뽑는 (Z, 기기 조합, 전력) 에서 따로 잰다.

무엇을 재나
----------
같은 개방 전압에서 출발해 `SmpsCircuit.solve_terminal` 로 단자 전압을 풀고, 그 전압에서 낸
전류를 **잠근 풀개**(relax 0.5 x 200회, 잔차 ~1e-20)와 견준다. 모형 오차는 양쪽에 같이 들어가
지워지므로 남는 것은 **풀개 오차뿐**이다.

쓰는 법
------
    python -X utf8 -m src.run_coupling_check                     # 지금 설정
    python -X utf8 -m src.run_coupling_check --relax 0.5 --n-iter 6   # 감쇠 후보
"""
from typing import Dict, List, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.synthesis.coupling import DEFAULT_N_ITER, SmpsCircuit
from src.synthesis.grid_simulator import GridSimulator
from src.synthesis.vtexture import default_library

#: 기기별 ON 전력 범위 — 단독 녹화의 p5~p95 (13.24.7 의 표와 같은 자료).
PW_RANGE: Dict[str, Tuple[float, float]] = {
    "laptop_charger": (30.0, 70.0),
    "minipc": (8.0, 27.0),
    "beam_projector": (35.0, 45.0),
}
#: 10% 를 넘으면 "틀렸다" 로 센다 — 회로 모델 자체의 실측 오차가 2~7% 이므로 그 위다.
BAD = 0.10


def measure(n: int, n_iter: int, relax: float, seed: int = 0,
            extrapolate: bool = True) -> List[dict]:
    gs = GridSimulator()
    bank = default_library()
    rng = np.random.default_rng(seed)
    now = SmpsCircuit(n_iter=n_iter, relax=relax, extrapolate=extrapolate)
    ref = SmpsCircuit(models=now.models, n_iter=200, relax=0.5)
    devs = list(PW_RANGE)
    out: List[dict] = []
    for _ in range(n):
        env = gs.sample_environment()
        Z = float(env.r_grid_ohm)
        k = int(rng.integers(1, len(devs) + 1))
        pick = list(rng.choice(devs, size=k, replace=False))
        p = {d: float(rng.uniform(*PW_RANGE[d])) for d in pick}
        tex = bank.sample(rng)
        rel = tex.source_rel()
        v1 = float(env.base_voltage_v)
        # L 도 그 환경 것을 쓴다. 100µH 로 고정하면 생성기 분포(x_grid 0.02~0.15Ω ->
        # 53~400µH, 중앙 225µH)보다 Z_h 가 작아져 **오차를 과소평가한다.**
        L = float(env.x_grid_ohm) / (2.0 * np.pi * 60.0)
        V_r = ref.solve_terminal(p, rel, v1, Z, L)
        V_n = now.solve_terminal(p, rel, v1, Z, L)
        if V_r is None or V_n is None:
            continue
        I_r = sum(np.asarray(ref.current(d, pw, V_r / v1, v1)) for d, pw in p.items())
        I_n = sum(np.asarray(now.current(d, pw, V_n / v1, v1)) for d, pw in p.items())
        out.append({"z": Z, "k": k, "p": sum(p.values()), "site": tex.site,
                    "err": float(np.max(np.abs(I_n - I_r) / np.maximum(np.abs(I_r), 1e-9)))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, default=300)
    # ⚠ 모듈 기본값을 따라간다. 여기에 숫자를 박으면 "지금 설정" 을 잰다면서
    #   다른 것을 재게 된다 (13.27 에서 실제로 그랬다 — n=4 라 이름 붙이고 n=3 을 쟀다).
    ap.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    ap.add_argument("--relax", type=float, default=1.0)
    ap.add_argument("--no-extrapolate", action="store_true",
                    help="Aitken 외삽을 끈다 — 옛 거동과 견줄 때 쓴다 (13.27)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = measure(a.windows, a.n_iter, a.relax, a.seed, not a.no_extrapolate)
    if not rows:
        print("표본이 없다 — 회로 모델이 전부 실패했다 (circuit_model/__pycache__ 를 지워라)")
        return 1
    z = np.array([r["z"] for r in rows])
    print("=" * 96)
    print(f"결합 풀개 검사 — n_iter={a.n_iter} relax={a.relax:g} "
          f"외삽={'끔' if a.no_extrapolate else '켬'} · 창 {len(rows)}개 "
          f"(기준: 잠근 풀개 relax 0.5 x 200)")
    print(f"   생성기의 Z: 중앙 {np.median(z):.2f}Ω  p10~p90 {np.percentile(z, 10):.2f}~"
          f"{np.percentile(z, 90):.2f}  최대 {z.max():.2f}")
    print("=" * 96)
    print(f"{'구간':24s} {'n':>4s} {'중앙':>8s} {'p90':>9s} {'p99':>10s} {'>10%':>7s}")

    def show(name, sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return
        e = np.array([x["err"] for x in r])
        print(f"{name:24s} {len(r):4d} {np.median(e):8.1%} {np.percentile(e, 90):9.1%} "
              f"{np.percentile(e, 99):10.1%} {(e > BAD).mean():7.1%}")

    show("전체", lambda x: True)
    show("Z < 0.6  (E 쪽)", lambda x: x["z"] < 0.6)
    show("0.6 <= Z < 1.3  (D 쪽)", lambda x: 0.6 <= x["z"] < 1.3)
    show("Z >= 1.3  (탐색 위쪽)", lambda x: x["z"] >= 1.3)
    for k in (1, 2, 3):
        show(f"SMPS {k}대", lambda x, k=k: x["k"] == k)
    e = np.array([r["err"] for r in rows])
    frac = float((e > BAD).mean())
    print(f"\n**{BAD:.0%} 를 넘게 틀리는 창: {int((e > BAD).sum())}/{len(rows)} = {frac:.1%}**")
    if frac > 0.05:
        print("  ⚠ 학습 자료의 상당 부분이 결합에서 틀린다. 감쇠를 켜라 — "
              "`--relax 0.5 --n-iter 6` 이면 모든 Z 에서 0.3% 아래다 (13.24.13)")
    if a.out:
        json.dump({"config": vars(a), "rows": rows}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
