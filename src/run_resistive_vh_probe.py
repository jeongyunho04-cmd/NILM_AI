"""저항 부하의 고조파가 정말 `V_h/R` 인가 — 자를 먼저 만든다 (12.151)

무엇을 묻나
----------
`L_harm` 은 기기마다 **와트당 고조파 전류가 상수**라고 둔다 (`harmonic_signatures`,
3.4절). 이것이 문헌이 말하는 *fixed current injection* 이고 실패가 기록돼 있다.

순저항 부하는 그 상수가 물리적으로 틀렸음을 **계산으로 안다**:

    I_h = V_h / R,   R = V_rms² / P   =>   I_h / P = V_h / V_rms²

즉 와트당 전류가 **그 순간 전압 고조파에 비례한다.** 고정지문은 `V_h` 가 안 변한다고
가정한 것이고, 3차 배경 전압은 장소 간 2.1배 다르다 (12.150).

그래서 손실을 고치기 **전에** 이걸 잰다 (규칙 25/39). 격리 녹화에서 저항 기기의
ON−OFF 차분 고조파를 뽑아 셋을 비교한다:

    실측 |ΔI_h|/ΔP        <- 참값
    옴 예측 |V_h|/V_rms²   <- 이번 처방
    고정지문 |sig[k,h]|    <- 지금 손실이 쓰는 값

위상도 본다. 옴이 맞다면 `∠ΔI_h` 는 **기기와 무관하게** `∠V_h` 여야 한다. 저항
기기 넷의 위상이 차수마다 모이면 그것이 옴의 증거고, 흩어지면 아니다.

쓰는 법
------
    python -X utf8 -m src.run_resistive_vh_probe
    python -X utf8 -m src.run_resistive_vh_probe --orders 1,3,5,7,9,11,13,15
"""
from typing import Dict, List, Tuple
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import pandas as pd

from src.preprocessing.raw_csv import read_raw_csv

#: 순저항으로 보는 기기와 그 격리 녹화. `fan`/`air_conditioner` 는 뺐다 —
#: 12.150.3 에서 선풍기는 선형이지만 저항이 아니고 에어컨은 비선형이다.
RESISTIVE: Dict[str, List[str]] = {
    "electiric_kettle": ["electiric_kettle", "electric_kettle_2_fixed"],
    "hair_dryer": ["hair_dryer_1", "hair_dryer_2", "hair_dryer_3"],
    "hotplate": ["hotplate_1", "hotplate_2", "hotplate_3_fixed"],
    "oven": ["oven", "oven_2", "oven_3_fixed"],
}
ORDERS = (1, 3, 5, 7, 9, 11, 13, 15)


def load(stem: str, orders=ORDERS):
    cols = ["p_w", "vrms"] + [f"ih{h}" for h in orders] \
        + [f"ihdeg{h}" for h in orders] + [f"vh{h}" for h in orders]
    # 정본 순서 (12.152). hotplate_1 은 순서 뒤바뀜이 210곳이다.
    d, _ = read_raw_csv(f"data/{stem}.csv", usecols=cols)
    I = np.stack([d[f"ih{h}"].to_numpy(np.float64)
                  * np.exp(1j * np.deg2rad(d[f"ihdeg{h}"].to_numpy(np.float64)))
                  for h in orders], 1)                       # (N, H) 복소
    V = np.stack([d[f"vh{h}"].to_numpy(np.float64) for h in orders], 1)   # 크기만
    return d["p_w"].to_numpy(np.float64), d["vrms"].to_numpy(np.float64), I, V


def split(p: np.ndarray, hi=0.6, lo=0.1, need=60) -> Tuple[np.ndarray, np.ndarray]:
    """통전/비통전 창을 가른다. 배경은 차분이 지운다 (12.140 의 교훈)."""
    bg, top = np.percentile(p, 5), np.percentile(p, 95)
    if top - bg < 100.0:
        return np.zeros(0, bool), np.zeros(0, bool)
    on = p > bg + hi * (top - bg)
    off = p < bg + lo * (top - bg)
    if on.sum() < need or off.sum() < need:
        return np.zeros(0, bool), np.zeros(0, bool)
    return on, off


def cmedian(z: np.ndarray) -> np.ndarray:
    return np.median(z.real, 0) + 1j * np.median(z.imag, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", default=",".join(map(str, ORDERS)))
    a = ap.parse_args()
    orders = tuple(int(x) for x in a.orders.split(","))

    from src.model.net import harmonic_signatures
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    apps = pool.get_appliance_types()
    sig = harmonic_signatures(pool, apps)          # (K, 15, 2) A/W
    del pool

    print("=" * 92)
    print("저항 부하 고조파 = V_h/R 인가 — 격리 녹화 ON-OFF 차분 (12.151)")
    print("=" * 92)

    rows, phases = [], {h: [] for h in orders}
    for app, stems in RESISTIVE.items():
        k = apps.index(app)
        for stem in stems:
            try:
                p, v, I, V = load(stem, orders)
            except Exception as e:                       # 파일/열이 없으면 건너뛴다
                print(f"  [건너뜀] {stem}: {e}")
                continue
            on, off = split(p)
            if on.size == 0:
                print(f"  [건너뜀] {stem}: 통전/비통전 창 부족 "
                      f"(p5={np.percentile(p,5):.0f} p95={np.percentile(p,95):.0f}W)")
                continue
            dI = cmedian(I[on]) - cmedian(I[off])
            dP = float(np.median(p[on]) - np.median(p[off]))
            vh = np.median(V[on], 0)
            vr = float(np.median(v[on]))
            meas = np.abs(dI) / dP * 1e3                                  # mA/W
            ohm = vh / vr ** 2 * 1e3                                      # mA/W
            fix = np.abs(sig[k, [h - 1 for h in orders], 0]
                         + 1j * sig[k, [h - 1 for h in orders], 1]) * 1e3
            rows.append((app, stem, dP, vr, int(on.sum()), meas, ohm, fix))
            for j, h in enumerate(orders):
                phases[h].append((stem, float(np.rad2deg(np.angle(dI[j])))))

    if not rows:
        print("쓸 수 있는 녹화가 없습니다."); return

    for app, stem, dP, vr, non, meas, ohm, fix in rows:
        print(f"\n■ {app} / {stem}   ΔP={dP:.0f}W  V_rms={vr:.1f}V  통전창 {non:,}")
        print("   h   실측 mA/W   옴 V_h/V²   고정지문    실측/옴   실측/고정")
        for j, h in enumerate(orders):
            r_o = meas[j] / ohm[j] if ohm[j] > 1e-9 else np.nan
            r_f = meas[j] / fix[j] if fix[j] > 1e-9 else np.inf
            print(f"  {h:>2}   {meas[j]:9.4f}   {ohm[j]:9.4f}   {fix[j]:9.4f}   "
                  f"{r_o:8.2f}   {r_f:8.2f}")

    # ── 종합: 차수별 중앙 비 ────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("차수별 중앙값 (저항 기기 전부)")
    print("   h   실측 mA/W   옴 예측    고정지문   실측/옴   실측/고정   위상 폭(도)")
    M = np.stack([r[5] for r in rows]); O = np.stack([r[6] for r in rows])
    F = np.stack([r[7] for r in rows])
    for j, h in enumerate(orders):
        ang = np.array([x[1] for x in phases[h]])
        # 원형 산포 — 평균벡터 길이로 잰다
        R = np.abs(np.mean(np.exp(1j * np.deg2rad(ang))))
        spread = np.rad2deg(np.sqrt(max(0.0, -2 * np.log(max(R, 1e-9)))))
        r_o = np.median(M[:, j] / np.maximum(O[:, j], 1e-9))
        r_f = np.median(M[:, j] / np.maximum(F[:, j], 1e-9))
        print(f"  {h:>2}   {np.median(M[:,j]):9.4f}   {np.median(O[:,j]):9.4f}   "
              f"{np.median(F[:,j]):9.4f}   {r_o:7.2f}   {r_f:9.2f}   {spread:9.1f}")

    print("\n[위상 — 옴이면 기기와 무관하게 ∠V_h 여야 한다]")
    for h in orders:
        s = "  ".join(f"{st.split('_')[0][:6]}:{ang:+6.1f}" for st, ang in phases[h])
        print(f"  h{h:<2} {s}")


if __name__ == "__main__":
    main()
