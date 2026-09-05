#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""교정 상수가 바뀌기 전에 모은 주기별 CSV 를 새 눈금으로 옮긴다.

    python migrate_cal.py data/*.csv
    python migrate_cal.py data/old.csv --out data/migrated

nilm_receiver.py 가 남긴 79열 주기별 CSV 가 대상이다. 입력은 절대 덮어쓰지
않고 <이름>.cal2.csv 로 내보내며, 무엇을 어떻게 바꿨는지 <이름>.cal2.json 에
같이 남긴다(데이터셋은 출처가 없으면 나중에 못 믿는다).

기본값은 2026-09-05 의 두 트림이다:
  - NILM_LOW_TRIM 1.0058  : LOW 경로가 HIGH 대비 0.55% 높게 읽던 것을 맞춤
  - NILM_VOLT_SCALE 307.9 -> 309.34 : 무부하 기준계측기 215V vs 보드 214V
위상 교정 상수(NILM_CAL_DEFAULT_*)는 아직 안 바뀌었으므로 기본 0 이다.
나중에 바꾸면 --phase-low / --phase-high 에 (새 값 - 옛 값)을 도(deg)로 준다.

[무엇이 정확하고 무엇이 근사인가]

 전압 트림은 정확하다. 전압 경로에는 레인지가 없어서 모든 행에 같은 배수가
 걸린다: vrms, vh1..vh15, 그리고 p_w.

 LOW 트림은 "그 주기가 전부 LOW 경로로 재구성됐을 때만" 정확하다. LOW 는
 피크 약 2.22A 에서 레일에 닿으므로, 그보다 큰 주기는 봉우리 근처만 HIGH 로
 넘어간 혼재 주기다. 주기별 CSV 에는 그 혼재 비율이 안 남으므로 되돌릴 수가
 없다. 그래서 이 스크립트는 피크 상한(sqrt2 * 고조파 rms 합)으로 순수 LOW 를
 판별해 그 행만 고치고, 나머지는 손대지 않고 개수를 보고한다.
 (--mixed apply 로 강제할 수 있다. 어차피 보정폭 자체가 0.58% 라 혼재 주기의
  참값은 "적용"과 "미적용" 사이에 있고, 어느 쪽을 골라도 오차는 0.58% 안이다)

 위상은 회전이라 정확히 되돌아간다. 크기(irms/vrms/ih/vh)는 회전에 안 변하고,
 phase_deg 와 ihdeg 는 각각 +delta, -h*delta 로 옮기면 된다. p_w 는 기본파
 성분만 다시 계산해 더한다 - 실측으로 vh1*ih1*cos(phase_deg) 가 p_w 와 0.08%
 안에서 맞는 것을 확인했다(나머지가 고조파 전력이고, 그건 h*delta 로 돌아야
 하지만 그 차이는 delta 1도 기준 P 의 0.3% 안이다).

[이 스크립트가 못 하는 것]
 2026-09-05 의 차동 -> 단일 종단 전환 이전 데이터는 눈금이 아니라 "측정 자체"가
 다르다. 같은 부하에서 전류 짝수 고조파가 1.3~1.5% -> 0.20%, 전압 h2 가
 2~3% -> 0.04% 로 줄었다. 그건 배수로 못 옮긴다. 고조파를 특징으로 쓰는
 학습에서는 그 경계를 데이터셋에 표시해 두는 편이 안전하다.
 off_low 열도 그대로 둔다. 카운트 단위가 402.8uV -> 201.4uV 로 바뀌었지만
 옛 데이터는 옛 단위로 적힌 것이 맞다.
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

IH = [f"ih{h}" for h in range(1, 16)]
IHDEG = [f"ihdeg{h}" for h in range(1, 16)]
VH = [f"vh{h}" for h in range(1, 16)]
NEEDED = ["irms", "p_w", "phase_deg", "range", "over_count", "vrms"] + IH + IHDEG + VH

# 옛 눈금에서 LOW 경로가 레일에 닿는 1차측 전류 [A] = 클리핑 문턱 / LOW 감도
SENS_HIGH = 27.0 * (1.0 + 6800 / 2700) / 2000.0
I_LOW_MAX = 1.55 / (SENS_HIGH * (75000 / 5100))          # 2.2189 A


def migrate(path, a):
    d = pd.read_csv(path)
    miss = [c for c in NEEDED if c not in d.columns]
    if miss:
        print(f"  [건너뜀] {os.path.basename(path)}: 열이 없다 {miss[:4]}")
        return None

    k_v = a.volt_new / a.volt_old
    k_i = 1.0 / a.low_trim

    # --- 1) 전압: 모든 행에 동일 ---
    d[["vrms"] + VH] *= k_v
    d["p_w"] *= k_v

    # --- 2) LOW 전류: 순수 LOW 주기만 ---
    peak_ub = np.sqrt(2.0) * d[IH].sum(axis=1)           # 고조파가 전부 동상일 때의 상한
    pure_low = (peak_ub <= I_LOW_MAX) & (d["over_count"] == 0)
    rows = slice(None) if a.mixed == "apply" else pure_low
    d.loc[rows, ["irms"] + IH] *= k_i
    d.loc[rows, "p_w"] *= k_i

    # --- 3) 위상: 회전 ---
    n_ph = 0
    if a.phase_low or a.phase_high:
        delta = np.where(d["range"] == 1, a.phase_high, a.phase_low)   # [deg]
        p1_old = d.vh1 * d.ih1 * np.cos(np.radians(d.phase_deg))
        d["phase_deg"] = (d["phase_deg"] + delta + 180.0) % 360.0 - 180.0
        for h, c in enumerate(IHDEG, start=1):
            d[c] = (d[c] - h * delta + 180.0) % 360.0 - 180.0
        d["p_w"] += d.vh1 * d.ih1 * np.cos(np.radians(d.phase_deg)) - p1_old
        n_ph = int((delta != 0).sum())

    stem = os.path.splitext(os.path.basename(path))[0]
    out_dir = a.out or os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, stem + ".cal2.csv")
    info = {
        "source": os.path.abspath(path), "rows": int(len(d)),
        "volt_scale": [a.volt_old, a.volt_new], "volt_factor": round(k_v, 8),
        "low_trim": a.low_trim, "low_current_factor": round(k_i, 8),
        "low_rows_corrected": int(pure_low.sum()) if a.mixed != "apply" else int(len(d)),
        "mixed_rows_left_alone": 0 if a.mixed == "apply" else int((~pure_low).sum()),
        "mixed_policy": a.mixed, "i_low_rail_a": round(I_LOW_MAX, 4),
        "phase_delta_deg": {"low": a.phase_low, "high": a.phase_high},
        "phase_rows": n_ph,
        "note": "차동->단일종단 전환 이전 데이터는 고조파 함량 자체가 다르다. 배수로 못 옮긴다.",
    }
    if not a.dry_run:
        d.to_csv(out_csv, index=False, float_format="%.6f")
        with open(out_csv[:-4] + ".json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    pct = 100.0 * info["low_rows_corrected"] / max(len(d), 1)
    print(f"  {os.path.basename(path)}: {len(d)}행  전압 x{k_v:.6f}  "
          f"LOW전류 x{k_i:.6f} ({pct:.1f}% 적용, 혼재 {info['mixed_rows_left_alone']}행 보류)"
          + (f"  위상 {n_ph}행" if n_ph else ""))
    return info


def main():
    p = argparse.ArgumentParser(description="주기별 CSV 를 새 교정 눈금으로 옮긴다")
    p.add_argument("csv", nargs="+")
    p.add_argument("--low-trim", type=float, default=1.00580, dest="low_trim")
    p.add_argument("--volt-old", type=float, default=307.9, dest="volt_old")
    p.add_argument("--volt-new", type=float, default=309.34, dest="volt_new")
    p.add_argument("--phase-low", type=float, default=0.0, dest="phase_low",
                   help="LOW 레인지 위상 상수의 (새 값 - 옛 값) [deg]")
    p.add_argument("--phase-high", type=float, default=0.0, dest="phase_high")
    p.add_argument("--mixed", choices=["skip", "apply"], default="skip",
                   help="혼재 주기의 LOW 전류 보정: 보류(기본) / 강제 적용")
    p.add_argument("--out", default=None, help="출력 폴더 (기본: 입력과 같은 곳)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    print(f"LOW 레일 한계 {I_LOW_MAX:.4f} A 기준으로 순수 LOW 주기를 가린다"
          + ("  [dry-run]" if a.dry_run else ""))
    ok = 0
    for f in a.csv:
        if migrate(f, a) is not None:
            ok += 1
    print(f"완료: {ok}/{len(a.csv)} 파일")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
