"""
레시피 믹스 후보의 동시성과 2차 배경을 미리 잰다 (설계 문서 12.67절)
======================================================================
12.66 이 격차를 닫았다 — 합성 창이 실측보다 한산해서(평균 1.41 vs 2.33,
빈 창 28% vs 5%) 2차 배경이 2.3배 조용하고, 그래서 |I2| 판별 신호 6.1 mA 가
합성에서만 살아남는다. 처방은 동시성을 올리는 것이다.

**믹스를 바꾸면 유령 전력이 돌아올 위험이 있다** — `standby_only` 와
`unplugged_baseline` 은 저부하 오탐 방지용이다 (0.3절). 그래서 학습 45분을
쓰기 전에 후보 믹스가 실제로 목표 동시성에 닿는지 여기서 먼저 본다.

    python -m src.run_recipe_mix_probe --preset half
    python -m src.run_recipe_mix_probe --mix standby_only=0.08 high_low_mixed=0.21

목표 (12.66.2, 12.66.3):
    실측    평균 2.33   빈 창  5%   |ΔI2| p95 12.26 mA
    현재    평균 1.41   빈 창 28%   |ΔI2| p95  5.42 mA
    절반    평균 1.87   빈 창 16%   |ΔI2| p95  8.8 mA  <- 12.68 의 목표

**SMPS 동시성 (2026-08-25 추가, HANDOFF_2026-08-25 5.1절)**
12.68 이 겨냥한 것은 **전체** 동시성이었고 실패했다. 12.88.4 가 남긴 축은
그것이 아니라 **SMPS 3종(프로젝터·충전기·미니PC)이 겹치는 비율**이다 —
12.81 이 미검출의 조건을 "경쟁 SMPS 가 함께 켜져 있을 때" 로 좁혔기 때문이다.
그래서 평균 활성 기기 수가 아니라 아래 두 지표를 본다.

    SMPS≥2   두 대 이상이 동시에 켜진 시간 비율
    SMPS=3   세 대가 모두 켜진 시간 비율

실측 기준선은 `--real` 로 같은 정의로 뽑는다 (정답 구간에서 직접).
합성은 창 전체(시간 점유)와 타깃 시점 두 가지로 잰다 — 손실이 실제로 먹는 것은
타깃 시점 쪽이다 (seq2point).
"""
from typing import Dict
import argparse
import io
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.synthesis.dataset import DEFAULT_RECIPE_MIX, NILMBatchGenerator
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.synthesizer import LoadSynthesizer

SMPS = ("beam_projector", "laptop_charger", "minipc")

#: 12.66.5 가 지목한 이동. 빈 창을 만드는 셋에서 덜어 동시 부하 쪽으로 옮긴다.
PRESETS: Dict[str, Dict[str, float]] = {
    # ── 동시성 탈상관 (2026-09-10, 13.83) ────────────────────────────────
    # 미니PC↔형제 SMPS 의 phi 를 **+0.333 -> 0** 으로 보낸다. 레시피별 phi 는
    # 전부 +0.13 이하인데 섞으면 +0.333 이 된다 (혼합 유도 상관) — 한계확률이
    # 극단이면서 같은 방향인 것들을 섞어서다. 그래서 믹스만으로는 +0.164 에서
    # 멈추고, **`--smps-focus-off-p 0.4` 와 같이** 써야 0 에 간다.
    # 곁들여 전력대 분포도 실측 쪽으로 간다 (30~100W 창 16.6% -> 30.9%,
    # 실측 32.6%). `smps_overlap` 은 오히려 0.26 -> 0.30 으로 올린다 —
    # 실측 SMPS>=2 가 79% 라 그 레시피를 줄이면 안 된다.
    # ⚠ `decorr` 는 **에어컨을 굶긴다** (2026-09-10, 13.83.13). 아래 `decorr2` 를 써라.
    #   에어컨은 `random_realistic`(양성률 0.107) 과 `random_uniform`(0.170) **에서만**
    #   나온다 — 나머지 일곱은 0% 다 (`smps_hi_fix` 주석에 이미 적혀 있었는데 내가
    #   최적화에 제약으로 안 걸었다). 둘을 0.08+0.11 -> 0.03+0.03 으로 깎으면
    #   0.06/0.19 = 0.32 배가 되어 캐시 양성률이 **0.028 -> 0.009 (111:1)** 로 떨어진다.
    #   v28 캐시가 그 상태로 구워졌다.
    "decorr": {
        "smps_overlap": 0.30,
        "low_load_among_standby": 0.296,
        "high_low_mixed": 0.219,
        "resistive_overlap": 0.035,
        "high_power_resistive": 0.03,
        "random_realistic": 0.03,
        "random_uniform": 0.03,
        "standby_only": 0.03,
        "unplugged_baseline": 0.03,
    },
    # `decorr` 에 **에어컨 노출 하한**(v27 수준 0.0272)을 걸고 다시 푼 것 (13.83.13).
    # 그 하한이 `random_realistic`+`random_uniform` 을 0.19 근처로 묶어 두므로
    # 탈상관이 조금 후퇴한다 — 대가는 표로 남긴다:
    #     phi 미↔충 / 미↔프     SMPS>=2   에어컨 양성률
    #   v27      +0.355 / +0.358   48.6%     0.0272
    #   decorr   +0.032 / +0.029   48.8%     0.0083   <- 에어컨 굶음
    #   decorr2  +0.074 / +0.088   47.9%     0.0272
    # `--smps-focus-off-p 0.4` 와 **같이** 쓴다는 전제는 `decorr` 와 같다.
    "decorr2": {
        "smps_overlap": 0.30,
        "high_low_mixed": 0.205,
        "low_load_among_standby": 0.185,
        "random_uniform": 0.111,
        "random_realistic": 0.079,
        "high_power_resistive": 0.03,
        "resistive_overlap": 0.03,
        "standby_only": 0.03,
        "unplugged_baseline": 0.03,
    },
    # 13.83.19 — `steady_loaded` 를 넣은 판. 실측은 **부하 있는 창의 11.7%** 가
    # 창 안 전이 0개인데 합성은 **0.8%** 였다 (133창 중 1창). 그 결핍이
    # "큰 고조파 계단이 있으면 충전기 · 잠잠하면 미니PC" 라는 지름길을 만들었고,
    # 충전기가 몇 분씩 잠잠한 실측 배포에서 통째로 뒤집힌다 (13.83.16~18).
    # ⚠ `steady_loaded` 는 에어컨의 **세 번째 공급원**이다 (레시피 안 양성률 0.19).
    #   13.83.13 의 굶김이 자동으로 완화되지만 굽기 전에 `positive_rate` 로 확인한다.
    # `--smps-focus-off-p 0.4` 와 같이 쓴다는 전제는 `decorr2` 와 같다.
    # ⚠ `steady` 는 **저항성을 굶긴다** — 쓰지 말고 아래 `steady2` 를 써라.
    #   핫플(0.883)과 오븐(0.808)의 유일 공급원이 `resistive_overlap` 인데
    #   `decorr` 가 그것을 0.12 -> 0.035 로 깎았고 나는 0.03 으로 더 깎았다.
    #   결과: 오븐 0.079 (v27 0.160 의 절반) · 핫플 0.100.
    #   에어컨 때(13.83.13)와 **완전히 같은 구조**인데 또 반복했다.
    "steady": {
        "steady_loaded": 0.15,
        "smps_overlap": 0.26,
        "high_low_mixed": 0.17,
        "low_load_among_standby": 0.155,
        "random_uniform": 0.095,
        "random_realistic": 0.07,
        "high_power_resistive": 0.03,
        "resistive_overlap": 0.03,
        "standby_only": 0.02,
        "unplugged_baseline": 0.02,
    },
    # 13.83.19 — **모든 부류에 하한**을 걸고 `results/_recipe_app_table.json`
    # (레시피 x 기기 양성률 + 결합분포) 위에서 푼 것. 굽지 않고 계산된다.
    #   제약: 에어컨>=0.027 · 오븐>=0.20 · 핫플>=0.15 · 포트>=0.15 · 드라이기>=0.12
    #        |phi 미↔충| · |phi 미↔프| <= 0.10 · steady_loaded >= 0.12
    #   목적: SMPS>=2 최대화 (실측 0.79)
    # 결과는 v27 을 **모든 축에서** 이긴다 (focus_off 0.4 를 건 같은 조건에서):
    #                oven  hotpl kettle dryer  a/c   phi미충  SMPS>=2
    #   v27          0.160 0.176 0.193 0.219 0.029  +0.180   0.485
    #   steady2      0.201 0.220 0.224 0.268 0.040  +0.072   0.568
    # `--smps-focus-off-p 0.4` 와 같이 쓴다.
    "steady2": {
        "high_low_mixed": 0.238,
        "smps_overlap": 0.188,
        "resistive_overlap": 0.162,
        "steady_loaded": 0.121,
        "high_power_resistive": 0.091,
        "random_uniform": 0.056,
        "low_load_among_standby": 0.044,
        "unplugged_baseline": 0.041,
        "random_realistic": 0.030,
        "standby_only": 0.029,
    },
    "half": {                        # 실측까지의 절반
        "random_realistic": 0.23,
        "random_uniform": 0.14,
        "standby_only": 0.08,
        "low_load_among_standby": 0.16,
        "high_power_resistive": 0.10,
        "high_low_mixed": 0.21,
        "resistive_overlap": 0.05,
        "unplugged_baseline": 0.03,
    },
    # ── SMPS 겹침을 겨냥한 후보 (2026-08-25, 12.88.4 의 1번) ──────────────
    # 지분은 `random_*` 와 `low_load_among_standby` 에서 덜어 온다. **`standby_only`
    # 와 `unplugged_baseline` 은 손대지 않는다** - 저부하 오탐(유령) 방지용이고
    # (0.3절), 12.66.5 가 거기서 덜어냈다가 유령 위험을 떠안았다.
    "smps": {
        "random_realistic": 0.12,
        "random_uniform": 0.12,
        "standby_only": 0.16,
        "low_load_among_standby": 0.14,
        "high_power_resistive": 0.10,
        "high_low_mixed": 0.12,
        "resistive_overlap": 0.05,
        "unplugged_baseline": 0.05,
        "smps_overlap": 0.14,
    },
    # 12.90.2 뒤의 판 (2026-08-25). smps_hi(0.26)가 저항 레시피를 함께 깎아
    # 포트 환각을 test_4 -> test3 으로 옮겼다. 여기서는 **저항 3종을 손대지
    # 않는다** - high_power_resistive / high_low_mixed / resistive_overlap 고정.
    # 용량도 0.26 -> 0.15 로 낮춘다.
    "smps_mid": {
        "random_realistic": 0.11,        # -0.05
        "random_uniform": 0.10,          # -0.04
        "standby_only": 0.14,            # -0.02
        "low_load_among_standby": 0.16,  # -0.04  (smps_overlap 이 부분 대체한다:
                                         #         SMPS 3종은 전부 저부하 기기이고
                                         #         force_plugged_all 로 대기전력이 깔린다)
        "high_power_resistive": 0.10,    # 고정
        "high_low_mixed": 0.14,          # 고정
        "resistive_overlap": 0.05,       # 고정
        "unplugged_baseline": 0.05,      # 고정
        "smps_overlap": 0.15,
    },
    # 12.92.4 의 후보 (2026-08-25). smps_hi 와 같은 용량(0.26)인데 **저항 3종을
    # 고정**한다 - adapt_smpsx 단독의 유령 13.20W 가 포트 환각이었고, 그것이
    # 저항 레시피 축소 탓이라는 12.91.1 의 읽기를 직접 시험한다.
    # 예산 0.26 은 random_* 에서 크게 덜고 유령 방지 계열은 최대한 남긴다.
    "smps_hi_fix": {
        # **에어컨은 random_realistic(15%)과 random_uniform(19%) 에서만 나온다** -
        # 레시피별 양성률을 재 보니 나머지 일곱은 전부 0% 였다. 그래서 그 둘을
        # 크게 깎으면 굶는다: 0.06/0.08 판에서 512창 중 6개(불균형 31.7:1)로
        # 떨어져 저장소 가드(30:1)를 넘었다. 예산은 low_load_among_standby 에서
        # 주로 덜어 온다 - smps_overlap 이 그 성격(대기전력 속 저부하)을 가장
        # 많이 대체하는 레시피다. **유령 위험은 이쪽으로 옮겨간다** (판정 ②).
        "random_realistic": 0.08,        # -0.08
        "random_uniform": 0.14,          #  0
        "standby_only": 0.14,            # -0.02
        "low_load_among_standby": 0.04,  # -0.16  <- 예산의 대부분
        "high_power_resistive": 0.10,    # 고정
        "high_low_mixed": 0.14,          # 고정
        "resistive_overlap": 0.05,       # 고정
        "unplugged_baseline": 0.05,      # 고정
        "smps_overlap": 0.26,
    },
    # 13.48: `smps_hi_fix` 에서 **저항 겹침을 5 -> 12%** 로 올린다. 재원은
    # standby_only(14->10) 와 random_uniform(14->11).
    #   실측 저항2+ 31.9% 인데 합성이 11.7% 였다 (`resistive_overlap` 이 5% 뿐).
    #   test_5 의 포트 창(저항 3~4종 동시)이 학습 분포 밖에 있던 것이 그 결과다.
    # `smps_overlap` 비중은 건드리지 않는다 — 그 안은 이미 SMPS2+ 100% 로 꽉 찼고,
    # 모자란 SMPS 동시성은 저항 레시피의 `RESISTIVE_SMPS_BACKGROUND` 로 채운다.
    "smps_hi_res3": {
        "smps_overlap": 0.26,
        "standby_only": 0.10,
        "high_low_mixed": 0.14,
        "random_uniform": 0.11,
        "high_power_resistive": 0.10,
        "random_realistic": 0.08,
        "unplugged_baseline": 0.05,
        "resistive_overlap": 0.12,
        "low_load_among_standby": 0.04,
    },
    "smps_hi": {
        "random_realistic": 0.10,
        "random_uniform": 0.10,
        "standby_only": 0.16,
        "low_load_among_standby": 0.10,
        "high_power_resistive": 0.08,
        "high_low_mixed": 0.10,
        "resistive_overlap": 0.05,
        "unplugged_baseline": 0.05,
        "smps_overlap": 0.26,
    },
    "full": {                        # 실측 수준을 겨냥
        "random_realistic": 0.28,
        "random_uniform": 0.14,
        "standby_only": 0.04,
        "low_load_among_standby": 0.12,
        "high_power_resistive": 0.10,
        "high_low_mixed": 0.26,
        "resistive_overlap": 0.05,
        "unplugged_baseline": 0.01,
    },
}


def measure(mix, n_windows=400, window_cycles=3600, seed=0, half=5.0, guard=1.0):
    """동시성과 |ΔI2| 널을 `audit_synth` 와 같은 기하로 잰다."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", holdout_frac=0.2)
    syn = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)
    gen = NILMBatchGenerator(segment_pool=pool, window_size_cycles=window_cycles,
                            recipe_mix=mix, synthesizer=syn, compute_gt_harmonics=False)
    nulls, n_active = [], []
    time_hist = np.zeros(4, dtype=np.int64)    # SMPS 동시 ON 수의 시간 점유
    tgt_hist = np.zeros(4, dtype=np.int64)     # 같은 것을 타깃 시점에서만
    for _ in range(n_windows):
        r, _ = gen._synthesize_window()
        I = np.asarray(r.harmonics_complex)
        n = len(I)
        t = np.arange(n) / 60.0
        n_active.append(len(r.active_appliances))
        on_mat = np.stack([np.asarray(r.gt_is_on.get(app, np.zeros(n, np.int8)))[:n]
                           for app in SMPS]).astype(np.int8)
        k_series = on_mat.sum(0)
        time_hist += np.bincount(k_series, minlength=4)[:4]
        tgt_hist[int(k_series[min(gen.target_index, n - 1)])] += 1
        tt = []
        for app in SMPS:
            on = on_mat[SMPS.index(app)]
            tt.extend(np.flatnonzero(np.diff(on) != 0) / 60.0)
        tt = np.array(tt)
        cand = [c for c in np.arange(half + 7, t[-1] - half - 7, 2.0)
                if len(tt) == 0 or np.min(np.abs(tt - c)) > 12]
        if not cand:
            continue
        for c in rng.choice(cand, min(2, len(cand)), replace=False):
            pre = (t >= c - half) & (t <= c - guard)
            post = (t >= c + guard) & (t <= c + half)
            if pre.sum() < 30 or post.sum() < 30:
                continue
            nulls.append(np.abs((np.median(I[post].real, 0) + 1j * np.median(I[post].imag, 0))
                                - (np.median(I[pre].real, 0) + 1j * np.median(I[pre].imag, 0))))
    na = np.array(n_active)
    nl = np.array(nulls)
    th = time_hist / max(time_hist.sum(), 1)
    gh = tgt_hist / max(tgt_hist.sum(), 1)
    return {"mean_active": float(na.mean()), "median_active": float(np.median(na)),
            "empty_share": float((na == 0).mean()), "n_null": len(nl),
            "i2_med_ma": float(np.median(nl[:, 1]) * 1000),
            "i2_p95_ma": float(np.percentile(nl[:, 1], 95) * 1000),
            "smps_time_ge2": float(th[2] + th[3]), "smps_time_eq3": float(th[3]),
            "smps_tgt_ge2": float(gh[2] + gh[3]), "smps_tgt_eq3": float(gh[3]),
            "smps_time_mean": float((np.arange(4) * th).sum())}


def real_smps_occupancy(path="processed_data/real_events.json", dt=0.5):
    """실측 정답 구간에서 SMPS 동시 ON 의 시간 점유를 잰다 (합성과 같은 정의).

    `uncertain` 구간은 OFF 로 세고 그 비율을 따로 돌려준다 — 큰 파일은
    기준선에서 빼야 한다 (test_4 는 SMPS 두 종이 701초 중 535초가 uncertain 이다).
    파일을 합치지 않고 하나씩 돌려준다 — 서로 구성이 달라 평균이 무엇의 평균인지
    알 수 없기 때문이다 (측정의 규칙 5).
    """
    files = json.load(io.open(path, encoding="utf-8"))["files"]
    out = {}
    for name, d in files.items():
        dur = float(d.get("duration_s", 0.0))
        if dur <= 0:
            continue
        n = int(dur / dt)
        on = np.zeros((len(SMPS), n), dtype=np.int8)
        unc = np.zeros(n, dtype=np.int8)
        iv = d.get("intervals", {})
        for i, app in enumerate(SMPS):
            spans = iv.get(app, {})
            for a, b in spans.get("on", []):
                on[i, int(a / dt):int(b / dt)] = 1
            for a, b in spans.get("uncertain", []):
                unc[int(a / dt):int(b / dt)] = 1
        k = on.sum(0)
        h = np.bincount(k, minlength=4)[:4] / max(n, 1)
        out[name] = {
            "present": sorted(set(d.get("appliances_present", [])) & set(SMPS)),
            "dur_s": dur, "uncertain_share": float(unc.mean()),
            "smps_time_ge2": float(h[2] + h[3]), "smps_time_eq3": float(h[3]),
            "smps_time_mean": float((np.arange(4) * h).sum()),
        }
    return out


def print_real(rows):
    print()
    print("=" * 92)
    print("[실측 기준선] SMPS 동시 ON 시간 점유 — 정답 구간에서 직접 (12.81 의 조건)")
    print("=" * 92)
    print(f"  {'파일':10s}{'길이s':>8s}{'SMPS 있음':>11s}{'평균':>7s}{'≥2':>8s}{'=3':>8s}{'uncertain':>11s}")
    for name, r in rows.items():
        star = " *" if len(r["present"]) == 3 and r["uncertain_share"] < 0.05 else ""
        print(f"  {name:10s}{r['dur_s']:>8.0f}{len(r['present']):>11d}"
              f"{r['smps_time_mean']:>7.2f}{100*r['smps_time_ge2']:>7.0f}%"
              f"{100*r['smps_time_eq3']:>7.0f}%{100*r['uncertain_share']:>10.0f}%{star}")
    good = [r for r in rows.values() if len(r["present"]) == 3 and r["uncertain_share"] < 0.05]
    if good:
        w = np.array([r["dur_s"] for r in good])
        ge2 = float(np.average([r["smps_time_ge2"] for r in good], weights=w))
        eq3 = float(np.average([r["smps_time_eq3"] for r in good], weights=w))
        mean = float(np.average([r["smps_time_mean"] for r in good], weights=w))
        print(f"  {'* 가중평균':10s}{w.sum():>8.0f}{3:>11d}{mean:>7.2f}{100*ge2:>7.0f}%{100*eq3:>7.0f}%")
    print("\n  * = SMPS 3종이 다 있고 uncertain 이 5% 미만인 파일. 기준선은 이것들만 쓴다")
    return good


def main() -> int:
    ap = argparse.ArgumentParser(description="레시피 믹스 후보를 미리 잰다")
    ap.add_argument("--preset", nargs="*", default=["half"], choices=list(PRESETS) + ["none"])
    ap.add_argument("--mix", nargs="*", default=None, metavar="KEY=VAL")
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--real", action="store_true",
                    help="실측 SMPS 동시성 기준선도 함께 낸다")
    ap.add_argument("--no-synth", action="store_true", help="합성 측정을 건너뛴다 (실측만)")
    a = ap.parse_args()

    real_rows = real_smps_occupancy() if (a.real or a.no_synth) else None
    if a.no_synth:
        print_real(real_rows)
        return 0

    cands = {"현재 (DEFAULT_RECIPE_MIX)": DEFAULT_RECIPE_MIX}
    for name in a.preset:
        if name != "none":
            cands[f"프리셋 {name}"] = PRESETS[name]
    if a.mix:
        m = dict(DEFAULT_RECIPE_MIX)
        for kv in a.mix:
            k, v = kv.split("=")
            m[k] = float(v)
        cands["직접 지정"] = m

    results = {}
    for name, mix in cands.items():
        tot = sum(mix.values())
        if abs(tot - 1.0) > 1e-6:
            print(f"  {name:28s}  ⚠ 합이 {tot:.3f} 입니다 — 건너뜁니다")
            continue
        results[name] = measure(mix, n_windows=a.windows)

    print("=" * 92)
    print("[레시피 믹스 후보] 전체 동시성과 2차 배경 (12.67절)")
    print("=" * 92)
    print(f"  {'믹스':28s}{'평균 활성':>10s}{'빈 창':>9s}{'|ΔI2| 중앙':>13s}{'|ΔI2| p95':>12s}")
    for name, r in results.items():
        print(f"  {name:28s}{r['mean_active']:>10.2f}{100*r['empty_share']:>8.0f}%"
              f"{r['i2_med_ma']:>11.2f}mA{r['i2_p95_ma']:>10.2f}mA")
    print(f"  {'실측 test_5/6/7':28s}{2.33:>10.2f}{5:>8.0f}%{1.21:>11.2f}mA{12.26:>10.2f}mA")
    print("\n  판별 신호 6.1 mA 가 널 p95 아래로 내려가야 |I2| 단서가 죽는다 (12.66.4)")

    print()
    print("=" * 92)
    print("[SMPS 동시성] 프로젝터·충전기·미니PC 가 겹치는 비율 — 5.1절이 겨냥하는 축")
    print("=" * 92)
    print(f"  {'믹스':28s}{'시간 평균':>10s}{'시간 ≥2':>10s}{'시간 =3':>10s}"
          f"{'타깃 ≥2':>10s}{'타깃 =3':>10s}")
    for name, r in results.items():
        print(f"  {name:28s}{r['smps_time_mean']:>10.2f}{100*r['smps_time_ge2']:>9.1f}%"
              f"{100*r['smps_time_eq3']:>9.1f}%{100*r['smps_tgt_ge2']:>9.1f}%"
              f"{100*r['smps_tgt_eq3']:>9.1f}%")
    print("\n  타깃 = seq2point 가 실제로 채점하는 시점. 손실이 먹는 것은 이쪽이다")

    if real_rows is not None:
        print_real(real_rows)
    else:
        print("  실측 기준선은 --real 로 함께 낸다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
