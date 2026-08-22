"""
Sim-to-Real 검증: 합성 데이터가 실측 복합 부하를 얼마나 재현하는가
====================================================================
data/test*.csv 는 여러 가전을 실제로 동시에 돌리며 얻은 실측 복합 부하다.
합성 엔진이 만든 신호를 같은 조합의 실측 구간과 맞대어, 모델이 보게 될
입력 특징이 얼마나 어긋나는지 수치로 낸다.

# 실행:
python -m src.run_sim_to_real

[비교 방법]
파일 전체 통계를 맞대면 안 된다. 실측 파일마다 기기 구성과 동작 시간이 달라
그 차이가 물리 오차로 둔갑한다. 대신 '같은 기기 조합이 켜져 있는 구간'끼리
비교한다. 예를 들어 test3 에서 오븐이 꺼져 있는 구간은 미니PC+빔프로젝터이므로,
합성 쪽에서도 그 둘만 켠 신호와 비교한다.

[비교하는 항목]
모델 입력 33채널에 직접 들어가는 값들이다.
  P, Q, PF, THD_i : 전력 특징
  I1, I3, I5, I7  : 고조파 전류 (기기 판별의 핵심 단서)
  R_grid          : 부하가 걸릴 때 전압이 얼마나 내려가는가

[이 검증으로 찾아 고친 것들]
1. 계통 임피던스가 실측의 1/4~1/6 (0.25 Ohm vs 실측 1.55 Ohm)
2. 무효전력 Q 의 부호가 통째로 반대 (phase_deg = -ihdeg1 관계를 놓침)
3. 활성화 균등 추첨으로 오븐 통전율이 실제의 3배 이상
4. 전압 환산 기준(v_ref)에 파일 전체 중앙값을 써서, 자기 부하로 전압이
   내려간 채 측정된 구간을 높은 전압에서 측정한 것으로 오인
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import sys
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.preprocessing import load_nilm_npz
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.grid_simulator import GridSimulator
from src.synthesis.synthesizer import ApplianceSchedule, LoadSynthesizer

SAMPLING_HZ = 60.0

# 실측 파일에서 잘라낼 비교 구간.
# (파일, 구간명, 그때 켜져 있던 가전, 전력 하한, 전력 상한, 시작 시각(초))
# 시작 시각은 그 조합이 성립한 이후만 보기 위한 것이다.
REAL_SEGMENTS: List[dict] = [
    dict(file="test3", name="미니PC+프로젝터", apps=["minipc", "beam_projector"],
         p_lo=40, p_hi=200, after_s=110),
    dict(file="test3", name="미니PC+프로젝터+오븐히터", apps=["minipc", "beam_projector", "oven"],
         p_lo=800, p_hi=1e9, after_s=110),
    dict(file="test.2", name="미니PC+프로젝터+오븐히터 (2)", apps=["minipc", "beam_projector", "oven"],
         p_lo=800, p_hi=1e9, after_s=110),
    dict(file="test.2", name="미니PC 단독", apps=["minipc"],
         p_lo=15, p_hi=40, after_s=240),
    dict(file="test", name="에어컨+충전기+선풍기", apps=["air_conditioner", "laptop_charger", "fan"],
         p_lo=800, p_hi=1200, after_s=250),
    dict(file="test", name="드라이기 고열+저부하", apps=["hair_dryer", "laptop_charger", "fan"],
         p_lo=1000, p_hi=1300, after_s=660),
    # ── test_4 (2026-08-22 추가) ────────────────────────────────────────────
    # 0.2절이 "저항성끼리 겹칠 때가 진짜 시험대" 라 한 구간을 sim-to-real 로 처음 잰다.
    # 여기 없으면 이 검증은 저항 부하가 겹친 적이 한 번도 없는 채로 "양호" 를 낸다.
    # 전력 구간은 실측 라벨로 확인했다 (real_events.json 의 오븐 `_heater_pulses`
    # 와 핫플 통전 구간 교집합):
    #   핫플 단독      P 491~602W  (11,917 사이클)
    #   오븐히터 단독  P 1146~1224W ( 8,563 사이클)
    #   **동시**       P 1558~1643W ( 5,093 사이클, 85초)
    # 세 구간이 안 겹치므로 전력만으로 갈린다. 프로젝터·충전기는 파일 내내 켜져 있어
    # (둘만 있을 때 P 중앙값 73W) 조합에 반드시 넣어야 비교가 성립한다.
    # test_4 는 218.4V 회선이지만 `compare_segment` 가 파일에서 V0·R 을 추정해
    # 합성 쪽에 그대로 물리므로 전압 차이는 비교를 깨지 않는다.
    dict(file="test_4", name="핫플 단독+저부하", apps=["hotplate", "beam_projector", "laptop_charger"],
         p_lo=420, p_hi=700, after_s=120),
    dict(file="test_4", name="오븐히터+저부하", apps=["oven", "beam_projector", "laptop_charger"],
         p_lo=1050, p_hi=1300, after_s=120),
    dict(file="test_4", name="오븐히터+핫플 동시 (저항 겹침)",
         apps=["oven", "hotplate", "beam_projector", "laptop_charger"],
         p_lo=1450, p_hi=1800, after_s=120),
    # 저부하 SMPS 기저 두 개. **위 세 구간의 고조파 오차를 읽으려면 이 둘이 필요하다.**
    # 실측에서는 두 조합이 거의 같다 (P 89.5 vs 92.9W, I3 0.3157 vs 0.3261).
    # 그런데 합성은 서로 **반대 방향**으로 틀린다 - 충전기 쪽이 P +15.5%/I3 +36.7%,
    # 미니PC 쪽이 P -35.3%/I3 -22.5% 다. 그래서 같은 오븐 히터(~1198W)를 재도
    # 동반 저부하가 충전기면 I3 오차가 +43.1%, 미니PC면 +8.7% 로 갈린다.
    # 이 구간이 없으면 그 차이가 저항 부하의 문제로 잘못 읽힌다.
    # 기존 구간에서 안 보였던 이유: 충전기가 든 유일한 구간(`test` 에어컨+충전기+선풍기)은
    # 에어컨 I3=2.09 가 충전기의 10배라 충전기 오차가 묻힌다.
    dict(file="test_4", name="저부하 기저 (프로젝터+충전기)", apps=["beam_projector", "laptop_charger"],
         p_lo=40, p_hi=140, after_s=120),
    dict(file="test.2", name="저부하 기저 (미니PC+프로젝터)", apps=["minipc", "beam_projector"],
         p_lo=40, p_hi=140, after_s=120),
]

FEATURES = [
    ("P", "P (W)"), ("Q", "Q (VAR)"), ("PF", "PF"), ("THD", "THD_i"),
    ("I1", "I1 (A)"), ("I3", "I3 (A)"), ("I5", "I5 (A)"), ("I7", "I7 (A)"),
]


def signature(power_features: np.ndarray, harmonics: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    """한 구간의 전기적 지문. 표본이 모자라면 None."""
    if int(mask.sum()) < 60:
        return None
    mg = np.abs(harmonics[mask])
    pf = power_features[mask]
    return {
        "P": float(np.median(pf[:, 0])), "Q": float(np.median(pf[:, 1])),
        "PF": float(np.median(pf[:, 3])), "THD": float(np.median(pf[:, 5])),
        "I1": float(np.median(mg[:, 0])), "I3": float(np.median(mg[:, 2])),
        "I5": float(np.median(mg[:, 4])), "I7": float(np.median(mg[:, 6])),
        "_n": int(mask.sum()),
    }


def estimate_grid_resistance(power_features: np.ndarray, harmonics: np.ndarray) -> Tuple[float, float, float]:
    """V = V0 - R*I_re 회귀로 배선 저항을 추정한다.

    Returns: (V0, R, 결정계수)
    """
    v = power_features[:, 4]
    i_re = np.real(harmonics[:, 0])
    A = np.column_stack([np.ones_like(i_re), -i_re])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    pred = A @ coef
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2 = 1.0 - float(np.sum((v - pred) ** 2)) / ss_tot if ss_tot > 0 else 0.0
    return float(coef[0]), float(coef[1]), r2


def load_real(stem: str) -> dict:
    path = Path("processed_data/composite_eval") / f"{stem}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 python -m src.run_preprocess_and_label 을 실행하세요."
        )
    return load_nilm_npz(path)


def synthesize_combo(
    synth: LoadSynthesizer, apps: Sequence[str], base_voltage: float, r_grid: float,
    n_cycles: int = 18000, seed: int = 0,
) -> "object":
    """지정한 가전만 켠 신호를 합성한다. 실측 구간과 같은 전압 환경을 쓴다."""
    np.random.seed(seed)
    grid = GridSimulator(
        nominal_voltage=base_voltage, r_grid=r_grid, x_grid=0.05,
        voltage_variation_std=0.6, sag_rate_per_min=0.0,
    )
    local = LoadSynthesizer(
        segment_pool=synth.pool, grid_simulator=grid,
        compute_gt_harmonics=False, sustained_power_limit_w=None,
    )
    usable = [a for a in apps if a in local.known_appliances]
    schedules = [ApplianceSchedule(a, 0, n_cycles) for a in usable]
    plugged = {a: (a in usable) for a in local.known_appliances}
    return local.synthesize_scenario(
        n_cycles, schedules, plugged_in_appliances=plugged,
        include_noise=True, simulate_voltage_drop=True,
    )


def compare_segment(
    pool: SegmentPool, synth: LoadSynthesizer, spec: dict, n_seeds: int = 5
) -> Optional[dict]:
    """실측 구간 하나와 같은 조합의 합성 신호를 비교한다.

    합성 쪽 표본은 '요청한 가전이 전부 켜져 있는' 시점만 쓴다. 전력 범위만으로
    거르면 비교가 무너진다. 짧은 활성화가 뽑히면 증강 한도(원본의 3배) 때문에
    요청한 길이를 못 채우고 중간에 꺼지는데, 그 구간이 섞이면 없는 기기의
    고조파를 실측과 맞대게 된다.

    풀에서 어떤 활성화가 뽑히느냐에 따라 값이 흔들리므로 여러 시드로 반복해
    중앙값을 쓴다.
    """
    real = load_real(spec["file"])
    pf, hc = real["power_features"], real["harmonics_complex"]
    p = pf[:, 0]

    mask = (p > spec["p_lo"]) & (p < spec["p_hi"])
    start = int(spec.get("after_s", 0) * SAMPLING_HZ)
    if start:
        mask[:start] = False
    real_sig = signature(pf, hc, mask)
    if real_sig is None:
        return None

    # 실측 구간의 전압 환경을 그대로 재현해야 비교가 성립한다
    v0, r_grid, r2 = estimate_grid_resistance(pf, hc)

    trials: List[Dict[str, float]] = []
    for seed in range(n_seeds):
        sample = synthesize_combo(synth, spec["apps"], v0, max(0.1, r_grid), seed=seed)
        s_pf, s_hc = sample.power_features, sample.harmonics_complex
        s_mask = (s_pf[:, 0] > spec["p_lo"]) & (s_pf[:, 0] < spec["p_hi"])
        for a in spec["apps"]:
            if a in sample.gt_is_on:
                s_mask &= sample.gt_is_on[a] == 1
        sig_i = signature(s_pf, s_hc, s_mask)
        if sig_i is not None:
            trials.append(sig_i)

    if not trials:
        return None
    syn_sig = {k: float(np.median([t[k] for t in trials])) for k, _ in FEATURES}
    syn_sig["_n"] = int(np.median([t["_n"] for t in trials]))
    syn_sig["_trials"] = len(trials)

    errors = {}
    for key, _ in FEATURES:
        r, s = real_sig[key], syn_sig[key]
        errors[key] = (s - r) / abs(r) * 100.0 if abs(r) > 1e-9 else float("nan")

    return dict(
        file=spec["file"], name=spec["name"], apps=list(spec["apps"]),
        real=real_sig, synth=syn_sig, errors=errors,
        real_v0=round(v0, 2), real_r_grid=round(r_grid, 3), real_r2=round(r2, 3),
    )


def run(output_dir: str = "synthetic_data") -> dict:
    print("\n" + "=" * 92)
    print("[NILM AI] Sim-to-Real 검증 - 합성 데이터 vs 실측 복합 부하")
    print("=" * 92)

    pool = SegmentPool(npz_dir="processed_data/npz")
    synth = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False)

    # ── 1. 계통 임피던스 ────────────────────────────────────────────────────
    print("\n[1] 계통 임피던스 - 부하가 걸릴 때 전압이 얼마나 내려가는가")
    print(f"  {'실측 파일':12s}{'V0(V)':>9s}{'R(옴)':>9s}{'R^2':>8s}{'1kW당 강하(V)':>15s}")
    grid_rows = {}
    for stem in ["test", "test.2", "test3"]:
        z = load_real(stem)
        v0, r, r2 = estimate_grid_resistance(z["power_features"], z["harmonics_complex"])
        grid_rows[stem] = dict(v0=round(v0, 2), r_ohm=round(r, 3), r2=round(r2, 3),
                               drop_per_kw=round(r * 1000 / v0, 2))
        flag = "" if r2 > 0.7 else "  (R^2 낮음 - 인버터 부하라 분리가 어려움)"
        print(f"  {stem:12s}{v0:>9.2f}{r:>9.3f}{r2:>8.3f}{r * 1000 / v0:>14.2f}{flag}")

    gs = synth.grid_sim
    envs = [gs.sample_environment() for _ in range(400)]
    r_syn = np.array([e.r_grid_ohm for e in envs])
    print(f"  {'합성 모델':12s}{'':>9s}{np.median(r_syn):>9.3f}{'':>8s}"
          f"{np.median(r_syn) * 1000 / 222:>14.2f}   (범위 {r_syn.min():.2f}~{r_syn.max():.2f})")

    # ── 2. 조합별 지문 비교 ─────────────────────────────────────────────────
    print("\n[2] 같은 기기 조합끼리 전기적 지문 비교  (오차 = (합성-실측)/실측)")
    results = []
    for spec in REAL_SEGMENTS:
        res = compare_segment(pool, synth, spec)
        if res is None:
            print(f"\n  [{spec['file']}] {spec['name']}: 표본 부족으로 건너뜀")
            continue
        results.append(res)
        print(f"\n  [{res['file']}] {res['name']}  ({', '.join(res['apps'])})")
        print(f"    실측 표본 {res['real']['_n']:,} 사이클 | 실측 환경 V0={res['real_v0']}V R={res['real_r_grid']}옴")
        print(f"    {'항목':10s}{'실측':>12s}{'합성':>12s}{'오차':>10s}")
        for key, label in FEATURES:
            e = res["errors"][key]
            mark = "" if abs(e) <= 15 else ("  <" if abs(e) <= 40 else "  <<")
            print(f"    {label:10s}{res['real'][key]:>12.4f}{res['synth'][key]:>12.4f}{e:>9.1f}%{mark}")

    # ── 3. 요약 ─────────────────────────────────────────────────────────────
    print("\n[3] 특징별 평균 절대 오차")
    print(f"  {'항목':10s}{'평균|오차|':>12s}{'최대|오차|':>12s}{'판정':>10s}")
    summary = {}
    for key, label in FEATURES:
        errs = [abs(r["errors"][key]) for r in results if np.isfinite(r["errors"][key])]
        if not errs:
            continue
        mean_e, max_e = float(np.mean(errs)), float(np.max(errs))
        verdict = "양호" if mean_e <= 15 else ("주의" if mean_e <= 40 else "불일치")
        summary[key] = dict(mean_abs_err_pct=round(mean_e, 1), max_abs_err_pct=round(max_e, 1))
        print(f"  {label:10s}{mean_e:>11.1f}%{max_e:>11.1f}%{verdict:>10s}")

    report = {
        "grid_impedance": grid_rows,
        "synthetic_r_grid_median": round(float(np.median(r_syn)), 3),
        "segments": results,
        "feature_summary": summary,
    }
    out = Path(output_dir) / "sim_to_real_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    print("\n" + "=" * 92)
    print(f"리포트 저장: {out.resolve()}")
    print("=" * 92 + "\n")
    return report


if __name__ == "__main__":
    run()
