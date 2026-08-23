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

from src.evaluation.real_events import load_events
from src.preprocessing import load_nilm_npz
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.grid_simulator import GridSimulator
from src.synthesis.synthesizer import ApplianceSchedule, LoadSynthesizer

SAMPLING_HZ = 60.0

# 오븐 히터 통전으로 볼 최소 전력. 팬/조명(약 15W)과 가르는 선이다.
OVEN_HEATER_W = 300.0

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
    # ── test_5 / test_6 (2026-08-23 추가) — 라벨로 자른다 ────────────────────
    # 여기부터는 전력 범위가 아니라 **사람 기록 라벨**로 구간을 정한다 (`select`).
    # test_5/6/7 은 스위치를 누른 사람이 기록했으므로(`_label_provenance` =
    # human_switching_log) 조합을 정확히 지정할 수 있다. 전력 범위로 자르면
    # 오븐의 팬/조명(15W) 구간과 히터 통전(1.1kW) 구간이 한 덩어리로 섞인다.
    # 실제로 test_6 의 오븐 `on` 구간을 그대로 쓰면 P 중앙값이 45W 로 나온다 —
    # 히터가 아니라 팬이다. 히터는 `_heater_pulses` 에만 있다.
    #
    # [무엇을 재려는 것인가] **저항 부하가 SMPS 지문을 얼마나 덮는가**이다.
    # 같은 SMPS 를 오븐 히터가 꺼졌을 때와 통전 중일 때 각각 재서 맞댄다.
    # 실측에서 세 기기 모두 I7/I3 가 +12~+33% 올라 0.75~0.86 으로 뭉친다 —
    # 12.32.1 의 SMPS 판별자가 그때 소멸한다.
    #
    # **원인은 전압강하가 아니라 오븐 고조파의 선형 중첩이다** (2026-08-23 확인).
    # 오븐 히터는 1156W 에서 I3 0.0124 / I5 0.0572 / I7 0.0725 A 를 흘린다 -
    # 3차가 거의 없고 5·7차가 크다 (I7/I3 = 5.85). 복합 구간 I5 의 38~97%,
    # I7 의 40~72% 가 오븐 몫이다. (SMPS 단독)+(오븐 단독) 을 복소로 더하면
    # 실제 합계가 ±10% 안에서 재현된다 - SMPS 자신의 고조파는 거의 안 변한다.
    # 전압 응답으로 읽으면 안 된다.
    #
    # 그래서 이 구간들이 재는 것은 **합성기가 그 중첩을 맞히는가**이다.
    # 합성 오븐 단독은 잘 맞으므로(I7 오차 0.0%, I5 +3.6%), 여기서 남는 오차는
    # SMPS 단독 크기 오차(충전기 -50%, 미니PC -46%)가 넘어온 것이다.
    dict(file="test_6", name="미니PC 단독 (오븐 OFF)", apps=["minipc"],
         select=dict(on=["minipc"], oven="off"), vpair="test_6:minipc", vstate="off"),
    dict(file="test_6", name="미니PC + 오븐 히터 통전", apps=["minipc", "oven"],
         select=dict(on=["minipc"], oven="heater"), vpair="test_6:minipc", vstate="heater"),
    dict(file="test_6", name="프로젝터 단독 (오븐 OFF)", apps=["beam_projector"],
         select=dict(on=["beam_projector"], oven="off"), vpair="test_6:beam_projector", vstate="off"),
    dict(file="test_6", name="프로젝터 + 오븐 히터 통전", apps=["beam_projector", "oven"],
         select=dict(on=["beam_projector"], oven="heater"), vpair="test_6:beam_projector", vstate="heater"),
    dict(file="test_6", name="충전기 단독 (오븐 OFF)", apps=["laptop_charger"],
         select=dict(on=["laptop_charger"], oven="off"), vpair="test_6:laptop_charger", vstate="off"),
    dict(file="test_6", name="충전기 + 오븐 히터 통전", apps=["laptop_charger", "oven"],
         select=dict(on=["laptop_charger"], oven="heater"), vpair="test_6:laptop_charger", vstate="heater"),
    # test_5 는 같은 조합을 다른 날 다른 충전 상태에서 잡은 것이다.
    # 충전기 단독 P 가 test_6 43.7W vs test_5 72.6W 로 1.7배 다르다(배터리 잔량).
    # 합성은 둘 다 36.9W 로 같은 값을 내므로 이 두 구간이 함께 있어야
    # '충전 궤적이 없다'는 것이 오차로 드러난다.
    dict(file="test_5", name="충전기 단독 (오븐 OFF, 급속충전)", apps=["laptop_charger"],
         select=dict(on=["laptop_charger"], oven="off"), vpair="test_5:laptop_charger", vstate="off"),
    dict(file="test_5", name="충전기 + 오븐 히터 통전 (급속충전)", apps=["laptop_charger", "oven"],
         select=dict(on=["laptop_charger"], oven="heater"), vpair="test_5:laptop_charger", vstate="heater"),
    dict(file="test_5", name="프로젝터 단독 (오븐 OFF)", apps=["beam_projector"],
         select=dict(on=["beam_projector"], oven="off")),
]

# `select` 로 자를 때 "정확히 이것만 켜져 있어야 한다" 를 확인할 SMPS 무리.
# 저항 부하는 `oven` 항목과 `hotplate` 강제 OFF 로 따로 다룬다.
SELECTABLE_SMPS = ("minipc", "beam_projector", "laptop_charger")

FEATURES = [
    ("P", "P (W)"), ("Q", "Q (VAR)"), ("PF", "PF"), ("THD", "THD_i"),
    ("I1", "I1 (A)"), ("I3", "I3 (A)"), ("I5", "I5 (A)"), ("I7", "I7 (A)"),
    # 고조파 **형상**. 크기(I3/I5/I7)만 보면 형상 오차가 크기 오차에 묻힌다.
    # SMPS 판별은 크기가 아니라 이 비율로 한다 (12.32.1절). 저항 부하가 켜지면
    # 이 값이 어디로 가는지를 보려면 비율을 따로 찍어야 한다.
    ("I5_I3", "I5/I3"), ("I7_I3", "I7/I3"),
]


def signature(power_features: np.ndarray, harmonics: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, float]]:
    """한 구간의 전기적 지문. 표본이 모자라면 None."""
    if int(mask.sum()) < 60:
        return None
    mg = np.abs(harmonics[mask])
    pf = power_features[mask]
    i3 = float(np.median(mg[:, 2]))
    i5 = float(np.median(mg[:, 4]))
    i7 = float(np.median(mg[:, 6]))
    return {
        "P": float(np.median(pf[:, 0])), "Q": float(np.median(pf[:, 1])),
        "PF": float(np.median(pf[:, 3])), "THD": float(np.median(pf[:, 5])),
        "I1": float(np.median(mg[:, 0])), "I3": i3, "I5": i5, "I7": i7,
        # 비율은 중앙값끼리 나눈다. 사이클마다 나눈 뒤 중앙값을 내면 I3 가
        # 0 근처인 사이클에서 발산한다.
        "I5_I3": i5 / i3 if i3 > 1e-9 else float("nan"),
        "I7_I3": i7 / i3 if i3 > 1e-9 else float("nan"),
        "V": float(np.median(pf[:, 4])),
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


def _spans_to_mask(spans: Optional[Sequence[Sequence[float]]], n_cycles: int) -> np.ndarray:
    """[[시작초, 끝초], ...] -> 사이클 단위 불리언 마스크."""
    m = np.zeros(n_cycles, dtype=bool)
    for t0, t1 in spans or []:
        m[max(0, int(t0 * SAMPLING_HZ)):min(n_cycles, int(t1 * SAMPLING_HZ))] = True
    return m


def label_mask(stem: str, sel: dict, n_cycles: int, events: dict) -> np.ndarray:
    """사람 기록 라벨로 실측 구간을 자른다.

    전력 범위(`p_lo`/`p_hi`)로 자르는 것보다 정확하다. 특히 오븐은 `on` 이
    스위치 기준이라 팬/조명(15W)과 히터 통전(1.1kW)이 같은 구간에 들어 있다.
    히터는 `_heater_pulses` 로만 갈린다.

    sel:
        on    : 반드시 켜져 있어야 하는 SMPS. 나머지 SMPS 는 꺼져 있어야 한다.
        oven  : "off" | "fan" | "heater"  (None 이면 오븐을 조건에서 뺀다)
        hotplate_off : 기본 True. 핫플이 끼면 전압 환경이 달라진다.
    """
    iv = events[stem]["intervals"]

    def on_of(app: str) -> np.ndarray:
        return _spans_to_mask(iv.get(app, {}).get("on"), n_cycles)

    def unc_of(app: str) -> np.ndarray:
        return _spans_to_mask(iv.get(app, {}).get("uncertain"), n_cycles)

    want = set(sel.get("on", []))
    m = np.ones(n_cycles, dtype=bool)
    for app in SELECTABLE_SMPS:
        m &= on_of(app) if app in want else ~on_of(app)

    if sel.get("hotplate_off", True):
        m &= ~on_of("hotplate") & ~unc_of("hotplate")

    oven = sel.get("oven")
    if oven is not None:
        heater = _spans_to_mask(iv.get("oven", {}).get("_heater_pulses"), n_cycles)
        if oven == "heater":
            m &= heater
        elif oven == "fan":                      # 스위치는 켜졌지만 히터는 쉬는 중
            m &= on_of("oven") & ~heater & ~unc_of("oven")
        elif oven == "off":
            m &= ~on_of("oven") & ~unc_of("oven")
        else:
            raise ValueError(f"oven 은 off/fan/heater 중 하나여야 합니다: {oven!r}")
    return m


def synth_select_mask(sample: "object", sel: dict) -> np.ndarray:
    """합성 쪽에서 `label_mask` 와 같은 상태를 고른다.

    실측은 라벨, 합성은 정답(`gt_is_on` / `gt_target_power_w`)을 쓴다.
    오븐 히터 여부는 양쪽 다 전력으로 가르므로 기준이 일치한다.
    """
    n = int(sample.duration_cycles)
    m = np.ones(n, dtype=bool)
    for app in sel.get("on", []):
        if app in sample.gt_is_on:
            m &= sample.gt_is_on[app].astype(bool)
    oven = sel.get("oven")
    if oven in ("heater", "fan") and "oven" in sample.gt_target_power_w:
        p_oven = np.asarray(sample.gt_target_power_w["oven"], dtype=np.float64)
        m &= (p_oven >= OVEN_HEATER_W) if oven == "heater" else (
            (p_oven > 0.0) & (p_oven < OVEN_HEATER_W))
    return m


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
    pool: SegmentPool, synth: LoadSynthesizer, spec: dict, n_seeds: int = 20,
    events: Optional[dict] = None,
) -> Optional[dict]:
    # n_seeds 는 5 였는데 부족했다 (2026-08-23). 충전기는 활성화가 48.5분짜리
    # 충전 곡선이라 어느 구간이 잘리느냐에 따라 60초 창 중앙 전력이 31~74W 로
    # 흔들린다. 5시드로는 중앙값이 35.0W 로 나와 실측 43.7W 대비 -19.8% 로
    # 읽혔는데, 30시드에서는 50.0W 로 +14.6% 다. **부호까지 뒤집힌다.**
    # 프로젝터(폭 44~55W)처럼 좁은 기기는 5시드로도 안정적이라 안 드러났다.
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
    sel = spec.get("select")

    if sel is not None:
        # 라벨로 자른다. `events` 의 사이클 수와 npz 길이가 몇 사이클 어긋날 수
        # 있어 짧은 쪽에 맞춘다.
        ev = events if events is not None else load_events()
        n = min(int(ev[spec["file"]]["cycles"]), len(pf))
        pf, hc = pf[:n], hc[:n]
        mask = label_mask(spec["file"], sel, n, ev)
    else:
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
        if sel is not None:
            s_mask = synth_select_mask(sample, sel)
        else:
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
    syn_sig["V"] = float(np.median([t["V"] for t in trials]))
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
    for stem in ["test", "test.2", "test3", "test_4", "test_5", "test_6", "test_7"]:
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
    events = load_events()
    for spec in REAL_SEGMENTS:
        res = compare_segment(pool, synth, spec, events=events)
        if res is None:
            print(f"\n  [{spec['file']}] {spec['name']}: 표본 부족으로 건너뜀")
            continue
        results.append(res)
        how = "라벨" if spec.get("select") else "전력범위"
        print(f"\n  [{res['file']}] {res['name']}  ({', '.join(res['apps'])})  [{how}로 선택]")
        print(f"    실측 표본 {res['real']['_n']:,} 사이클 ({res['real']['_n']/SAMPLING_HZ:.0f}초)"
              f" | 실측 V={res['real']['V']:.1f}V, 합성 V={res['synth']['V']:.1f}V"
              f" | 환경 V0={res['real_v0']}V R={res['real_r_grid']}옴")
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

    # ── 4. 저항 부하가 SMPS 지문을 덮는 정도 ────────────────────────────────────────────────────
    # [3] 의 평균으로는 이 오차가 안 보인다. SMPS 단독(221V)에서는 고조파 형상이
    # 잘 맞아서(I7/I3 오차 -1.8%) 히터 통전 구간의 큰 오차(-18.8%)가 희석된다.
    # 그래서 같은 기기를 오븐 히터 OFF/통전 두 상태로 짝지어 **변화량**을 본다.
    #
    # 실측: 세 기기 전부 I7/I3 가 올라가 0.75~0.86 으로 뭉친다 (판별자 소멸).
    #       원인은 오븐 고조파(I3 0.012 / I5 0.057 / I7 0.073 A)의 선형 중첩이다.
    # 합성: 오븐 단독은 잘 맞으므로, 여기서 어긋나면 SMPS 쪽 크기 오차이거나
    #       중첩 위상이 틀린 것이다. 부호까지 반대면 실패로 본다.
    print("\n[4] 저항 부하가 SMPS 지문을 덮는 정도 - 오븐 히터 OFF/통전 짝 비교")
    by_pair: Dict[str, Dict[str, dict]] = {}
    idx = {(r["file"], r["name"]): r for r in results}
    for spec in REAL_SEGMENTS:
        key, st = spec.get("vpair"), spec.get("vstate")
        r = idx.get((spec["file"], spec["name"]))
        if key and st and r is not None:
            by_pair.setdefault(key, {})[st] = r

    vdrop_rows = {}
    print(f"  {'기기':24s}{'dV%':>8s}{'':4s}{'ΔI5/I3 (%)':>22s}{'ΔI7/I3 (%)':>22s}")
    print(f"  {'':24s}{'':8s}{'':4s}{'실측':>11s}{'합성':>11s}{'실측':>11s}{'합성':>11s}")
    for key, st in sorted(by_pair.items()):
        if "off" not in st or "heater" not in st:
            continue
        off, hot = st["off"], st["heater"]
        row = {"dv_pct_real": (hot["real"]["V"] - off["real"]["V"]) / off["real"]["V"] * 100}
        for k in ("I5_I3", "I7_I3"):
            for side in ("real", "synth"):
                a, b = off[side][k], hot[side][k]
                row[f"d{k}_{side}"] = (b - a) / abs(a) * 100 if abs(a) > 1e-9 else float("nan")
        # 부호가 갈리면 합성이 물리를 반대로 재현하는 것이다
        row["sign_ok"] = {
            k: bool(row[f"d{k}_real"] * row[f"d{k}_synth"] > 0
                    or abs(row[f"d{k}_real"]) < 5.0)
            for k in ("I5_I3", "I7_I3")
        }
        vdrop_rows[key] = row
        bad = "".join("" if v else "  <- 부호 반대" for v in row["sign_ok"].values())
        print(f"  {key:24s}{row['dv_pct_real']:>+8.2f}{'':4s}"
              f"{row['dI5_I3_real']:>+11.1f}{row['dI5_I3_synth']:>+11.1f}"
              f"{row['dI7_I3_real']:>+11.1f}{row['dI7_I3_synth']:>+11.1f}{bad}")
    if vdrop_rows:
        n_bad = sum(1 for r in vdrop_rows.values() for v in r["sign_ok"].values() if not v)
        print(f"\n  부호가 어긋난 항목 {n_bad} / {2 * len(vdrop_rows)}"
              f"   -> {'합격' if n_bad == 0 else '불합격 (중첩 재현 실패 — SMPS 크기 또는 위상)'}")

    # ── 5. 전압-고조파 결합 함수 직접 검사 (참고) ──────────────────────────────────
    # 같은 고조파 벡터 하나에 전압만 두 번 물려 함수의 응답만 본다. 결정적이다.
    #
    # **이 값에 판정 기준을 걸지 말 것.** 이 모델은 전 차수를 같은 배율로 곱하고
    # 3차만 보정하므로 형상이 -1.4% 밖에 안 움직인다. 한때 이것을 실측과
    # 어긋나는 결함으로 읽었는데 틀렸다 - 실측의 형상 변화는 전압이 아니라
    # 오븐 고조파의 중첩이었다 ([4] 주석 참조). 전압 응답이 필요하다는 증거는
    # 아직 없다. 기기별 응답이 없다는 사실만 눈에 보이게 남겨 둔다.
    print("\n[5] apply_cross_appliance_coupling 직접 검사 (참고, 판정 아님)")
    probe = np.zeros((2, 15), dtype=np.complex64)
    probe[:, 0], probe[:, 2], probe[:, 4], probe[:, 6] = 0.25, 0.21, 0.17, 0.135
    kappa_lo = 1.0 - 0.034                    # 실측 평균 전압강하 -3.4%
    coupling = {}
    print(f"  {'기기':18s}{'ΔI5/I3 (%)':>14s}{'ΔI7/I3 (%)':>14s}")
    for app in ("minipc", "beam_projector", "laptop_charger", "oven"):
        vals = {}
        for tag, k in (("hi", 1.0), ("lo", kappa_lo)):
            out = np.abs(gs.apply_cross_appliance_coupling(
                app, probe.copy(), np.full(2, k, dtype=np.float32))[0])
            vals[tag] = (out[4] / out[2], out[6] / out[2])
        d5 = (vals["lo"][0] - vals["hi"][0]) / vals["hi"][0] * 100
        d7 = (vals["lo"][1] - vals["hi"][1]) / vals["hi"][1] * 100
        coupling[app] = {"d_i5_i3_pct": round(float(d5), 3), "d_i7_i3_pct": round(float(d7), 3)}
        print(f"  {app:18s}{d5:>+14.2f}{d7:>+14.2f}")
    print("  세 SMPS 의 응답이 서로 같고 오븐(저항)은 0 이다 - 기기별 전압 응답이 없다.")
    print("  (실측에 전압 응답이 필요하다는 증거는 없다. 위 주석 참조.)")

    report = {
        "grid_impedance": grid_rows,
        "synthetic_r_grid_median": round(float(np.median(r_syn)), 3),
        "segments": results,
        "feature_summary": summary,
        "voltage_drop_response": vdrop_rows,
        "coupling_probe": coupling,
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
