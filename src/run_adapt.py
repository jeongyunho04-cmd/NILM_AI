"""
Phase 5 — 2단계 실측 준지도 적응 (설계 문서 4.2절)
====================================================
1단계(합성 사전학습)가 끝난 모델을 **기기별 라벨 없이** 실측 복합 부하에 맞춘다.

    L_adapt = L_cons(실측) + L_harm(실측)  +  λ · L_전체(합성)

고치려는 것은 0.6절의 sim-to-real 편차다. 합성 미니PC 17.6W vs 실측 28.4W 처럼
**저부하 기기가 체계적으로 17~34% 낮게 예측되는 것**을 실측 총전력으로 끌어올린다.
적응 전 v11 은 실측 총전력을 평균 −13.9W 로 과소 예측한다.

    python -m src.run_adapt --init results/cnn_v11.pt --tag adapt_v1

[왜 두 항을 함께 걸어야 하는가]
`L_cons` 는 합이 얼마나 모자란지만 알고 **어느 기기인지 모른다.** 실측에는 기기별
라벨이 아예 없으므로 `L_harm` 없이는 배분이 미결정인 채 아무 기기에나 붙는다.
이 두 항이 2단계의 분해를 통째로 떠받친다 (3.3·3.4·4.2절).

[왜 합성 손실을 함께 유지하는가]
파괴적 망각 방지다. λ=0.5 로 1단계 손실 전체를 계속 건다. 학습률은 1단계의 1/10.

[봉인]
`test.csv` 는 최종 평가 전용이라 적응에도 채점에도 쓰지 않는다 (4.3절).
`RealWindows` 가 목록에서 빼고, 채점도 봉인 해제 없이 도는 파일만 한다.
"""
from pathlib import Path
from typing import Optional, Sequence
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation import load_holdout, resistive_confusion, score_appliances, summarize, total_power_residual
from src.evaluation.power_ref import REFERENCE_W
from src.evaluation.real_events import load_events, score_absent, score_events, score_on_off
from src.evaluation.sealing import is_sealed
from src.model.losses import LossWeights, NILMLoss, build_state_scales
from src.model.inputs import FINE_CYCLES, TARGET_LOOKAHEAD, LEGACY_FINE_CHANNELS
from src.run_gate_check import assert_target_config
from src.model.net import (
    NILMNet, appliance_state_counts, harmonic_scales, harmonic_signatures,
    noise_reactive, reactive_signatures,
    noise_signature, standby_signatures,
)
from src.model.realdata import RealWindows, dense_targets, upsample_to_cycles
from src.model.traincache import CachedWindows
from src.run_baseline import LOW_LOAD, S_I
from src.run_train_cnn import evaluate, prepare_holdout_inputs, report, to_targets

HOLDOUT_DIR = "processed_data/holdout60"


#: SMPS 3종이 다 켜져도 150W 를 넘지 않는다 (프로젝터 48 + 충전기 45 + 미니PC 19).
SMPS_ONLY_W = 150.0


def real_sample_weights(p_observed, mode: str, boost: float = 4.0):
    """실측 창별 가중 (12.94절). 평균 1 로 정규화해 `w_cons` 의 뜻을 유지한다.

    `none` 은 None 을 돌려주어 예전 경로(단순 평균)를 그대로 탄다 — 기존
    체크포인트를 재현할 수 있어야 한다.
    """
    if mode == "none":
        return None
    p = p_observed.detach().float()
    if mode == "inv-power":
        w = 1.0 / p.clamp(min=SMPS_ONLY_W)
    elif mode == "smps-boost":
        w = torch.where(p < SMPS_ONLY_W, torch.full_like(p, float(boost)),
                        torch.ones_like(p))
    else:
        raise ValueError(mode)
    return w / w.mean().clamp(min=1e-6)


def real_targets(b, dev, human=None, qobs=None, hoff=None, vsc=None):
    """`human` 은 `RealWindows.human(idx)` 의 (on, mask). 없으면 기존과 같다.

    `qobs` 는 `RealWindows.reactive(idx)` — 무효전력 보존 항이 쓴다 (12.133).
    """
    fine, wide, pobs, oh, pn = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in b]
    tg = {"p_observed": pobs, "obs_harm": oh, "p_noise": pn}
    if qobs is not None:
        tg["q_observed"] = torch.from_numpy(np.ascontiguousarray(qobs)).to(dev)
    if hoff is not None:      # 교차주파수 어드미턴스 보정 (12.148)
        tg["harm_offset"] = torch.from_numpy(np.ascontiguousarray(hoff)).to(dev)
    if vsc is not None:       # h1 지문의 전압 보정 (12.151)
        tg["vscale"] = torch.from_numpy(np.ascontiguousarray(vsc)).to(dev)
    if human is not None:
        ho, hm = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in human]
        tg["human_on"], tg["human_mask"] = ho, hm
    return fine, wide, tg


@torch.no_grad()
def score_real_files(model, apps, dev, stride: int = 30) -> dict:
    """실측 파일별 채점 — 총전력 잔차 / on-off F1 / 이벤트 ΔP.

    12.4절: 실측에는 기기별 정답이 없으므로 MAE·FA·RE 는 못 낸다. 낼 수 있는 것만 낸다.
    창을 사이클마다 만들면 파일당 9만 창이라, 0.5초 간격 예측을 계단 보간한다.
    """
    model.eval()
    ev = load_events()
    out = {}
    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        rw = dense_targets(stem, stride=stride)
        P, ON = [], []
        for i in range(0, len(rw), 512):
            f, w, tg = real_targets(rw.batch(np.arange(i, min(i + 512, len(rw)))), dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = model(f, w)
            P.append(o["power"].float().cpu().numpy())
            ON.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
            resid = (o["power"].sum(1) + o["standby"].sum(1) + tg["p_noise"]
                     - tg["p_observed"]).float().cpu().numpy()
            out.setdefault(stem + "_resid", []).append(resid)
        P, ON = np.concatenate(P), np.concatenate(ON)
        r = np.concatenate(out.pop(stem + "_resid"))
        n_cycles = int(ev[stem]["cycles"])
        pc = upsample_to_cycles(P, rw.target_cycle, n_cycles)
        oc = upsample_to_cycles(ON > 0.5, rw.target_cycle, n_cycles)
        out[stem] = {
            "n_windows": len(rw),
            "residual_mean_w": float(r.mean()), "residual_abs_w": float(np.abs(r).mean()),
            "on_off": score_on_off(oc, stem, apps, events=ev),
            "events": score_events(pc, stem, apps, events=ev),
            # **2단계의 주 지표** — 없는 기기에 붙인 전력 (라벨 없이 잴 수 있다)
            "absent": score_absent(P, stem, apps, pred_on=ON > 0.5, s_i=S_I, events=ev),
        }
    model.train()
    return out


def summarize_real(rs: dict, only: Optional[Sequence[str]] = None) -> dict:
    """`only` 를 주면 그 파일들만 요약한다 (leave-one-file-out 채점용)."""
    if only is not None:
        rs = {k: v for k, v in rs.items() if k in set(only)}
    if not rs:
        return {k: float("nan") for k in
                ("residual_mean_w", "residual_abs_w", "on_off_f1_mean", "event_abs_rel_mean",
                 "absent_sum_w", "absent_share", "absent_fa_rel_max")} | {"n_scored_windows": 0}
    f1 = [v["f1"] for s in rs.values() for v in s["on_off"].values() if v["n_true_on"] > 0]
    er = [abs(e["error_rel"]) for s in rs.values() for e in s["events"] if e["error_rel"] == e["error_rel"]]
    n = sum(s["n_windows"] for s in rs.values())
    fa_rel = [v["fa_rel"] for s in rs.values() for v in s["absent"]["absent"].values()
              if v["fa_rel"] == v["fa_rel"]]
    return {
        "residual_mean_w": float(np.mean([s["residual_mean_w"] for s in rs.values()])),
        "residual_abs_w": float(np.mean([s["residual_abs_w"] for s in rs.values()])),
        "on_off_f1_mean": float(np.mean(f1)) if f1 else float("nan"),
        "event_abs_rel_mean": float(np.mean(er)) if er else float("nan"),
        # 없는 기기에 붙인 전력 — 2단계 주 지표
        "absent_sum_w": float(np.mean([s["absent"]["absent_sum_w"] for s in rs.values()])),
        "absent_share": float(np.mean([s["absent"]["absent_share"] for s in rs.values()])),
        "absent_fa_rel_max": float(np.max(fa_rel)) if fa_rel else float("nan"),
        "n_scored_windows": int(n),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5 — 2단계 실측 준지도 적응 (4.2절)")
    # 기본값은 12.9.5절 "확정된 것" 표와 일치시킨다. 4.2절이 적었던 첫 제안값
    # (w_cons 0.4 / w_harm 0.1 / w_hedge 0)은 **스윕 6개 조합 중 최악**이었다
    # (12.12.2절: L_cons 가 L_harm 보다 38.8배 강해 배분을 결정하지 못한다).
    # 기본값으로 남겨 두면 아무 옵션 없이 돌린 사람이 그 조합을 얻는다.
    ap.add_argument("--real-weight", default="none",
                    choices=("none", "inv-power", "smps-boost"),
                    help="실측 창별 가중 (12.94절). 창 하나당 기울기 노름이 고부하 창에서 "
                         "25배 크고, SMPS 만 있는 창은 개수 58%% 인데 기울기로는 10%% 미만이다. "
                         "inv-power = w ∝ 1/max(P_관측, 150W) (스케일을 직접 상쇄), "
                         "smps-boost = P<150W 창만 --smps-boost 배 (표본 가중과 같은 효과). "
                         "기본 none 은 이전과 동일하다")
    ap.add_argument("--smps-boost", type=float, default=4.0,
                    help="--real-weight smps-boost 의 배수")
    ap.add_argument("--harm-odd-only", action="store_true",
                    help="L_harm 에서 짝수차를 뺀다 (12.75절 — 계획만 있던 절이고 실행 기록은 "
                         "12.78, 단일 변수 재측정은 12.75.5). **2단계가 실측에서 도는 것이므로 "
                         "1단계보다 이쪽이 더 중요하다**")
    ap.add_argument("--init", default="results/cnn_v17.pt",
                    help="1단계 체크포인트. v17 은 듀티 주기 무작위화로 다시 학습한 것이고 "
                         "1단계만으로 v15 의 2단계 결과를 앞선다 (12.17절)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256, help="실측/합성 각각의 배치 크기")
    ap.add_argument("--lr", type=float, default=3e-5, help="1단계(3e-4)의 1/10")
    ap.add_argument("--lam", type=float, default=0.5, help="합성 replay 가중 (4.2절)")
    ap.add_argument("--w-cons", type=float, default=0.1,
                    help="실측 보존 손실. 4.2절 제안값 0.4 는 L_harm 을 압도해 배분을 "
                         "방치한다 (12.12.2절)")
    ap.add_argument("--w-harm", type=float, default=4.0,
                    help="실측 고조파 제약. 배분을 결정하는 유일한 항이라 0.1 로는 "
                         "발언권이 2.5%% 밖에 안 된다 (12.12.2절)")
    ap.add_argument("--w-over", type=float, default=0.0)
    ap.add_argument("--w-hedge", type=float, default=0.2,
                    help="게이트 헤지 벌점 (12.12.3절). 실측은 라벨이 없어 BCE 가 "
                         "확신을 강제하지 못한다. 이진 엔트로피로 결정을 요구한다")
    ap.add_argument("--holdout-real", default="",
                    help="적응에서 뺄 실측 파일 (쉼표 구분, 예: test_4). 뺀 파일도 "
                         "채점은 하므로 실측 홀드아웃이 된다. 12.12 의 가중치가 "
                         "튜닝 잔향인지 확인하는 용도")
    ap.add_argument("--real-stride", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--harm-grad-balance", default="off",
                    choices=("off", "smps", "all"),
                    help="L_harm 의 기울기를 기기별로 균등화한다 (12.120). "
                         "값은 안 바뀌고 기울기만 바뀐다. smps 는 SMPS 3종 "
                         "안에서만, all 은 9종 전부. 기본 off")
    # ── 사람 스위칭 로그 지도 (SMPS_PLAN 4.5절) ─────────────────────────
    ap.add_argument("--w-real-on", type=float, default=0.0,
                    help="사람 스위칭 로그 on/off 를 2단계에 거는 무게 (기본 0 = 끔). "
                         "test_5/6/7/8/13 의 human_switching_log 만 쓴다. 전력은 "
                         "감독하지 않는다 - 로그가 on/off 만 주기 때문 (SMPS_PLAN 4.5)")
    ap.add_argument("--real-on-scope", default="smps",
                    choices=("smps", "present", "all"),
                    help="어느 열을 감독할지. smps=SMPS 3종만(가설 그대로), "
                         "present=그 파일에 있던 기기만, all=9종 전부(없는 기기=OFF). "
                         "규칙 3 - 유령 억제와 SMPS 분해를 섞지 않으려고 가른다")
    ap.add_argument("--human-label-shuffle", action="store_true",
                    help="**귀무 대조.** 라벨 시간축을 순환 이동해 ON 비율·구간 "
                         "길이는 보존하고 시각 대응만 깬다. 여기서도 같은 이득이 "
                         "나오면 라벨이 아니라 BCE 항의 정규화 효과다 (규칙 3)")
    ap.add_argument("--human-label-files", default="",
                    help="지도에 쓸 파일을 직접 지정 (쉼표). 비우면 SMPS 가 든 "
                         "사람 라벨 5파일. **test_11/12 를 넣으면 규칙 20 대조가 죽는다**")
    ap.add_argument("--harm-weight", default="off",
                    choices=("off", "inv_h", "inv_h2", "inv_tau"),
                    help="`L_harm` 의 **차수별 신뢰도 가중** (12.135). `harm_scale` 이 "
                         "15차수를 균등화하는데, 실측에서는 고차가 신호가 아니라 "
                         "모델오차다. 모델오차/판별신호가 균등 2.23 -> 1/h 1.74 -> "
                         "**inv_h2 1.57** 로 단조로 준다. 바닥은 1.41(h1,h3)인데 "
                         "거기선 유효차원이 4 라 식별성이 죽는다. 기본 off = 이전과 동일")
    ap.add_argument("--w-consq", type=float, default=0.0, metavar="W",
                    help="**무효전력 보존 항** (12.133). `L_cons` 와 같은 꼴로 P 대신 "
                         "Q 를 맞춘다. 저항에는 등가저항(`resistive_match`)이라는 둘째 "
                         "판별자가 있는데 SMPS 에는 없어서 배분이 `L_harm` 하나에 걸려 "
                         "있었고, 12.122.2 가 그 항의 최소는 **오답 쪽**이라고 확정했다. "
                         "`Q/P` 는 SMPS 를 고조파보다 2.2~2.5배 잘 가른다. 0 이면 끔")
    ap.add_argument("--harm-deadzone", type=float, default=0.0, metavar="X",
                    help="L_harm 불감대 배수 (12.122.16). 정답 배분에서도 남는 "
                         "차수별 잔차의 X배까지는 벌하지 않는다. **줄일 수 없는 "
                         "잔차의 70%%를 줄이라고 밀어서 배분이 밀린다** — 그것을 "
                         "끊는다. 1.0 이 측정된 중앙값. 0 이면 끔(이전과 동일). "
                         "⚠ 너무 키우면 L_harm 이 죽어 '합만 맞추는 해' 로 간다 (12.12.2)")
    ap.add_argument("--w-pref", type=float, default=0.0, metavar="W",
                    help="**전력 사전** (12.145). 격리 녹화에서 통전 전력이 좁은 "
                         "기기(`power_ref.REFERENCE_W`)의 전력을 그 참값에 묶는다. "
                         "12.144.2 가 잰 것 — 저항 없는 창에서 프로젝터가 82.8W 인데 "
                         "참값은 46.9W 이고 **실측 손실의 최적점마저 60.0W** 다. "
                         "SMPS 3종 지문이 11.9도 안에 몰려 있어(cos 0.979) 고조파로는 "
                         "못 가르고, 그 안에서 프로젝터가 와트당 전류가 가장 작아 "
                         "**가장 싼 배출구**가 된다. 규칙 35 대로 기울기가 아니라 "
                         "값을 만진다. 최적점 훑기가 0.02 면 47.5W 로 옮겨지고 그 위로 "
                         "포화한다고 쟀다. **사람 라벨이 아니다** — 기기별 격리 녹화 "
                         "상수다 (인수인계의 `--snap` 과 같은 등급). 0 이면 끔")
    ap.add_argument("--harm-offset", default="", metavar="NPZ",
                    help="**교차주파수 어드미턴스 보정** (12.148, `run_norton_probe --save-coef` 의 "
                         "산출물). `harmonic_signatures` 는 fixed current injection "
                         "모형이라 기기 전류가 계통 조건과 무관하다고 보는데, 문헌이 "
                         "그 실패를 오래 전에 적었다 (attenuation & diversity). 표준 "
                         "처방은 Norton 등가 `I_h = I_source,h − Y_h·V_h` 이고 **여러 "
                         "차수의 전압**이 한 차수의 전류에 든다. 12.148 이 실측에서 "
                         "적합해 배분 오차가 파일 홀드아웃에서 39.4 -> 14.3W. "
                         "⚠ 원자료 CSV 의 `vh1~vh15` 를 읽는다 — 전처리가 그것을 버린다")
    ap.add_argument("--harm-offset-z", default="", metavar="NPZ",
                    help="`--harm-offset` 의 계수는 그대로 두고 **계통 임피던스만** "
                         "현장 값으로 갈아끼운다 (`run_fit_impedance --out` 의 산출물). "
                         "Z 는 그 집 배선이라 장소 간 3.2배까지 다르고 **라벨 없이** "
                         "11초에 다시 잰다. 기울기는 `ΣY`(기기 속성)라 기기 구성이 "
                         "같으면 전이될 것으로 보지만 **확인 안 됐다**. "
                         "⚠ 상수항은 학습 장소 것이 남는다 (배경 V_src 효과)")
    ap.add_argument("--pref-apps", default="beam_projector", metavar="LIST",
                    help="전력 사전을 걸 기기 (쉼표). **기본이 프로젝터 하나인 이유** — "
                         "`REFERENCE_W` 에는 포트·핫플·오븐도 있는데 저항을 참값에 "
                         "못 박는 것은 12.143 이 **이미 닫았다** (유령 2.5~20배). "
                         "규칙 3 — 절제 그룹은 가설을 가르도록 짠다")
    ap.add_argument("--sig-insitu", default="", metavar="NPZ",
                    help="`L_harm` 의 지문을 in-situ 적합본으로 갈아끼운다 "
                         "(12.122.11, run_fit_insitu_sig 의 산출물). 비우면 격리 지문. "
                         "**LOFO 로 검증했지만 사람 라벨 5파일에서 적합한 것이라, "
                         "그 파일을 --holdout-real 로 빼도 지문에는 남아 있다**")
    ap.add_argument("--harm-vnorm", action="store_true",
                    help="h1 지문에서 **녹화 전압**을 나눈다 (12.151.1). 지문의 h1 "
                         "실수부 역수가 그 격리 녹화의 선전압이라는 항등식에서 "
                         "나온다 — 기기 간 11.8% 의 가짜 판별자를 지운다")
    ap.add_argument("--harm-vnorm-both", action="store_true",
                    help="정규화를 **합성 갈래에도** 건다. 캐시의 `obs_harm` 은 원래 "
                         "지문으로 합성한 것이라 틀린 값이 된다 — **대조용**이다")
    ap.add_argument("--harm-vscale", type=float, default=0.0, metavar="X",
                    help="L_harm 의 h1 지문을 창별 전압으로 보정 (12.151). "
                         "1.0 이 물리값 — 와트당 h1 전류가 1/V_rms 라는 항등식이다. "
                         "기준 전압은 적응 창의 중앙값이라 평균 배율이 1 이고, "
                         "**부하와 상관된 변동만** 새 정보로 들어간다 "
                         "(그냥 지문을 상수배 한 것과 구별하기 위해서다)")
    ap.add_argument("--cache", default="cache/train60")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="adapt")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    env_guard.verify_numerics()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 84)
    print("[Phase 5] 2단계 실측 준지도 적응 — 기기별 라벨 없이 (4.2절)")
    print("=" * 84)

    hs = load_holdout(HOLDOUT_DIR)
    apps = hs.appliances
    prep = prepare_holdout_inputs(hs)
    hs.X = np.zeros((len(prep[0]), 1, 1), np.float32)

    held = [x.strip() for x in a.holdout_real.split(",") if x.strip()]
    hl = [x.strip() for x in a.human_label_files.split(",") if x.strip()]
    rw = RealWindows(stride=a.real_stride, exclude=held or None,
                     appliances=apps if a.w_real_on > 0 else None,
                     human_on_scope=a.real_on_scope if a.w_real_on > 0 else "off",
                     human_on_stems=hl or None,
                     human_on_shuffle=a.human_label_shuffle)
    adapted = list(rw.stems)
    print(rw.describe())
    if a.w_real_on > 0:
        print(f"  ** 사람 라벨 지도 켜짐: w_real_on={a.w_real_on:g} "
              f"scope={a.real_on_scope} (SMPS_PLAN 4.5) **"
              + ("\n  ** 라벨 순환 이동 (귀무 대조) **" if a.human_label_shuffle else ""))
        print(rw.human_coverage())
    if held:
        print(f"  ** 실측 홀드아웃: {', '.join(held)} - 적응에 안 쓰고 채점만 한다 **")
    print(f"합성 replay: {a.cache} | λ={a.lam} | w_cons={a.w_cons} w_harm={a.w_harm}")
    if a.real_weight != "none":
        extra = f" x{a.smps_boost:g}" if a.real_weight == "smps-boost" else ""
        print(f"  ** 실측 창별 가중: {a.real_weight}{extra} (12.94절) **")

    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig, sb, nz, hsc = (harmonic_signatures(pool, apps), standby_signatures(pool, apps),
                        noise_signature(pool), harmonic_scales(pool, apps))
    qp, qp_ok = reactive_signatures(pool, apps)
    nq = noise_reactive(pool)
    del pool

    if a.w_consq > 0:
        # 규칙 14 — 검증된 상수와 아닌 것을 갈라 찍는다. `usable` 은 창 폭과
        # **녹화 간** 일치를 둘 다 통과한 것이다 (`reactive_signatures` 주석).
        good = ", ".join(f"{apps[j]} {qp[j]:+.3f}" for j in range(len(apps)) if qp_ok[j])
        bad = ", ".join(f"{apps[j]} {qp[j]:+.3f}" for j in range(len(apps)) if not qp_ok[j])
        print(f"  ** 무효전력 보존 켜짐: w_consq={a.w_consq:g}, 계측 Q={nq:+.3f} VAR **")
        print(f"     검증된 Q/P : {good}")
        print(f"     ⚠ 미검증   : {bad}  (중앙값을 그대로 쓴다 — 12.133 주석)")

    if a.sig_insitu:
        # ── in-situ 지문 (12.122.11) ────────────────────────────────────
        # 격리 녹화가 아니라 **복합 파일에서** 푼 지문이다. 12.122.10 이
        # 격리->복합 전이 실패를 확정했고, 이쪽은 그 전이를 아예 건너뛴다.
        z = np.load(a.sig_insitu, allow_pickle=True)
        if list(z["appliances"]) != list(apps):
            raise SystemExit(f"{a.sig_insitu} 의 기기 목록이 다릅니다")
        new = np.asarray(z["sig"], np.float32)
        d = np.abs(new - sig).max()
        sig = new
        print(f"  ** in-situ 지문: {a.sig_insitu} (격리 대비 최대 차 {d:.5f} A/W) **")

    # ── h1 지문의 녹화전압 정규화 (12.151.1) ──────────────────────────────
    # `Re(I₁)/P = 1/V₁` 은 유효전력의 **정의**다 (12.151). 그러면 지문의 h1 실수부
    # 역수 `1/Re(sig[k,1,0])` 은 기기의 성질이 아니라 **그 격리 녹화의 선전압**이다.
    # 실제로 격리 녹화 전압과 몇 V 안에서 맞는다:
    #
    #     포트 212.9(녹화 213.5)  오븐 215.7(216.0)  핫플 217.7(219.8)
    #     프로젝터 224.4(223.4)   드라이기 234.5(228.7)  에어컨 238.1(232.9)
    #
    # 즉 손실은 기기 사이에 **11.8% 의 가짜 판별자**를 갖고 있다. 같은 드라이기를
    # 217.7V 와 233.8V 에서 찍은 두 녹화가 와트당 7.4% 다른 것이 그 증거다.
    # 여기서 그 편차를 나눠 버리면 전 기기의 h1 실수부가 `1/V_ref` 로 같아진다 —
    # **정의상 같아야 하는 값이다.** 허수부(변위)는 손대지 않는다.
    SIG_REAL = None
    if a.harm_vnorm:
        vref0 = float(np.median(np.asarray(rw.v_observed, np.float64)))
        vk = 1.0 / np.maximum(sig[:, 0, 0].astype(np.float64), 1e-9)   # 함의 전압
        f = (vk / vref0).astype(np.float32)
        SIG_REAL = sig.copy(); SIG_REAL[:, 0, :] *= f[:, None]
        if a.harm_vnorm_both:
            # ⚠ 합성 갈래에도 건다 — **캐시의 `obs_harm` 은 원래 지문으로 합성한
            #    것**이라 여기에는 정규화가 틀린 값이다. 대조용으로만 둔다.
            sig = SIG_REAL; SIG_REAL = None
        print(f"  ** h1 녹화전압 정규화: V_ref={vref0:.1f}V "
              f"({'양쪽 갈래 — 대조' if a.harm_vnorm_both else '실측 갈래만'}, 12.151.1) **")
        print("     " + "  ".join(f"{apps[j][:6]} {vk[j]:.0f}V(x{f[j]:.3f})"
                                  for j in range(len(apps))))

    _pref_set = {x for x in a.pref_apps.split(",") if x}
    if a.w_pref > 0:
        good = [f"{x} {REFERENCE_W[x][0]:.1f}W" for x in apps
                if x in _pref_set and x in REFERENCE_W]
        miss = sorted(_pref_set - set(REFERENCE_W))
        print(f"  ** 전력 사전 켜짐: w_pref={a.w_pref:g} | {', '.join(good)} **")
        if miss:
            print(f"     ⚠ 참값이 없어 뺀 기기: {miss} (격리 통전 폭이 넓다)")

    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    assert_target_config(ck, a.init)   # 12.45.3
    model = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                    wide_summary=ck.get("wide_summary", False),
                    periodicity=ck.get("periodicity", False),
                    fine_dropout=ck.get("fine_dropout", 0.0),
                    prior_kappa=ck.get("prior_kappa", 0.0),
                    prior_beta=ck.get("prior_beta", 0.5),
                    fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS)).to(dev)
    model.load_state_dict(ck["model"])
    print(f"1단계 체크포인트: {a.init} (ep{ck.get('epoch')}, width {ck.get('width')})")

    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig), standby_sig=torch.from_numpy(sb),
        noise_sig=torch.from_numpy(nz), harm_scale=torch.from_numpy(hsc),
        harm_odd_only=a.harm_odd_only,
        harm_grad_balance=a.harm_grad_balance,
        harm_deadzone=a.harm_deadzone, harm_weight=a.harm_weight,
        reactive_qp=torch.from_numpy(qp), noise_q=nq,
        smps_group=[apps.index(x) for x in
                    ("beam_projector", "laptop_charger", "minipc") if x in apps],
        weights=LossWeights(harm=0.1, cons=0.0, over=0.0),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
        power_ref=torch.tensor(
            [REFERENCE_W[x][0] if (x in REFERENCE_W and x in _pref_set) else 0.0
             for x in apps], dtype=torch.float32),
        sig_real=(None if SIG_REAL is None else torch.from_numpy(SIG_REAL)),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    cache = CachedWindows(a.cache)
    # ── 교차주파수 어드미턴스 보정 (12.148) — 창마다 상수라 미리 다 만든다 ──
    HOFF = None
    if a.harm_offset:
        from src.model.realdata import harmonic_offset
        HOFF = harmonic_offset(rw.stem, rw.target_cycle, a.harm_offset,
                               z_npz=a.harm_offset_z)
        z = np.load(a.harm_offset, allow_pickle=True)
        nz_ = float(np.abs(HOFF).max())
        print(f"  ** 교차주파수 보정: {a.harm_offset} "
              f"(적합 {list(z['stems'])}, 뺀 파일 {list(z['excluded'])}) **")
        if a.harm_offset_z:
            zz = np.load(a.harm_offset_z, allow_pickle=True)
            print(f"     계통 임피던스 교체: {a.harm_offset_z}  "
                  f"R {float(z['R']):.3f} -> {float(zz['R']):.3f} Ω,  "
                  f"L {float(z['X1']) / (2 * np.pi * 60) * 1e6:.0f} -> "
                  f"{float(zz['X1']) / (2 * np.pi * 60) * 1e6:.0f} µH")
            print(f"     ⚠ 상수항은 학습 장소 것이 남는다 (배경 V_src 효과, 12.150.1)")
        print(f"     창 {len(HOFF)}개, 최대 보정 {nz_ * 1000:.1f} mA, "
              f"h1 중앙 {np.median(np.linalg.norm(HOFF[:, 0], axis=1)) * 1000:.1f} mA")
    # ── h1 지문의 전압 보정 (12.151) — 이것도 창마다 상수다 ────────────────
    # 기준을 **적응 창의 중앙 전압**으로 잡는다. 그러면 배율의 중앙이 정확히 1 이라
    # "지문을 상수배 한 것" 과 구별된다 (12.135 가 `harm_weight` 에 건 것과 같은
    # 판정 기준이다). 남는 것은 부하와 상관된 변동뿐이다.
    VSC = None
    if a.harm_vscale > 0:
        v = np.asarray(rw.v_observed, np.float32)
        vref = float(np.median(v))
        VSC = (1.0 + a.harm_vscale
               * (vref / np.clip(v, 1.0, None) - 1.0)).astype(np.float32)
        print(f"  ** h1 전압 보정: x{a.harm_vscale:g} (12.151) **")
        print(f"     V_rms {v.min():.1f} ~ {v.max():.1f}V (중앙 {vref:.1f}), "
              f"배율 {VSC.min():.4f} ~ {VSC.max():.4f} (중앙 {np.median(VSC):.4f})")
    rng = np.random.default_rng(a.seed)

    def snapshot(tag: str) -> dict:
        pred, onp = evaluate(model, prep, dev)
        sc, summ, cm, resid = report(pred, onp, hs)
        rs = score_real_files(model, apps, dev)
        rsum = summarize_real(rs)
        print(f"\n  [{tag}] 합성 홀드아웃  MAE {summ['mae_w_mean']:.3f}W  F1 {summ['f1_mean']:.4f}  "
              f"저항3종 {cm['accuracy']:.3f}  잔차 {resid['mean_abs_w']:.1f}W")
        print(f"  {'':{len(tag)+4}s}실측          잔차 평균 {rsum['residual_mean_w']:+.2f}W  "
              f"절대 {rsum['residual_abs_w']:.2f}W  on/off F1 {rsum['on_off_f1_mean']:.3f}  "
              f"이벤트 |상대오차| {rsum['event_abs_rel_mean']:.3f}")
        print(f"  {'':{len(tag)+4}s}**오귀속**    없는 기기 합계 {rsum['absent_sum_w']:.1f}W "
              f"(예측 총합의 {100*rsum['absent_share']:.1f}%)  최악 FA_rel {rsum['absent_fa_rel_max']:.3f}")
        out = {"tag": tag, "synth": {"mae": summ["mae_w_mean"], "f1": summ["f1_mean"],
                                     "resistive_acc": cm["accuracy"],
                                     "resid_abs": resid["mean_abs_w"]},
               "real": rsum, "real_detail": rs,
               "summary": summ, "resistive_confusion": cm, "total_power_residual": resid,
               "per_appliance": [x.as_row() for x in sc]}
        if held:
            # **판정은 여기를 본다.** 위 `real` 은 적응에 쓴 파일이 섞여 있다.
            out["real_adapted"] = summarize_real(rs, only=adapted)
            out["real_heldout"] = summarize_real(rs, only=held)
            pad = " " * (len(tag) + 4)
            for lab, key in (("적응셋", "real_adapted"), ("**홀드아웃**", "real_heldout")):
                v = out[key]
                print(f"  {pad}{lab:12s}오귀속 {v['absent_sum_w']:6.1f}W  "
                      f"잔차 {v['residual_abs_w']:6.2f}W  on/off F1 {v['on_off_f1_mean']:.3f}  "
                      f"최악 FA_rel {v['absent_fa_rel_max']:.3f}")
        return out

    hist = [snapshot("적응 전")]
    t0 = time.time()
    agg, nb = {}, 0
    for step in range(1, a.steps + 1):
        ridx = rng.choice(len(rw), a.batch, replace=len(rw) < a.batch)
        rf, rwd, rtg = real_targets(rw.batch(ridx), dev,
                                    rw.human(ridx) if a.w_real_on > 0 else None,
                                    rw.reactive(ridx) if a.w_consq > 0 else None,
                                    HOFF[ridx] if HOFF is not None else None,
                                    VSC[ridx] if VSC is not None else None)
        sidx = np.sort(rng.choice(len(cache), a.batch, replace=False))
        sb_ = tuple(torch.from_numpy(x) for x in cache.batch(sidx))
        sf, swd, stg = to_targets(sb_, dev)

        sw = real_sample_weights(rtg["p_observed"], a.real_weight, a.smps_boost)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            rp = crit.unlabeled(model(rf, rwd), rtg, w_cons=a.w_cons,
                                w_harm=a.w_harm, w_over=a.w_over, w_hedge=a.w_hedge,
                                sample_w=sw, w_real_on=a.w_real_on,
                                w_consq=a.w_consq, w_pref=a.w_pref)
            sp = crit(model(sf, swd), stg)
            loss = rp["total"] + a.lam * sp["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        for k, v in (("real_cons", rp["cons"]), ("real_harm", rp["harm"]),
                     ("real_consq", rp["consq"]),
                     ("real_hedge", rp["hedge"]), ("real_on", rp["real_on"]),
                     ("synth_total", sp["total"]), ("loss", loss)):
            d = v.detach()
            agg[k] = d if k not in agg else agg[k] + d
        nb += 1

        if step % a.eval_every == 0 or step == a.steps:
            m = {k: float(v) / nb for k, v in agg.items()}
            print(f"\n  step {step:>5d}/{a.steps}  loss {m['loss']:.4f} "
                  f"(실측 cons {m['real_cons']:.2f} harm {m['real_harm']:.3f} "
                  f"hedge {m['real_hedge']:.3f}"
                  + (f" consQ {m['real_consq']:.2f}" if a.w_consq > 0 else "")
                  + (f" on {m['real_on']:.3f}" if a.w_real_on > 0 else "") + f" / "
                  f"합성 {m['synth_total']:.4f})  [{time.time()-t0:.0f}s]", flush=True)
            hist.append(snapshot(f"step {step}"))
            agg, nb = {}, 0

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "appliances": apps,
                "width": ck.get("width", 1.0), "epoch": a.steps,
                "prior_kappa": ck.get("prior_kappa", 0.0),
                "prior_beta": ck.get("prior_beta", 0.5),
                "wide_summary": ck.get("wide_summary", False),
                "periodicity": ck.get("periodicity", False),
                "fine_dropout": ck.get("fine_dropout", 0.0),
                # 짝수차 배제 (12.77). 부모 체크포인트 값을 그대로 물려받는다.
                "zero_even_harmonics": ck.get("zero_even_harmonics", False),
                "fine_channels": model.fine_channels,
                "target_lookahead": TARGET_LOOKAHEAD, "fine_cycles": FINE_CYCLES,
                "select": "final", "stage": 2, "init": a.init},
               out / f"{a.tag}.pt")

    first, last = hist[0], hist[-1]
    print(f"\n{'=' * 84}\n적응 전 -> 후\n{'=' * 84}")
    print(f"  {'':26s}{'전':>12s}{'후':>12s}")
    for lab, kk in [("**실측 오귀속 합계 (W)**", ("real", "absent_sum_w")),
                    ("**오귀속 비중**", ("real", "absent_share")),
                    ("**최악 FA_rel**", ("real", "absent_fa_rel_max")),
                    ("실측 잔차 평균 (W)", ("real", "residual_mean_w")),
                    ("실측 잔차 절대 (W)", ("real", "residual_abs_w")),
                    ("실측 on/off F1", ("real", "on_off_f1_mean")),
                    ("실측 이벤트 |상대오차|", ("real", "event_abs_rel_mean")),
                    ("합성 MAE (W)", ("synth", "mae")),
                    ("합성 F1", ("synth", "f1")),
                    ("합성 저항3종", ("synth", "resistive_acc")),
                    ("합성 잔차 (W)", ("synth", "resid_abs"))]:
        g, k = kk
        print(f"  {lab:26s}{first[g][k]:>12.4f}{last[g][k]:>12.4f}")

    if held:
        # 위 표는 적응에 쓴 파일이 섞여 있다. 일반화 판정은 이 표를 본다.
        bar = "-" * 84
        print("")
        print(bar)
        print(f"실측 홀드아웃 ({', '.join(held)}) - 적응에 쓰지 않은 파일")
        print(bar)
        print(f"  {'':26s}{'전':>12s}{'후':>12s}")
        for lab, k in [("**오귀속 합계 (W)**", "absent_sum_w"),
                       ("**최악 FA_rel**", "absent_fa_rel_max"),
                       ("잔차 절대 (W)", "residual_abs_w"),
                       ("on/off F1", "on_off_f1_mean"),
                       ("이벤트 |상대오차|", "event_abs_rel_mean")]:
            print(f"  {lab:26s}{first['real_heldout'][k]:>12.4f}{last['real_heldout'][k]:>12.4f}")

    payload = {"phase": "5-adapt", "config": vars(a),
               "adapted_on": adapted, "held_out": held,
               "holdout": {k: hs.meta[k] for k in ("n_windows", "content_sha256", "target_index")},
               "history": [{k: v for k, v in h.items() if k != "real_detail"} for h in hist],
               "real_detail": last["real_detail"],
               "per_appliance": last["per_appliance"], "summary": last["summary"],
               "resistive_confusion": last["resistive_confusion"],
               "total_power_residual": last["total_power_residual"],
               "elapsed_s": round(time.time() - t0, 1)}
    (out / f"{a.tag}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n완료 {time.time()-t0:.0f}s | {out / f'{a.tag}.json'}")
    print("=" * 84 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
