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
from src.evaluation.power_ref import REFERENCE_W, REFERENCE_W_STEMWISE
from src.evaluation.real_events import load_events, score_absent, score_events, score_on_off
from src.evaluation.sealing import is_sealed
from src.model.losses import LossWeights, NILMLoss, build_state_scales
from src.model.postproc import HALFWAVE_OHM, RESISTIVE_OHM
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


def real_targets(b, dev, human=None, qobs=None, hoff=None, vsc=None, vrms=None,
                 prefm=None):
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
    if vrms is not None:      # 단자 전압. `L_res` 가 P = V²/R 을 푼다 (12.156)
        tg["v_rms"] = torch.from_numpy(np.ascontiguousarray(vrms)).to(dev)
    if prefm is not None:     # 창별 참값 마스크 (12.159)
        tg["pref_mask"] = torch.from_numpy(np.ascontiguousarray(prefm)).to(dev)
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


def _cache_says_background(cache: str) -> bool:
    """캐시 meta 의 `background` 플래그. 없거나 못 읽으면 False."""
    if not cache or str(cache).lower() == "none":
        return False
    try:
        return bool(json.load(open(Path(cache) / "meta.json",
                                   encoding="utf-8")).get("background", False))
    except Exception:
        return False


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
    ap.add_argument("--w-res", type=float, default=0.0, metavar="W",
                    help="**저항 컨덕턴스 정합** `L_res` (12.156). 니크롬선은 "
                         "P = V^2/R 이고 R 이 기기 고유값이라(같은 기기 다른 녹화에서 "
                         "0.1~1.3%%) 저항 몫을 컨덕턴스로 옮기면 조합을 셀 수 있다. "
                         "포트 35.8Ω 과 오븐 40.6Ω 은 222V 에서 163W 벌어지는데 "
                         "`L_harm` 은 그 둘을 h1(=전력) 으로만 가른다 — 판별 기여 "
                         "18.27 중 17.38(97.6%%)이 h1 이고 모양은 4.9%% 다. 그래서 "
                         "12.155.6 의 반파 채널이 포트 1,209W 를 **장소 B 에 없는 "
                         "오븐**에게 넘겼다. 잃은 창에서 16조합 최적을 세면 포트가 "
                         "93/88/93%% 이고 오븐은 상위 4위에 한 번도 없다 — 정보는 "
                         "깨끗한데 모델에게 준 적이 없어 `resistive_match` 후처리가 "
                         "뒤늦게, 그것도 절반만 고치고 있었다 (포트 F1 0.403 -> 0.767). "
                         "**포트·오븐에만 건다** (핫플은 장소 B 에서 230~240W 로 참조 "
                         "460W 와 다르고, 드라이기는 강 54.3Ω / 약 108.6Ω 로 상태마다 "
                         "다르다 — 규칙 14).")
    ap.add_argument("--w-swap", type=float, default=0.0, metavar="W",
                    help="**저항 조합 맞바꿈 감독** `L_swap` (12.158). `L_res` 는 "
                         "`σ(on)` 을 곱해 걸리므로 게이트가 바닥이면 안 닿는다 — "
                         "12.157.4b 가 그것을 확정했다 (포트 σ중앙 0.0344 -> 효과 "
                         "+0.563, 핫플 0.0033 -> +0.009, 드라이기 강풍 0.0001 -> "
                         "−0.011. 완전한 단조이고 유일한 차이가 게이트다). 이 항은 "
                         "`σ` 를 안 거치고 **로짓에 직접** BCE 를 건다. 무엇을 "
                         "가르칠지는 `resistive_match`(12.112) 처럼 컨덕턴스 조합을 "
                         "세서 정하므로 **라벨이 필요 없다**. 12.112 의 제한을 "
                         "그대로 쓴다 — 개수는 안 바꾸고 맞바꿈만, tol 밖은 안 건드림. "
                         "여기에 하나 더: `best==현재` 인 창은 감독하지 않는다 "
                         "(후처리와 달리 손실은 1,000 스텝을 밀므로 틀린 결정이 굳는다).")
    ap.add_argument("--swap-tol", type=float, default=0.02, metavar="TOL",
                    help="`L_swap` 의 상대오차 문턱. 12.112.3 이 0.02 를 최적으로 "
                         "쟀다 (0.01 은 아무것도 안 고치고 0.05 이상은 엉뚱한 조합을 문다).")
    ap.add_argument("--swap-slack", type=int, default=0, metavar="N",
                    help="`L_swap` 이 허용할 켜진 기기 **개수의 변화**. 0 이면 "
                         "12.112 처럼 맞바꿈만 한다. 12.158.1 이 잰 것 — 드라이기 "
                         "강풍 정답 ON 창의 44%%가 저항이 하나도 안 켜진 창이라 "
                         "개수 고정으로는 감독에서 통째로 빠진다. 1 이면 하나를 "
                         "켜거나 끌 수 있다. ⚠ 후처리에서 이 제한을 풀었을 때 "
                         "없는 기기를 발명했다 (test_9 유령 3.94 -> 86.98W).")
    ap.add_argument("--harm-offset-skip-stems", default="", metavar="LIST",
                    help="이 파일들에는 `harm_offset` 을 **안 건다** (12.160.2). "
                         "`norton_coef` 는 장소 A 8파일에서 적합한 것이고, 장소 B 로 "
                         "전이하면 보정이 관측을 넘는다 (h9 111%%, h13 143%%). "
                         "12.148 이 *'전이될 것으로 보지만 확인 안 됐다'* 고 유보한 "
                         "것을 12.155 의 라벨로 확인한 결과다.")
    ap.add_argument("--harm-offset-z-stems", default="", metavar="LIST",
                    help="`--harm-offset-z` 를 **이 파일들에만** 건다 (12.160). "
                         "안 주면 전 창에 걸린다(기존 동작). 적응 자료에 장소가 "
                         "섞여 있으면 하나를 전역으로 걸 때 한쪽이 반드시 틀린다 — "
                         "장소 A 의 Z(L 455µH)로 장소 B 를 보정하면 h3 보정량이 "
                         "244~503mA 어긋나고, 그것은 미니PC IDLE 의 |I3| 43.6mA 의 "
                         "6~11배다.")
    ap.add_argument("--res-apps", default="electiric_kettle,oven", metavar="LIST",
                    help="`--w-res` 가 저항을 못 박을 기기. 기본은 포트·오븐 — "
                         "축퇴인 쌍이면서 등가저항이 13%% 벌어진 유일한 쌍이다.")
    ap.add_argument("--swap-tiebreak", default="off", choices=("off", "h3", "mag"),
                    help="`L_swap` 이 허용오차 안에서 **여러 조합**을 만났을 때 "
                         "무엇으로 고르는가 (12.165.6). 기본 `off` 는 컨덕턴스 "
                         "argmin 인데, 컨덕턴스가 같으면 **전력도 같아서**(포트 "
                         "1377W vs 드라이기강+핫플 1392W, 15W 차이) 정보가 0 이다. "
                         "`h3` 는 관측 고조파에 가장 가까운 조합을 고른다 — 같은 두 "
                         "조합의 h3/h1 이 0.37%% vs 2.14%% 로 5.8배 갈린다. "
                         "h1 은 안 쓴다 (거기가 축퇴인 축이다). "
                         "`h3` 는 복소 거리인데 **반증됐다** — `harm_offset` 이 안 "
                         "빠져 위상이 돌아가 있다 (12.165.7). `mag` 는 차수별 크기만 "
                         "비교해 공통 위상 회전에 면역이다. `mag` 를 쓸 것.")
    ap.add_argument("--swap-tb-orders", default="3", metavar="LIST",
                    help="동점깨기가 쓸 차수. **기본은 3 하나다** — 실측에서 "
                         "h5 는 약하고 h7 은 오히려 틀린 쪽을 고른다 (12.165.7).")
    ap.add_argument("--harm-max-order", type=int, default=0, metavar="H",
                    help="2단계 `L_harm` 에서 이 차수 위를 뺀다 (12.171.4 의 B). "
                         "실측 창에서 L_harm 값의 **56%%가 h11~h15** 이고 그 예측이 "
                         "관측의 1/4~2/3 다 (h15 21.6 vs 92.4 mA). 9 를 권장. 0 이면 끔.")
    ap.add_argument("--w-impl", type=float, default=0.0, metavar="W",
                    help="**함의 제약** `L_impl = relu(on_logit − plugged_logit)` "
                         "(12.164.9). 꽂히지 않은 기기가 켜질 수는 없다 — 합성 30만 "
                         "창에서 `on=1 & plugged=0` 은 9종 전부 0건이다. 그런데 두 "
                         "머리가 독립 시그모이드라 2단계에서는 그 모순을 벌하는 항이 "
                         "없었고, 12.164 가 `gt_plugged` 를 '동작 세션 중' 으로 바꾸자 "
                         "모델이 그 틈으로 갔다 (장소B 오븐 유령 1.1 -> 134W). "
                         "로짓에 직접 건다 — σ 를 곱하면 포화 게이트에 안 닿는다(규칙 51). "
                         "0 이면 끔")
    ap.add_argument("--impl-side", default="both", choices=("both", "on"),
                    help="`--w-impl` 에서 **어느 쪽이 양보하는가**. `both` 는 두 로짓 "
                         "모두에 기울기를 준다 — 12.164.10 에서 모델이 틀린 쪽을 골랐다 "
                         "(장소 B 에서 on 을 내리는 대신 plugged 를 0.02 -> 0.96 으로 "
                         "올렸다). `on` 은 `plugged_logit` 을 detach 해 on 쪽만 민다.")
    ap.add_argument("--standby-operating", nargs="?", const="all", default="off",
                    choices=("off", "session", "all"),
                    help="`standby_sig` 를 **동작 중 휴지**의 지문으로 바꾼다 (12.163). "
                         "기본값(`get_standby_profile`)은 `OFF_STANDBY` 인데, 합성은 "
                         "activation 휴지의 전력을 `net_power_features` 에서 가져오므로 "
                         "오븐의 경우 FAN_LIGHT 15.02W 다. 즉 **전력은 FAN_LIGHT 인데 "
                         "고조파는 OFF_STANDBY** 이라 6.44 vs 67.3mA 로 10배 어긋난다. "
                         "이 플래그가 둘을 같은 상태로 맞춘다. "
                         "`session` 은 `gt_plugged` 의 뜻이 '동작 중' 으로 바뀐 "
                         "기기(오븐)에만 건다 (12.164). 값 없이 주면 `all` — "
                         "12.163 의 자와 같다(오븐+핫플).")
    ap.add_argument("--companion", action="store_true",
                    help="**동반 부하 항** (12.156). 오븐은 `OFF_STANDBY / FAN_LIGHT / "
                         "HEATING` 세 상태인데 라벨이 FAN_LIGHT 를 `is_on=0, "
                         "target_power_w=0` 으로 적어서, 조명·컨벡션 팬의 14.2W 가 "
                         "**어느 기기 몫도 아니게** 배경으로 흘러간다. 그래서 오븐의 "
                         "`standby_profile` 이 0.40W(OFF_STANDBY) 이고, 오븐이 존재하는 "
                         "시간의 52~73%% 를 차지하는 상태가 순방향 모형에 없다. "
                         "이 항은 그것을 `σ(plugged)·companion_sig` 로 되돌린다 — "
                         "**`(1−σ(on))` 을 곱하지 않는다.** 팬·조명은 히터와 동시에 "
                         "돌기 때문이다 (격리 HEATING 의 |I3| 중 27~58%%가 팬/조명 몫이고, "
                         "빼고 남은 잔차의 |u3| 0.0010~0.0028 이 순수 니크롬이다). "
                         "겨냥은 크기가 아니라 **신원**이다 — 오븐이 켜졌다면 64mA/"
                         "|u3| 0.058 의 SMPS 전류가 같이 있어야 하고 포트에는 그런 "
                         "상태가 없다. 상수는 녹화 3개에서 폭/중앙 0.010 이다.")
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
                         "나온다 — 기기 간 11.8%% 의 가짜 판별자를 지운다")
    ap.add_argument("--harm-vnorm-both", action="store_true",
                    help="정규화를 **합성 갈래에도** 건다. 캐시의 `obs_harm` 은 원래 "
                         "지문으로 합성한 것이라 틀린 값이 된다 — **대조용**이다")
    ap.add_argument("--harm-vscale", type=float, default=0.0, metavar="X",
                    help="L_harm 의 h1 지문을 창별 전압으로 보정 (12.151). "
                         "1.0 이 물리값 — 와트당 h1 전류가 1/V_rms 라는 항등식이다. "
                         "기준 전압은 적응 창의 중앙값이라 평균 배율이 1 이고, "
                         "**부하와 상관된 변동만** 새 정보로 들어간다 "
                         "(그냥 지문을 상수배 한 것과 구별하기 위해서다)")
    ap.add_argument("--zero-channels", default="", metavar="LIST",
                    help="세밀 입력의 이 채널들을 0 으로 (쉼표). 1단계에서 같은 "
                         "인자로 학습한 모델을 2단계에서도 같은 입력으로 돌리려면 "
                         "여기서도 줘야 한다 (12.114 재시험의 조인 대조)")
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

    ZERO_CH = [int(x) for x in a.zero_channels.split(",") if x.strip()]
    if ZERO_CH:
        print(f"  ** 세밀 채널 {ZERO_CH} 를 0 으로 (조인 대조, 12.114 재시험) **")
    hs = load_holdout(HOLDOUT_DIR)
    apps = hs.appliances
    prep = prepare_holdout_inputs(hs)
    if ZERO_CH:
        prep[0][:, ZERO_CH] = 0.0
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
    # 상시 배경 (12.166.3). 실측에도 실재하므로 캐시가 넣었으면 여기도 넣는다.
    if _cache_says_background(a.cache):
        from src.synthesis.sp_curves import background_signature, background_power
        _bg = background_signature()
        nz = nz + _bg
        print(f"  ** 상시 배경 (12.166): +{background_power():.2f}W, "
              f"|I1| +{np.hypot(_bg[0,0], _bg[0,1])*1000:.1f} mA -> noise_sig **")
    # ── 동작 중 휴지의 지문 (2026-09-03, 12.163) ─────────────────────────
    # `standby_signatures` 는 `OFF_STANDBY` 을 준다. 그런데 **합성이 실제로 넣는
    # 값은 다르다** — `synthesizer` 가 activation 휴지의 `gt_standby_p` 를
    # `net_power_features[:,0]` 에서 가져오므로 오븐은 FAN_LIGHT 의 15.02W 다.
    # 전력 15W / 고조파 6.44mA 로 **10배 어긋나 있었다** (실제 67.3mA).
    if a.standby_operating != "off":
        from src.model.companion import standby_operating_signatures
        from src.synthesis.synthesizer import SESSION_PLUGGED_APPS
        _only = SESSION_PLUGGED_APPS if a.standby_operating == "session" else None
        sb_op, sb_pw, sb_used = standby_operating_signatures(pool, apps, only=_only)
        if sb_used:
            print(f"  ** 동작 중 휴지 지문 ({a.standby_operating}, 12.163) "
                  f"— 합성이 넣는 것과 같은 자 **")
            for x in sb_used:
                _j = apps.index(x)
                _o = float(np.hypot(sb[_j, 0, 0], sb[_j, 0, 1])) * 1000
                _n = float(np.hypot(sb_op[_j, 0, 0], sb_op[_j, 0, 1])) * 1000
                print(f"     {x:18s} |I1| {_o:6.2f} -> {_n:6.2f} mA "
                      f"({_n/max(_o,1e-9):.1f}배), 전력 {sb_pw[_j]:.2f}W")
                sb[_j] = sb_op[_j]
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
    # ── 저항 컨덕턴스 정합이 붙잡을 기기 (12.156) ────────────────────────
    # **포트·오븐만이다.** 이 둘이 `L_harm` 에서 축퇴이고(판별의 97.6%가 h1),
    # 그러면서 등가저항은 13% 벌어져 있다. 핫플·드라이기를 넣으면 안 된다 —
    # 핫플은 장소 B 에서 230~240W 로 돌고(참조 460W), 드라이기는 상태마다
    # 저항이 다르다. 규칙 14: 안 잰 것을 측정처럼 쓰지 않는다.
    _res_set = ({x for x in a.res_apps.split(",") if x}
                if (a.w_res > 0 or a.w_swap > 0) else set())
    # ── 동반 부하 상수 (12.156) ─────────────────────────────────────────
    COMP_SIG = COMP_W = None
    if a.companion:
        from src.model.companion import companion_constants
        COMP_SIG, COMP_W, _cnames = companion_constants(apps)
        print(f"  ** 동반 부하 항 켜짐 (12.156): {', '.join(_cnames)} **")
        for _n in _cnames:
            _j = apps.index(_n)
            _m = float(np.hypot(COMP_SIG[_j, 0, 0], COMP_SIG[_j, 0, 1]))
            print(f"     {_n}: {COMP_W[_j]:.2f}W, |I1| {_m*1000:.2f}mA "
                  f"(σ(plugged) 로만 건다 — 히터와 동시에 돈다)")
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
        harm_max_order=a.harm_max_order,
        harm_grad_balance=a.harm_grad_balance,
        harm_deadzone=a.harm_deadzone, harm_weight=a.harm_weight,
        reactive_qp=torch.from_numpy(qp), noise_q=nq,
        smps_group=[apps.index(x) for x in
                    ("beam_projector", "laptop_charger", "minipc") if x in apps],
        weights=LossWeights(harm=0.1, cons=0.0, over=0.0),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
        power_ref=torch.tensor(
            [(REFERENCE_W[x][0] if x in REFERENCE_W
              else REFERENCE_W_STEMWISE[x][0][0]) if (x in _pref_set and
              (x in REFERENCE_W or x in REFERENCE_W_STEMWISE)) else 0.0
             for x in apps], dtype=torch.float32),
        sig_real=(None if SIG_REAL is None else torch.from_numpy(SIG_REAL)),
        companion_sig=(None if COMP_SIG is None else torch.from_numpy(COMP_SIG)),
        companion_w=(None if COMP_W is None else torch.from_numpy(COMP_W)),
        res_ohm=torch.tensor(
            [RESISTIVE_OHM[x] if (x in RESISTIVE_OHM and x in _res_set) else 0.0
             for x in apps], dtype=torch.float32),
        # 반파 상태의 겉보기 저항. 드라이기만 있고 나머지는 0 이다 (12.157).
        res_ohm_half=torch.tensor(
            [HALFWAVE_OHM[x] if (x in HALFWAVE_OHM and x in _res_set) else 0.0
             for x in apps], dtype=torch.float32),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    cache = CachedWindows(a.cache)
    # ── 교차주파수 어드미턴스 보정 (12.148) — 창마다 상수라 미리 다 만든다 ──
    HOFF = None
    if a.harm_offset:
        from src.model.realdata import harmonic_offset
        # ── 창별 임피던스 (2026-09-03, 12.160) ──────────────────────────
        # `Z` 는 **그 집 배선의 값**이라 장소가 바뀌면 다르다. 지금까지는
        # `--harm-offset-z` 가 전 창에 하나를 걸었는데, 적응 자료에 장소 A 와 B
        # 가 섞여 있으므로 **한쪽은 반드시 틀린다.**
        #
        # 얼마나 틀리는가 (12.160.1): 장소 A 의 Z(L 455µH)로 장소 B 를 보정하면
        # h3 보정량이 186~269mA 인데, 장소 B 의 Z(L 6µH)로는 456~688mA 다 —
        # **차 244~503mA.** 미니PC IDLE 의 |I3| 가 43.6mA 이므로 그 지문의
        # **6~11배**가 보정 오차로 들어간다. 장소 B 에서만 미니PC `p_raw` 가
        # 0.64W (참값 9.90) 로 무너진 것이 이것으로 설명된다.
        #
        # `--harm-offset-z-stems` 를 주면 **그 파일들만** 다른 Z 로 계산한다.
        if a.harm_offset_z and a.harm_offset_z_stems:
            zs = {x for x in a.harm_offset_z_stems.split(",") if x}
            HOFF = harmonic_offset(rw.stem, rw.target_cycle, a.harm_offset)
            sel = np.isin(rw.stem, list(zs))
            if sel.any():
                alt = harmonic_offset(rw.stem, rw.target_cycle, a.harm_offset,
                                      z_npz=a.harm_offset_z)
                HOFF[sel] = alt[sel]
            print(f"  ** 창별 임피던스 (12.160): {sorted(zs)} 만 "
                  f"{a.harm_offset_z} 로 계산 — 적응 창의 {sel.mean()*100:.1f}% **")
        else:
            HOFF = harmonic_offset(rw.stem, rw.target_cycle, a.harm_offset,
                                   z_npz=a.harm_offset_z)
        # ── 보정을 **안 걸 파일** (2026-09-03, 12.160.2) ────────────────────
        # 12.148 의 독스트링이 유보해 둔 것: *"기울기는 물리적으로 ΣY 라 기기
        # 구성이 같으면 전이될 것으로 보지만 **확인 안 됐다** — 다른 집 라벨이
        # 없어 못 쟀다."* 12.155 가 그 라벨을 만들었고, **전이가 안 된다.**
        #
        # 보정량 / 관측 (중앙):
        #     장소 A (적합한 곳)  h1 9.4%  h3 20.9%  h5 30.4%  h9 31.0%  h13 64.8%
        #     장소 B (전이)      h1 2.3%  h3 79.5%  h5 94.7%  h9 **111.4%**  h13 **142.5%**
        #
        # 장소 B 에서는 h9 이상이 **보정 > 관측** 이다. h3 만 봐도 216.7mA 로
        # 미니PC IDLE 의 |I3| 43.6mA 의 5배다 — 그 지문이 보정에 묻힌다.
        if a.harm_offset_skip_stems:
            sk = {x for x in a.harm_offset_skip_stems.split(",") if x}
            msk = np.isin(rw.stem, list(sk))
            HOFF[msk] = 0.0
            print(f"  ** 보정 제외 (12.160.2): {sorted(sk)} — 적응 창의 "
                  f"{msk.mean()*100:.1f}% 에서 harm_offset = 0 **")
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
    # `L_res` 가 쓰는 단자 전압 (12.156). 창마다 상수라 미리 다 만든다.
    VRMS = (np.asarray(rw.v_observed, np.float32)
            if (a.w_res > 0 or a.w_swap > 0) else None)
    if VRMS is not None:
        print(f"  ** 저항 제약 켜짐: w_res={a.w_res:g} (12.156) / "
              f"w_swap={a.w_swap:g} tol={a.swap_tol:g} (12.158), 대상 {sorted(_res_set)} **")
        _v0 = float(np.median(VRMS))
        print(f"     V_rms {VRMS.min():.1f} ~ {VRMS.max():.1f}V (중앙 {_v0:.1f})")
        for x in sorted(_res_set):
            if x not in RESISTIVE_OHM:
                continue
            line = f"     {x:18s} {RESISTIVE_OHM[x]:6.1f}Ω -> {_v0**2/RESISTIVE_OHM[x]:6.0f}W"
            if x in HALFWAVE_OHM:
                line += (f"   |  반파 {HALFWAVE_OHM[x]:.1f}Ω -> "
                         f"{_v0**2/HALFWAVE_OHM[x]:.0f}W  (12.157, 관문 |I2|−|I4|>0.1A)")
            print(line)
    # ── 창별 참값 마스크 (12.159) ────────────────────────────────────────
    # `REFERENCE_W_STEMWISE` 의 기기는 **적힌 파일의 창에서만** 참값이 성립한다.
    # 나머지 창에서는 0 이라 `L_pref` 가 그 기기를 안 건드린다.
    PREFM = None
    if a.w_pref > 0:
        pm = np.ones((len(rw), len(apps)), np.float32)
        _named = []
        for x, (val, stems) in REFERENCE_W_STEMWISE.items():
            if x not in _pref_set or x not in apps:
                continue
            j = apps.index(x)
            pm[:, j] = np.isin(rw.stem, list(stems)).astype(np.float32)
            _named.append((x, val[0], stems, float(pm[:, j].mean())))
        if _named:
            PREFM = pm
            print("  ** 창별 참값 (12.159) — 그 파일의 창에서만 건다 **")
            for x, v, stems, frac in _named:
                print(f"     {x}: {v:.2f}W, 파일 {list(stems)} "
                      f"-> 적응 창의 {frac*100:.1f}%")
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
                                    VSC[ridx] if VSC is not None else None,
                                    VRMS[ridx] if VRMS is not None else None,
                                    PREFM[ridx] if PREFM is not None else None)
        sidx = np.sort(rng.choice(len(cache), a.batch, replace=False))
        sb_ = tuple(torch.from_numpy(x) for x in cache.batch(sidx))
        sf, swd, stg = to_targets(sb_, dev)
        if ZERO_CH:
            rf[:, ZERO_CH] = 0.0
            sf[:, ZERO_CH] = 0.0

        sw = real_sample_weights(rtg["p_observed"], a.real_weight, a.smps_boost)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            rp = crit.unlabeled(model(rf, rwd), rtg, w_cons=a.w_cons,
                                w_harm=a.w_harm, w_over=a.w_over, w_hedge=a.w_hedge,
                                sample_w=sw, w_real_on=a.w_real_on,
                                w_consq=a.w_consq, w_pref=a.w_pref,
                                w_res=a.w_res, w_swap=a.w_swap,
                                swap_tol=a.swap_tol, swap_slack=a.swap_slack,
                                swap_tiebreak=a.swap_tiebreak,
                                swap_tb_orders=tuple(
                                    int(x) for x in a.swap_tb_orders.split(",") if x),
                                w_impl=a.w_impl, impl_side=a.impl_side,
                                companion=bool(a.companion))
            sp = crit(model(sf, swd), stg)
            loss = rp["total"] + a.lam * sp["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        for k, v in (("real_cons", rp["cons"]), ("real_harm", rp["harm"]),
                     ("real_consq", rp["consq"]), ("real_res", rp["res"]),
                     ("real_swap", rp["swap"]), ("swap_frac", rp["swap_frac"]),
                     ("swap_ties", rp["swap_ties"]),
                     ("real_impl", rp["impl"]), ("impl_frac", rp["impl_frac"]),
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
                  + (f" on {m['real_on']:.3f}" if a.w_real_on > 0 else "")
                  + (f" res {m['real_res']:.2f}" if a.w_res > 0 else "")
                  # 감독 창 비율을 같이 찍는다 — 맞바꿈이 드물면 항이 켜져 있어도
                  # 아무 일이 안 일어난다 (`_criteria_hwL.md` 의 미리 적은 위험).
                  + (f" swap {m['real_swap']:.3f} (창 {m['swap_frac']*100:.1f}%"
                     + (f", 동점 {m['swap_ties']:.2f}개"
                        if a.swap_tiebreak != "off" else "") + ")"
                     if a.w_swap > 0 else "")
                  + (f" impl {m['real_impl']:.4f} (위반 {m['impl_frac']*100:.2f}%)"
                     if a.w_impl > 0 else "") + f" / "
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
