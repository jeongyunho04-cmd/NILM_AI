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
from src.evaluation.real_events import load_events, score_absent, score_events, score_on_off
from src.evaluation.sealing import is_sealed
from src.model.losses import LossWeights, NILMLoss, build_state_scales
from src.model.inputs import FINE_CYCLES, TARGET_LOOKAHEAD, LEGACY_FINE_CHANNELS
from src.run_gate_check import assert_target_config
from src.model.net import (
    NILMNet, appliance_state_counts, harmonic_scales, harmonic_signatures,
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


def real_targets(b, dev, human=None):
    """`human` 은 `RealWindows.human(idx)` 의 (on, mask). 없으면 기존과 같다."""
    fine, wide, pobs, oh, pn = [torch.from_numpy(np.ascontiguousarray(x)).to(dev) for x in b]
    tg = {"p_observed": pobs, "obs_harm": oh, "p_noise": pn}
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
    ap.add_argument("--harm-deadzone", type=float, default=0.0, metavar="X",
                    help="L_harm 불감대 배수 (12.122.16). 정답 배분에서도 남는 "
                         "차수별 잔차의 X배까지는 벌하지 않는다. **줄일 수 없는 "
                         "잔차의 70%%를 줄이라고 밀어서 배분이 밀린다** — 그것을 "
                         "끊는다. 1.0 이 측정된 중앙값. 0 이면 끔(이전과 동일). "
                         "⚠ 너무 키우면 L_harm 이 죽어 '합만 맞추는 해' 로 간다 (12.12.2)")
    ap.add_argument("--sig-insitu", default="", metavar="NPZ",
                    help="`L_harm` 의 지문을 in-situ 적합본으로 갈아끼운다 "
                         "(12.122.11, run_fit_insitu_sig 의 산출물). 비우면 격리 지문. "
                         "**LOFO 로 검증했지만 사람 라벨 5파일에서 적합한 것이라, "
                         "그 파일을 --holdout-real 로 빼도 지문에는 남아 있다**")
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
    del pool

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
        harm_deadzone=a.harm_deadzone,
        smps_group=[apps.index(x) for x in
                    ("beam_projector", "laptop_charger", "minipc") if x in apps],
        weights=LossWeights(harm=0.1, cons=0.0, over=0.0),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    cache = CachedWindows(a.cache)
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
                                    rw.human(ridx) if a.w_real_on > 0 else None)
        sidx = np.sort(rng.choice(len(cache), a.batch, replace=False))
        sb_ = tuple(torch.from_numpy(x) for x in cache.batch(sidx))
        sf, swd, stg = to_targets(sb_, dev)

        sw = real_sample_weights(rtg["p_observed"], a.real_weight, a.smps_boost)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            rp = crit.unlabeled(model(rf, rwd), rtg, w_cons=a.w_cons,
                                w_harm=a.w_harm, w_over=a.w_over, w_hedge=a.w_hedge,
                                sample_w=sw, w_real_on=a.w_real_on)
            sp = crit(model(sf, swd), stg)
            loss = rp["total"] + a.lam * sp["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        for k, v in (("real_cons", rp["cons"]), ("real_harm", rp["harm"]),
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
