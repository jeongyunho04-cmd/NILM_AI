"""
학습된 CNN 이 어떻게 예측하는지 그림으로 본다
==============================================
네 장을 만든다.

  1_timeline_synthetic.png  합성 홀드아웃 10분 타임라인 — 정답과 나란히
  2_timeline_real.png       실측 test3 — 정답이 없으므로 타임라인 이벤트만 표시
  3_scatter.png             기기별 예측 vs 정답 산점도 (홀드아웃 8,000창)
  4_training.png            학습 곡선

**실측에는 기기별 정답이 없다** (12.4절). 2번 그림의 세로선은 타임라인 문서에서
시각·ΔP 가 정확하다고 밝힌 이벤트뿐이고, 정상상태 귀속은 옮기지 않았다.

python -m src.run_plot_predictions --ckpt results/cnn_s0.pt
"""
from pathlib import Path
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.evaluation import load_holdout
from src.evaluation.real_events import SAMPLING_HZ, load_events
from src.model.inputs import build_inputs
from src.model.net import NILMNet, appliance_state_counts
from src.run_baseline import S_I, baseline_reference

WINDOW = 3600
KO = {"air_conditioner": "에어컨", "beam_projector": "빔프로젝터", "electiric_kettle": "전기포트",
      "fan": "선풍기", "hair_dryer": "헤어드라이기", "hotplate": "핫플레이트",
      "laptop_charger": "노트북충전기", "minipc": "미니PC", "oven": "오븐"}
COL = plt.get_cmap("tab10")


def _font():
    for f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.family"] = f
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def load_model(ckpt: str, dev: str):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    m = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                prior_kappa=ck.get("prior_kappa", 0.0),
                prior_beta=ck.get("prior_beta", 0.5)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, apps, ck.get("epoch", -1)


@torch.no_grad()
def predict_windows(model, x33: np.ndarray, dev: str, batch: int = 256) -> np.ndarray:
    out = []
    for i in range(0, len(x33), batch):
        f, w = build_inputs(x33[i:i + batch])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(torch.from_numpy(f).to(dev), torch.from_numpy(w).to(dev))
        out.append(o["power"].float().cpu().numpy())
    return np.concatenate(out)


def slide(sig33: np.ndarray, stride: int = 60):
    """(33, T) 연속 신호 -> 슬라이딩 창 (N, 33, 3600) 과 각 창의 타깃 시각(사이클)."""
    T = sig33.shape[1]
    starts = np.arange(0, max(1, T - WINDOW), stride, dtype=int)
    xs = np.stack([sig33[:, s:s + WINDOW] for s in starts]).astype(np.float32)
    return xs, starts + (WINDOW - 1 - 60)          # 타깃 = 창 끝 -1초


def plot_timeline(t_s, agg, pred, apps, truth=None, events=None, title="", path="",
                  top_k: int = 5):
    order = np.argsort(-(pred.max(0) if truth is None else np.maximum(pred, truth).max(0)))
    sel = [j for j in order[:top_k]]
    n = len(sel) + 1
    fig, ax = plt.subplots(n, 1, figsize=(15, 2.0 * n), sharex=True,
                           gridspec_kw={"hspace": 0.15})
    ax[0].plot(t_s, agg, lw=0.7, color="0.25")
    ax[0].set_ylabel("총전력 (W)"); ax[0].set_title(title, fontsize=12, loc="left")
    ax[0].grid(alpha=.3)
    for i, j in enumerate(sel):
        a = ax[i + 1]
        if truth is not None:
            a.fill_between(t_s, truth[:, j], color=COL(i % 10), alpha=.25, label="정답")
        a.plot(t_s, pred[:, j], lw=1.1, color=COL(i % 10), label="예측")
        a.set_ylabel(f"{KO.get(apps[j], apps[j])}\n(W)", fontsize=9)
        a.grid(alpha=.3); a.legend(loc="upper right", fontsize=8, framealpha=.9)
        if events:
            for e in events:
                if e["appliance"] == apps[j]:
                    a.axvline(e["t_s"], color="crimson", ls="--", lw=1.2, alpha=.8)
                    a.annotate(f"{e['kind']} ΔP{e['delta_p_w']:+.0f}W",
                               (e["t_s"], a.get_ylim()[1]), fontsize=7, color="crimson",
                               ha="left", va="top")
    ax[-1].set_xlabel("시간 (초)")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  저장 {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="CNN 예측 시각화")
    ap.add_argument("--ckpt", default="results/cnn_s0.pt")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=60)
    a = ap.parse_args()

    _font()
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, ep = load_model(a.ckpt, dev)
    print(f"모델 {a.ckpt} (epoch {ep}) | 장치 {dev}")

    # ── 1. 합성 홀드아웃 타임라인 (정답 있음) ────────────────────────────
    print("[1/4] 합성 홀드아웃 타임라인")
    from src.synthesis.scenario_generator import ScenarioGenerator
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import LoadSynthesizer
    np.random.seed(7)
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="holdout")
    sg = ScenarioGenerator(LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False))
    smp = sg.create_long_timeline(duration_min=a.minutes)
    sig = np.empty((33, smp.duration_cycles), np.float32)
    sig[0:15] = smp.harmonics_ri[:, :, 0].T; sig[15:30] = smp.harmonics_ri[:, :, 1].T
    sig[30] = smp.power_features[:, 0]; sig[31] = smp.power_features[:, 1]
    sig[32] = smp.power_features[:, 4]
    xs, tidx = slide(sig, a.stride)
    pred = predict_windows(model, xs, dev)
    truth = np.stack([smp.gt_target_power_w[x][tidx] for x in apps], axis=1)
    plot_timeline(tidx / SAMPLING_HZ, sig[30][tidx], pred, apps, truth=truth,
                  title=f"합성 홀드아웃 {a.minutes:.0f}분 — 예측(실선) vs 정답(음영)  "
                        f"[학습에 쓰지 않은 뒤 20% 구간]",
                  path=str(outd / "1_timeline_synthetic.png"))

    # ── 2. 실측 test3 (기기별 정답 없음) ──────────────────────────────────
    print("[2/4] 실측 test3 타임라인")
    d = np.load("processed_data/composite_eval/test3.npz", allow_pickle=True)
    T = len(d["power_features"])
    rsig = np.empty((33, T), np.float32)
    rsig[0:15] = d["harmonics_ri"][:, :, 0].T; rsig[15:30] = d["harmonics_ri"][:, :, 1].T
    rsig[30] = d["power_features"][:, 0]; rsig[31] = d["power_features"][:, 1]
    rsig[32] = d["power_features"][:, 4]
    rxs, rtidx = slide(rsig, a.stride)
    rpred = predict_windows(model, rxs, dev)
    ev = load_events()["test3"]["events"]
    plot_timeline(rtidx / SAMPLING_HZ, rsig[30][rtidx], rpred, apps, truth=None, events=ev,
                  title="실측 test3 (미니PC+빔프로젝터+오븐) — 예측만. "
                        "기기별 정답이 없어 비교 불가, 빨간 선은 타임라인의 확실한 이벤트",
                  path=str(outd / "2_timeline_real.png"))
    resid = rsig[30][rtidx] - rpred.sum(1)
    print(f"    실측 총전력 잔차: 평균 {resid.mean():+.1f}W | 절대평균 {np.abs(resid).mean():.1f}W")

    # ── 3. 기기별 산점도 ──────────────────────────────────────────────────
    print("[3/4] 기기별 산점도")
    hs = load_holdout("processed_data/holdout60")
    hp = predict_windows(model, np.asarray(hs.X), dev)
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for j, (axx, app) in enumerate(zip(axes.ravel(), apps)):
        t, p = hs.y_power[:, j], hp[:, j]
        m = hs.y_on[:, j].astype(bool)
        lim = max(float(np.percentile(np.concatenate([t, p]), 99.5)), S_I[app] * 1.2, 1.0)
        axx.scatter(t[~m], p[~m], s=3, alpha=.25, color="0.6", label="꺼짐")
        axx.scatter(t[m], p[m], s=4, alpha=.45, color=COL(j % 10), label="켜짐")
        axx.plot([0, lim], [0, lim], "k--", lw=.9)
        mae = float(np.mean(np.abs(p[m] - t[m]))) if m.any() else float("nan")
        axx.set_title(f"{KO.get(app, app)}  MAE(on) {mae:.1f}W", fontsize=10)
        axx.set_xlim(-lim * .03, lim); axx.set_ylim(-lim * .03, lim)
        axx.set_xlabel("정답 (W)", fontsize=8); axx.set_ylabel("예측 (W)", fontsize=8)
        axx.grid(alpha=.3); axx.legend(fontsize=7, loc="upper left")
    fig.suptitle("홀드아웃 8,000창 — 기기별 예측 vs 정답 (점선 = 완벽)", fontsize=13)
    fig.tight_layout(); fig.savefig(outd / "3_scatter.png", dpi=110, bbox_inches="tight")
    plt.close(fig); print(f"  저장 {outd / '3_scatter.png'}")

    # ── 4. 학습 곡선 ──────────────────────────────────────────────────────
    print("[4/4] 학습 곡선")
    jp = Path(a.ckpt).with_suffix(".json")
    if jp.exists():
        h = json.loads(jp.read_text(encoding="utf-8"))["history"]
        e = [r["epoch"] for r in h]
        fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
        ref = baseline_reference()
        for axx, key, lab, base in [
                (axs[0], "mae", "기기 평균 MAE (W)", ref["mae"]),
                (axs[1], "f1", "F1 평균", ref["f1"]),
                (axs[2], "resistive_acc", "저항3종 정확도", ref["resistive_acc"]),
                (axs[3], "resid_abs", "총전력 잔차 (W)", ref["resid_abs"])]:
            axx.plot(e, [r[key] for r in h], "o-", ms=3, color=COL(0))
            axx.axhline(base, color="crimson", ls="--", lw=1.2, label="Phase1 GBM")
            axx.set_xlabel("epoch"); axx.set_title(lab, fontsize=10)
            axx.grid(alpha=.3); axx.legend(fontsize=8)
            if key in ("mae", "resid_abs"):
                axx.set_yscale("log")
        fig.suptitle("Phase 3 CNN 학습 곡선 (빨간 점선 = Phase 1 GBM 기준선)", fontsize=12)
        fig.tight_layout(); fig.savefig(outd / "4_training.png", dpi=110, bbox_inches="tight")
        plt.close(fig); print(f"  저장 {outd / '4_training.png'}")

    print(f"\n완료: {outd.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
