"""
실측에서 어디가 틀어지는가 — 진단 플롯
========================================
합성 지표만 보고 손실 가중치를 만지는 접근은 여러 번 실패했다 (12.9.3절).
2단계 설계 전에 **실측에서 무엇이 어긋나는지** 눈으로 확인한다.

파일마다 네 칸을 그린다.

  1) 관측 총전력 vs 예측 합계      — 합이 맞는가
  2) 잔차 (관측 − 예측)            — 어느 구간에서 벌어지는가
  3) 기기별 예측 전력 (누적)        — 그 와트를 누가 가져갔는가
  4) 알려진 정답 구간              — 그때 실제로 무엇이 켜져 있었나
  5) **모델의 on/off 판정**        — 모델은 무엇이 켜졌다고 봤나 (게이트)

5번은 **모든 기기**를 행으로 놓는다. 정답이 없는 기기에 게이트가 서는 것이 곧
오귀속이라, 정답 있는 기기만 그리면 그게 안 보인다. 정답 구간은 검은 테두리로
겹쳐 두었으므로 **테두리 밖의 색 = 헛detect, 테두리 안의 빈칸 = 놓침**이다.

**실측에는 기기별 정답이 없다** (12.4절). 4번 칸은 `real_events.json` 의
*확실* 구간과 이벤트뿐이고, 정상상태 귀속은 애초에 옮기지 않았다 (4.2절).

    python -m src.run_plot_real --ckpt results/cnn_v11.pt
    python -m src.run_plot_real --ckpt results/adapt_v1.pt --tag adapt
"""
from pathlib import Path
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.model.inputs import LEGACY_FINE_CHANNELS
from src.model.net import NILMNet, appliance_state_counts
from src.model.realdata import dense_targets
from src.preprocessing import load_nilm_npz

KO = {"air_conditioner": "에어컨", "beam_projector": "빔프로젝터",
      "electiric_kettle": "전기포트", "fan": "선풍기", "hair_dryer": "헤어드라이기",
      "hotplate": "핫플레이트", "laptop_charger": "노트북충전기",
      "minipc": "미니PC", "oven": "오븐"}
COL = plt.get_cmap("tab10")
SAMPLING_HZ = 60.0


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
                prior_beta=ck.get("prior_beta", 0.5),
                aux_z=ck.get("aux_z", False),
                fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, apps, ck


@torch.no_grad()
def predict(model, rw, dev: str, batch: int = 512):
    """(전력, 대기, 관문, 관측P, 관측고조파, 계측P) — 후처리가 뒤의 셋을 쓴다."""
    P, S, G, PO, OH, PN = [], [], [], [], [], []
    for i in range(0, len(rw), batch):
        f, w, pobs, oh, pn = rw.batch(np.arange(i, min(i + batch, len(rw))))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                      torch.from_numpy(np.ascontiguousarray(w)).to(dev))
        P.append(o["power"].float().cpu().numpy())
        S.append(o["standby"].float().cpu().numpy())
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
        PO.append(pobs); OH.append(oh); PN.append(pn)
    return (np.concatenate(P), np.concatenate(S), np.concatenate(G),
            np.concatenate(PO), np.concatenate(OH), np.concatenate(PN))


def run_postproc(mode, apps, pred, standby, gate, pobs, oh, pn, v_rms):
    """`cap` = 물리 상한 넘기기, `full` = 거기에 저항 조합 정합까지 (12.112).

    ⚠ `resistive_match` 는 `min_w=150W` 문턱이라 **SMPS 전용 창(93W)은 안 건드린다.**
      자리 D 배분은 후처리로 못 고친다 — 고치는 것은 저항 창의 조합이다.
    """
    if mode == "off":
        return pred, gate
    from src.model.postproc import apply_postproc, resistive_match
    P, g = apply_postproc(pred, gate, apps)
    if mode == "full":
        P, g = resistive_match(P, g, apps, pobs, v_rms, standby, pn, obs_harm=oh)
    return np.asarray(P, np.float32), np.asarray(g, np.float32)


def plot_file(stem, apps, t_pred, pred, standby, t_obs, obs, spec, path, title,
              gate=None, gate_thr=0.5):
    n_ax = 4 if gate is None else 5
    hr = [3, 2, 3, 2.2] if gate is None else [3, 2, 3, 2.2, 2.4]
    fig, ax = plt.subplots(n_ax, 1, figsize=(16, 11 if gate is None else 13.4), sharex=True,
                           gridspec_kw={"height_ratios": hr, "hspace": 0.12})

    total = pred.sum(1) + standby.sum(1)
    obs_at = np.interp(t_pred, t_obs, obs)

    # 1) 관측 vs 예측 합계
    ax[0].plot(t_obs, obs, lw=0.4, color="0.35", label="관측 총전력 (60Hz)")
    ax[0].plot(t_pred, total, lw=1.4, color="crimson", label="예측 합계 (활성+대기)")
    ax[0].set_ylabel("전력 (W)")
    ax[0].legend(loc="upper right", fontsize=9, framealpha=.9)
    ax[0].grid(alpha=.3)
    ax[0].set_title(title, fontsize=13, loc="left")

    # 2) 잔차
    r = obs_at - total
    ax[1].axhline(0, color="0.3", lw=0.8)
    ax[1].fill_between(t_pred, r, 0, where=r >= 0, color="tab:red", alpha=.45,
                       label="과소 예측 (관측 > 예측)")
    ax[1].fill_between(t_pred, r, 0, where=r < 0, color="tab:blue", alpha=.45,
                       label="과대 예측")
    ax[1].set_ylabel("잔차 (W)")
    ax[1].legend(loc="upper right", fontsize=9, framealpha=.9)
    ax[1].grid(alpha=.3)
    ax[1].annotate(f"평균 {r.mean():+.1f}W   절대평균 {np.abs(r).mean():.1f}W",
                   (0.01, 0.06), xycoords="axes fraction", fontsize=10, color="0.2")

    # 3) 기기별 예측 (누적)
    order = np.argsort(-pred.mean(0))
    ax[2].stackplot(t_pred, *[pred[:, j] for j in order],
                    labels=[KO.get(apps[j], apps[j]) for j in order],
                    colors=[COL(i % 10) for i in range(len(order))], alpha=.85)
    ax[2].plot(t_obs, obs, lw=0.4, color="0.15", alpha=.6)
    ax[2].set_ylabel("기기별 예측 (W)")
    ax[2].legend(loc="upper right", fontsize=8, ncol=3, framealpha=.9)
    ax[2].grid(alpha=.3)

    # 4) 알려진 정답 구간
    iv = spec["intervals"]
    rows = [a for a in apps if iv.get(a, {}).get("on") or iv.get(a, {}).get("uncertain")]
    for i, a in enumerate(rows):
        for t0, t1 in iv[a].get("uncertain", []):
            ax[3].barh(i, t1 - t0, left=t0, height=0.55, color="0.75",
                       hatch="///", edgecolor="0.5", linewidth=0.4)
        for t0, t1 in iv[a].get("on", []):
            ax[3].barh(i, t1 - t0, left=t0, height=0.55,
                       color=COL(apps.index(a) % 10), alpha=.9)
    # 이벤트 라벨은 **세로로 쓰고 층을 번갈아 놓는다.** 가로로 쓰면 test_5/test_8
    # 처럼 20~24개가 몰린 파일에서 글자가 서로를 덮어 아무것도 안 읽힌다.
    # 그래도 가까운 것끼리는 겹치므로, 앞 라벨과 최소 간격을 두고 층을 올린다.
    ev_sorted = sorted(spec.get("events", []), key=lambda e: e["t_s"])
    span = max(t_obs[-1] - t_obs[0], 1.0) if len(t_obs) else 1.0
    min_gap = span * 0.035          # 이보다 가까우면 다음 층으로 올린다
    last_t, level, n_level = -1e9, 0, 3
    for e in ev_sorted:
        ax[3].axvline(e["t_s"], color="crimson", ls="--", lw=1.0, alpha=.55)
        level = (level + 1) % n_level if (e["t_s"] - last_t) < min_gap else 0
        last_t = e["t_s"]
        # ΔP 가 없는 이벤트가 있다 (test_6/test_9 는 신호 유추라 `_note` 만 있다).
        dp = e.get("delta_p_w")
        ax[3].annotate(f"{KO.get(e['appliance'], e['appliance'])} {e['kind']}"
                       + (f" {dp:+.0f}W" if dp is not None else ""),
                       (e["t_s"], len(rows) - 0.45 - 0.16 * level),
                       fontsize=6.5, color="crimson", ha="left", va="top",
                       rotation=90, rotation_mode="anchor", alpha=.85)
    ax[3].set_yticks(range(len(rows)))
    ax[3].set_yticklabels([KO.get(a, a) for a in rows], fontsize=9)
    # 세로 라벨이 들어갈 자리를 위쪽에 비워 둔다.
    ax[3].set_ylim(-0.6, len(rows) + 1.1)
    if gate is None:
        ax[3].set_xlabel("시간 (초)")
    ax[3].set_ylabel("알려진 정답")
    ax[3].grid(alpha=.3, axis="x")
    ax[3].annotate("색칠 = 확실히 켜짐   빗금 = uncertain(채점 제외)",
                   (0.01, 0.04), xycoords="axes fraction", fontsize=9, color="0.3")

    # 5) 모델의 on/off 판정 — **모든 기기**를 행으로
    if gate is not None:
        g = np.asarray(gate, float)
        iv = spec["intervals"]
        dt = float(np.median(np.diff(t_pred))) if len(t_pred) > 1 else 1.0
        for i, a in enumerate(apps):
            col = COL(apps.index(a) % 10)
            on = g[:, i] > gate_thr
            # 게이트 세기를 진하기로 (0.5 -> 옅게, 1.0 -> 진하게)
            for k in np.flatnonzero(on):
                ax[4].barh(i, dt, left=t_pred[k], height=0.62, color=col,
                           alpha=float(np.clip((g[k, i] - gate_thr) / (1 - gate_thr), 0.15, 1.0)),
                           linewidth=0)
            # 정답 구간을 테두리로 겹친다 — 테두리 밖의 색이 헛detect 다
            for t0, t1 in iv.get(a, {}).get("on", []):
                ax[4].add_patch(plt.Rectangle((t0, i - 0.34), t1 - t0, 0.68, fill=False,
                                              edgecolor="0.1", linewidth=1.1, zorder=5))
            for t0, t1 in iv.get(a, {}).get("uncertain", []):
                ax[4].add_patch(plt.Rectangle((t0, i - 0.34), t1 - t0, 0.68, fill=False,
                                              edgecolor="0.55", linewidth=0.9, ls=":", zorder=5))
        ax[4].set_yticks(range(len(apps)))
        ax[4].set_yticklabels([KO.get(a, a) for a in apps], fontsize=9)
        ax[4].set_ylim(-0.6, len(apps) - 0.4)
        ax[4].set_xlabel("시간 (초)")
        ax[4].set_ylabel(f"모델 판정 (게이트>{gate_thr:g})")
        ax[4].grid(alpha=.3, axis="x")
        # 기기별로 얼마나 켰다고 봤는지 — 정답과 견줄 수 있게 옆에 적는다
        for i, a in enumerate(apps):
            frac = float((g[:, i] > gate_thr).mean())
            true_s = sum(t1 - t0 for t0, t1 in iv.get(a, {}).get("on", []))
            span = max(t_pred[-1] - t_pred[0], 1.0) if len(t_pred) else 1.0
            ax[4].annotate(f"{100*frac:.0f}% (참 {100*true_s/span:.0f}%)",
                           (1.002, i), xycoords=("axes fraction", "data"),
                           fontsize=7.5, va="center", color="0.25")
        ax[4].annotate("진하기 = 게이트 세기   검은 테두리 = 참 ON   점선 = uncertain   "
                       "**테두리 밖의 색 = 헛detect, 테두리 안 빈칸 = 놓침**",
                       (0.01, 0.03), xycoords="axes fraction", fontsize=8.5, color="0.3")

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장 {path}")
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="실측 진단 플롯")
    ap.add_argument("--ckpt", default="results/cnn_v11.pt")
    ap.add_argument("--stride", type=int, default=30, help="예측 간격 (사이클)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--gate-thr", type=float, default=0.5,
                    help="5번 칸에서 '켜졌다' 로 볼 게이트 문턱")
    ap.add_argument("--no-gate", action="store_true", help="5번 칸을 안 그린다 (옛 4칸 그림)")
    ap.add_argument("--postproc", default="off", choices=("off", "cap", "full"),
                    help="`cap` 물리 상한 넘기기 · `full` 거기에 저항 조합 정합까지. "
                         "⚠ resistive_match 는 min_w=150W 라 SMPS 전용 창은 안 건드린다")
    a = ap.parse_args()

    _font()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outd = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    model, apps, ck = load_model(a.ckpt, dev)
    ev = load_events()
    name = a.tag or Path(a.ckpt).stem
    print(f"[실측 진단] {a.ckpt} (ep{ck.get('epoch')}) | 장치 {dev}")

    rows = []
    for stem in sorted(ev):
        if is_sealed(stem):
            print(f"  {stem}: 봉인 — 건너뜀 (4.3절)")
            continue
        rw = dense_targets(stem, stride=a.stride)
        pred, standby, gate, pobs, oh, pn = predict(model, rw, dev)
        if a.postproc != "off":
            pred, gate = run_postproc(a.postproc, apps, pred, standby, gate,
                                      pobs, oh, pn,
                                      np.asarray(rw.v_observed, np.float32))
        raw = load_nilm_npz(f"processed_data/composite_eval/{stem}.npz")
        obs = np.asarray(raw["power_features"])[:, 0]
        t_obs = np.arange(len(obs)) / SAMPLING_HZ
        t_pred = rw.target_cycle / SAMPLING_HZ
        r = plot_file(stem, apps, t_pred, pred, standby, t_obs, obs, ev[stem],
                      str(outd / f"real_{stem}_{name}.png"),
                      f"{stem}  —  {a.ckpt} 예측  ({len(rw):,}창, {a.stride/60:.1f}초 간격)",
                      gate=None if a.no_gate else gate, gate_thr=a.gate_thr)
        rows.append((stem, r, pred))

    print(f"\n{'파일':10s}{'잔차 평균':>11s}{'절대':>9s}  기기별 평균 예측 (W)")
    for stem, r, pred in rows:
        top = np.argsort(-pred.mean(0))[:4]
        s = "  ".join(f"{KO.get(apps[j], apps[j])} {pred[:, j].mean():.0f}" for j in top)
        print(f"{stem:10s}{r.mean():>+10.1f}W{np.abs(r).mean():>8.1f}W  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
