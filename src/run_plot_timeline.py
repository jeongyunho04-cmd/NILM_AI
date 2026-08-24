"""
실측 예측 타임라인 — 정답과 나란히 그린다
============================================
`run_plot_real` 은 전력 위주의 진단 플롯이다. 여기서는 **무엇이 언제 켜졌다고
예측했는가**를 정답과 같은 축에 놓고 본다.

파일마다 세 칸:

  1) 관측 총전력 vs 예측 합계
  2) 기기별 예측 전력 (누적)
  3) **타임라인** — 기기마다 두 줄. 위=정답, 아래=예측.
     정답의 판정보류(`uncertain`) 구간은 빗금으로 표시하고 채점에서 빠진다 (12.25절).

    python -m src.run_plot_timeline                      # 운영 조합이 기본값
    python -m src.run_plot_timeline --stride 6           # 핫플 펄스(1초)를 보려면

[정답이 없는 기기는 어떻게 그리나]
`appliances_present` 에 없는 기기는 **정답이 0 으로 확정**이다 (12.12.1절).
그 기기에 예측이 붙으면 그것이 오귀속이고, 빨간색으로 그린다.
"""
from pathlib import Path
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.run_gate_check import forward_file, load_model
from src.run_live import SMPS_GROUP

KOR = {"oven": "오븐", "hotplate": "핫플레이트", "electiric_kettle": "전기포트",
       "hair_dryer": "헤어드라이기", "minipc": "미니PC", "beam_projector": "빔프로젝터",
       "laptop_charger": "노트북충전기", "fan": "선풍기", "air_conditioner": "에어컨"}
# 저항 발열 무리를 위에, 저부하를 아래에 (0.1절의 세 무리)
ORDER = ["oven", "hotplate", "electiric_kettle", "hair_dryer",
         "air_conditioner", "beam_projector", "laptop_charger", "minipc", "fan"]


def _korean_font() -> None:
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
        if any(name.lower() in f.name.lower() for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _spans(mask: np.ndarray, t: np.ndarray):
    """불리언 마스크 -> [(시작초, 끝초)] 구간 목록."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    out, b = [], idx[0]
    for i in range(1, len(idx)):
        if idx[i] != idx[i - 1] + 1:
            out.append((t[b], t[idx[i - 1]])); b = idx[i]
    out.append((t[b], t[idx[-1]]))
    return out


def plot_file(model, apps, stem: str, ev: dict, dev: str, out_dir: Path, tag: str,
              stride: int = 15, model_smps=None) -> Path:
    """`model_smps` 를 주면 SMPS 3종만 그 체크포인트에서 가져온다.

    `run_live --ckpt-smps` 와 같은 구성이다 (12.31.5). 운영에서 실제로 도는 것이
    두 체크포인트의 조합이므로, 그림도 그 조합으로 그려야 실물과 맞는다.
    """
    d = forward_file(model, stem, dev, stride=stride)
    if model_smps is not None:
        ds = forward_file(model_smps, stem, dev, stride=stride)
        six = [apps.index(x) for x in SMPS_GROUP if x in apps]
        for k in ("gate", "p_raw", "standby"):
            d[k][:, six] = ds[k][:, six]
    t = d["targets"] / 60.0
    P = d["gate"] * d["p_raw"]
    on = d["gate"] > 0.5
    n_cycles = int(ev[stem]["cycles"])
    present = set(ev[stem].get("appliances_present", []))
    iv = ev[stem]["intervals"]

    shown = [a for a in ORDER if a in apps and (a in present or on[:, apps.index(a)].any())]
    fig, ax = plt.subplots(3, 1, figsize=(15, 4.0 + 0.52 * len(shown)),
                           gridspec_kw={"height_ratios": [2.0, 2.0, 0.42 * len(shown) + 0.6]})

    # ── 1) 총전력 ────────────────────────────────────────────────────────
    ax[0].plot(t, d["p_observed"], lw=1.0, color="black", label="관측 총전력")
    ax[0].plot(t, P.sum(1) + d["standby"].sum(1) + d["p_noise"], lw=1.0, color="crimson",
               ls="--", label="예측 합계 (기기 + 대기 + 계측잡음)")
    ax[0].set_ylabel("W"); ax[0].legend(loc="upper right", fontsize=9)
    ax[0].set_title(f"{stem}  —  {tag}", fontsize=13, weight="bold")
    ax[0].grid(alpha=0.25)

    # ── 2) 기기별 예측 전력 ──────────────────────────────────────────────
    cmap = plt.get_cmap("tab10")
    cols = {a: cmap(i % 10) for i, a in enumerate(ORDER)}
    js = [apps.index(a) for a in shown]
    ax[1].stackplot(t, *[P[:, j] for j in js],
                    labels=[KOR.get(a, a) + ("" if a in present else " (없는 기기)") for a in shown],
                    colors=[cols[a] for a in shown], alpha=0.85)
    ax[1].set_ylabel("기기별 예측 W")
    ax[1].legend(loc="upper right", fontsize=8, ncol=2)
    ax[1].grid(alpha=0.25)

    # ── 3) 타임라인 ──────────────────────────────────────────────────────
    a3 = ax[2]
    for row, app in enumerate(shown):
        j = apps.index(app)
        y = len(shown) - 1 - row
        # 정답
        if app in present and iv.get(app, {}).get("on"):
            for s0, s1 in iv[app]["on"]:
                a3.broken_barh([(s0, s1 - s0)], (y + 0.52, 0.34), color="0.35")
            for s0, s1 in iv[app].get("uncertain", []):
                a3.broken_barh([(s0, s1 - s0)], (y + 0.52, 0.34),
                               facecolor="0.85", hatch="///", edgecolor="0.5", lw=0.4)
        elif app in present:
            a3.text(0.2, y + 0.60, "정답 구간 없음", fontsize=7, color="0.5")
        else:
            a3.text(0.2, y + 0.60, "이 파일에 없는 기기 (정답 = 항상 OFF)", fontsize=7, color="firebrick")
        # 예측
        col = cols[app] if app in present else "firebrick"
        for s0, s1 in _spans(on[:, j], t):
            a3.broken_barh([(s0, max(s1 - s0, 0.4))], (y + 0.12, 0.34), color=col)
        a3.text(-0.012, y + 0.5, KOR.get(app, app), ha="right", va="center",
                transform=a3.get_yaxis_transform(), fontsize=9)

    a3.set_ylim(0, len(shown)); a3.set_yticks([])
    a3.set_xlim(0, n_cycles / 60.0); a3.set_xlabel("시간 (초)")
    ax[0].set_xlim(0, n_cycles / 60.0); ax[1].set_xlim(0, n_cycles / 60.0)
    a3.grid(axis="x", alpha=0.25)
    a3.legend(handles=[Patch(color="0.35", label="정답 ON (위 줄)"),
                       Patch(facecolor="0.85", hatch="///", edgecolor="0.5", label="판정보류"),
                       Patch(color="tab:blue", label="예측 ON (아래 줄)"),
                       Patch(color="firebrick", label="없는 기기에 붙인 예측")],
              loc="upper right", fontsize=8, ncol=4)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"timeline_{stem}_{tag}.png"
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="실측 예측 타임라인 플롯")
    ap.add_argument("--ckpt", default="results/adapt_ph1.pt")
    ap.add_argument("--ckpt-smps", default="results/cnn_ov1.pt", metavar="PT",
                    help="SMPS 3종만 이 체크포인트로 예측한다 (run_live --ckpt-smps 와 동일). "
                         "빈 문자열을 주면 --ckpt 단독")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--out", default="results/plots")
    a = ap.parse_args()

    _korean_font()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, ck = load_model(a.ckpt, dev)
    model_s = None
    if a.ckpt_smps:
        model_s, apps_s, ck_s = load_model(a.ckpt_smps, dev)
        if list(apps_s) != list(apps):
            raise SystemExit("두 체크포인트의 가전 목록이 다릅니다")
        print(f"  SMPS 3종은 {a.ckpt_smps} 에서 가져옵니다 -> {', '.join(SMPS_GROUP)}")
    tag = a.tag or (Path(a.ckpt).stem if not a.ckpt_smps
                    else f"{Path(a.ckpt).stem}+{Path(a.ckpt_smps).stem}")
    ev = load_events()
    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        p = plot_file(model, apps, stem, ev, dev, Path(a.out), tag,
                      stride=a.stride, model_smps=model_s)
        print(f"  저장 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
