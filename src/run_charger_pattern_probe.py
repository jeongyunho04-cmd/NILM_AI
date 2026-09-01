"""충전기의 시간 패턴 — **참값 없이** 쓸 수 있는 제약인가 (12.130)

배경
----
12.129 가 배분 난제를 프로젝터↔충전기(±17W)에서 충전기↔미니PC(±2.5W)로 옮겼다.
남은 것을 못 고치는 이유는 원리가 아니라 **자가 없어서**다 — 충전기 격리 통전이
33~68W (폭/중앙 0.722) 라 `power_ref` 에서 제외됐고, 프로젝터처럼 스냅을 못 건다.

그런데 격리 로그를 보면 **모든 통전 구간이 같은 모양**이다:

    0~30s    상승   21~52W
    30~60s   고원 진입
    60s~     거의 평평. **고원 높이가 구간 안에서는 안정적**이고 구간 사이에서만 다르다
             (laptop_charger_4 는 두 구간 다 68.3~68.7W, ±0.4W)

**그러면 참값 하나가 없어도 제약이 하나 생긴다** — *"통전 구간 안에서 충전기
전력은 거의 상수다"*. 이것은

    (1) 고조파 잔차를 안 쓴다        12.124.1 이 유일한 지렛대로 지목한 종류다
    (2) 참값이 필요 없다             절대값이 아니라 **평평함**을 건다
    (3) 관측에서 결정된다             통전 구간은 게이트가 준다 (규칙 31)
    (4) 값을 바꾼다                  기울기 조작이 아니다 (규칙 35)

이 스크립트가 그 제약의 **여지**를 잰다. 격리에서 참 전력이 얼마나 평평한지와,
모델이 복합에서 얼마나 흔들리는지를 같은 자로 비교한다. 후자가 크게 넘으면
그 차이가 곧 벌 수 있는 양이다.

    python -m src.run_charger_pattern_probe
"""
from pathlib import Path
from typing import Dict, List, Sequence
import argparse
import glob
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

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.preprocessing import classify_file, load_nilm_npz
from src.run_gate_check import forward_file, gated, load_model
from src.run_snap_reverses_probe import pipeline

RAMP_S = 60          # 상승 구간 — 고원 통계에서 뺀다
MIN_SEG_S = 90       # 이보다 짧은 구간은 고원을 못 본다


def _korean_font() -> None:
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
        if any(name.lower() in f.name.lower()
               for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def segments(mask: np.ndarray, gap: int = 60) -> List[np.ndarray]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1)


def isolated_plateaus(app: str = "laptop_charger") -> List[dict]:
    """격리 녹화의 통전 구간별 궤적과 고원 통계."""
    out: List[dict] = []
    for f in sorted(glob.glob("processed_data/npz/*.npz")):
        try:
            if classify_file(f).appliance_type != app:
                continue
        except Exception:
            continue
        z = load_nilm_npz(f)
        p = np.asarray(z["p_denoised_w"])
        m = (np.asarray(z["is_on"]).astype(bool)
             & np.asarray(z["is_valid"]).astype(bool) & (p > 1.0))
        for s in segments(m):
            if len(s) < MIN_SEG_S * 60:
                continue
            el = (s - s[0]) / 60.0
            pl = p[s][el >= RAMP_S]                       # 고원만
            if len(pl) < 60:
                continue
            out.append({"file": Path(f).stem, "dur_s": len(s) / 60.0,
                        "elapsed": el, "power": p[s],
                        "plateau_med": float(np.median(pl)),
                        "plateau_sd": float(np.std(pl)),
                        "plateau_iqr": float(np.subtract(*np.percentile(pl, [75, 25])))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/adapt_ovh.pt")
    ap.add_argument("--snap", type=float, default=46.9)
    ap.add_argument("--resmatch", type=float, default=0.02)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--app", default="laptop_charger")
    ap.add_argument("--out", default="results/charger_pattern.json")
    ap.add_argument("--plot", default="results/plots_op/charger_pattern.png")
    a = ap.parse_args()

    print("=" * 88)
    print(f"{a.app} 의 시간 패턴 — 참값 없이 쓸 수 있는 제약인가")
    print("=" * 88)

    iso = isolated_plateaus(a.app)
    print(f"\n[1] 격리 녹화 — 통전 구간 {len(iso)}개 ({MIN_SEG_S}s 이상), "
          f"고원 = 통전 {RAMP_S}s 이후")
    print(f"  {'녹화':<24s}{'길이s':>8s}{'고원 중앙W':>12s}{'고원 SD':>10s}"
          f"{'SD/중앙':>10s}")
    print("  " + "-" * 66)
    for r in iso:
        print(f"  {r['file']:<24s}{r['dur_s']:>8.0f}{r['plateau_med']:>12.1f}"
              f"{r['plateau_sd']:>10.2f}{r['plateau_sd'] / r['plateau_med']:>10.3f}")
    iso_cv = float(np.mean([r["plateau_sd"] / r["plateau_med"] for r in iso]))
    lv = [r["plateau_med"] for r in iso]
    print(f"\n  구간 **안**의 변동  SD/중앙 평균 {iso_cv:.3f}   <- 참 전력은 이만큼 평평하다")
    print(f"  구간 **사이**의 변동 고원 {min(lv):.1f}~{max(lv):.1f}W "
          f"(SD/중앙 {np.std(lv) / np.mean(lv):.3f})   <- 여기가 넓어 참값을 못 만든다")

    # ── 모델은 복합에서 얼마나 흔들리는가 ────────────────────────────────
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, _ = load_model(a.ckpt, dev)
    j = apps.index(a.app)
    ev = load_events()
    print(f"\n[2] 복합 실측 — 모델 예측이 통전 구간 안에서 얼마나 흔들리는가")
    print(f"  ({a.ckpt}, 운영점 + 스냅 {a.snap:g})")
    print(f"  {'파일':<10s}{'구간':>5s}{'길이s':>8s}{'중앙W':>9s}{'SD':>8s}{'SD/중앙':>10s}")
    print("  " + "-" * 52)
    rows: List[dict] = []
    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        d = forward_file(model, stem, dev, stride=a.stride)
        P = pipeline(d, apps, snap=a.snap, resmatch=a.resmatch)
        g = d["gate"][:, j] > 0.5
        for s in segments(g, gap=2):
            if len(s) * a.stride < MIN_SEG_S * 60:
                continue
            v = P[s, j]
            el = (s - s[0]) * a.stride / 60.0
            v = v[el >= RAMP_S]
            if len(v) < 3 or np.median(v) < 1.0:
                continue
            med, sd = float(np.median(v)), float(np.std(v))
            rows.append({"stem": stem, "med": med, "sd": sd, "cv": sd / med,
                         "n": len(v)})
            print(f"  {stem:<10s}{len(rows):>5d}{len(s) * a.stride / 60:>8.0f}"
                  f"{med:>9.1f}{sd:>8.2f}{sd / med:>10.3f}")
    if rows:
        mdl_cv = float(np.mean([r["cv"] for r in rows]))
        print(f"\n  모델 구간 안 변동  SD/중앙 평균 **{mdl_cv:.3f}**")
        print(f"  격리 참 전력       SD/중앙 평균   {iso_cv:.3f}")
        print(f"  -> 모델이 참보다 **{mdl_cv / max(iso_cv, 1e-9):.1f}배** 흔들린다. "
              f"그 차이가 평탄 제약으로 벌 수 있는 양이다.")

    # ── 그림 ─────────────────────────────────────────────────────────────
    _korean_font()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for r in iso:
        ax[0].plot(r["elapsed"], r["power"], lw=0.8, alpha=0.85,
                   label=f"{r['file']} ({r['dur_s']:.0f}s)")
    ax[0].axvspan(0, RAMP_S, color="0.85", zorder=0)
    ax[0].text(RAMP_S / 2, ax[0].get_ylim()[1] * 0.05, "상승", ha="center", fontsize=9)
    ax[0].set_xlabel("통전 후 경과 (초)"); ax[0].set_ylabel("전력 W")
    ax[0].set_title("격리 녹화 — 구간마다 상승 뒤 **고원**", fontsize=12, weight="bold")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25); ax[0].set_xlim(0, 900)

    if rows:
        ax[1].bar(np.arange(len(rows)), [r["cv"] for r in rows],
                  color="tab:orange", label=f"모델 예측 (복합, n={len(rows)})")
        ax[1].axhline(iso_cv, color="black", ls="--", lw=1.6,
                      label=f"격리 참 전력 {iso_cv:.3f}")
        ax[1].set_xlabel("통전 구간"); ax[1].set_ylabel("구간 안 SD / 중앙")
        ax[1].set_title("구간 안에서 얼마나 흔들리는가 (낮을수록 평평)",
                        fontsize=12, weight="bold")
        ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25, axis="y")
    Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.plot, dpi=130); plt.close(fig)
    print(f"\n  그림 저장: {a.plot}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "isolated": [{k: v for k, v in r.items() if k not in ("elapsed", "power")}
                     for r in iso],
        "isolated_within_cv": iso_cv,
        "isolated_between_cv": float(np.std(lv) / np.mean(lv)),
        "model_segments": rows,
        "model_within_cv": (float(np.mean([r["cv"] for r in rows])) if rows else None),
        "_config": {"argv": sys.argv, "args": vars(a)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
