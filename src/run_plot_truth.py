"""
실측 정답 감사용 플롯 — 예측 없이 정답과 신호만
==================================================
12.25절에서 미니PC 라벨이 신호와 모순되는 것을 찾았다. 나머지 라벨도 눈으로
확인할 수 있게, **예측을 빼고** 정답과 그 근거만 크게 그린다.

파일마다 네 칸:

  1) 총전력 전체 스케일        — 오븐/핫플 같은 고부하가 보인다
  2) **총전력 저전력 확대**     — 여기가 핵심이다. 1번에서는 20~50W 전환이
                                오븐 펄스에 눌려 안 보인다
  3) `I3` (3차 고조파 전류)     — SMPS 부하의 존재량. 저항 부하는 거의 안 올린다
  4) 정답 타임라인 + **신호에서 검출한 전환점**

4번의 전환점은 라벨이 아니라 신호에서 뽑은 것이다. 라벨이 놓친 전환이 있으면
여기서 드러난다.

    python -m src.run_plot_truth
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
from matplotlib.patches import Patch
import numpy as np

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.preprocessing import load_nilm_npz

KOR = {"oven": "오븐", "hotplate": "핫플레이트", "electiric_kettle": "전기포트",
       "hair_dryer": "헤어드라이기", "minipc": "미니PC", "beam_projector": "빔프로젝터",
       "laptop_charger": "노트북충전기", "fan": "선풍기", "air_conditioner": "에어컨"}
ORDER = ["oven", "hotplate", "electiric_kettle", "hair_dryer",
         "air_conditioner", "beam_projector", "laptop_charger", "minipc", "fan"]


def _korean_font() -> None:
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
        if any(name.lower() in f.name.lower() for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11


def detect_steps(p_sec: np.ndarray, i3_sec: np.ndarray, thr_w: float = 4.0):
    """1초 중앙값 계열에서 계단을 찾는다.

    `I3` 가 함께 뛰면 SMPS, 안 뛰면 저항성이다 — 이 구분이 라벨 감사의 핵심이다
    (12.25절: test3 t=61s 의 +14.2W 는 `I3` 가 +0.0017 이라 오븐 팬이었다).
    """
    out = []
    d, di = np.diff(p_sec), np.diff(i3_sec)
    for t in range(len(d)):
        if np.isnan(d[t]) or abs(d[t]) < thr_w:
            continue
        # SMPS 판정: 1W 당 I3 가 0.002 이상 움직이면 SMPS 로 본다
        kind = "SMPS" if abs(di[t]) / max(abs(d[t]), 1e-6) > 0.002 else "저항성"
        out.append((t + 1, float(d[t]), kind))
    return out


def plot_file(stem: str, ev: dict, out_dir: Path, zoom_w: float = 150.0) -> Path:
    r = load_nilm_npz(f"processed_data/composite_eval/{stem}.npz")
    pf, hc = r["power_features"], r["harmonics_complex"]
    p, n = pf[:, 0], len(pf)
    t = np.arange(n) / 60.0
    i3 = np.abs(hc[:, 2])

    nsec = n // 60
    p_sec = np.array([np.median(p[i * 60:(i + 1) * 60]) for i in range(nsec)])
    i3_sec = np.array([np.median(i3[i * 60:(i + 1) * 60]) for i in range(nsec)])
    steps = detect_steps(p_sec, i3_sec)

    present = set(ev[stem].get("appliances_present", []))
    iv = ev[stem]["intervals"]
    shown = [a for a in ORDER if a in present]

    fig, ax = plt.subplots(4, 1, figsize=(17, 12),
                           gridspec_kw={"height_ratios": [2.2, 3.0, 2.2, 0.75 * len(shown) + 1.0]})
    dur = n / 60.0

    ax[0].plot(t, p, lw=0.8, color="black")
    ax[0].set_ylabel("총전력 W\n(전체)", fontsize=12)
    ax[0].set_title(f"{stem}  —  실측 정답 감사 (예측 없음)", fontsize=16, weight="bold")

    ax[1].plot(t, p, lw=1.0, color="black")
    ax[1].set_ylim(0, zoom_w)
    ax[1].set_ylabel(f"총전력 W\n(0~{zoom_w:.0f}W 확대)", fontsize=12)
    ax[1].axhline(7.8, color="tab:blue", ls=":", lw=1.2)
    ax[1].text(dur * 0.995, 9.0, "미니PC 최저 유휴 7.8W", ha="right", fontsize=9, color="tab:blue")
    ax[1].axhline(48.8, color="tab:brown", ls=":", lw=1.2)
    ax[1].text(dur * 0.995, 50.5, "프로젝터 단독 48.8W", ha="right", fontsize=9, color="tab:brown")

    ax[2].plot(t, i3, lw=0.9, color="tab:purple")
    ax[2].set_ylabel("I3 (A)\nSMPS 존재량", fontsize=12)

    for k in range(3):
        ax[k].grid(alpha=0.3)
        ax[k].set_xlim(0, dur)
        for sec, dw, kind in steps:
            ax[k].axvline(sec, color="tab:red" if kind == "SMPS" else "tab:green",
                          lw=0.8, alpha=0.55, ls="--")

    a3 = ax[3]
    for row, app in enumerate(shown):
        y = len(shown) - 1 - row
        spec = iv.get(app, {})
        for s0, s1 in spec.get("on", []):
            a3.broken_barh([(s0, s1 - s0)], (y + 0.25, 0.5), color="tab:blue", alpha=0.9)
        for s0, s1 in spec.get("uncertain", []):
            a3.broken_barh([(s0, s1 - s0)], (y + 0.25, 0.5), facecolor="0.88",
                           hatch="///", edgecolor="0.55", lw=0.5)
        if app == "oven" and spec.get("_heater_pulses"):
            for s0, s1 in spec["_heater_pulses"]:
                a3.broken_barh([(s0, s1 - s0)], (y + 0.30, 0.18), color="crimson")
        if not spec.get("on") and not spec.get("uncertain"):
            a3.text(dur * 0.01, y + 0.5, "정답 구간이 아예 없음", fontsize=10,
                    color="firebrick", va="center")
        a3.text(-0.008, y + 0.5, KOR.get(app, app), ha="right", va="center",
                transform=a3.get_yaxis_transform(), fontsize=12, weight="bold")
        a3.axhline(y, color="0.85", lw=0.6)

    for sec, dw, kind in steps:
        a3.axvline(sec, color="tab:red" if kind == "SMPS" else "tab:green", lw=0.8,
                   alpha=0.55, ls="--")
    a3.set_ylim(0, len(shown)); a3.set_yticks([])
    a3.set_xlim(0, dur); a3.set_xlabel("시간 (초)", fontsize=13)
    a3.grid(axis="x", alpha=0.3)
    a3.legend(handles=[Patch(color="tab:blue", label="정답 ON"),
                       Patch(facecolor="0.88", hatch="///", edgecolor="0.55", label="판정보류"),
                       Patch(color="crimson", label="오븐 히터 통전"),
                       plt.Line2D([], [], color="tab:red", ls="--", label="신호 계단 (SMPS)"),
                       plt.Line2D([], [], color="tab:green", ls="--", label="신호 계단 (저항성)")],
              loc="upper right", fontsize=10, ncol=5)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"truth_{stem}.png"
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)

    smps = [s for s in steps if s[2] == "SMPS"]
    print(f"  {stem}: 신호 계단 {len(steps)}개 (SMPS {len(smps)} / 저항성 {len(steps)-len(smps)})"
          f"  -> {path}")
    for sec, dw, kind in steps:
        print(f"      t={sec:4d}s  ΔP {dw:+7.1f}W  {kind}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="실측 정답 감사 플롯 (예측 없음)")
    ap.add_argument("--zoom-w", type=float, default=150.0)
    ap.add_argument("--out", default="results/plots")
    a = ap.parse_args()
    _korean_font()
    ev = load_events()
    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        plot_file(stem, ev, Path(a.out), zoom_w=a.zoom_w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
