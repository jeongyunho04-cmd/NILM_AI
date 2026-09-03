"""12.155.4 그림 — 정밀화 라벨 재채점과 새 장소 모델

    python -X utf8 -m src.run_plot_rescore --out results/plots/rescore.png
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

SUF = "+pp+rm0.02"
A = ["test_5", "test_6", "test_7", "test_8", "test_11", "test_12", "test_13", "test_14"]
B = ["test_15", "test_16", "test_17", "test_18"]
NAMES = {"adapt_ovh": "기준선", "adapt_zi_s0": "Z·I (장소A) s0",
         "adapt_zi_s1": "Z·I (장소A) s1", "adapt_zi_s2": "Z·I (장소A) s2",
         "adapt_zi_newsite": "Z·I (장소B 재적응)"}
ORDER = ["adapt_ovh", "adapt_zi_s0", "adapt_zi_s1", "adapt_zi_s2", "adapt_zi_newsite"]
#: 12.155.4 에서 잰 값. 그림에 넣으려고 상수로 둔다 (재계산은 오래 걸린다).
DP_REL = {"adapt_ovh": (0.998, 0.718), "adapt_zi_s0": (0.861, 0.474),
          "adapt_zi_newsite": (0.688, 0.457)}


def _korean_font() -> None:
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
        if any(name.lower() in f.name.lower()
               for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", default="results/gc_ref.json")
    ap.add_argument("--out", default="results/plots/rescore.png")
    a = ap.parse_args()
    _korean_font()

    gc = json.load(open(a.gc, encoding="utf-8"))
    G = {k[:-len(SUF)]: v for k, v in gc.items() if k != "_config" and k.endswith(SUF)}
    mods = [m for m in ORDER if m in G]

    def agg(m, S, key):
        return float(np.mean([G[m][s]["soft"][key] for s in S if s in G[m]]))

    def oven_ghost(m, S):
        return float(np.mean([G[m][s]["soft"]["absent"].get("oven", {}).get("mean_w", 0.0)
                              for s in S if s in G[m]]))

    fig, ax = plt.subplots(1, 3, figsize=(17.0, 5.4))
    x = np.arange(len(mods))
    w = 0.38

    # ① 유령 — 장소 A vs B
    ax[0].bar(x - w / 2, [agg(m, A, "absent_sum_w") for m in mods], w,
              color="#3b6ea5", label="장소 A (기존 8파일)")
    ax[0].bar(x + w / 2, [agg(m, B, "absent_sum_w") for m in mods], w,
              color="#b4453c", label="장소 B (신규 4파일)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([NAMES[m] for m in mods], rotation=22, ha="right", fontsize=8.5)
    ax[0].set_ylabel("없는 기기에 준 전력 (W)")
    ax[0].set_title("① 유령 — 장소를 옮기면 20배가 된다\n"
                    "그리고 장소 B 의 Z 로 재적응하면 3분의 1 이하로 떨어진다",
                    fontsize=11, weight="bold")
    ax[0].legend(fontsize=8.5)
    ax[0].grid(axis="y", alpha=0.25)
    for i, m in enumerate(mods):
        ax[0].annotate(f"{agg(m, B, 'absent_sum_w'):.0f}",
                       (x[i] + w / 2, agg(m, B, "absent_sum_w")),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    # ② 장소 B 의 유령은 **오븐**이다 — 그 집에 없는 기기
    ov = [oven_ghost(m, B) for m in mods]
    rest = [agg(m, B, "absent_sum_w") - o for m, o in zip(mods, ov)]
    ax[1].bar(x, ov, 0.6, color="#7a4b8f", label="오븐 — 장소 B 에 없는 기기")
    ax[1].bar(x, rest, 0.6, bottom=ov, color="#c9c9c9", label="나머지 (주로 에어컨)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([NAMES[m] for m in mods], rotation=22, ha="right", fontsize=8.5)
    ax[1].set_ylabel("유령 전력 (W)")
    ax[1].set_title("② 그 유령의 정체 — 오븐\n"
                    "재적응이 오븐 유령만 90% 지운다",
                    fontsize=11, weight="bold")
    ax[1].legend(fontsize=8.5)
    ax[1].grid(axis="y", alpha=0.25)
    for i, o in enumerate(ov):
        ax[1].annotate(f"{o:.0f}", (x[i], o / 2), ha="center", va="center",
                       fontsize=9, color="white", weight="bold")

    # ③ 이벤트 ΔP 상대오차 — 기존 라벨 vs 정밀화 라벨
    md = [m for m in ORDER if m in DP_REL]
    x3 = np.arange(len(md))
    ax[2].bar(x3 - w / 2, [DP_REL[m][0] for m in md], w, color="#9a9a9a",
              label="기존 라벨 (n=37)")
    ax[2].bar(x3 + w / 2, [DP_REL[m][1] for m in md], w, color="#2f6f4f",
              label="정밀화 라벨 (n=126)")
    ax[2].set_xticks(x3)
    ax[2].set_xticklabels([NAMES[m] for m in md], rotation=22, ha="right", fontsize=8.5)
    ax[2].set_ylabel("이벤트 ΔP |상대오차| 중앙")
    ax[2].set_title("③ 기존 라벨이 개선을 가리고 있었다\n"
                    "기준선 대비 이득 14% -> 34%",
                    fontsize=11, weight="bold")
    ax[2].legend(fontsize=8.5)
    ax[2].grid(axis="y", alpha=0.25)
    for i, m in enumerate(md):
        ax[2].annotate(f"{DP_REL[m][0]:.2f}", (x3[i] - w / 2, DP_REL[m][0]),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax[2].annotate(f"{DP_REL[m][1]:.2f}", (x3[i] + w / 2, DP_REL[m][1]),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    fig.suptitle("12.155.4  정밀화 라벨로 재채점 — 그리고 장소 B 재적응 모델",
                 fontsize=13.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
