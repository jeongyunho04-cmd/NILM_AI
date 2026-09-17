# -*- coding: utf-8 -*-
"""판 비교 그림 — **씨앗 점을 다 찍는다** (14.404).

⚠ 평균만 그리면 씨앗 바닥이 안 보인다. 오늘 §60 이 그것으로 ①을 과하게 읽었다
([[measure-the-seed-floor-before-reading-any-effect]]). 그래서 **씨앗마다 점**을 찍고
**짝지은 선**으로 잇는다 — 같은 씨앗끼리 어떻게 움직였는지가 판정의 자다.

    python -X utf8 -m src.run_plot_arms
"""
import numpy as np

from src import env_guard  # noqa: F401

from src.run_plot_real import _font  # noqa: E402

#: 실측 AUC (`run_scorecard` 규약 · 5파일 통합 · scorable 마스크) — 씨앗 6
AUC = {
    "cnn_comb": {"minipc": [.784, .784, .769, .783, .770, .786],
                 "laptop_charger": [.997, .996, .997, .992, .997, .998],
                 "beam_projector": [.991, .976, .990, .982, .966, .971],
                 "hotplate": [.910, .974, .917, .940, .970, .891]},
    "cnn_pjit": {"minipc": [.744, .712, .755, .736, .749, .753],
                 "laptop_charger": [.987, .995, .996, .993, .995, .995],
                 "beam_projector": [.995, .995, .993, .997, .997, .998],
                 "hotplate": [.955, .964, .898, .952, .960, .937]},
    "cnn_amp3": {"minipc": [.765, .745, .784, .773, .728, .771],
                 "laptop_charger": [.995, .993, .996, .993, .991, .993],
                 "beam_projector": [.988, .983, .996, .985, .989, .984],
                 "hotplate": [.926, .961, .921, .945, .958, .834]},
}
#: 그 판이 그 기기에 **실제로 넣은 지터 진폭** (도/차수)
JIT = {"cnn_comb": {"minipc": 4.0, "laptop_charger": 4.0, "beam_projector": 4.0},
       "cnn_pjit": {"minipc": 0.94, "laptop_charger": 2.13, "beam_projector": 3.66},
       "cnn_amp3": {"minipc": 1.16, "laptop_charger": 4.73, "beam_projector": 4.76}}
ARMS = ("cnn_comb", "cnn_pjit", "cnn_amp3")
NAME = {"minipc": "미니PC", "laptop_charger": "충전기",
        "beam_projector": "빔프로젝터", "hotplate": "핫플 (대조군 · 지터 안 넣음)"}
COL = {"cnn_comb": "#4C72B0", "cnn_pjit": "#DD8452", "cnn_amp3": "#55A868"}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _font()

    apps = ("minipc", "laptop_charger", "beam_projector", "hotplate")
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.2))

    # ── 윗줄: 판별 AUC · **씨앗마다 점 + 짝지은 선** ─────────────────────────
    for c, app in enumerate(apps):
        a = ax[0, c]
        for s in range(6):                       # 같은 씨앗을 잇는다
            a.plot(range(3), [AUC[m][app][s] for m in ARMS],
                   "-", color="0.75", lw=.9, zorder=1)
        for i, m in enumerate(ARMS):
            v = np.array(AUC[m][app])
            a.scatter([i] * 6, v, s=46, color=COL[m], zorder=3,
                      edgecolor="white", linewidth=.8)
            a.plot([i - .22, i + .22], [v.mean()] * 2, color=COL[m], lw=3, zorder=4)
        a.set_xticks(range(3))
        a.set_xticklabels(["comb\n(±4.0 일괄)", "pjit\n(v50 깎음)", "amp3\n(v51 √3)"],
                          fontsize=9)
        a.set_title(NAME[app], fontsize=12, pad=8)
        a.grid(alpha=.25, axis="y")
        if c == 0:
            a.set_ylabel("실측 AUC  (5파일 통합)", fontsize=11)
        #: 짝지은 t 를 적는다 — 눈으로 볼 때 폭과 같이 봐야 한다
        for i, (m0, m1) in enumerate(((0, 1), (1, 2))):
            d = np.array(AUC[ARMS[m1]][app]) - np.array(AUC[ARMS[m0]][app])
            t = d.mean() / (d.std(ddof=1) / np.sqrt(6)) if d.std(ddof=1) > 0 else 0.0
            a.annotate("Δ %+.4f\nt %+.1f" % (d.mean(), t), (m0 + .5, 0.02),
                       xycoords=("data", "axes fraction"), ha="center", fontsize=8.5,
                       color=("#C44E52" if abs(t) >= 2 else "0.45"),
                       fontweight=("bold" if abs(t) >= 2 else "normal"))

    # ── 아랫줄: **지터 진폭 대 AUC** — 최적점이 기기마다 다른가 ──────────────
    for c, app in enumerate(apps):
        a = ax[1, c]
        if app == "hotplate":
            a.axis("off")
            a.annotate("핫플에는 지터를 **안 넣었다**.\n그런데도 씨앗 폭이 0.834~0.974 다 —\n"
                       "**씨앗 바닥이 이만큼**이라는 뜻이고,\n위 판정은 그 폭과 같이 봐야 한다.",
                       (.02, .62), xycoords="axes fraction", fontsize=11, color="0.25")
            continue
        xs = [JIT[m][app] for m in ARMS]
        ys = [np.mean(AUC[m][app]) for m in ARMS]
        es = [np.std(AUC[m][app], ddof=1) / np.sqrt(6) for m in ARMS]
        o = np.argsort(xs)
        a.plot(np.array(xs)[o], np.array(ys)[o], "-", color="0.6", lw=1.2, zorder=1)
        for i, m in enumerate(ARMS):
            a.errorbar(xs[i], ys[i], yerr=es[i], fmt="o", ms=11, color=COL[m],
                       capsize=4, zorder=3, label=m.replace("cnn_", ""))
        best = ARMS[int(np.argmax(ys))]
        a.annotate("최적 **%s**\n(진폭 %.2f)" % (best.replace("cnn_", ""), JIT[best][app]),
                   (.04, .06), xycoords="axes fraction", fontsize=10, color="#C44E52")
        a.set_xlabel("주입한 위상 지터 진폭  a  [도/차수]", fontsize=10)
        a.grid(alpha=.25)
        if c == 0:
            a.set_ylabel("실측 AUC 평균 (±SE)", fontsize=11)
        a.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("판 셋 비교 — 실측 AUC · 씨앗 6개를 **전부** 찍었다 (회색 선 = 같은 씨앗)",
                 fontsize=13.5, y=.985)
    fig.tight_layout(rect=(0, 0, 1, .96))
    p = "results/plots/arms_auc_seeds.png"
    fig.savefig(p, dpi=115, bbox_inches="tight")
    print("  저장 %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
