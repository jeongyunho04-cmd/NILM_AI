"""12.151 그림 — h1 항등식, 저항 고조파, 캐시 크기

    python -X utf8 -m src.run_plot_h1v --out results/plots/h1v.png
"""
from typing import Dict, List
import argparse
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _korean_font() -> None:
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
        if any(name.lower() in f.name.lower()
               for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


FILES = {
    "air_conditioner": ["air_conditioner"],
    "beam_projector": ["beam_projector", "beam_projector_2", "beam_projector_3_fixed"],
    "electiric_kettle": ["electiric_kettle", "electric_kettle_2_fixed"],
    "fan": ["fan_1", "fan_2", "fan_3"],
    "hair_dryer": ["hair_dryer_1", "hair_dryer_2", "hair_dryer_3"],
    "hotplate": ["hotplate_1", "hotplate_2", "hotplate_3_fixed"],
    "laptop_charger": ["laptop_charger_1", "laptop_charger_2",
                       "laptop_charger_3_fixed", "laptop_charger_4_fixed"],
    "minipc": ["minipc_1", "minipc_2", "minipc_3"],
    "oven": ["oven", "oven_2", "oven_3_fixed"],
}
SHORT = {"air_conditioner": "에어컨", "beam_projector": "프로젝터",
         "electiric_kettle": "전기포트", "fan": "선풍기", "hair_dryer": "드라이기",
         "hotplate": "핫플레이트", "laptop_charger": "충전기", "minipc": "미니PC",
         "oven": "오븐"}


def rec_voltage() -> Dict[str, float]:
    """격리 녹화의 통전 구간 선전압. 전력 변화가 작은 녹화는 뺀다."""
    out = {}
    for app, stems in FILES.items():
        vs = []
        for st in stems:
            f = f"data/{st}.csv"
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f, usecols=["p_w", "vrms"])
            p, v = d.p_w.values, d.vrms.values
            bg, top = np.percentile(p, 5), np.percentile(p, 95)
            # 문턱 20W — 프로젝터(46W)·충전기(33W)·선풍기(38W)를 살린다.
            # 미니PC 는 통전 폭이 12~18W 라 어느 문턱으로도 안 살아난다.
            if top - bg < 20:
                continue
            vs.append(float(np.median(v[p > bg + 0.6 * (top - bg)])))
        if vs:
            out[app] = float(np.mean(vs))
    return out


def cache_curve() -> List[tuple]:
    """(GB, 합성 MAE 평균, 표준편차, 이름, 시드수). 없는 판은 건너뛴다."""
    pans = [(0.0, "adapt_zl0", "λ=0"), (0.22, "adapt_zc3", "3k"),
            (0.73, "adapt_zc10", "10k"), (2.18, "adapt_zc30", "30k"),
            (20.9, "adapt_zi", "300k")]
    out = []
    for gb, tag, name in pans:
        v = []
        for s in range(3):
            p = f"results/{tag}_s{s}.json"
            if os.path.exists(p):
                v.append(json.load(open(p, encoding="utf-8"))
                         ["history"][-1]["synth"]["mae"])
        if v:
            out.append((gb, float(np.mean(v)), float(np.std(v)), name, len(v)))
    return out


#: 판정 칸에 올릴 판 — 이름은 `run_summarize_pan.NAMES` 와 맞춘다
VERDICT = [("adapt_ovh", "기준선 w4.0"), ("adapt_zi", "복소 Z·I (현행)"),
           ("adapt_zv", "  + h1 추종만"), ("adapt_znr", "  + h1 정규화만"),
           ("adapt_zvnr", "  + 정규화 + 추종"), ("adapt_vs", "h1 전압만 (Z 없음)")]


def panel(ax) -> None:
    """④ 판 표 — 프로젝터 오차와 유령8. `pc_v1`/`gc_v1` 이 없으면 비운다."""
    import re
    try:
        pc = json.load(open("results/pc_v1.json", encoding="utf-8"))
        gc = json.load(open("results/gc_v1.json", encoding="utf-8"))
    except Exception:
        ax.axis("off")
        return
    ctrl = ("test_9", "test_11", "test_12")
    rows = []
    for tag, name in VERDICT:
        pj = [v["summary"]["beam_projector"]["mean_err_w"] for k, v in pc.items()
              if k != "_config" and re.sub(r"_s\d+$", "", k) == tag]
        gh = []
        for k, v in gc.items():
            if k == "_config" or not k.endswith("+pp+rm0.02"):
                continue
            if re.sub(r"_s\d+$", "", k[:-len("+pp+rm0.02")]) != tag:
                continue
            w = [f["soft"]["absent_sum_w"] for s, f in v.items()
                 if not s.startswith("_") and s not in ctrl]
            if w:
                gh.append(float(np.mean(w)))
        if pj and gh:
            rows.append((name, float(np.mean(pj)), float(np.std(pj)),
                         float(np.mean(gh)), float(np.std(gh))))
    if not rows:
        ax.axis("off")
        return
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.19, [r[1] for r in rows], 0.36, xerr=[r[2] for r in rows],
            color="#3b6ea5", capsize=3, label="프로젝터 오차 W  (0 이 참값 46.9W)")
    ax.barh(y - 0.19, [r[3] for r in rows], 0.36, xerr=[r[4] for r in rows],
            color="#b4453c", capsize=3, label="유령8 W  (없는 기기에 준 전력)")
    ax.axvline(0, color="k", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("W")
    ax.set_title("④ 판정 — h1 처방 셋은 전부 **시드 폭 안**이다\n"
                 "정규화만 걸면 유령이 충전기로 옮겨 간다 (규칙 29)",
                 fontsize=11, weight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="x", alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/plots/h1v.png")
    a = ap.parse_args()
    _korean_font()

    from src.model.net import harmonic_signatures
    from src.synthesis.segment_pool import SegmentPool
    from src.run_resistive_vh_probe import RESISTIVE, ORDERS, load, split, cmedian
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    apps = pool.get_appliance_types()
    sig = harmonic_signatures(pool, apps)
    del pool

    fig, axs = plt.subplots(2, 2, figsize=(16.5, 11.0))
    ax = [axs[0][0], axs[0][1], axs[1][0]]
    ax4 = axs[1][1]

    # ── ① h1 함의 전압 vs 격리 녹화 전압 ────────────────────────────────
    rv = rec_voltage()
    imp = {a_: 1.0 / sig[j, 0, 0] for j, a_ in enumerate(apps)}
    order = sorted(apps, key=lambda x: imp[x])
    x = np.arange(len(order))
    w = 0.38
    ax[0].bar(x - w / 2, [imp[k] for k in order], w, color="#3b6ea5",
              label="지문 함의 전압  1/Re(sig[k,h1])")
    ax[0].bar(x + w / 2, [rv.get(k, np.nan) for k in order], w, color="#c4703a",
              label="격리 녹화 실측 전압")
    for i, k in enumerate(order):
        if k not in rv:
            # 미니PC 뿐이다 — 통전 폭이 12~18W 라 선전압을 정할 구간이 없다.
            ax[0].text(x[i] + w / 2, 206.0, "통전폭 12~18W\n못 잼", ha="center",
                       va="bottom", fontsize=6.5, color="0.45")
    ax[0].axhline(221.5, color="k", ls="--", lw=1.1)
    ax[0].text(len(order) - 0.4, 221.9, "적응 창 중앙 221.5V", ha="right", fontsize=8.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([SHORT[k] for k in order], rotation=30, ha="right", fontsize=9)
    ax[0].set_ylim(203, 244)
    ax[0].set_ylabel("전압 (V)")
    ax[0].set_title("① 지문의 h1 실수부는 그 녹화의 전압이다\n"
                    "212.9 ~ 238.1V, 폭 11.8% — 손실은 이걸 판별자로 쓰고 있다",
                    fontsize=11, weight="bold")
    ax[0].legend(fontsize=8.5, loc="upper left")
    ax[0].grid(axis="y", alpha=0.25)

    # ── ② 저항 고조파: 실측 / 옴 / 고정지문 ─────────────────────────────
    M, O, F = [], [], []
    for app, stems in RESISTIVE.items():
        k = apps.index(app)
        for st in stems:
            try:
                p, v, I, V = load(st, ORDERS)
            except Exception:
                continue
            on, off = split(p)
            if on.size == 0:
                continue
            dI = cmedian(I[on]) - cmedian(I[off])
            dP = float(np.median(p[on]) - np.median(p[off]))
            vh = np.median(V[on], 0)
            vr = float(np.median(v[on]))
            M.append(np.abs(dI) / dP * 1e3)
            O.append(vh / vr ** 2 * 1e3)
            F.append(np.abs(sig[k, [h - 1 for h in ORDERS], 0]
                            + 1j * sig[k, [h - 1 for h in ORDERS], 1]) * 1e3)
    M, O, F = np.median(M, 0), np.median(O, 0), np.median(F, 0)
    xo = np.arange(1, len(ORDERS))          # h1 은 척도가 100배라 뺀다
    w2 = 0.27
    ax[1].bar(xo - w2, M[1:], w2, color="#2f2f2f", label="실측 (ON-OFF 차분)")
    ax[1].bar(xo, O[1:], w2, color="#b4453c", label="옴 예측  $V_h/V_{rms}^2$")
    ax[1].bar(xo + w2, F[1:], w2, color="#3b6ea5", label="고정지문 (현행)")
    ax[1].set_xticks(xo)
    ax[1].set_xticklabels([f"h{h}" for h in ORDERS[1:]])
    ax[1].set_ylabel("와트당 전류 (mA/W)")
    ax[1].set_title("② 저항의 고조파는 $V_h/R$ 이 아니다\n"
                    f"h1 은 셋 다 {M[0]:.3f} mA/W 로 같다 (막대에서 뺌)",
                    fontsize=11, weight="bold")
    ax[1].legend(fontsize=8.5)
    ax[1].grid(axis="y", alpha=0.25)
    for i in range(1, len(ORDERS)):
        r = M[i] / max(O[i], 1e-9)
        ax[1].text(i, max(M[i], O[i], F[i]) * 1.06, f"x{r:.2f}", ha="center",
                   fontsize=7.5, color="#b4453c")

    # ── ③ 캐시 크기 ────────────────────────────────────────────────────
    cc = cache_curve()
    if cc:
        mu = [c[1] for c in cc]
        sd = [c[2] for c in cc]
        xs = np.arange(len(cc))
        ax[2].errorbar(xs, mu, yerr=sd, marker="o", ms=7, lw=1.8, capsize=4,
                       color="#2f6f4f")
        base = [c for c in cc if c[3] == "300k"]
        if base:
            ax[2].axhline(base[0][1], color="k", ls="--", lw=1.0)
            ax[2].text(0.02, base[0][1] + 0.06, "20.9GB 기준선", fontsize=8.5)
        ax[2].set_xticks(xs)
        ax[2].set_xticklabels([f"{c[3]}\n{c[0]:.2f}GB" for c in cc], fontsize=9)
        ax[2].set_ylabel("합성 홀드아웃 MAE (W)   낮을수록 좋다")
        ax[2].set_title("③ 현장 꾸러미는 21GB 가 아니다\n"
                        "replay 는 못 빼지만 100배 솎아도 같다",
                        fontsize=11, weight="bold")
        ax[2].grid(alpha=0.25)
        for i, c in enumerate(cc):
            ax[2].annotate(f"{c[1]:.2f}", (xs[i], mu[i]), textcoords="offset points",
                           xytext=(0, 12), ha="center", fontsize=8.5)

    # ── ④ 판정 (12.151.2) ──────────────────────────────────────────────
    panel(ax4)

    fig.suptitle("12.151  유효전력의 정의가 지문에 남긴 것 — 그리고 replay 절제",
                 fontsize=14.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=140)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
