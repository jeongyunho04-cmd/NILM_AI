# -*- coding: utf-8 -*-
"""지금 상태를 한 장에 — **오차가 어디 있나** (13.84.73).

다음에 무엇을 고칠지 정하려면 오차 예산을 봐야 한다. 네 칸:

  A 기기별 on/off      실측 5파일. 누가 혼자 나쁜가
  B 미니PC 파일별      유령 파일(test_3·4) 대 미탐 파일(test_1·2). 자기보정 효과까지
  C 상태별 전력 오차   격리 녹화 (참 상태를 아는 유일한 자료). 빈 슬롯 고침 전/후
  D 맞바꿈             전력 오차 대 on/off 창별. 13.84.72 의 결정이 걸린 자리

⚠ A·B·D 는 **on/off 라벨**로만 잰다 — 실측 복합에는 기기별 참 전력이 없다 (13.84.68 [7]).
  C 만 전력이고 그것은 **격리 녹화**다. 이 비대칭이 지금 판단을 왜곡한다.

    python -X utf8 src/run_plot_status.py [--out results/_fig_status.png]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.model.losses import S_STATE
from src.run_diag_states import windows as iso_windows, PAIRS
from src.run_gate_check import load_model
from src.run_score_seq import FILES, RESISTIVE_ALL, SMPS_TRIO, score

#: 볼 판들. (표시 이름, 파일)
ARMS = [("cnn_v37 (1단계)", "results/cnn_v37.pt"),
        ("v36p_mask (대조)", "results/seq_v36p_mask.pt"),
        ("slot_init ①", "results/seq_slot_init.pt"),
        ("slot_both ①+②", "results/seq_slot_both.pt")]
SEQ = [a for a in ARMS if "cnn_v37" not in a[0]]
GHOST, MISS = ("test_3", "test_4"), ("test_1", "test_2")


def ko_font():
    from matplotlib import font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    for c in ("Malgun Gothic", "NanumGothic", "Gulim", "AppleGothic"):
        if c in names:
            plt.rcParams["font.family"] = c
            # 이 폰트들은 U+2212(진짜 빼기표)와 경고표를 안 갖고 있다 — ASCII 로 쓴다
            plt.rcParams["axes.unicode_minus"] = False
            return c
    return None


def state_power(cks, dev):
    """격리 녹화의 상태별 (참 W, 판별 예측 W). {(기기,상태): (참, {판: 예측})}"""
    cache = {app: iso_windows(stem) for app, stem in PAIRS}
    out = {}
    for lbl, cp in cks:
        if not os.path.exists(cp):
            continue
        ck = torch.load(cp, map_location=dev, weights_only=False)
        apps = list(ck["appliances"])
        ref = ck.get("ref", "results/cnn_v37.pt") if "heads" in ck else cp
        m = load_model(ref, dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
        m.load_state_dict(ck["model"]); m.eval()
        for app, _ in PAIRS:
            if cache.get(app) is None or app not in apps:
                continue
            F, W, S, P, _ = cache[app]
            K = apps.index(app)
            PW = []
            with torch.no_grad():
                for i in range(0, len(F), 256):
                    PW.append(m(torch.from_numpy(F[i:i + 256]).to(dev),
                                torch.from_numpy(W[i:i + 256]).to(dev))["power"].float().cpu())
            PW = torch.cat(PW).numpy()
            thr = 0.5 * np.median(P[P > 1]) if (P > 1).any() else 1.0
            for st in S_STATE.get(app, {}):
                mm = (S == st) & (P > thr)
                if mm.sum() < 3:
                    continue
                e = out.setdefault((app, st), [float(np.median(P[mm])), {}])
                e[1][lbl] = float(np.median(PW[mm][:, K]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/_fig_status.png")
    a = ap.parse_args()
    f = ko_font()
    print("한글 폰트: %s" % (f or "**없다 — 글자가 깨진다**"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("상태별 전력 재는 중...", flush=True)
    SP = state_power(ARMS, dev)
    print("실측 채점 중...", flush=True)
    cache, SC = {}, {}
    for lbl, cp in SEQ:
        if os.path.exists(cp):
            SC[lbl] = score(cp, dev, cache, detail=True)

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 10.5))
    fig.suptitle("NILM 오차 예산 — 2026-09-12 (13.84.73)", fontsize=15, fontweight="bold")

    # ── A 기기별 on/off ─────────────────────────────────────────────────────
    A = ax[0, 0]
    apps_order = list(SMPS_TRIO) + list(RESISTIVE_ALL) + ["fan"]
    x = np.arange(len(apps_order)); w = 0.8 / max(len(SC), 1)
    for i, (lbl, r) in enumerate(SC.items()):
        v = [np.mean([q[0] for q in r[3]["app"].get(ap_, {}).values()])
             if r[3]["app"].get(ap_) else np.nan for ap_ in apps_order]
        A.bar(x + i * w - 0.4 + w / 2, v, w, label=lbl)
    A.axhline(1.0, color="k", lw=0.6, ls=":")
    A.set_xticks(x); A.set_xticklabels([s[:9] for s in apps_order], rotation=30, ha="right")
    A.set_ylim(0.5, 1.02); A.set_ylabel("사슬 on/off 정확도")
    A.set_title("A. 기기별 on/off — **미니PC 만 혼자 나쁘다**", loc="left", fontweight="bold")
    A.axvspan(-0.5, 2.5, color="tab:red", alpha=0.05)
    A.text(1.0, 0.53, "SMPS", ha="center", color="tab:red", fontsize=9)
    A.text(4.5, 0.53, "저항 4종", ha="center", color="tab:blue", fontsize=9)
    A.legend(fontsize=8, loc="lower right"); A.grid(axis="y", alpha=0.3)

    # ── B 미니PC 파일별 ─────────────────────────────────────────────────────
    B = ax[0, 1]
    x = np.arange(len(FILES))
    for i, (lbl, r) in enumerate(SC.items()):
        v = [r[2][s][1] if s in r[2] else np.nan for s in FILES]
        B.bar(x + i * w - 0.4 + w / 2, v, w, label=lbl)
    B.set_xticks(x); B.set_xticklabels(FILES)
    B.set_ylim(0, 1.05); B.set_ylabel("미니PC 시간 정확도")
    B.set_title("B. 미니PC 파일별 — **test_2 가 0.2 다**", loc="left", fontweight="bold")
    for s, c, t in ((MISS, "tab:orange", "미탐(실패③)"), (GHOST, "tab:purple", "유령(실패②)")):
        i0 = FILES.index(s[0]); i1 = FILES.index(s[1])
        B.axvspan(min(i0, i1) - 0.45, max(i0, i1) + 0.45, color=c, alpha=0.08)
        B.text((i0 + i1) / 2, 1.0, t, ha="center", color=c, fontsize=9)
    B.legend(fontsize=8, loc="lower left"); B.grid(axis="y", alpha=0.3)

    # ── C 상태별 전력 오차 ──────────────────────────────────────────────────
    # ⚠ **모든 상태**를 본다. 처음에 눈에 띄는 6개만 보고 "2.5%" 라고 적었다가
    #   15개 전부로 재니 11.8% 였다 — 부분집합이 ① 에 유리한 쪽으로 치우쳐 있었다
    #   ([[check-conditioning-before-believing-a-fit]] 과 같은 종류).
    C = ax[1, 0]
    base = ARMS[1][0]
    keys = sorted(SP, key=lambda k: 100 * abs(SP[k][1].get(base, np.nan) - SP[k][0])
                  / max(SP[k][0], 1e-9))
    y = np.arange(len(keys)); h = 0.8 / len(ARMS)
    for i, (lbl, _) in enumerate(ARMS):
        v = [100 * abs(SP[k][1].get(lbl, np.nan) - SP[k][0]) / max(SP[k][0], 1e-9) for k in keys]
        C.barh(y + i * h - 0.4 + h / 2, v, h, label=lbl)
    C.set_yticks(y)
    C.set_yticklabels(["%s s%d (%.0fW)" % (k[0][:12], k[1], SP[k][0]) for k in keys], fontsize=8)
    C.set_xlim(0, 46); C.set_xlabel("|상대오차| %")
    C.axvline(10, color="k", lw=0.8, ls=":")
    C.text(10.4, len(keys) - 0.6, "10%", fontsize=8)
    C.set_title("C. 상태별 전력 오차 (격리 녹화, **15개 상태 전부**)", loc="left", fontweight="bold")
    med = {lbl: np.median([100 * abs(SP[k][1].get(lbl, np.nan) - SP[k][0]) / max(SP[k][0], 1e-9)
                           for k in SP if lbl in SP[k][1]]) for lbl, _ in ARMS}
    C.text(0.98, 0.02, "중앙  " + " · ".join("%s %.1f%%" % (l.split()[0], m) for l, m in med.items()),
           transform=C.transAxes, ha="right", fontsize=8.5,
           bbox=dict(fc="w", ec="0.7", alpha=0.9))
    C.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.10))
    C.grid(axis="x", alpha=0.3)

    # ── D 맞바꿈 ────────────────────────────────────────────────────────────
    D = ax[1, 1]
    for lbl, r in SC.items():
        D.scatter(med[lbl], r[1], s=170, zorder=3)
        D.annotate(lbl, (med[lbl], r[1]), textcoords="offset points", xytext=(9, 6), fontsize=9)
    D.axvline(med[ARMS[0][0]], color="tab:blue", ls="--", lw=1.2)
    D.text(med[ARMS[0][0]] + 0.15, D.get_ylim()[0] + 0.0005,
           "cnn_v37(1단계)의 전력 오차 — 사슬 머리가 없어 on/off 는 못 잰다",
           fontsize=8, color="tab:blue", va="bottom", rotation=90)
    D.set_xlabel("상태별 전력 오차 중앙 %  (왼쪽이 좋다)")
    D.set_ylabel("실측 on/off 창별 정확도  (위가 좋다)")
    D.set_title("D. 맞바꿈 — **1단계의 전력을 2단계가 잃는다**", loc="left", fontweight="bold")
    D.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.text(0.01, 0.005,
             "[주의] A·B·D 의 세로축은 **on/off** 다. 실측 복합에 기기별 참 전력이 없어서 "
             "전력은 C(격리 녹화)에서만 잰다 — 이 비대칭이 지금 판단을 왜곡한다 (13.84.68 [7]).",
             fontsize=8.5, color="tab:red")
    fig.savefig(a.out, dpi=130)
    print("저장: %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
