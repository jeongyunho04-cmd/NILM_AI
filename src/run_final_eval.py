"""봉인 파일 최종 평가 — 개봉 1회 (설계 문서 4.3절, 12.105)

`test.csv` 는 최종 평가 전용으로 봉인돼 있었다. **이 스크립트가 그것을 연다.**
`processed_data/SEAL_BROKEN.json` 에 사유가 기록된다.

**개봉 근거 (사용자 판단, 2026-08-25).** 4.3절이 지목한 구조적 결함 — 봉인 파일만
234V/0.45Ω 다른 콘센트이고 에어컨·드라이기가 검증셋에 없다 — 때문에 이 파일의
숫자는 원래도 해석이 제한적이었다. 그리고 **더 나은 실측을 새로 만들 수 있으므로**
봉인을 유지할 값어치가 낮다고 판단했다.

**기기별 정답이 없다.** `real_events.json` 에 `test` 항목이 없다. 그래서 여기서
재는 것은 라벨이 덜 필요한 둘이다:

    총전력 잔차          |관측 − 예측 합계|
    없는 기기 전력        오븐·핫플·포트·프로젝터·미니PC 는 이 파일에 없다

이벤트는 `TEST_DATASET_TIMELINE_ANALYSIS.txt` 의 신호 유추 타임라인이다
(사람 기록이 아니다 — test_9 와 같은 등급).

    python -m src.run_final_eval
"""
from typing import List
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

from src.evaluation.sealing import seal_status, unseal
from src.model.postproc import apply_postproc
from src.run_gate_check import forward_file, gated, load_model, merge_smps
from src.run_plot_real import _font  # 한글 폰트 설정

STEM = "test"
#: 이 파일에 실제로 있던 기기 (타임라인 문서 [1]절).
PRESENT = ("air_conditioner", "hair_dryer", "laptop_charger", "fan")
#: 신호 유추 타임라인의 이벤트 (t초, 기기, on/off, ΔP W).
EVENTS = [
    (33.0, "air_conditioner", "on", +195.0),
    (195.0, "hair_dryer", "on", +965.6),
    (213.0, "hair_dryer", "off", -960.0),
    (657.0, "air_conditioner", "off", -766.2),
    (664.0, "hair_dryer", "on", +990.5),
    (679.0, "hair_dryer", "warm", -480.9),
    (694.0, "hair_dryer", "off", -511.0),
    (837.0, "air_conditioner", "on", +220.0),
    (1205.0, "air_conditioner", "off", -762.5),
]
KO = {"air_conditioner": "에어컨", "hair_dryer": "드라이기", "laptop_charger": "충전기",
      "fan": "선풍기", "beam_projector": "프로젝터", "minipc": "미니PC",
      "electiric_kettle": "전기포트", "oven": "오븐", "hotplate": "핫플레이트"}


def main() -> int:
    ap = argparse.ArgumentParser(description="봉인 파일 최종 평가 (개봉 1회)")
    ap.add_argument("--ckpt", default="results/adapt_smpsf.pt")
    ap.add_argument("--ckpt-smps", default="", metavar="PT")
    ap.add_argument("--postproc", default="on", choices=("off", "on", "sync"))
    ap.add_argument("--absorb", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--reason", default="최종 평가 1회 — 운영점 adapt_smpsf + 후처리 "
                                        "(2026-08-25, 사용자 판단: 더 나은 실측을 새로 "
                                        "만들 수 있어 봉인 유지 값어치가 낮다)")
    ap.add_argument("--out", default="results/plots/final_test.png")
    a = ap.parse_args()

    _font()
    st = seal_status()
    print("=" * 92)
    print(f"[봉인 상태] 개봉 전: {'유지' if st['intact'] else '이미 열림'} "
          f"(개봉 기록 {st['openings']}회)")
    print("=" * 92)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, ck = load_model(a.ckpt, dev)
    m_smps = load_model(a.ckpt_smps, dev)[0] if a.ckpt_smps else None

    with unseal(a.reason):
        d = forward_file(model, STEM, dev, stride=a.stride)
        if m_smps is not None:
            d = merge_smps(d, forward_file(m_smps, STEM, dev, stride=a.stride), apps)

    P = gated(d, False)
    if a.postproc != "off":
        P, g = apply_postproc(P, d["gate"], apps, gate_sync=(a.postproc == "sync"))
    else:
        g = d["gate"]
    if a.absorb > 0:
        from src.run_gate_check import _signatures
        from src.model.postproc import absorb_residual
        P = absorb_residual(P, g, apps, d["standby"], d["p_noise"], d["p_observed"],
                            d["obs_harm"], *_signatures(apps), frac=a.absorb)

    t = d["targets"] / 60.0
    total = P.sum(1) + d["standby"].sum(1) + d["p_noise"]
    resid = d["p_observed"] - total
    absent = [j for j, x in enumerate(apps) if x not in PRESENT]

    print(f"\n  모델 {a.ckpt} (stage {ck.get('stage', 1)}) | 후처리 {a.postproc}"
          f" | 흡수 {a.absorb:g}")
    print(f"  창 {len(t):,}개 | 길이 {t[-1]:.0f}초 | 관측 평균 {d['p_observed'].mean():.1f}W")
    print(f"\n  총전력 잔차   평균 {resid.mean():+.2f}W   절대 평균 {np.abs(resid).mean():.2f}W"
          f"   p95 {np.percentile(np.abs(resid), 95):.1f}W")
    print(f"  없는 기기 전력 합계 {P[:, absent].mean(0).sum():.2f}W "
          f"(예측 총합의 {100*P[:, absent].mean(0).sum()/max(P.mean(0).sum(), 1e-9):.1f}%)")
    print(f"\n  {'기기':10s}{'평균W':>9s}{'p95W':>9s}{'있음?':>7s}")
    for j, x in enumerate(apps):
        mark = "O" if x in PRESENT else "-"
        print(f"  {KO.get(x, x):10s}{P[:, j].mean():>9.2f}{np.percentile(P[:, j], 95):>9.2f}{mark:>7s}")

    # ── 그림 ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1.4, 3]})
    ax[0].plot(t, d["p_observed"], color="0.35", lw=0.8, label="관측 총전력")
    ax[0].plot(t, total, color="crimson", lw=1.0, label="예측 합계 (활성+대기)")
    ax[0].set_ylabel("전력 (W)"); ax[0].legend(loc="upper right", fontsize=9)
    ax[0].grid(alpha=.3)
    ax[0].set_title(f"{STEM}.csv (봉인 해제) — {a.ckpt} | 후처리 {a.postproc}"
                    f"{' + 흡수 ' + format(a.absorb, 'g') if a.absorb else ''}")

    ax[1].fill_between(t, 0, np.clip(resid, 0, None), color="salmon", label="과소 예측")
    ax[1].fill_between(t, 0, np.clip(resid, None, 0), color="steelblue", label="과대 예측")
    ax[1].axhline(0, color="0.3", lw=.8)
    ax[1].set_ylabel("잔차 (W)"); ax[1].legend(loc="upper right", fontsize=8); ax[1].grid(alpha=.3)
    ax[1].annotate(f"평균 {resid.mean():+.1f}W   절대평균 {np.abs(resid).mean():.1f}W",
                   (0.01, 0.08), xycoords="axes fraction", fontsize=9)

    order = sorted(range(len(apps)), key=lambda j: -P[:, j].mean())
    ax[2].stackplot(t, *[P[:, j] for j in order],
                    labels=[KO.get(apps[j], apps[j]) + ("" if apps[j] in PRESENT else " (없는 기기)")
                            for j in order],
                    colors=[plt.cm.tab10(j % 10) for j in order])
    ax[2].plot(t, d["p_observed"], color="0.25", lw=0.6, alpha=.7)
    ax[2].set_ylabel("기기별 예측 (W)"); ax[2].set_xlabel("시간 (초)")
    ax[2].legend(loc="upper right", fontsize=8, ncol=3); ax[2].grid(alpha=.3)

    for ts, app, kind, dp in EVENTS:
        for k in (0, 2):
            ax[k].axvline(ts, color="crimson", ls="--", lw=0.9, alpha=.5)
        ax[0].annotate(f"{KO.get(app, app)} {kind} {dp:+.0f}W", (ts, ax[0].get_ylim()[1]),
                       fontsize=6.5, color="crimson", rotation=90,
                       ha="right", va="top", rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"\n  저장: {a.out}")
    print(f"  개봉 기록: processed_data/SEAL_BROKEN.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
