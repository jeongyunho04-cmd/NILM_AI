# -*- coding: utf-8 -*-
"""**세밀 갈래가 보는 10초를 시간축으로 전부 편다** (14.157).

사용자: *"근데 그 지점의 입력만 보지 말고 세밀갈래가 볼 10초 동안을 모두 봐 줄 수 있어?"*

14.156 은 창을 **요약통계 4개**(타깃·평균·최대·최소)로 눌러서 봤다. 세밀 갈래가 실제로
보는 것은 **600샘플 × 10초**이고, 차이가 *언제* 있는지가 요점이다.

```
  세밀 창  타깃 −4.0초 ~ +6.0초   (600사이클 @60Hz, 타깃 index 239)
  광역 창  타깃 −54.0초 ~ +6.0초  (3600사이클을 2Hz 로)
```

세 가지를 한다.

```
  [A] **시간 해상 차이 지도**  채널마다 Δ(t)/σ(t) 를 600샘플 전부에 대해.
      σ(t) 는 그 동네 창들의 (채널,시각)별 표준편차다 — 시각마다 자연 변동폭이 다르다
  [B] **시간 구간별 되돌리기**  원시 채널을 **그 구간 안에서만** 앞 창 값으로 갈아 끼운다.
      광역 전용(−54~−4초) / 세밀 5토막(−4~−2, −2~0, 0~+2, +2~+4, +4~+6초).
      **어느 2초가 답을 정하는지**가 나온다. 0 으로 죽이지 않는다
  [C] 그림  핵심 채널의 10초 파형을 앞/뒤로 겹쳐 그린다
```

    python -X utf8 -m src.run_diag_finewindow --stem test_2 --at 231.4 --gap 2.5 \
        --app oven --ckpt results/cnn_pcap_s0.pt
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.inputs import build_inputs, fine_target_index  # noqa: E402
from src.model.net import P_CH_FINE, POWER_SCALE  # noqa: E402
from src.model.realdata import RealWindows, target_index  # noqa: E402
from src.preprocessing import load_nilm_npz  # noqa: E402
from src.run_diag_flipedge import SH, fine_names, fwd, raw_groups, raw_names  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

FS = 60
WIN = 3600
FINE = 600


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_pcap_s0.pt"])
    ap.add_argument("--stem", default="test_2")
    ap.add_argument("--at", type=float, default=231.4)
    ap.add_argument("--gap", type=float, default=2.5)
    ap.add_argument("--app", default="oven")
    ap.add_argument("--also", default="electiric_kettle", help="같이 찍을 기기")
    ap.add_argument("--out", default="results/plots")
    ap.add_argument("--even-median", type=int, default=0,
                    help="짝수차 중앙값을 **강제**한다 (14.160) — 분포 밖 시험용. "
                         "안 주면 체크포인트가 적어 둔 값을 따라간다")
    a = ap.parse_args()
    from src.run_gate_check import sync_even_median
    sync_even_median(a.ckpt, a.even_median)   #: 창을 짓기 **전에** 전역을 맞춘다

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    ev = json.load(open("processed_data/real_events.json",
                        encoding="utf-8"))["files"][a.stem]
    raw = load_nilm_npz("processed_data/composite_eval/%s.npz" % a.stem)
    x = RealWindows._to_33ch(raw)
    off = target_index(WIN)
    ti = fine_target_index()
    f0 = WIN - FINE                         # 세밀 창이 시작하는 원시 좌표
    tau = (np.arange(FINE) - ti) / FS       # 세밀 창의 시간축 (타깃 = 0)

    def win(t_s):
        c = int(round(t_s * FS))
        return x[:, c - off:c - off + WIN].copy()

    w_lo, w_hi = win(a.at - a.gap), win(a.at + a.gap)
    f2, wd2 = build_inputs(np.stack([w_lo, w_hi]))
    k = apps.index(a.app)
    k2 = apps.index(a.also) if a.also in apps else None
    absent = [v for v in apps if v not in ev["appliances_present"]]
    print("%s · 경계 %.2f초 · 앞 %.2f / 뒤 %.2f초 · 세밀 창 %.1f ~ %+.1f초"
          % (a.stem, a.at, a.at - a.gap, a.at + a.gap, tau[0], tau[-1]))
    if absent:
        print("   ⚠ 이 파일에 **없는** 기기: " + " · ".join(SH.get(v, v) for v in absent))

    # 그 동네 창들 — (채널, 시각)별 σ
    nb = np.arange(a.at - 6.0, a.at + 6.0 + 1e-9, 0.25)
    fn, _ = build_inputs(np.stack([win(t) for t in nb]))
    sig = fn.std(0) + 1e-9                                   # (57, 600)

    for ck in a.ckpt:
        m = load_model(ck, dev)[0]
        m.eval()
        g2, p2 = fwd(m, f2, wd2, dev)
        print("\n" + "=" * 96)
        print("■ %s   %s 게이트 %.3f -> %.3f · 전력 %.0f -> %.0fW"
              % (ck.split("/")[-1].replace(".pt", ""), SH.get(a.app, a.app),
                 g2[0, k], g2[1, k], p2[0, k], p2[1, k]))
        if k2 is not None:
            print("      %s 게이트 %.3f -> %.3f · 전력 %.0f -> %.0fW"
                  % (SH.get(a.also, a.also), g2[0, k2], g2[1, k2],
                     p2[0, k2], p2[1, k2]))
        print("=" * 96)

        # ── [A] 시간 해상 차이 지도 ─────────────────────────────────────
        d = (f2[1] - f2[0]) / sig                            # (57,600)
        nm = fine_names()
        sc = np.abs(d).mean(-1)                              # 채널별 평균 |Δ|/σ
        order = np.argsort(-sc)[:14]
        print("\n[A] **시간 해상 차이** — 채널별 평균 |Δ|/σ 와 그 차이가 **어디에 있나**")
        print("  %-20s %7s %8s %9s | %s"
              % ("채널", "평균|Δ|/σ", "최대", "최대시각", "10초 프로필 (−4 ...0... +6초)"))
        for c in order:
            row = d[c]
            b = np.abs(row).reshape(20, 30).mean(-1)         # 0.5초 눈금 20칸
            lv = " .:-=+*#@"
            mx = max(b.max(), 1e-9)
            spark = "".join(lv[min(int(v / mx * (len(lv) - 1)), len(lv) - 1)] for v in b)
            j = int(np.argmax(np.abs(row)))
            print("  %-20s %7.2f %8.2f %8.2f초 | %s"
                  % (nm[c], sc[c], np.abs(row).max(), tau[j], spark))
        print("       (프로필 눈금: 0.5초 x 20칸. 왼쪽 끝 −4.0초 · 8번째 칸이 타깃 0초 · 오른쪽 끝 +6.0초)")

        # ── [B] 시간 구간별 되돌리기 ────────────────────────────────────
        segs = [("광역만 −54~−4초", 0, f0)]
        for s0 in (-4, -2, 0, 2, 4):
            i0 = f0 + int((s0 + 4.0) * FS)
            i1 = min(f0 + int((s0 + 6.0) * FS), WIN)
            segs.append(("세밀 %+.0f~%+.0f초" % (s0, s0 + 2), i0, i1))
        segs.append(("세밀 10초 전부", f0, WIN))
        segs.append(("창 전부", 0, WIN))
        cand, tags = [], []
        for nm_, i0, i1 in segs:
            y = w_hi.copy(); y[:, i0:i1] = w_lo[:, i0:i1]
            cand.append(y); tags.append(nm_)
        # 무리별 x 시간 (세밀 10초 안에서만)
        for gn, idx in raw_groups().items():
            y = w_hi.copy(); y[np.ix_(idx, np.arange(f0, WIN))] = w_lo[np.ix_(idx, np.arange(f0, WIN))]
            cand.append(y); tags.append("세밀10초 · " + gn)
        f3, w3 = build_inputs(np.stack(cand))
        g3, p3 = fwd(m, f3, w3, dev)
        print("\n[B] **시간 구간별 되돌리기** — 뒤 창의 그 구간만 앞 창 값으로")
        print("  %-22s %10s %10s %10s"
              % ("구간", "게이트", "전력(W)", SH.get(a.also, "짝")))
        for i, nm_ in enumerate(tags):
            print("  %-22s %10.3f %10.0f %10.0f"
                  % (nm_, g3[i, k], p3[i, k], p3[i, k2] if k2 is not None else np.nan))
        print("  기준(안 건드림)        %10.3f %10.0f %10.0f"
              % (g2[1, k], p2[1, k], p2[1, k2] if k2 is not None else np.nan))

        # ── [C] 그림 ───────────────────────────────────────────────────
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from src.plotting.style import use_korean_font  # noqa
            use_korean_font()
        except Exception:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except Exception:
                continue
        show = [P_CH_FINE] + [int(c) for c in order[:5] if c != P_CH_FINE][:5]
        fig, axs = plt.subplots(len(show), 1, figsize=(11, 1.7 * len(show)),
                                sharex=True)
        for ax, c in zip(np.atleast_1d(axs), show):
            v0, v1 = f2[0, c], f2[1, c]
            if c == P_CH_FINE:
                v0, v1 = np.sinh(v0) * POWER_SCALE, np.sinh(v1) * POWER_SCALE
            ax.plot(tau, v0, lw=1.0, label="앞 %.2f초" % (a.at - a.gap))
            ax.plot(tau, v1, lw=1.0, label="뒤 %.2f초" % (a.at + a.gap))
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_ylabel(("P (W)" if c == P_CH_FINE else nm[c]), fontsize=8)
            ax.grid(alpha=.3)
        np.atleast_1d(axs)[0].legend(fontsize=8, ncol=2)
        np.atleast_1d(axs)[0].set_title(
            "%s · %s 경계 %.2f초 — 세밀 갈래가 보는 10초 (타깃 = 0)"
            % (a.stem, SH.get(a.app, a.app), a.at), fontsize=10)
        np.atleast_1d(axs)[-1].set_xlabel("타깃 기준 시각 (초)")
        fig.tight_layout()
        import os
        os.makedirs(a.out, exist_ok=True)
        pth = "%s/fine10_%s_%s_%.0f_%s.png" % (
            a.out, a.stem, a.app, a.at, ck.split("/")[-1].replace(".pt", ""))
        fig.savefig(pth, dpi=130)
        plt.close(fig)
        print("\n[C] 저장 %s" % pth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
