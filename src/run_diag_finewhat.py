# -*- coding: utf-8 -*-
"""`wtap` 의 넓은 탭은 **무엇을 보고** 오븐을 끄는가 (14.90).

왜 만드나
---------
14.89 가 남긴 수수께끼. `t=128초` 창의 **세밀** 입력은 `[124.02, 134.0]` 이고 그 구간
관측은 **평평하다** (1629 -> 1616W). 123초의 통전->팬 계단은 **창 밖**이다. 그런데
```
  오븐 통전 확률 s2     기준선 **0.806**      wtap A **0.051**
```
계단을 읽어서가 아니다. 넓은 탭(`--fine-extra-dilations 32,64 --tap-layers 0,1,4`)이
**창 안의 다른 무언가**를 읽고 있다. 그것을 짚는다.

어떻게 재나 — **지우고 본다**
-----------------------------
세밀 입력의 한 조각을 **타깃 열의 값으로 덮는다** (그 조각이 시간 구조를 하나도 안 담게).
두 축으로 자른다:

```
  구역   과거먼 [0,120) · 과거가까이 [120,239) · 타깃근방 [239,300)
         · 미래중 [300,450) · 미래끝 [450,600)       (타깃 = 239)
  채널   I 홀수(0~15) · I 짝수(16~22) · P(23) · Q(24) · V(25) · 비(26~28)
         · 리플(29,30) · 위상(31~38) · PF(39) · I9/I3(40) · **전방탭(41,42)**
         · 반파(43) · 전압고조파(45~56)
```

⚠ **양성 대조가 필수다** — 전부 덮으면 예측이 크게 달라져야 한다. 안 달라지면 내 덮기가
안 먹은 것이다 ([[the-gate-must-build-the-real-object]], 14.87 에서 양성 대조 없이
갔다가 두 번 틀린 자를 붙들었다).

    python -X utf8 src/run_diag_finewhat.py --plot results/plots/finewhat.png
"""
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.run_train_seq import real_windows  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402
from src.run_train_cnn import FINE_TPOS  # noqa: E402

T = FINE_TPOS                       # 239
REGIONS = [("과거 먼", 0, 120), ("과거 가까이", 120, T), ("타깃 근방", T, 300),
           ("미래 중", 300, 450), ("미래 끝", 450, 600)]
CHGRP = [("I 홀수 Re/Im", list(range(0, 16))),
         ("I 짝수 |·|", list(range(16, 23))),
         ("P", [23]), ("Q", [24]), ("V", [25]),
         ("|I3|/|I1| 등 비", [26, 27, 28]),
         ("리플 P−이동평균", [29, 30]),
         ("위상 φ3·5·7·9", list(range(31, 39))),
         ("PF", [39]), ("|I9|/|I3|", [40]),
         ("전방탭 P−P(+3s,+5.5s)", [41, 42]),
         ("반파 |I2|−|I4|", [43]),
         ("전압 고조파", list(range(45, 57)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/cnn_v32h_off_s1.pt", "results/cnn_wtap_s1.pt"])
    ap.add_argument("--stem", default="test_5")
    ap.add_argument("--lo", type=float, default=128.0)
    ap.add_argument("--hi", type=float, default=140.0)
    ap.add_argument("--plot", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    jo = apps.index("oven")
    cache = real_windows(apps, 2.0, dev)
    d = cache[a.stem]
    t = d["t"]
    sel = np.nonzero((t >= a.lo) & (t <= a.hi))[0]
    print("%s %.0f~%.0f초 · 창 %d개 · 세밀 타깃 자리 %d (과거 %.2f초 | 미래 %.2f초)"
          % (a.stem, a.lo, a.hi, len(sel), T, T / 60.0, (599 - T) / 60.0))
    print("덮기 = 그 조각을 **타깃 열의 값**으로 (시간 구조를 지운다). 값은 오븐 s2 의 평균.")

    res = {}
    for path in a.ckpt:
        m, _ = load_model(path, dev)[:2]
        m.eval()
        name = path.split("/")[-1][:-3]
        F = torch.from_numpy(d["fine"][sel]).to(dev)
        W = torch.from_numpy(d["wide"][sel]).to(dev)

        def s2(f):
            with torch.no_grad():
                return float(m(f, W)["state"][:, jo].softmax(-1)[:, 2].mean())

        base = s2(F)
        allf = F.clone()
        allf[:, :, :] = F[:, :, T:T + 1]
        pos = s2(allf)
        print("")
        print("=" * 84)
        print("%s   기준 s2 = **%.3f** · 세밀 전부 덮으면 %.3f (양성 대조: 크게 달라야 한다)"
              % (name, base, pos))
        print("=" * 84)

        print("  [1] 구역별 (채널 전부)")
        row_r = []
        for rn, x, z in REGIONS:
            f = F.clone()
            f[:, :, x:z] = F[:, :, T:T + 1]
            v = s2(f)
            row_r.append(v - base)
            print("      %-12s [%3d,%3d)  s2 %.3f   Δ %+.3f" % (rn, x, z, v, v - base))

        print("  [2] 채널 무리별 (창 전체)")
        row_c = []
        for cn, chs in CHGRP:
            f = F.clone()
            f[:, chs, :] = F[:, chs, T:T + 1]
            v = s2(f)
            row_c.append(v - base)
            print("      %-22s s2 %.3f   Δ %+.3f" % (cn, v, v - base))
        res[name] = (base, pos, np.array(row_r), np.array(row_c))

    if a.plot:
        plot(res, a.plot, a.stem, a.lo, a.hi)
        print("")
        print("저장 %s" % a.plot)
    return 0


def plot(res, path, stem, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.run_plot_real import _font
    _font()
    names = list(res)
    fig, ax = plt.subplots(2, 1, figsize=(13, 9.5))
    for k, (labels, idx) in enumerate(((["%s" % r[0] for r in REGIONS], 2),
                                       ([c[0] for c in CHGRP], 3))):
        x = np.arange(len(labels))
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            ax[k].bar(x + i * w - 0.4 + w / 2, res[n][idx], w,
                      label="%s (기준 s2 %.3f)" % (n, res[n][0]))
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(labels, rotation=20, ha="right")
        ax[k].axhline(0, color="k", lw=0.8)
        ax[k].grid(axis="y", alpha=0.3)
        ax[k].legend(loc="best", fontsize=9)
        ax[k].set_ylabel("Δ 오븐 통전확률 s2")
    ax[0].set_title("%s %.0f~%.0f초 — 세밀 입력의 **구역**을 타깃 열로 덮었을 때"
                    % (stem, lo, hi))
    ax[1].set_title("같은 창 — 세밀 입력의 **채널 무리**를 덮었을 때 (창 전체)")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
