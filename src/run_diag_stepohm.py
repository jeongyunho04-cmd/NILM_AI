# -*- coding: utf-8 -*-
"""그 계단은 **누구의 저항**인가 — 오븐↔포트 오탐의 원인을 자료에서 (14.114).

사용자(여러 번): *"test_5 140초 부근은 통째로 포트를 오븐으로 오탐한다. 포트 켜질
때랑 오븐 통전이 꺼질 때 공통으로 오탐이 생긴다. 왜 원인을 못 찾느냐."*

맞는 지적이다. 여태 **증상**(헛게이트 비율·유형·축퇴 %)만 쟀다. 정작 물어야 할 것은
하나다 — **그 순간의 관측 전력 계단이 물리적으로 누구를 가리키는가.**

```
  컨덕턴스 표 (postproc.RESISTIVE_OHM)   포트 **35.8Ω**  ·  오븐 **40.6Q**
  222V 에서                               포트 1377W    ·  오븐 1214W   (13% 차)
```
계단 하나에서 `R = V^2 / dP` 를 내면 둘 중 누구인지 **자료가 말해 준다**. 그래서:

```
  [1] 참 전이마다 R 을 내고 표값과 견준다 — 포트 전이가 정말 35.8Ω 로 보이나
  [2] 두 무리가 **겹치나** (겹치면 자료가 애매한 것이고, 손실로 못 고친다)
  [3] 모델이 틀린 전이만 따로 — 거기서 R 은 누구를 가리켰나
      가리키는데 틀렸으면 **정보는 있고 모델이 안 쓴다** -> `--w-swap` 이 답이다
```

    python -X utf8 src/run_diag_stepohm.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.postproc import RESISTIVE_OHM  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402
from src.run_train_seq import real_windows  # noqa: E402

KO = {"oven": "오븐", "electiric_kettle": "포트", "hotplate": "핫플", "hair_dryer": "드라이"}
RES = ("electiric_kettle", "oven", "hotplate", "hair_dryer")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_hzwt_s0.pt")
    ap.add_argument("--grid-s", type=float, default=0.5)
    ap.add_argument("--min-w", type=float, default=300.0, help="이만큼 큰 계단만")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"])
    cache = real_windows(apps, a.grid_s, dev)
    m = load_model(a.ckpt, dev)[0]
    m.eval()

    print("계단의 컨덕턴스 (14.114) · 격자 %.1f초 · %s" % (a.grid_s, a.ckpt.split("/")[-1]))
    print("  표값  " + " · ".join("%s %.1fΩ (222V 에서 %.0fW)"
                                  % (KO[r], RESISTIVE_OHM[r], 222.0 ** 2 / RESISTIVE_OHM[r])
                                  for r in RES))
    print("")

    rows = {r: [] for r in RES}
    wrong = []
    for stem, d in cache.items():
        y = d["y"].astype(bool)
        p = np.asarray(d["p_obs"], float)
        v = np.asarray(d["v_obs"], float)
        with torch.no_grad():
            gl = []
            for i in range(0, len(d["t"]), 512):
                o = m(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                      torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                gl.append(o["on_logit"].float().cpu())
            g = torch.cat(gl).numpy() > 0
        for r in RES:
            k = apps.index(r)
            e = np.diff(np.r_[0, y[:, k].astype(int), 0])
            for sgn, idx in ((+1, np.nonzero(e == 1)[0]), (-1, np.nonzero(e == -1)[0])):
                for i in idx:
                    i0, i1 = i - 2, i + 2
                    if i0 < 1 or i1 >= len(p) - 1:
                        continue
                    # 그 전이에서 **다른 기기는 안 바뀌어야** 한다 (한 스텝 한 기기, 11.x)
                    if (y[i0, :] != y[min(i1, len(y) - 1), :]).sum() > 1:
                        continue
                    dp = sgn * (p[i1] - p[i0])
                    if dp < a.min_w:
                        continue
                    vv = float(np.median(v[i0:i1 + 1]))
                    R = vv * vv / dp
                    rows[r].append(R)
                    g_ov = bool(g[i, apps.index("oven")])
                    if r == "electiric_kettle" and g_ov:
                        wrong.append((stem, float(d["t"][i]), "+" if sgn > 0 else "-",
                                      dp, vv, R))

    print("  [1][2] 참 전이의 `R = V^2/dP` (계단 %.0fW 이상 · 한 기기만 바뀐 전이)" % a.min_w)
    print("    %-8s %5s %9s %9s %9s %9s   표값" % ("기기", "전이", "p10", "중앙", "p90", "산포"))
    for r in RES:
        x = np.asarray(rows[r])
        if not len(x):
            print("    %-8s %5d  (없다)" % (KO[r], 0))
            continue
        print("    %-8s %5d %9.1f %9.1f %9.1f %9.1f   %.1fΩ"
              % (KO[r], len(x), np.percentile(x, 10), np.median(x),
                 np.percentile(x, 90), x.std(), RESISTIVE_OHM[r]))
    ke, ov = np.asarray(rows["electiric_kettle"]), np.asarray(rows["oven"])
    if len(ke) and len(ov):
        lo, hi = max(ke.min(), ov.min()), min(ke.max(), ov.max())
        ovl = float(((ke >= lo) & (ke <= hi)).mean() + ((ov >= lo) & (ov <= hi)).mean()) / 2
        mid = 0.5 * (np.median(ke) + np.median(ov))
        acc = float((ke < mid).mean() * 0.5 + (ov > mid).mean() * 0.5)
        print("")
        print("    포트 중앙 %.1f · 오븐 중앙 %.1f · 가운데 문턱 %.1fΩ 로만 갈라도 **%.0f%%**"
              % (np.median(ke), np.median(ov), mid, 100 * acc))
        print("    두 무리가 겹치는 범위에 든 비율 %.0f%%" % (100 * ovl))

    print("")
    print("  [3] **참 포트 전이인데 모델이 오븐 게이트를 세운** 자리 (최대 12개)")
    print("    %-8s %7s %3s %8s %7s %9s   R 이 가리키는 쪽" % ("파일", "t(초)", "on", "ΔP(W)", "V", "R(Ω)"))
    for x in sorted(wrong, key=lambda z: -z[3])[:12]:
        near = "포트" if abs(x[5] - RESISTIVE_OHM["electiric_kettle"]) < \
            abs(x[5] - RESISTIVE_OHM["oven"]) else "**오븐**"
        print("    %-8s %7.1f %3s %8.0f %7.1f %9.1f   %s"
              % (x[0], x[1], x[2], x[3], x[4], x[5], near))
    if wrong:
        w = np.asarray([x[5] for x in wrong])
        k_near = np.abs(w - RESISTIVE_OHM["electiric_kettle"]) < np.abs(w - RESISTIVE_OHM["oven"])
        print("    => 그 %d개 중 R 이 **포트를 가리킨 것 %d개 (%.0f%%)**"
              % (len(w), int(k_near.sum()), 100 * k_near.mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
