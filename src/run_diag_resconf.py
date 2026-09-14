# -*- coding: utf-8 -*-
"""저항 기기끼리의 **교차 오귀속**을 본다 (14.71).

왜 만드나
---------
`저항 4종 신원`(`run_baseline_nilmtk.score_arm`)은 이렇게 센다:

```python
solo = (yg.sum(1) == 1) & gate          # 저항 넷 중 **정확히 하나만** 켜진 창
hit  = argmax(예측 저항 4) == argmax(참 저항 4)
```

**저항 기기가 하나만 켜진 창만 본다.** 그런데 2026-09-14 에 사용자가 그림에서 짚은
두 자리는 둘 다 여러 개가 같이 켜진 창이다:

```
  test_5  50~150초   참 {포트, 오븐}      -> 포트의 와트가 오븐으로
  test_2 235~265초   참 {포트, 드라이기}  -> 포트의 와트가 통째로 오븐으로
```

**혼동이 일어나는 창이 지표에서 통째로 빠진다.** 그 지표가 0.9815 로 안 움직인다고
세 번(14.60 · 14.61 · 14.63) 축을 기각했는데, 그 자가 이 병을 볼 수 없는 자였다.

무엇을 보나
-----------
```
(1) 저항 창 분포 — solo 대 multi. `저항 4종 신원` 이 몇 %를 안 보는가
(2) 오귀속 — 참 저항 집합 -> 예측 저항 집합. 어느 기기가 어느 기기를 먹는가
(3) 자리별로 가른다 — 가설은 **지문의 고조파 모양이 녹화 자리의 전압**이라는 것이다.
    자리 D 녹화(오븐·핫플) h3/h1 ~ 0.005 · 자리 E 녹화(포트·드라이기) ~ 0.03.
    시험 파일은 test_2 만 E 고 나머지는 D 다.
```

    python -X utf8 src/run_diag_resconf.py --ckpt results/cnn_vhrc_50_s0.pt
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

from src.run_diag_rollback import predict  # noqa: E402
from src.run_train_seq import real_windows  # noqa: E402

RES = ("electiric_kettle", "oven", "hotplate", "hair_dryer")
KO = {"electiric_kettle": "포트", "oven": "오븐", "hotplate": "핫플", "hair_dryer": "드라이"}


def _site(stem):
    try:
        from src.preprocessing.file_registry import site_of
        return site_of(stem) or "?"
    except Exception:
        return "?"


def _nm(mask, idx):
    return "{" + "+".join(KO[RES[i]] for i in range(len(RES)) if mask[idx[i]]) + "}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--postproc", default="off", choices=("off", "cap", "full"))
    ap.add_argument("--top", type=int, default=8, help="오귀속 표에 몇 줄")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps0 = list(torch.load(a.ckpt[0], map_location="cpu",
                            weights_only=False)["appliances"])
    gi = [apps0.index(x) for x in RES]
    cache = real_windows(apps0, a.grid_s, dev)

    for path in a.ckpt:
        apps, pr = predict(path, cache, dev, postproc=a.postproc)
        assert apps == apps0
        print()
        print("=" * 86)
        print("%s   (후처리 %s)" % (path.split("/")[-1], a.postproc))
        print("=" * 86)

        print("  (1) 저항 창 분포 — `저항 4종 신원` 은 **solo 만** 본다")
        print("      %-9s %4s %9s %9s %9s %9s" % ("파일", "자리", "저항켜짐", "solo", "multi", "solo비율"))
        tot = [0, 0]
        for stem, d in sorted(cache.items()):
            y = d["y"].astype(bool)[:, gi]
            on = y.any(1)
            solo = int((y.sum(1) == 1).sum())
            multi = int((y.sum(1) >= 2).sum())
            tot[0] += solo
            tot[1] += multi
            print("      %-9s %4s %9d %9d %9d %8.1f%%"
                  % (stem, _site(stem), int(on.sum()), solo, multi,
                     100.0 * solo / max(solo + multi, 1)))
        print("      %-9s %4s %9d %9d %9d %8.1f%%   <- **multi %d창이 지표에서 빠진다**"
              % ("전부", "", tot[0] + tot[1], tot[0], tot[1],
                 100.0 * tot[0] / max(sum(tot), 1), tot[1]))

        print()
        print("  (2) 오귀속 — 참 저항 집합 -> 예측 저항 집합 (자리별, 어긋난 것만)")
        for site in ("D", "E"):
            cnt = {}
            for stem, d in sorted(cache.items()):
                if _site(stem) != site:
                    continue
                y = d["y"].astype(bool)[:, gi]
                p = pr[stem][0].astype(bool)[:, gi]
                w = pr[stem][1][:, gi]
                for i in range(len(y)):
                    if not y[i].any():
                        continue
                    if (y[i] == p[i]).all():
                        continue
                    k = (tuple(y[i]), tuple(p[i]))
                    c = cnt.setdefault(k, [0, [], []])
                    c[0] += 1
                    # ⚠ **평균으로 모으지 마라** (14.85). 이 분포는 쌍봉이다 —
                    #   `{오븐+핫플}->{오븐}` 134창은 중앙 15W(팬·조명)인데 평균은 349W
                    #   라서 "핫플이 오븐으로 삼켜진다" 로 읽었다. 틀렸다.
                    c[1].append(w[i])
                    # 그 창에 **실제로 부하가 있었나.** 핫플은 듀티로 꺼져 있는데
                    # 라벨만 ON 인 창이 절반이라(13.84.x: 1998창 중 1011창 총전력 42.1W)
                    # 혼동표가 오답으로 세지만 모델이 맞는 자리가 많다.
                    c[2].append(float(d["p_obs"][i] - d["p_base"]))
            if not cnt:
                print("      자리 %s — 어긋난 창 없음" % site)
                continue
            n = sum(v[0] for v in cnt.values())
            print("      자리 %s — 어긋난 창 %d개   (예측W 는 **중앙값**, 부하 = 관측−기준선)"
                  % (site, n))
            for k, v in sorted(cnt.items(), key=lambda x: -x[1][0])[:a.top]:
                ty, tp = k
                wt = np.median(np.asarray(v[1]), axis=0)
                ld = np.asarray(v[2])
                # 부하가 없는 창이 많으면 그 줄은 **오귀속이 아니라 라벨**이다.
                flag = "  <- 부하 없음 %.0f%% (듀티 라벨)" % (100.0 * (ld < 100).mean()) \
                    if (ld < 100).mean() > 0.4 else ""
                print("        %-22s -> %-22s %6d창 (%4.1f%%)  부하중앙 %5.0fW  예측W %s%s"
                      % ("{" + "+".join(KO[RES[j]] for j in range(4) if ty[j]) + "}",
                         "{" + "+".join(KO[RES[j]] for j in range(4) if tp[j]) + "}" if any(tp) else "{}",
                         v[0], 100.0 * v[0] / n, np.median(ld),
                         " ".join("%s %.0f" % (KO[RES[j]], wt[j]) for j in range(4) if wt[j] > 20),
                         flag))

        print()
        print("  (3) 기기별 — 참으로 켜졌을 때 **자기 게이트가 서는 비율** (자리별)")
        print("      %-8s %8s %8s      %s" % ("기기", "자리 D", "자리 E", "(참 ON 창 수)"))
        for j, app in enumerate(RES):
            row, cn = [], []
            for site in ("D", "E"):
                t = f = 0
                for stem, d in cache.items():
                    if _site(stem) != site:
                        continue
                    y = d["y"].astype(bool)[:, gi][:, j]
                    p = pr[stem][0].astype(bool)[:, gi][:, j]
                    t += int(y.sum())
                    f += int((y & p).sum())
                row.append(f / t if t else float("nan"))
                cn.append(t)
            print("      %-8s %8.3f %8.3f      (D %d · E %d)"
                  % (KO[app], row[0], row[1], cn[0], cn[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
