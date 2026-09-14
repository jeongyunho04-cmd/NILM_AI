# -*- coding: utf-8 -*-
"""**고전력 상황의 과소예측** — 사용자가 다섯 번 말한 그 숫자 하나만 잰다 (14.49/14.51).

    잔차 = 관측 − (예측합 + 기준선)          ⚠ **양수 = 과소예측**
    (`run_baseline_nilmtk.score_arm` 은 부호가 반대다 — 거기서는 양수가 과대예측이다.
     여기서는 사용자가 말하는 '과소' 를 양수로 읽히게 뒤집어 놓았다.)

자르는 곳 — 증상이 나타난 그 창:
    ① 핫플 통전 (참 라벨) **그리고** 관측 총전력 > 1500W      <- 판정 줄
    ② 관측 총전력 > 1500W (기기 무관)
    ③ 오븐 통전 (참 라벨) 그리고 관측 > 1000W
    ④ 전체 창                                               <- 대조

    python -X utf8 src/run_diag_hipow.py --ckpt results/a.pt results/b.pt

⚠ 체크포인트를 여러 개 주면 **같은 창·같은 자**로 나란히 낸다. 짝비교가 목적이다.
⚠ 판정은 **시드 셋**으로 한다 ([[measure-the-seed-floor-before-reading-any-effect]]).
  `--pair N` 을 주면 앞 N개를 A팔, 뒤 N개를 B팔로 보고 짝차의 평균±표준편차까지 낸다.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch  # noqa: E402

from src.run_diag_rollback import predict  # noqa: E402
from src.run_train_seq import real_windows  # noqa: E402

CUTS = (("핫플통전·P>1500", "hotplate", 1500.0),
        ("P>1500 (전부)", None, 1500.0),
        ("오븐통전·P>1000", "oven", 1000.0),
        ("전체 창", None, 0.0))


def residuals(apps, pred, cache):
    """{stem: (잔차 (T,), p_obs (T,), y (T,K))}. 잔차는 **양수 = 과소예측**."""
    out = {}
    for stem, (on, pw) in pred.items():
        d = cache[stem]
        base = float(d.get("p_base", 0.0))
        r = d["p_obs"].astype(np.float64) - (np.where(on, pw, 0.0).sum(1) + base)
        out[stem] = (r, d["p_obs"].astype(np.float64), d["y"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--postproc", default="off", choices=("off", "cap", "full"))
    ap.add_argument("--pair", type=int, default=0, metavar="N",
                    help="앞 N개를 A팔, 뒤 N개를 B팔로 보고 짝차를 낸다 (시드 셋이면 N=3)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck0 = torch.load(a.ckpt[0], map_location="cpu", weights_only=False)
    apps = list(ck0["appliances"])
    cache = real_windows(apps, a.grid_s, dev)
    R = {}
    for p in a.ckpt:
        _, pr = predict(p, cache, dev, postproc=a.postproc)
        R[p] = residuals(apps, pr, cache)
        print("  %s 끝" % p.split("/")[-1], flush=True)
    print()

    names = [p.split("/")[-1].replace(".pt", "") for p in a.ckpt]
    stems = sorted(cache)
    for label, app, thr in CUTS:
        ai = apps.index(app) if app else None
        print(f"■ {label} — 잔차 중앙값 (W)   **양수 = 과소예측**")
        print("    %-8s%6s" % ("파일", "창") + "".join("%14s" % n[-13:] for n in names))
        allsel = {p: [] for p in a.ckpt}
        for stem in stems:
            r0, pobs, y = R[a.ckpt[0]][stem]
            m = pobs > thr
            if ai is not None:
                m &= y[:, ai] == 1
            if int(m.sum()) < 5:
                continue
            line = "    %-8s%6d" % (stem, int(m.sum()))
            for p in a.ckpt:
                r, _, _ = R[p][stem]
                line += "%14.1f" % float(np.median(r[m]))
                allsel[p].append(r[m])
            print(line)
        line = "    %-8s%6d" % ("**전부**", sum(len(x) for x in allsel[a.ckpt[0]]))
        med = {}
        for p in a.ckpt:
            v = np.concatenate(allsel[p]) if allsel[p] else np.zeros(0)
            med[p] = float(np.median(v)) if len(v) else float("nan")
            line += "%14.1f" % med[p]
        print(line)
        if a.pair and len(a.ckpt) == 2 * a.pair:
            A, B = a.ckpt[:a.pair], a.ckpt[a.pair:]
            d = []
            for x, y_ in zip(A, B):
                da, db = [], []
                for stem in stems:
                    ra, pobs, yy = R[x][stem]
                    rb, _, _ = R[y_][stem]
                    m = pobs > thr
                    if ai is not None:
                        m &= yy[:, ai] == 1
                    if int(m.sum()) < 5:
                        continue
                    da.append(ra[m]); db.append(rb[m])
                if da:
                    d.append(float(np.median(np.concatenate(db)))
                             - float(np.median(np.concatenate(da))))
            if d:
                d = np.array(d)
                sd = d.std(ddof=1) if len(d) > 1 else float("nan")
                mark = ("  ✅ **줄었다**" if (d.mean() < 0 and abs(d.mean()) > 2 * sd)
                        else ("  ~ 씨앗 폭 안" if abs(d.mean()) <= 2 * (sd if np.isfinite(sd) else 0)
                              else "  ⚠ 늘었다"))
                print("    짝차 (B − A)  %+.1f ± %.1f W%s" % (d.mean(), sd, mark))
        print()
    print("⚠ 대조 — 14.48 이 잰 값 (`cnn_seg2_*` 계열, 후처리 off):")
    print("     핫플통전·P>1500 잔차 중앙  test_1 +37.6 · test_3 +47.8 · test_5 +16.9")
    print("⚠ 판정은 **시드 셋의 짝차**로 한다. 한 판의 숫자는 씨앗 폭 안일 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
