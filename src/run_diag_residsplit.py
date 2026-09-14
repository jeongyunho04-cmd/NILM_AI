# -*- coding: utf-8 -*-
"""고전력 창의 잔차를 **유령 몫**과 **수준 몫**으로 가른다 (14.55).

14.52 가 남긴 것: 앵커 뒤에도 `핫플통전·P>1500` 잔차 중앙이 **−16.3W(과대)** 다.
그 중 `vrel` 10초평균 구멍이 **+4.5W** 를 설명하고 **~12W 가 미귀속**이다. 그것을 가른다.

    잔차 = P_obs − base − Σ_k pred_k·on_k
         = [P_obs − base − Σ_{참ON} pred_k]   −   [Σ_{참OFF} pred_k·on_k]
            └ **수준 몫** (참으로 켜진 것들의 크기 오차)  └ **유령 몫** (꺼진 것에 준 전력)

참 on/off 는 `processed_data/real_events.json` 이 준다 — 모호함이 없다.

⚠ **중앙값으로 본다.** 14.48 이 과대 스파이크(+500W 까지)를 봤는데, 스파이크는 평균은
  움직여도 중앙값은 안 움직인다. 중앙이 −16.3 이면 그것은 **고른 이동**이지 스파이크가 아니다.
  둘을 같이 찍어 어느 쪽인지 가른다.

    python -X utf8 src/run_diag_residsplit.py --ckpt results/a.pt ...
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
        ("오븐통전·P>1000", "oven", 1000.0),
        ("P>1500 (전부)", None, 1500.0),
        ("전체 창", None, 0.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--postproc", default="off", choices=("off", "cap", "full"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu", weights_only=False)["appliances"])
    cache = real_windows(apps, a.grid_s, dev)
    R = {}
    for p in a.ckpt:
        _, pr = predict(p, cache, dev, postproc=a.postproc)
        R[p] = pr
        print("  %s 끝" % p.split("/")[-1], flush=True)
    print()

    names = [p.split("/")[-1].replace(".pt", "") for p in a.ckpt]
    for label, app, thr in CUTS:
        ai = apps.index(app) if app else None
        print(f"■ {label}    **양수 = 과소예측**")
        print("    %-20s%9s%10s%10s%10s%10s" %
              ("체크포인트", "창", "잔차중앙", "유령중앙", "수준중앙", "잔차평균"))
        ph_by = {}
        for p, nm in zip(a.ckpt, names):
            on, pw = R[p][list(cache)[0]][0], None
            res, gho, lvl = [], [], []
            per = np.zeros(len(apps))
            nw = 0
            for stem, d in cache.items():
                on, pw = R[p][stem]
                base = float(d.get("p_base", 0.0))
                y = d["y"].astype(bool)
                pobs = d["p_obs"].astype(np.float64)
                m = pobs > thr
                if ai is not None:
                    m &= y[:, ai]
                if int(m.sum()) < 5:
                    continue
                use = np.where(on, pw, 0.0)                 # 게이트가 켠 것만
                ghost = np.where(~y, use, 0.0).sum(1)       # **참으로 꺼진 것**에 준 전력
                r = pobs - (use.sum(1) + base)
                res.append(r[m]); gho.append(ghost[m]); lvl.append((r + ghost)[m])
                per += np.where(~y, use, 0.0)[m].sum(0)
                nw += int(m.sum())
            if not res:
                continue
            res = np.concatenate(res); gho = np.concatenate(gho); lvl = np.concatenate(lvl)
            print("    %-20s%9d%10.1f%10.1f%10.1f%10.1f" %
                  (nm[-20:], nw, np.median(res), np.median(gho), np.median(lvl), res.mean()))
            ph_by[nm] = per / max(nw, 1)
        if ph_by and thr > 0:
            print("    유령 전력 평균 (W/창) — 어느 기기인가")
            top = sorted(range(len(apps)),
                         key=lambda j: -max(v[j] for v in ph_by.values()))[:5]
            print("      %-20s" % "" + "".join("%14s" % apps[j][:13] for j in top))
            for nm, v in ph_by.items():
                print("      %-20s" % nm[-20:] + "".join("%14.2f" % v[j] for j in top))
        print()
    print("  ⚠ 중앙 잔차 ≈ 수준중앙 − 유령중앙 이다 (부호에 주의: 유령은 과대 쪽).")
    print("  ⚠ 잔차중앙과 잔차평균이 크게 다르면 **스파이크**가 있다는 뜻이다 (14.48).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
