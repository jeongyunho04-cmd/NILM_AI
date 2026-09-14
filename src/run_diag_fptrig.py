# -*- coding: utf-8 -*-
"""**어떤 전이가** 오븐 헛detect 를 부르는가 (14.86).

왜 만드나
---------
14.72(`run_diag_fpage`)는 *"전환 **이후 시간**"* 만 봤다 — 전이가 **무엇이었는지**는 안 봤다.
2026-09-15 에 사용자가 `984295` A 갈래 그림에서 짚었다:

    *"원래 혼동하던 곳은 상당부분 고쳐졌지만 **드라이기 꺼짐에 반응한 건지** 국소적으로
      오븐으로 오탐하고 있어"*

그러면 자는 **전이의 정체와 방향**별 헛detect 율이다. "오래되면 나빠진다" 와
"드라이기 off 가 방아쇠다" 는 **다른 주장**이고, 후자는 `드라이 off` 칸만 튀어야 한다.

무엇을 세나
-----------
```
  대상 창 : 참 오븐 **OFF** 인 창
  헛detect: 예측 오븐 게이트 ON **이고 와트 > --min-w**
            ⚠ 와트 문턱이 있어야 한다 — 오븐 팬·조명(14~16W) 게이트가 섞이면
              와트가 안 움직이는 깜빡임까지 같이 세어 표가 뜻을 잃는다 (14.85).
  가르기 : 그 창에서 **가장 가까운 참값 전이**의 (기기, 방향) 과 부호 있는 오프셋
  대조   : |오프셋| > --band 인 창 = "전이 없음" 칸. 방아쇠 가설이면 여기가 바닥이어야 한다.
```

    python -X utf8 src/run_diag_fptrig.py --ckpt results/cnn_wtap_s1.pt
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

KO = {"electiric_kettle": "포트", "oven": "오븐", "hotplate": "핫플",
      "hair_dryer": "드라이", "air_conditioner": "에어컨", "fan": "선풍기",
      "beam_projector": "빔프", "laptop_charger": "충전기", "minipc": "미니PC"}


def events(t, y, apps):
    """참값 전이 (시각, '기기 on|off'). 창 격자에서 라벨이 바뀐 자리."""
    out = []
    for k, app in enumerate(apps):
        d = np.diff(y[:, k].astype(np.int16))
        for i in np.nonzero(d != 0)[0]:
            out.append((float(t[i + 1]), "%s %s" % (KO.get(app, app[:4]),
                                                    "on" if d[i] > 0 else "off")))
    out.sort()
    return out


def nearest(t, ev, step=None):
    """창마다 (가장 가까운 전이 이름, 부호 있는 오프셋 t−t_전이, 그 전이의 관측 계단)."""
    if not ev:
        return ([None] * len(t), np.full(len(t), np.inf), np.full(len(t), np.nan))
    te = np.array([e[0] for e in ev])
    nm = [e[1] for e in ev]
    j = np.abs(t[:, None] - te[None, :]).argmin(1)
    sp = np.full(len(t), np.nan) if step is None else np.asarray(step)[j]
    return [nm[x] for x in j], t - te[j], sp


def obs_step(t, p, ev, half=3.0):
    """전이마다 **관측 총전력의 계단** (W). 앞뒤 `half` 초의 중앙값 차.

    라벨이 바뀌어도 관측이 안 움직이는 전이가 있다 — 오븐의 팬·조명 꺼짐(−2W),
    선풍기(−20W). 그 창에는 **모델이 쓸 증거가 없다.**
    """
    out = []
    for te, _ in ev:
        a = p[(t >= te - half) & (t < te)]
        b = p[(t > te) & (t <= te + half)]
        out.append(float(np.median(b) - np.median(a))
                   if (a.size and b.size) else float("nan"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--app", default="oven", help="헛detect 를 볼 기기")
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--band", type=float, default=20.0, help="전이 근방으로 볼 폭 (초)")
    ap.add_argument("--min-w", type=float, default=100.0,
                    help="이 와트를 넘어야 헛detect 로 센다 (14.85 — 팬·조명 깜빡임 제외)")
    ap.add_argument("--list", default="", metavar="STEM",
                    help="이 파일의 헛detect 창을 시각과 함께 낱낱이 찍는다")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    j = apps.index(a.app)
    cache = real_windows(apps, a.grid_s, dev)

    for path in a.ckpt:
        _, pr = predict(path, cache, dev, postproc="off")
        print("")
        print("=" * 88)
        print("%s — %s 헛detect 를 **전이의 정체**로 가른다 (>%.0fW 만)"
              % (path.split("/")[-1], KO.get(a.app, a.app), a.min_w))
        print("=" * 88)

        tab, none_n, none_f, none_w = {}, 0, 0, []
        rows_list = []
        SB = [0.0, 50.0, 200.0, 600.0, 1e9]          # |관측 계단| 칸
        sbin = [[0, 0] for _ in SB[:-1]]
        for stem, d in sorted(cache.items()):
            y = d["y"].astype(bool)
            t = d["t"]
            off = ~y[:, j]                                  # 참 오븐 OFF 인 창만
            if not off.any():
                continue
            fp = pr[stem][0].astype(bool)[:, j] & (pr[stem][1][:, j] > a.min_w)
            ev = events(t, y, apps)
            nm, dt, sp = nearest(t, ev, obs_step(t, d["p_obs"], ev))
            for i in np.nonzero(off)[0]:
                if abs(dt[i]) > a.band or nm[i] is None:
                    none_n += 1
                    none_f += int(fp[i])
                    if fp[i]:
                        none_w.append(pr[stem][1][i, j])
                    continue
                c = tab.setdefault(nm[i], [0, 0, []])
                c[0] += 1
                c[1] += int(fp[i])
                if np.isfinite(sp[i]):
                    b = int(np.searchsorted(SB, abs(sp[i]), side="right") - 1)
                    b = min(max(b, 0), len(sbin) - 1)
                    sbin[b][0] += 1
                    sbin[b][1] += int(fp[i])
                if fp[i]:
                    c[2].append(pr[stem][1][i, j])
                    rows_list.append((stem, t[i], nm[i], dt[i], pr[stem][1][i, j]))

        print("  %-14s %7s %9s %8s %11s" % ("가장 가까운 전이", "창", "헛detect", "비율", "유령 중앙W"))
        for k, v in sorted(tab.items(), key=lambda x: -(x[1][1] / max(x[1][0], 1))):
            if v[0] < 3:
                continue
            print("  %-14s %7d %9d %7.1f%% %11s"
                  % (k, v[0], v[1], 100.0 * v[1] / v[0],
                     "%.0f" % np.median(v[2]) if v[2] else "-"))
        print("  %-14s %7d %9d %7.1f%% %11s   <- **대조: 전이에서 %.0f초 넘게 떨어진 창**"
              % ("(전이 없음)", none_n, none_f, 100.0 * none_f / max(none_n, 1),
                 "%.0f" % np.median(none_w) if none_w else "-", a.band))

        print("")
        print("  [2] **관측 계단 크기**별 — 라벨은 바뀌는데 전력이 안 움직이는 전이가 있다")
        print("      %-16s %7s %9s %8s" % ("|관측 계단|", "창", "헛detect", "비율"))
        for b in range(len(sbin)):
            lo, hi = SB[b], SB[b + 1]
            nm_ = ("%.0f~%.0fW" % (lo, hi)) if hi < 1e8 else ("%.0fW 이상" % lo)
            n_, f_ = sbin[b]
            if n_ == 0:
                continue
            print("      %-16s %7d %9d %7.1f%%" % (nm_, n_, f_, 100.0 * f_ / n_))

        if a.list:
            sel = [r for r in rows_list if r[0] == a.list]
            print("")
            print("  [%s] 헛detect 창 낱낱이 (%d개)" % (a.list, len(sel)))
            for stem, tt, nmm, dd, ww in sorted(sel, key=lambda r: r[1]):
                print("      %7.1f초  %6.0fW   가장 가까운 전이  %-12s %+6.1f초"
                      % (tt, ww, nmm, dd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
