# -*- coding: utf-8 -*-
"""오븐 헛detect 가 **전환 이후 시간**에 어떻게 달리는가 (14.72).

사용자가 그림에서 짚은 것: *"참값 전력이 전혀 안 맞는데 오븐으로 혼동된 게 오래 지속되고,
또 다른 대부하가 켜지거나 꺼지면 원래대로 돌아간다."*

검정할 기전
-----------
모델은 창 하나를 독립으로 본다. 기억은 **입력 창 길이**뿐이다.

```
  세밀  600사이클(10초), 타깃 239  ->  [t-4.0초, t+6.0초]
  광역  120블록 x 0.5초(60초), 타깃 54초  ->  **[t-54초, t+6초]**
```

저항 기기는 고조파로 안 갈린다 (14.71: 코사인 0.9994~0.99999). 그러면 **무엇이 켜졌는지는
켜지는 순간의 계단으로만 알 수 있다.** 그 계단이 광역 창 밖으로 밀려나면(= 마지막 전환 이후
**54초**가 지나면) 모델은 "정상상태 저항 부하" 만 보고 사전확률로 떨어진다 — 오븐은
`--carrier-on` 이고 학습 양성률이 제일 높다(0.315).

⇒ **떨어지는 예측: 마지막 전환 이후 시간이 54초를 넘으면 오븐 헛detect 가 뛴다.**
⇒ 음성 대조: 그 경계가 30초나 90초가 아니라 **54초 언저리**여야 한다. 아무 데서나
   단조 증가하기만 하면 "오래되면 나빠진다" 는 하나 마나 한 말이다.

    python -X utf8 src/run_diag_fpage.py --ckpt results/cnn_vhrc_50_s0.pt
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
EDGES = (0, 10, 20, 30, 40, 54, 70, 100, 150, 1e9)


def _site(stem):
    try:
        from src.preprocessing.file_registry import site_of
        return site_of(stem) or "?"
    except Exception:
        return "?"


def _trans(t, y):
    """참값 전환 시각 (어떤 기기든 on/off 가 바뀐 창의 시각)."""
    d = np.abs(np.diff(y.astype(np.int16), axis=0)).sum(1) > 0
    return t[1:][d]


def _age(t, tr):
    """창마다 **마지막 전환 이후 시간**. 전환이 앞에 없으면 창 시작부터."""
    if len(tr) == 0:
        return t - t[0]
    j = np.searchsorted(tr, t, side="right") - 1
    out = np.where(j >= 0, t - tr[np.clip(j, 0, len(tr) - 1)], t - t[0])
    return out


def _runs(m):
    """참(True) 이 이어지는 토막 길이들."""
    out, k = [], 0
    for v in m:
        if v:
            k += 1
        elif k:
            out.append(k); k = 0
    if k:
        out.append(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--app", default="oven", help="헛detect 를 볼 기기")
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--postproc", default="off", choices=("off", "cap", "full"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps0 = list(torch.load(a.ckpt[0], map_location="cpu",
                            weights_only=False)["appliances"])
    k = apps0.index(a.app)
    cache = real_windows(apps0, a.grid_s, dev)

    for path in a.ckpt:
        apps, pr = predict(path, cache, dev, postproc=a.postproc)
        assert apps == apps0
        print()
        print("=" * 84)
        print("%s — %s 헛detect 대 **마지막 전환 이후 시간**" % (path.split("/")[-1], KO[a.app]))
        print("=" * 84)
        for only_absent in (True, False):
            rows = {}
            runs = []
            for stem, d in sorted(cache.items()):
                y = d["y"].astype(bool)
                if only_absent and y[:, k].any():
                    continue                      # 그 기기가 **한 번도 안 켜진** 파일만
                t = d["t"]
                p = pr[stem][0].astype(bool)[:, k]
                fp = p & ~y[:, k]
                age = _age(t, _trans(t, y))
                for i in range(len(EDGES) - 1):
                    m = (age >= EDGES[i]) & (age < EDGES[i + 1])
                    if not m.any():
                        continue
                    r = rows.setdefault(i, [0, 0])
                    r[0] += int(fp[m].sum()); r[1] += int(m.sum())
                runs += [x * float(np.median(np.diff(t))) for x in _runs(fp)]
            if not rows:
                continue
            print("  %s" % ("[그 기기가 한 번도 안 켜진 파일만]" if only_absent
                            else "[전체 파일 · 참 OFF 창만]"))
            print("      %-14s %8s %8s   %s" % ("전환 이후", "창", "헛detect", "비율"))
            for i in sorted(rows):
                f, n = rows[i]
                lo, hi = EDGES[i], EDGES[i + 1]
                lab = ("%.0f~%.0f초" % (lo, hi)) if hi < 1e8 else ("%.0f초+" % lo)
                bar = "#" * int(round(40 * f / max(n, 1)))
                print("      %-14s %8d %8d   %6.1f%%  %s" % (lab, n, f, 100.0 * f / n, bar))
            if runs:
                runs = np.array(runs)
                print("      이어지는 헛detect 토막: %d개 · 중앙 %.1f초 · 최대 %.1f초 · 60초 넘는 것 %d개"
                      % (len(runs), float(np.median(runs)), float(runs.max()),
                         int((runs > 60).sum())))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
