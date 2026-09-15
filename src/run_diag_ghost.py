# -*- coding: utf-8 -*-
"""**유령 채점** — 그 파일에 **없는 기기**에 모델이 무엇을 주는가 (14.159).

사용자: *"유령 채점 고치고 wh03 판정해줘"*

자에 구멍이 있었다. `run_baseline_nilmtk.score_arm` 은

```python
    for k in np.nonzero(d["present"])[0]:        # <- 없는 기기는 **한 칸도 안 센다**
```

라서 **그 파일에 안 꽂힌 기기의 유령이 판정 줄에 전혀 안 나타난다.** 실제로 잡힌 것:

```
  test_2  오븐 유령 **46초 연속 · 1,254W**   (test_2 에 오븐은 없다)
  test_1  드라이 유령 스파이크 208~402W      (test_1 에 드라이기는 없다)
```

둘 다 그림(5번 칸)에만 보이고 숫자에는 없었다.

⚠ **`score_arm` 은 안 건드린다.** 그 줄은 로그의 모든 옛 숫자와 이어져 있다
([[match-the-scoring-convention-before-comparing]]). 대신 **독립된 자**를 하나 더 세운다.

없는 기기는 참값이 **OFF 확정 · 0W 확정**이라 라벨 애매함이 전혀 없다 — 가장 깨끗한
모집단이다.

```
  유령 게이트율   없는 기기 칸 중 σ(on_logit) > 0.5 인 비율
  유령 전력      그 칸들의 평균 예측 W
  유령 에너지 몫  Σ(없는 기기 W) / Σ(모든 기기 W)
  유령 최장      가장 긴 연속 유령 구간 (초)
  (곁들임) 있는데 꺼진 기기의 헛ON율 — `score_arm` 의 정확도가 이미 섞어 보던 것
```

    python -X utf8 -m src.run_diag_ghost --ckpt results/cnn_pcap_s0.pt results/cnn_wh03_s0.pt
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

from src.run_gate_check import load_model  # noqa: E402

SH = {"oven": "오븐", "electiric_kettle": "포트", "hotplate": "핫플", "hair_dryer": "드라이",
      "beam_projector": "빔", "laptop_charger": "충전", "minipc": "미니PC",
      "fan": "선풍", "air_conditioner": "에어컨"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--per-app", action="store_true", help="기기별 표도 찍는다")
    ap.add_argument("--even-median", type=int, default=0,
                    help="짝수차 크기에 k탭 이동중앙값 (14.160)을 **강제**한다 — 분포 밖 시험용. "
                         "안 주면 체크포인트가 적어 둔 값을 **판마다 따라간다**")
    a = ap.parse_args()

    from src.model import inputs as _I
    if a.even_median > 1:
        _I.EVEN_MEDIAN = int(a.even_median)
        print("⚠ 짝수차 이동중앙값 k=%d 를 **강제**한다 — 그것으로 학습되지 않은 판은 "
              "**분포 밖 시험**이다" % a.even_median)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    #: ⚠ `EVEN_MEDIAN` 은 모듈 전역이라 한 표에 k=0 판과 k=5 판을 같이 놓으면 한쪽이
    #  **조용히 분포 밖**으로 간다. 판마다 그 판의 값으로 **창을 다시 짓는다**
    #  ([[verify-the-input-path-not-just-the-model]]).
    _em = {p_: max(int(torch.load(p_, map_location="cpu",
                                  weights_only=False).get("even_median", 0) or 0), 1)
           for p_ in a.ckpt}
    if a.even_median > 1:
        _em = {p_: int(a.even_median) for p_ in a.ckpt}
    from src.run_train_seq import real_windows
    _cur = [None]

    def _need(k):
        if _cur[0] == k:
            return
        _I.EVEN_MEDIAN = int(k)
        cache.clear()
        cache.update(real_windows(apps, a.grid_s, dev))
        _cur[0] = k
        if len(set(_em.values())) > 1:
            print("  -- 창 다시 지음: 짝수차 중앙값 k=%d --" % k, flush=True)

    cache = {}
    _need(_em[a.ckpt[0]])
    if len(set(_em.values())) > 1:
        print("  ** 짝수차 중앙값이 갈린다: "
              + " · ".join("%s k=%d" % (p_.split("/")[-1][:-3], v)
                           for p_, v in _em.items()))
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]

    # 없는 기기 / 있는데 꺼진 칸
    ABS, OFFP, STEM, DT = [], [], [], []
    for stem, d in cache.items():
        n = len(d["t"])
        pres = np.array([v in ev[stem]["appliances_present"] for v in apps])
        unc = np.zeros((n, len(apps)), bool)
        t = np.asarray(d["t"], float)
        for k, v in enumerate(apps):
            for t0, t1 in ev[stem]["intervals"].get(v, {}).get("uncertain", []):
                unc[(t >= t0) & (t <= t1), k] = True
        y = d["y"].astype(bool)
        ABS.append(np.tile(~pres, (n, 1)))
        OFFP.append(pres[None] & (~y) & (~unc))
        STEM += [stem] * n
        DT.append(np.r_[a.grid_s, np.diff(t)])
    ABS, OFFP = np.concatenate(ABS), np.concatenate(OFFP)
    DT = np.concatenate(DT)
    STEM = np.array(STEM)
    print("실측 5파일 · 창 %d · 격자 %.1fs" % (len(ABS), a.grid_s))
    print("  파일별 **없는** 기기 (참 OFF·0W 확정)")
    for stem in cache:
        miss = [SH.get(v, v) for v in apps if v not in ev[stem]["appliances_present"]]
        print("    %-8s %d종: %s" % (stem, len(miss), " · ".join(miss)))
    print("  없는 기기 칸 **%d개** · 있는데 꺼진 칸 %d개" % (ABS.sum(), OFFP.sum()))

    rows = []
    for ck in a.ckpt:
        _need(_em[ck])                 # 이 판의 짝수차 규약으로 창을 맞춘다
        m = load_model(ck, dev)[0]
        m.eval()
        G, PW = [], []
        with torch.no_grad():
            for stem, d in cache.items():
                for i in range(0, len(d["t"]), 256):
                    o = m(torch.from_numpy(np.ascontiguousarray(d["fine"][i:i + 256])).to(dev),
                          torch.from_numpy(np.ascontiguousarray(d["wide"][i:i + 256])).to(dev))
                    G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
                    PW.append(o["power"].float().cpu().numpy())
        G, PW = np.concatenate(G), np.concatenate(PW)
        gon = G > a.thr
        # 최장 연속 유령 (기기별로 파일 안에서)
        longest, who = 0.0, ""
        for stem in cache:
            sel = STEM == stem
            for k in range(len(apps)):
                if not ABS[sel, k].any():
                    continue
                b = gon[sel, k]
                if not b.any():
                    continue
                dd = np.diff(np.r_[0, b.astype(int), 0])
                for s_, e_ in zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1) - 1):
                    L = (e_ - s_ + 1) * a.grid_s
                    if L > longest:
                        longest, who = L, "%s·%s" % (stem, SH.get(apps[k], apps[k]))
        rows.append((ck.split("/")[-1].replace(".pt", ""),
                     100 * gon[ABS].mean(),
                     float(PW[ABS].mean()),
                     100 * PW[ABS].sum() / max(PW.sum(), 1e-9),
                     float(np.percentile(PW[ABS], 99)),
                     longest, who,
                     100 * gon[OFFP].mean(),
                     float(PW[OFFP].mean())))
        if a.per_app:
            print("\n  ■ %s 기기별 (없는 기기만)" % rows[-1][0])
            for k, v in enumerate(apps):
                if not ABS[:, k].any():
                    continue
                s = ABS[:, k]
                print("     %-18s 칸 %5d · 게이트>%.1f %5.1f%% · 평균 %6.1fW · p99 %7.1fW"
                      % (SH.get(v, v), s.sum(), a.thr, 100 * gon[s, k].mean(),
                         PW[s, k].mean(), np.percentile(PW[s, k], 99)))
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    print("\n★ **유령 채점** — 없는 기기(참 OFF·0W 확정)")
    print("  %-16s %9s %9s %9s %9s %9s  %s"
          % ("판", "게이트율", "평균W", "에너지몫", "p99 W", "최장초", "최장 자리"))
    for r in rows:
        print("  %-16s %8.2f%% %9.2f %8.2f%% %9.1f %9.1f  %s"
              % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    print("\n  (곁들임) 있는데 꺼진 칸 — `score_arm` 의 정확도가 섞어 보던 것")
    print("  %-16s %9s %9s" % ("판", "헛ON율", "평균W"))
    for r in rows:
        print("  %-16s %8.2f%% %9.2f" % (r[0], r[7], r[8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
