# -*- coding: utf-8 -*-
"""**스파이크·잔차 탐지기** — 경계 판정의 **0단계** (14.163).

어느 기기가 언제 얼마나 튀는지부터 찾는다. 이것이 `run_diag_flipedge` ->
`run_diag_finewindow` -> `run_diag_finesweep` 로 가는 입구다.

```
  [1] 합 잔차 순위   |Σ예측 − (P관측 − 기준선)| 이 큰 자리와 그때 누가 얼마를 갖고 있나
  [2] 기기별 스파이크  이웃 ±`--nb`초 중앙값의 `--ratio` 배 이상 & `--min-w` 이상
```

⚠ **`present` 로 안 거른다.** 그 파일에 없는 기기의 유령이 가장 큰 오류인데
`score_arm` 은 그것을 한 칸도 안 센다 ([[score-what-you-excluded]]). 없는 기기는
`(없음)` 으로 표시한다.

이 도구가 14.159 의 ③ 을 찾았다 — test_1 에 **드라이기가 없는데** 208~402W 유령이
7번 섰고, 릴레이 한 사이클이 물리적으로 반파였기 때문이다.

    python -X utf8 -m src.run_diag_spike --stem test_1 --t0 500 --t1 580 \
        --ckpt results/cnn_gfp_s0.pt
    python -X utf8 -m src.run_diag_spike --stem test_2 --t0 205 --t1 290 \
        --ckpt results/cnn_pcap_s0.pt --even-median 5
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

from src.model.realdata import RealWindows, target_index  # noqa: E402
from src.preprocessing import load_nilm_npz  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

FS = 60
WIN = 3600
SH = {"oven": "오븐", "electiric_kettle": "포트", "hotplate": "핫플", "hair_dryer": "드라이",
      "beam_projector": "빔", "laptop_charger": "충전", "minipc": "미니PC",
      "fan": "선풍", "air_conditioner": "에어컨"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--stem", default="test_1")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--nb", type=float, default=2.0, help="이웃 창 반경(초)")
    ap.add_argument("--ratio", type=float, default=3.0, help="이웃 중앙값의 몇 배")
    ap.add_argument("--min-w", type=float, default=80.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--only-absent", action="store_true",
                    help="그 파일에 **없는** 기기만 (유령 확정)")
    ap.add_argument("--gate-max", type=float, default=1.01,
                    help="게이트가 이 값 이하인 것만 — 게이트는 껐는데 와트가 나오는 자리")
    ap.add_argument("--even-median", type=int, default=0,
                    help="짝수차 크기 k탭 중앙값 (14.160). 0/1 이면 비트 동일")
    a = ap.parse_args()

    if a.even_median > 1:
        from src.model import inputs as _I
        _I.EVEN_MEDIAN = int(a.even_median)
        print("⚠ 짝수차 이동중앙값 k=%d — 체크포인트가 그것으로 학습되지 않았다면 "
              "**분포 밖 시험**이다" % a.even_median)
    from src.model.inputs import build_inputs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    ev = json.load(open("processed_data/real_events.json",
                        encoding="utf-8"))["files"][a.stem]
    absent = {v for v in apps if v not in ev["appliances_present"]}
    raw = load_nilm_npz("processed_data/composite_eval/%s.npz" % a.stem)
    x = RealWindows._to_33ch(raw)
    P = np.asarray(raw["power_features"], np.float64)[:, 0]
    off = target_index(WIN)
    tt = np.arange(a.t0, min(a.t1, len(P) / FS) + 1e-9, a.step)
    tc = np.round(tt * FS).astype(np.int64)
    ok = (tc >= off) & (tc < x.shape[1] - (WIN - 1 - off))
    tt, tc = tt[ok], tc[ok]
    if not len(tt):
        print("창이 없다 — 구간을 넓혀라")
        return 1
    # 전부-꺼짐 기준선 (`run_train_seq._alloff_base` 와 같은 규약)
    try:
        from src.run_train_seq import _alloff_base
        base = float(_alloff_base(ev, P, len(P)))
    except Exception:
        base = 0.0
    print("%s · %.1f~%.1f초 · %d창 (%.2f초 눈금) · 기준선 %.1fW"
          % (a.stem, tt[0], tt[-1], len(tt), a.step, base))
    if absent:
        print("  ⚠ 이 파일에 **없는** 기기 (참 OFF·0W 확정): "
              + " · ".join(SH.get(v, v) for v in sorted(absent)))

    for ck in a.ckpt:
        m = load_model(ck, dev)[0]
        m.eval()
        G, PW = [], []
        with torch.no_grad():
            for i in range(0, len(tc), 96):
                f, w = build_inputs(np.stack([x[:, c - off:c - off + WIN]
                                              for c in tc[i:i + 96]]))
                o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                      torch.from_numpy(np.ascontiguousarray(w)).to(dev))
                G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
                PW.append(o["power"].float().cpu().numpy())
        G, PW = np.concatenate(G), np.concatenate(PW)
        S = PW.sum(1)
        res = S - (P[tc] - base)
        print("\n" + "=" * 96)
        print("■ %s" % ck.split("/")[-1].replace(".pt", ""))
        print("=" * 96)
        print("[1] **합 잔차** (Σ예측 − (P관측−기준선)) 가 큰 자리 %d" % a.top)
        for j in np.argsort(-np.abs(res))[:a.top]:
            top = np.argsort(-PW[j])[:3]
            print("   %8.2f초 잔차 %+7.0fW  관측 %6.0fW  Σ %6.0fW | %s"
                  % (tt[j], res[j], P[tc[j]], S[j],
                     "  ".join("%s%s %.0fW(g%.2f)"
                               % (SH.get(apps[k], apps[k]),
                                  "**없음**" if apps[k] in absent else "",
                                  PW[j, k], G[j, k]) for k in top)))
        w = max(int(round(a.nb / a.step)), 3)
        print("\n[2] **기기별 스파이크** (이웃 ±%.1f초 중앙값의 %.1f배 이상 & %.0fW 이상)"
              % (a.nb, a.ratio, a.min_w))
        hits = []
        for k, v in enumerate(apps):
            u = PW[:, k]
            ab = v in absent
            if a.only_absent and not ab:
                continue
            for j in range(w, len(u) - w):
                nb = np.concatenate([u[j - w:j - 2], u[j + 3:j + w]])
                med = float(np.median(nb)) if len(nb) else 0.0
                if (u[j] >= a.min_w and u[j] >= a.ratio * max(med, 5.0)
                        and u[j] >= u[j - 1] and u[j] >= u[j + 1]
                        and G[j, k] <= a.gate_max):
                    hits.append((ab, u[j] / max(med, 1.0), tt[j], v, u[j], med, G[j, k],
                                 P[tc[j]]))
        #: **없는 기기를 먼저**, 그 다음 튄 배수 순. 듀티 기기의 진짜 펄스가 목록을 덮는 것을 막는다
        hits.sort(key=lambda r: (not r[0], -r[1]))
        found = len(hits)
        for ab, rt, t_, v, u_, med, g_, p_ in hits[:a.top]:
            print("   %8.2f초 %-8s%s %6.0fW (이웃 중앙 %5.0fW · x%.1f) "
                  "게이트 %.3f · 관측 %6.0fW"
                  % (t_, SH.get(v, v), " **없음**" if ab else "       ",
                     u_, med, rt, g_, p_))
        if found > a.top:
            print("   ... 그리고 %d개 더 (`--top` 으로 늘려라)" % (found - a.top))
        if not found:
            print("   (없음)")
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("\n다음 단계: `run_diag_flipedge`(경계 찾기) -> `run_diag_finewindow`(10초 펴기)"
          " -> `run_diag_finesweep`(전부 훑기)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
