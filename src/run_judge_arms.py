# -*- coding: utf-8 -*-
"""**여러 팔을 한 표에 놓고 짝 검정한다** (14.415).

`run_scorecard` 는 팔마다 수를 내지만 **씨앗 짝 검정**을 안 한다. 이 프로젝트의
판정줄은 늘 *"씨앗을 맞춘 차이의 t"* 였는데 (§58·§65 …) 그걸 그때그때 손으로 냈다.
여기서 못 박는다.

```
  ★1 실측 게이트 AUC — 기준선 팔과 **같은 씨앗끼리** 뺀 차이의 평균과 t
  ★2 **자리별로 가른다** — D/E (기존) 대 **F** (test_7, 배포 도메인).
     14.409 가 쟀다: 자리 F 에서 미니PC 0.784 -> 0.445~0.583 (동전)
  ★3 저항 넷도 같이 본다 — SMPS 전용 처치가 저항으로 **새는지** 본다
     (14.35 의 `L_swap` 이 저항을 겨냥했는데 **SMPS 가 −0.043** 나빴다. 반대 방향 경계)
  ★4 예측합/관측 — 게이트가 아니라 **와트**가 맞는지. 자리 F 에서 0.72~0.79 였다
```

    python -X utf8 -m src.run_judge_arms --base v52base --arms v52dp v52mg --seeds 0 1 2 3 4
    python -X utf8 -m src.run_judge_arms --base comb --arms v52base --seeds 0 1 2 3 4

⚠ **시간 (14.416 실측).** 판당 `forward_file` 이 **파일 크기에 비례**한다:
  test_4(29,880 사이클) 6.4초 · test_1(48,000) · test_2(51,810) 은 그 1.6~1.8배다.
  5파일+test_7 x 15판 = **약 30분** (노트북 CPU). 처음에 test_4 로만 재고 "10분" 이라
  했다가 3배 틀렸다 — **제일 작은 파일로 재지 마라**.
  ⇒ 그래서 `patches/judge_arms.sbatch` 로 HPC 에서 돌린다 (14.416).
⚠ 캐시가 다른 팔끼리 견줄 때는 (예: v49 `comb` 대 v52 `v52base`) **풀이 달라져
  `sig`·`harm_scale` 이 바뀌었다**. 추론에는 안 쓰이지만 후처리를 켜면 갈린다 —
  여기서는 후처리를 **안 켠다**.
"""
from typing import Dict, List
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.evaluation.sealing import is_sealed  # noqa: E402
from src.model.realdata import dense_targets  # noqa: E402
from src.run_gate_check import forward_file, load_model, sync_even_median  # noqa: E402
from src.run_scorecard import auc, build_on_off_truth, load_events  # noqa: E402

SMPS = ("minipc", "laptop_charger", "beam_projector")
RES = ("oven", "hotplate", "electiric_kettle", "hair_dryer")
SITE_F = ("test_6", "test_7")


def tstat(d: np.ndarray) -> float:
    """짝 차이의 t (표본 표준편차, n−1)."""
    if len(d) < 2 or np.allclose(d.std(ddof=1), 0):
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="기준선 팔 이름 (cnn_<이름>_s<씨앗>.pt)")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--dev", default="cpu")
    a = ap.parse_args()

    names = [a.base] + list(a.arms)
    sync_even_median(["results/cnn_%s_s%d.pt" % (a.base, a.seeds[0])])
    ev = load_events()
    stems = [s for s in ev if not is_sealed(s)]

    #: (팔, 씨앗) -> {기기: [점수], 기기: [라벨]} · 자리별로 따로 쌓는다
    acc: Dict[tuple, Dict[str, Dict[str, list]]] = {}
    ratio: Dict[tuple, Dict[str, list]] = {}
    apps: List[str] = []
    #: ⚠ 14.415 — 모델은 **한 번만** 읽는다. 파일 고리 안에서 읽으면 파일수 x 판수 만큼
    #  다시 읽는다 (6 x 15 = 90번). `cpu` 에선 0.0초라 안 보이지만 `cuda` 에선 1.9초씩이라
    #  **171초가 샌다**. `real_auc_many` 가 `dense_targets` 를 밖으로 뺀 것과 같은 자리다.
    MODELS = []
    for nm in names:
        for sd in a.seeds:
            p = "results/cnn_%s_s%d.pt" % (nm, sd)
            try:
                m, ap_list, _ = load_model(p, a.dev)
            except Exception as e:
                print("  %s 못 읽음: %s" % (p, e))
                continue
            apps = list(ap_list)
            MODELS.append((nm, sd, m))
    print("  판 %d개 적재" % len(MODELS), flush=True)
    for stem in sorted(stems):
        n_cyc = int(ev[stem]["cycles"])
        rw = dense_targets(stem, stride=a.stride, site_transfer=None)
        grp = "F" if stem in SITE_F else "DE"
        #: ⚠ 14.416 — `build_on_off_truth` 도 **파일당 한 번**이다. 모델 고리 안에 두면
        #  (n_cyc, K) 배열을 판수만큼 다시 짓는다 — test_2 면 51,810x9 를 **15번**.
        #  `dense_targets`·모델 적재와 같은 자리다 (`real_auc_many` 의 교훈).
        on, sc = build_on_off_truth(stem, apps, n_cyc, ev)
        for nm, sd, m in MODELS:
                d = forward_file(m, stem, a.dev, stride=a.stride, _rw=rw)
                t = np.asarray(d["targets"], int).clip(0, n_cyc - 1)
                key = (nm, sd)
                acc.setdefault(key, {})
                for j, app in enumerate(apps):
                    k = sc[t, j]
                    if not k.any():
                        continue
                    e = acc[key].setdefault(app, {"DE": [[], []], "F": [[], []]})
                    e[grp][0].append(d["gate"][k, j]); e[grp][1].append(on[t, j][k])
                pr = (d["gate"] * d["p_raw"]).sum(1)
                ob = d["p_observed"]
                w = ob > 20.0
                ratio.setdefault(key, {"DE": [], "F": []})[grp].append(
                    float(np.mean(pr[w] / ob[w])) if w.any() else np.nan)
        print("  %s 채점 완료 (판 %d)" % (stem, len(MODELS)), flush=True)

    def pooled(nm, sd, app, grp):
        e = acc.get((nm, sd), {}).get(app)
        if not e or not e[grp][0]:
            return float("nan")
        s = np.concatenate(e[grp][0]); y = np.concatenate(e[grp][1])
        return auc(s, y) if 0 < y.sum() < len(y) else float("nan")

    for grp, lab in (("DE", "기존 자리 D/E"), ("F", "**자리 F** (배포 도메인)")):
        print("\n" + "=" * 86)
        print("  ★ 실측 게이트 AUC — %s" % lab)
        print("  %-18s %9s | %s" % ("기기", a.base,
                                    " | ".join("%-24s" % x for x in a.arms)))
        for app in list(SMPS) + list(RES):
            if app not in apps:
                continue
            b = np.array([pooled(a.base, s, app, grp) for s in a.seeds], float)
            if np.all(np.isnan(b)):
                continue
            cells = []
            for nm in a.arms:
                v = np.array([pooled(nm, s, app, grp) for s in a.seeds], float)
                d = v - b
                ok = np.isfinite(d)
                cells.append("%7.4f %+7.4f t%+5.1f"
                             % (np.nanmean(v), np.nanmean(d), tstat(d[ok])))
            star = "★" if app in SMPS else " "
            print(" %s%-18s %9.4f | %s" % (star, app, np.nanmean(b), " | ".join(cells)))

        print("  %-18s %9s | %s" % ("예측합/관측", "", ""))
        b = np.array([np.nanmean(ratio.get((a.base, s), {}).get(grp, [np.nan]))
                      for s in a.seeds], float)
        cells = []
        for nm in a.arms:
            v = np.array([np.nanmean(ratio.get((nm, s), {}).get(grp, [np.nan]))
                          for s in a.seeds], float)
            d = v - b
            cells.append("%7.4f %+7.4f t%+5.1f" % (np.nanmean(v), np.nanmean(d),
                                                   tstat(d[np.isfinite(d)])))
        print("  %-18s %9.4f | %s" % ("", np.nanmean(b), " | ".join(cells)))
    print("\n  ⓘ t 는 **씨앗을 맞춘 차이**의 t (n=%d). |t| >= 2.8 이면 5씨앗에서 p<0.05 다."
          % len(a.seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
