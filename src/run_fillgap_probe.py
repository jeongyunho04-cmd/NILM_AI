"""
후처리 — 미설명 잔여를 물리로 메운다 (설계 문서 12.51절)
==========================================================
12.50.4 가 그림을 다시 그렸다. 실측 `>=1300W & 오븐 통전` 구간에서 핫플은 98.8%
통전 중이고, 판별력도 충분하다 (합성 AUC 0.917). **모델은 못 가르는 것이 아니라
지나치게 보수적이고, 비운 자리를 유령 포트가 채운다.**

12.9.8 의 물리 프라이어는 **하한**만 건다 — "창 최대가 최소 ON 전력보다 작으면
못 켜진다". 그 거울(**함의**)이 없다:

    관측이 켜졌다고 한 것들로 설명이 안 되면, 무언가가 더 켜져 있다.

    python -m src.run_fillgap_probe --ckpt results/cnn_ov1.pt

[규칙 — 기기 이름을 안 쓴다]
```
미설명 = 관측 − (Σ 기기 예측 + 대기 + 계측잡음)
미설명 > margin 이면:
    후보 = 꺼져 있고(게이트<=0.5) **최소 ON 전력 <= 미설명** 인 기기
    그중 게이트가 가장 높은 것을 켠다.  전력 = min(p_raw, 미설명)
```

[왜 이번에는 게이트 순서를 써도 되나 — 12.46 과 다른 점]
12.46 은 **끄는** 규칙이었고, 그 창에서 순서가 뒤집혀 있었다 (없는 포트 0.952 >
있는 핫플 0.310). 여기는 **켜는** 규칙이고, 미탐 창에서 순서가 맞다:

    꺼진 기기 중 핫플이 1위 74.3% (2위까지 94%),  게이트 중앙 0.136

그리고 순서만 쓰지 않는다. **최소 ON 전력 필터가 물리로 후보를 자른다** —
미설명 중앙 286W 는 핫플(214W)은 허용하고 전기포트(578W)는 **배제**한다.
유령의 주범을 이름이 아니라 물리로 막는 것이 요점이다.

[판정 기준 (돌리기 전에 적는다)]
1. `>=1300W` 핫플 재현율이 오른다 (현재 `cnn_ov1` 0.413)
2. **참으로 꺼진 기기를 켜는 비율이 5% 미만** — 12.46 이 49~64% 로 실패한 그 기준
3. 유령이 안 오른다 (규칙이 기기를 **켜므로** 유령을 만들 수 있다)
4. 잔차가 준다 (미설명을 메우는 규칙이다)
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.run_gate_check import forward_file, load_model
from src.run_live import KOR
from src.run_seed_variance_probe import _mask_from

MARGINS = (100.0, 150.0, 200.0, 300.0, 400.0)


def min_on_w(model) -> np.ndarray:
    """모델 프라이어가 쓰는 기기별 최소 ON 전력 (W)."""
    th = model.on_threshold_asinh.detach().cpu().numpy()
    return np.sinh(th) * 100.0


def truth_mask(ev: dict, stem: str, app: str, targets, n: int):
    """그 기기의 정답 ON. 없는 기기면 전부 False, 라벨 없으면 None."""
    if app not in ev[stem].get("appliances_present", []):
        return np.zeros(len(targets), bool)
    iv = ev[stem]["intervals"].get(app, {})
    key = "_heater_pulses" if app == "oven" and iv.get("_heater_pulses") else "on"
    if not iv.get(key):
        return None                      # 정답 구간이 없다 — 셀 수 없다
    return _mask_from(iv.get(key), n, targets)


def main() -> int:
    ap = argparse.ArgumentParser(description="미설명 잔여 메우기 (12.51절)")
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_ov1.pt"])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--out", default="results/fillgap_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = load_events()
    stems = [s for s in sorted(ev) if not is_sealed(s)]
    payload: Dict[str, dict] = {}

    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        tag = Path(ck).stem
        mow = min_on_w(model)
        jh = apps.index("hotplate")
        fwd = {s: forward_file(model, s, dev, stride=a.stride) for s in stems}
        truth = {s: {app: truth_mask(ev, s, app, fwd[s]["targets"], int(ev[s]["cycles"]))
                     for app in apps} for s in stems}

        print("=" * 96)
        print(f"[미설명 잔여 메우기] {tag}   (비봉인 {len(stems)}파일, stride {a.stride})")
        print("  최소 ON 전력(W): " + "  ".join(
            f"{KOR.get(x, x)} {mow[j]:.0f}" for j, x in enumerate(apps)
            if mow[j] > 100))
        print("=" * 96)
        print(f"    {'규칙':<16s}{'유령W':>9s}{'잔차W':>9s}"
              f"{'핫플재현(≥1300)':>16s}{'핫플정밀':>10s}{'켠 창':>8s}"
              f"{'참OFF 오점화':>13s}")
        print("    " + "-" * 82)

        rows = {}
        for margin in (None,) + MARGINS:
            g, r, non, nbad, nchk = [], [], 0, 0, 0
            hp_tp = hp_fp = hp_true = 0
            fired = {}
            for s in stems:
                d = fwd[s]
                P = (d["gate"] * d["p_raw"]).copy()
                on = d["gate"] > 0.5
                base = d["standby"].sum(1) + d["p_noise"]
                if margin is not None:
                    unexp = d["p_observed"] - (P.sum(1) + base)
                    for i in np.flatnonzero(unexp > margin):
                        cand = np.flatnonzero((~on[i]) & (mow <= unexp[i]))
                        if not len(cand):
                            continue
                        j = cand[np.argmax(d["gate"][i, cand])]
                        P[i, j] = min(float(d["p_raw"][i, j]), float(unexp[i]))
                        on[i, j] = True
                        non += 1
                        fired[apps[j]] = fired.get(apps[j], 0) + 1
                        t = truth[s][apps[j]]
                        if t is not None:
                            nchk += 1
                            nbad += int(not t[i])
                absent = [j for j, x in enumerate(apps)
                          if x not in ev[s]["appliances_present"]]
                g.append(float(P[:, absent].mean(0).sum()))
                r.append(float(np.abs(P.sum(1) + base - d["p_observed"]).mean()))
                th = truth[s]["hotplate"]
                if th is not None:
                    m = d["p_observed"] >= 1300.0
                    pred = on[m, jh]
                    hp_tp += int((pred & th[m]).sum())
                    hp_fp += int((pred & ~th[m]).sum())
                    hp_true += int(th[m].sum())
            rec = hp_tp / max(hp_true, 1)
            pre = hp_tp / max(hp_tp + hp_fp, 1)
            row = {"ghost_w": float(np.mean(g)), "resid_w": float(np.mean(r)),
                   "hp_recall_hi": rec, "hp_precision_hi": pre,
                   "n_fired": non, "n_checked": nchk, "n_wrong": nbad,
                   "wrong_rate": nbad / max(nchk, 1), "fired_by_app": fired}
            lab = "원본" if margin is None else f"미설명 > {margin:.0f}W"
            rows[lab] = row
            print(f"    {lab:<16s}{row['ghost_w']:>9.2f}{row['resid_w']:>9.2f}"
                  f"{rec:>16.3f}{pre:>10.3f}{non:>8d}"
                  + (f"{nbad:>6d}/{nchk:<6d}" if nchk else f"{'-':>13s}"))
        for lab, row in rows.items():
            if row["fired_by_app"]:
                print(f"      {lab}: " + ", ".join(
                    f"{KOR.get(k, k)} {v}" for k, v in
                    sorted(row["fired_by_app"].items(), key=lambda x: -x[1])))
        payload[tag] = rows
        print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
