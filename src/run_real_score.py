"""실측 채점 — **후처리를 켜고 끄고** 견준다 (13.65)
=================================================================
`run_scorecard` 는 관문(AUC)과 합성 홀드아웃 전력 헤드를 본다. 후처리는 **전력을
재배분**하므로 그 두 자로는 값어치가 안 잡힌다. 여기서는 `run_adapt` 가 학습 중에
쓰는 실측 지표 넷을 후처리 전후로 낸다:

    on/off F1        관문이 정답 구간을 맞히나 (사람 로그·이벤트 기반)
    이벤트 |상대오차|  계단 크기를 맞히나
    **오귀속**        그 파일에 **없는 기기**에 붙인 전력 — 라벨 없이 잴 수 있다
    총전력 잔차       |관측 − 예측|

⚠ `resistive_match` 는 `min_w=150W` 문턱이라 **SMPS 전용 창(자리 D 93W)은 안
  건드린다.** 자리 D 배분은 후처리로 못 고친다 — 고치는 것은 저항끼리의 조합이다.
  그 축은 `run_site_split_check` 로 따로 본다.

    python -m src.run_real_score --ckpt results/hwh1_nosb_s0.pt --postproc off full
"""
from typing import List, Sequence
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.real_events import load_events
from src.evaluation.sealing import is_sealed
from src.model.realdata import dense_targets, upsample_to_cycles
from src.run_baseline import S_I
from src.run_gate_check import load_model


def score_one(model, apps: List[str], dev: str, postproc: str, stride: int = 30) -> dict:
    from src.evaluation.real_events import score_absent, score_events, score_on_off
    from src.model.postproc import apply_postproc, resistive_match
    ev = load_events()
    out = {}
    for stem in sorted(ev):
        if is_sealed(stem):
            continue
        rw = dense_targets(stem, stride=stride,
                           site_transfer=getattr(model, "site_transfer", None))
        P, G, S, PO, OH, PN = [], [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(rw), 512):
                idx = np.arange(i, min(i + 512, len(rw)))
                f, w, pobs, oh, pn = rw.batch(idx)
                o = model(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                          torch.from_numpy(np.ascontiguousarray(w)).to(dev))
                P.append(o["power"].float().cpu().numpy())
                G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
                S.append(o["standby"].float().cpu().numpy())
                PO.append(pobs); OH.append(oh); PN.append(pn)
        P = np.concatenate(P); G = np.concatenate(G); S = np.concatenate(S)
        PO = np.concatenate(PO); OH = np.concatenate(OH); PN = np.concatenate(PN)
        if postproc != "off":
            P, G = apply_postproc(P, G, apps)
            if postproc == "full":
                P, G = resistive_match(P, G, apps, PO, np.asarray(rw.v_observed, np.float64),
                                       S, PN, obs_harm=OH)
            P = np.asarray(P, np.float32); G = np.asarray(G, np.float32)
        r = P.sum(1) + S.sum(1) + PN - PO
        n_cyc = int(ev[stem]["cycles"])
        pc = upsample_to_cycles(P, rw.target_cycle, n_cyc)
        oc = upsample_to_cycles(G > 0.5, rw.target_cycle, n_cyc)
        out[stem] = {
            "resid_abs": float(np.abs(r).mean()), "resid_mean": float(r.mean()),
            "on_off": score_on_off(oc, stem, apps, events=ev),
            "events": score_events(pc, stem, apps, events=ev),
            "absent": score_absent(P, stem, apps, pred_on=G > 0.5, s_i=S_I, events=ev),
        }
    return out


def summarize(rs: dict) -> dict:
    f1 = [v["f1"] for s in rs.values() for v in s["on_off"].values() if v["n_true_on"] > 0]
    er = [abs(e["error_rel"]) for s in rs.values() for e in s["events"]
          if e["error_rel"] == e["error_rel"]]
    fa = [v["fa_rel"] for s in rs.values() for v in s["absent"]["absent"].values()
          if v["fa_rel"] == v["fa_rel"]]
    return {
        "resid_abs": float(np.mean([s["resid_abs"] for s in rs.values()])),
        "f1": float(np.mean(f1)) if f1 else float("nan"),
        "ev_err": float(np.mean(er)) if er else float("nan"),
        "absent_w": float(np.sum([s["absent"]["absent_sum_w"] for s in rs.values()])),
        "fa_max": float(np.max(fa)) if fa else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--postproc", nargs="+", default=["off", "full"],
                    choices=("off", "cap", "full"))
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--detail", action="store_true",
                    help="오귀속을 파일 x 기기로 쪼개 찍는다 — 합계만 보면 '늘었다' 까지만 "
                         "알고 어디서 늘었는지는 모른다")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for p in a.ckpt:
        m, apps, _ = load_model(p, dev)
        m.eval()
        for mode in a.postproc:
            rs = score_one(m, apps, dev, mode, a.stride)
            rows.append((p, mode, summarize(rs), rs))
            print(f"  {p} · {mode} 완료", flush=True)
        del m
        torch.cuda.empty_cache()

    short = [p.replace("\\", "/").split("/")[-1].replace(".pt", "") for p, _, _, _ in rows]
    print("\n" + "=" * 100)
    print("실측 채점 — 후처리 전후")
    print("=" * 100)
    print(f"  {'판 · 후처리':34s}{'잔차|W|':>10s}{'on/off F1':>11s}"
          f"{'이벤트오차':>11s}{'오귀속W':>10s}{'최악FA':>9s}")
    for (p, mode, s, _), nm in zip(rows, short):
        print(f"  {nm + ' · ' + mode:34s}{s['resid_abs']:10.2f}{s['f1']:11.3f}"
              f"{s['ev_err']:11.3f}{s['absent_w']:10.1f}{s['fa_max']:9.3f}")

    print("\n  파일별 잔차 |W|")
    stems = sorted(rows[0][3])
    print(f"  {'판 · 후처리':34s}" + "".join(f"{s:>10s}" for s in stems))
    for (p, mode, _, rs), nm in zip(rows, short):
        print(f"  {nm + ' · ' + mode:34s}"
              + "".join(f"{rs[s]['resid_abs']:10.2f}" for s in stems))
    if a.detail:
        # ── 오귀속을 파일 x 기기로 쪼갠다 (13.65.2) ─────────────────────────
        # 없는 기기는 정답이 **0 으로 확정**이므로 칸마다 그대로 읽으면 된다.
        print("\n" + "=" * 100)
        print("오귀속 상세 — 그 파일에 **없는 기기**의 평균 예측 W (정답 0) / 켜짐비율")
        print("=" * 100)
        keys = sorted({(st, ap) for _, _, _, rs in rows for st in rs
                       for ap in rs[st]["absent"]["absent"]})
        print(f"  {'파일 · 없는 기기':30s}" + "".join(f"{n[:15]:>17s}" for n in short))
        for st, ap in keys:
            vals = [rs[st]["absent"]["absent"].get(ap) for _, _, _, rs in rows]
            if all(v is None or v["mean_w"] < 0.5 for v in vals):
                continue
            print(f"  {st + ' · ' + ap:30s}"
                  + "".join(f"{'—':>17s}" if v is None
                            else f"{v['mean_w']:11.1f}/{v['on_rate']:5.2f}" for v in vals))

    if a.detail:
        # ── 있는 기기의 검출 (13.66) ────────────────────────────────────────
        # 오귀속의 짝이다. 없는 기기에 붙였나(위)와 **있는 기기를 놓쳤나**(아래)를
        # 같이 봐야 잔차가 어디서 오는지 갈린다.
        print("\n" + "=" * 100)
        print("검출 상세 — 그 파일에 **있는 기기**의 on/off F1 (n_true_on>0 인 칸만)")
        print("=" * 100)
        keys2 = sorted({(st, ap) for _, _, _, rs in rows for st in rs
                        for ap, v in rs[st]["on_off"].items() if v["n_true_on"] > 0})
        print(f"  {'파일 · 있는 기기':30s}" + "".join(f"{n[:15]:>17s}" for n in short))
        for st, ap in keys2:
            vals = [rs[st]["on_off"].get(ap) for _, _, _, rs in rows]
            print(f"  {st + ' · ' + ap:30s}"
                  + "".join(f"{'—':>17s}" if v is None else f"{v['f1']:17.3f}"
                            for v in vals))

    print("\n  ⚠ `오귀속` 이 2단계의 주 지표다 — 그 파일에 없는 기기에 붙인 전력이고")
    print("    라벨 없이 잴 수 있다. `resistive_match` 가 노리는 축이 여기다.")


if __name__ == "__main__":
    main()
