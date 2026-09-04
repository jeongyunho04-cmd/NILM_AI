# -*- coding: utf-8 -*-
"""유령이 **어느 창**에 있는가 — 시각·참 ON 집합·기기별 게이트와 p_raw (12.180).

    python -m src.run_ghost_window_probe --stem test_15 --app air_conditioner \
        --ckpt results/adapt_ac_s0.pt results/adapt_ac_s1.pt results/cnn_ac.pt
    python -m src.run_ghost_window_probe --stem test_15 --range 388,414 --ckpt ... [--harm]

`score_absent` 의 유령 W 는 파일 평균이라 **한 구간에 몰린 것**과 **골고루 번진
것**을 못 가른다. 12.176.2 가 "test_15 에어컨 30.96W" 로 적은 것은 10초 단위로
찍으면 388~414초 한 구간(포트+드라이기 강풍 겹침, 2308W)의 **659W** 였다.

`--range` 는 그 구간의 기기별 게이트/p_raw 를 체크포인트마다 찍고, `--harm` 을
주면 그 구간에서 `L_harm`(inv_h2, harm_offset 포함) 과 |잔차| 를 같은 식으로
재서 **손실이 두 해를 실제로 가르는지** 본다. 12.180 에서 s0(에어컨 해) 3.70 vs
s1(드라이기 해) 1.30 — 손실은 가르는데 s0 이 못 넘어간다(국소 최소).
"""
import argparse
import json
import sys
from collections import Counter

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

SHORT = {"hair_dryer": "드라", "electiric_kettle": "포트", "hotplate": "핫플", "minipc": "PC",
         "beam_projector": "프로", "laptop_charger": "충전", "oven": "오븐", "fan": "선풍", "air_conditioner": "AC"}


def _mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs:
        m[int(s * 60):min(int(e * 60), n)] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stem", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--app", default="", help="타임라인으로 찍을 기기 (없으면 --range 만)")
    ap.add_argument("--range", default="", metavar="T0,T1", help="구간(초)의 기기별 게이트/p_raw")
    ap.add_argument("--bin", type=float, default=10.0)
    ap.add_argument("--harm", action="store_true", help="--range 구간에서 L_harm 오차도 잰다")
    ap.add_argument("--events", default="processed_data/real_events_refined.json")
    a = ap.parse_args()
    import torch
    from src.run_gate_check import forward_file, load_model
    ev = json.load(open(a.events, encoding="utf-8"))["files"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = int(ev[a.stem]["cycles"]); iv = ev[a.stem]["intervals"]
    truth = {x: _mask(iv[x].get("on", []), n) for x in iv}
    D, apps = {}, None
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        D[ck] = forward_file(model, a.stem, dev, stride=30)
    t = D[a.ckpt[0]]["targets"]
    tags = [c.split("/")[-1].replace(".pt", "") for c in a.ckpt]

    if a.app:
        j = apps.index(a.app)
        print(f"{a.stem}: {a.bin:.0f}초 구간별 {SHORT[a.app]} 예측 전력 (gate·p_raw)")
        print(f"{'t(s)':>6s}{'관측W':>8s}" + "".join(f"{x[-8:]:>10s}" for x in tags) + "   참 ON")
        for b0 in np.arange(0, n / 60, a.bin):
            m = (t / 60 >= b0) & (t / 60 < b0 + a.bin)
            if not m.any():
                continue
            row = f"{b0:6.0f}{D[a.ckpt[0]]['p_observed'][m].mean():8.1f}"
            row += "".join(f"{(D[c]['gate'][:, j] * D[c]['p_raw'][:, j])[m].mean():10.1f}" for c in a.ckpt)
            on = [SHORT[x] for x in truth if truth[x][int(b0 * 60):int((b0 + a.bin) * 60)].mean() > 0.5]
            print(row + "   " + "+".join(on))
        P0 = D[a.ckpt[0]]["gate"] * D[a.ckpt[0]]["p_raw"]; big = P0[:, j] > 10
        print(f"\n{tags[0]}: {SHORT[a.app]} > 10W 창 {int(big.sum())}/{len(big)}")
        if big.any():
            c = Counter("+".join(SHORT[x] for x in truth if truth[x][t[i]]) for i in np.where(big)[0])
            print("  그 창의 참 ON 집합:", c.most_common(6))
            print("  그 창의 기기별 예측:", {SHORT[apps[k]]: round(float(P0[big, k].mean()), 1) for k in range(len(apps)) if P0[big, k].mean() > 1})

    if a.range:
        t0, t1 = (float(x) for x in a.range.split(",")); m = (t / 60 >= t0) & (t / 60 < t1)
        if a.harm:
            from src.synthesis.segment_pool import SegmentPool
            from src.model.net import harmonic_signatures, standby_signatures, noise_signature, harmonic_scales
            from src.model.realdata import harmonic_offset
            pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
            assert list(pool.get_appliance_types()) == list(apps)
            SIG, SB, NZ, HS = (harmonic_signatures(pool, apps), standby_signatures(pool, apps),
                               noise_signature(pool), harmonic_scales(pool, apps))
            w_h = 1.0 / np.arange(1, 16) ** 2; w_h /= w_h.max()
            try:
                OFF = harmonic_offset([a.stem] * int(m.sum()), t[m], "results/norton_coef.npz")
            except Exception as e:      # noqa: BLE001
                print("harm_offset 없음:", e); OFF = 0.0
        for ck, tag in zip(a.ckpt, tags):
            d = D[ck]; P = d["gate"] * d["p_raw"]
            resid = d["p_observed"][m] - P[m].sum(1) - d["standby"][m].sum(1) - d["p_noise"][m]
            print(f"\n== {tag}  {a.stem} {t0:.0f}-{t1:.0f}s  창 {int(m.sum())}  관측 {d['p_observed'][m].mean():.0f}W"
                  f"  잔차 {resid.mean():+.1f}W (|잔차| {np.abs(resid).mean():.1f})  대기합 {d['standby'][m].sum(1).mean():.1f}")
            print("   " + "".join(f"{SHORT[x]:>10s}" for x in apps))
            print("   게이트" + "".join(f"{d['gate'][m, k].mean():10.3f}" for k in range(len(apps))))
            print("   p_raw " + "".join(f"{d['p_raw'][m, k].mean():10.1f}" for k in range(len(apps))))
            print("   전력  " + "".join(f"{P[m, k].mean():10.1f}" for k in range(len(apps))))
            if a.harm:
                pred = (np.einsum("bk,khc->bhc", P[m], SIG) + np.einsum("bk,khc->bhc", d["idle"][m], SB) + NZ[None] + OFF)
                err = np.abs(pred - d["obs_harm"][m]) / HS[None, :, None]
                print(f"   L_harm(inv_h2) {(err * w_h[None, :, None]).mean() / w_h.mean():.3f}   차수별: "
                      + " ".join(f"h{h + 1}:{err.mean(0).mean(1)[h]:.2f}" for h in range(0, 15, 2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
