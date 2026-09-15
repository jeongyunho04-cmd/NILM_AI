# -*- coding: utf-8 -*-
"""**오탐이 켜지는 경계에서 입력의 무엇이 바뀌었나** (14.156).

사용자: *"오탐이 일어난 곳 경계에서 오탐이 일어나기 전의 입력과 오탐이 일어난 후의
입력의 차이를 비교해보는 건 어때? 시간 많이 써도 되니까 정말 모든 걸 비교해보는 거지.
여기 사진에 보이는 포트가 켜지기 직전의 오탐을 한번 해봐"*

세 단계. **집계가 아니라 한 경계를 편다.**

```
  [1] 경계 찾기  게이트가 0.5 를 **올라서 넘는** 지점을 0.1초 눈금으로 (참 라벨은 OFF)
  [2] 무엇이 바뀌었나  경계 앞/뒤 창의 **원시 45채널 + 세밀 57채널**을 전부 견준다.
                    ⚠ 차이를 **그 동네의 자연 변동폭(σ)** 으로 나눈다 — 안 그러면
                      원래 크게 흔들리는 채널이 항상 1등이다
  [3] 되돌리기   경계 **뒤** 창에서 채널(또는 무리)을 **앞 창의 실제 값**으로 갈아 끼우고
                다시 돌린다. 게이트가 0.5 밑으로 내려가면 **그 채널이 범인**이다.
                0 으로 죽이지 않는다 ([[dont-ablate-to-zero-to-find-a-cause]]).
                원시에서 갈아 끼우고 `build_inputs` 를 다시 돌리므로 **파생 채널까지 정합**이다.
```

    python -X utf8 -m src.run_diag_flipedge --stem test_5 --t0 55 --t1 95 --list
    python -X utf8 -m src.run_diag_flipedge --stem test_5 --t0 55 --t1 95 \
        --app oven --at 78.0 --ckpt results/cnn_gfp_s0.pt
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

from src.model.inputs import (EVEN2_CH, EVEN_MAG0, EVEN_ORDERS, FINE_VOLT0,  # noqa: E402
                              HALFWAVE_CH, ODD_ORDERS, PHI0, PHI_ORDERS,
                              VOLT_ORDERS, build_inputs)
from src.model.realdata import RealWindows, target_index  # noqa: E402
from src.preprocessing import load_nilm_npz  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

FS = 60
WIN = 3600
SH = {"oven": "오븐", "electiric_kettle": "포트", "hotplate": "핫플", "hair_dryer": "드라이",
      "beam_projector": "빔", "laptop_charger": "충전", "minipc": "미니PC",
      "fan": "선풍", "air_conditioner": "에어컨"}

#: 원시 45채널 이름 (`realdata._to_33ch` 의 배치)
def raw_names():
    n = ["Re(I%d)" % h for h in range(1, 16)] + ["Im(I%d)" % h for h in range(1, 16)]
    n += ["P", "Q", "V"]
    n += ["Re(V%d)" % h for h in VOLT_ORDERS] + ["Im(V%d)" % h for h in VOLT_ORDERS]
    return n


#: 원시 채널 무리 — 하나씩 말고 무리로도 갈아 끼운다
def raw_groups():
    odd = [h for h in range(3, 16, 2)]
    even = [h for h in range(2, 16, 2)]
    ri = lambda hs: [h - 1 for h in hs] + [15 + h - 1 for h in hs]
    return {
        "I1 (기본파)": ri([1]),
        "I 홀수 3~15": ri(odd),
        "I 짝수 2~14": ri(even),
        "I 전체": list(range(30)),
        "P": [30], "Q": [31], "V(실효)": [32],
        "전압 고조파": list(range(33, 45)),
    }


def fine_names():
    n = {}
    for i, h in enumerate(ODD_ORDERS):
        n[2 * i], n[2 * i + 1] = "Re(I%d)" % h, "Im(I%d)" % h
    for i, h in enumerate(EVEN_ORDERS):
        n[EVEN_MAG0 + i] = "|I%d|" % h
    n.update({23: "asinh(P/100)", 24: "Q", 25: "V",
              26: "|I3|/|I1|", 27: "|I5|/|I1|", 28: "|I2|/|I1|",
              29: "P-이동평균(0.5s)", 30: "P-이동평균(2.5s)",
              39: "역률", 40: "|I9|/|I3|",
              41: "P-P(t+3.0s)", 42: "P-P(t+5.5s)",
              HALFWAVE_CH: "반파 |I2|-|I4|", EVEN2_CH: "짝수2차 |I2|"})
    for i, h in enumerate(PHI_ORDERS):
        n[PHI0 + 2 * i], n[PHI0 + 2 * i + 1] = "cosφ%d" % h, "sinφ%d" % h
    for i, h in enumerate(VOLT_ORDERS):
        n[FINE_VOLT0 + i] = "Re(V%d)" % h
        n[FINE_VOLT0 + len(VOLT_ORDERS) + i] = "Im(V%d)" % h
    return [n.get(i, "ch%d" % i) for i in range(57)]


def cut(x, targets, off=None):
    off = target_index(WIN) if off is None else off
    c = np.stack([x[:, t - off:t - off + WIN] for t in targets])
    return c


def fwd(m, fine, wide, dev, bs=128):
    G, PW = [], []
    with torch.no_grad():
        for i in range(0, len(fine), bs):
            o = m(torch.from_numpy(np.ascontiguousarray(fine[i:i + bs])).to(dev),
                  torch.from_numpy(np.ascontiguousarray(wide[i:i + bs])).to(dev))
            G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
            PW.append(o["power"].float().cpu().numpy())
    return np.concatenate(G), np.concatenate(PW)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_gfp_s0.pt"])
    ap.add_argument("--stem", default="test_5")
    ap.add_argument("--t0", type=float, default=55.0)
    ap.add_argument("--t1", type=float, default=95.0)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--list", action="store_true", help="경계만 찍고 끝낸다")
    ap.add_argument("--app", default="", help="[2][3] 을 걸 기기")
    ap.add_argument("--at", type=float, default=np.nan, help="[2][3] 을 걸 경계 시각(초)")
    ap.add_argument("--gap", type=float, default=1.0, help="경계 앞/뒤로 몇 초를 짝지을까")
    ap.add_argument("--big", type=float, default=300.0,
                    help="이 와트 이상인 띠를 **수백 W**(① 치환)로 따로 합친다")
    ap.add_argument("--even-median", type=int, default=0,
                    help="짝수차 중앙값을 **강제**한다 (14.160) — 분포 밖 시험용. "
                         "안 주면 체크포인트가 적어 둔 값을 따라간다")
    a = ap.parse_args()
    from src.run_gate_check import sync_even_median
    sync_even_median(a.ckpt, a.even_median)   #: 창을 짓기 **전에** 전역을 맞춘다

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"][a.stem]
    raw = load_nilm_npz("processed_data/composite_eval/%s.npz" % a.stem)
    x = RealWindows._to_33ch(raw)
    P = np.asarray(raw["power_features"], np.float64)[:, 0]
    off = target_index(WIN)
    tt = np.arange(a.t0, a.t1 + 1e-9, a.step)
    tc = np.round(tt * FS).astype(np.int64)
    ok = (tc >= off) & (tc < x.shape[1] - (WIN - 1 - off))
    tt, tc = tt[ok], tc[ok]
    #: ⚠ **없는 기기도 넣는다** — 안 꽂힌 기기에 게이트가 서는 것이 곧 유령이고,
    #: `present` 로 거르면 그게 판정에서 통째로 안 보인다 (`score_arm` 의 사각지대).
    absent = [v for v in apps if v not in ev["appliances_present"]]
    pres = list(apps)
    if absent:
        print("   ⚠ 이 파일에 **없는** 기기(참 OFF 확정): "
              + " · ".join(SH.get(v, v) for v in absent))

    def lab(app, key):
        m = np.zeros(len(tt), bool)
        for b0, b1 in ev["intervals"].get(app, {}).get(key, []):
            m |= (tt >= b0) & (tt <= b1)
        return m
    ON = {v: lab(v, "on") for v in pres}
    UN = {v: lab(v, "uncertain") for v in pres}

    fine, wide = build_inputs(cut(x, tc, off))
    print("%s · %.1f~%.1f초 · %d창 (%.2f초 눈금)" % (a.stem, a.t0, a.t1, len(tt), a.step))
    for e in sorted(ev.get("events", []), key=lambda z: z["t_s"]):
        if a.t0 - 2 <= e["t_s"] <= a.t1 + 2:
            print("   이벤트 %7.1fs  %s %s" % (e["t_s"], SH.get(e.get("appliance"), "?"),
                                             e.get("kind", e.get("type", ""))))

    for ck in a.ckpt:
        m = load_model(ck, dev)[0]
        m.eval()
        G, PW = fwd(m, fine, wide, dev)
        print("\n■ %s" % ck.split("/")[-1].replace(".pt", ""))
        print("  **헛ON 경계** (참 OFF·uncertain 아님인데 게이트가 0.5 를 올라섬)")
        found = []
        for v in pres:
            k = apps.index(v)
            g = G[:, k]
            bad = (~ON[v]) & (~UN[v])
            for i in range(1, len(tt)):
                if bad[i] and g[i] > 0.5 >= g[i - 1]:
                    found.append((tt[i], v, g[i - 1], g[i], PW[i, k]))
        for t_, v, g0, g1, p_ in sorted(found):
            print("    %7.2fs  %-8s 게이트 %.3f -> %.3f   전력 %5.0fW   P관측 %6.0fW"
                  % (t_, SH.get(v, v), g0, g1, p_, P[int(round(t_ * FS))]))
        if not found:
            print("    (없음)")
        # 그 구간의 헛ON 지속 구간도 같이
        print("  **헛ON 이 서 있는 구간** (연속)")
        #: `--big` 위/아래로 **갈라서 합친다**. ① 수백 W 치환과 ② 수십 W 표류는
        #  같은 줄에 섞이면 안 된다 ([[split-the-metric-before-sizing-the-disease]]).
        tot = {}
        for v in pres:
            k = apps.index(v)
            bad = (~ON[v]) & (~UN[v]) & (G[:, k] > 0.5)
            if not bad.any():
                continue
            d = np.diff(np.r_[0, bad.astype(int), 0])
            for s_, e_ in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
                if tt[e_] - tt[s_] >= 0.3:
                    w_ = float(np.median(PW[s_:e_ + 1, k]))
                    t_ = float(tt[e_] - tt[s_])
                    g_ = tot.setdefault(v, [0.0, 0.0])
                    g_[0 if w_ >= a.big else 1] += t_
                    print("    %-8s %7.2f ~ %7.2f초 (%.1f초) 게이트 중앙 %.3f · 전력 중앙 %4.0fW"
                          % (SH.get(v, v), tt[s_], tt[e_], t_,
                             np.median(G[s_:e_ + 1, k]), w_))
        if tot:
            b = sum(x[0] for x in tot.values())
            sm = sum(x[1] for x in tot.values())
            print("  ** 합 ** 수백 W 띠(>=%.0fW) **%.1f초** · 수십 W 띠 %.1f초   |   %s"
                  % (a.big, b, sm,
                     " · ".join("%s %.1f+%.1f" % (SH.get(v, v), x[0], x[1])
                                for v, x in sorted(tot.items(), key=lambda r: -r[1][0]))))

        # ── [2][3] 한 경계를 편다 ────────────────────────────────────────
        if not a.app or not np.isfinite(a.at):
            continue
        kk = apps.index(a.app)
        i_lo = int(np.argmin(np.abs(tt - (a.at - a.gap))))
        i_hi = int(np.argmin(np.abs(tt - (a.at + a.gap))))
        w_lo, w_hi = cut(x, [tc[i_lo]], off)[0], cut(x, [tc[i_hi]], off)[0]
        f2, wd2 = build_inputs(np.stack([w_lo, w_hi]))
        g2, p2 = fwd(m, f2, wd2, dev)
        print(chr(10) + "  [2] **경계를 편다** — %s · %s  앞 %.2f초(게이트 %.3f) 대 뒤 %.2f초(게이트 %.3f)"
              % (a.stem, SH.get(a.app, a.app), tt[i_lo], g2[0, kk], tt[i_hi], g2[1, kk]))
        nb = np.flatnonzero(np.abs(tt - a.at) <= 6.0)
        fn, wn = build_inputs(cut(x, tc[nb], off))

        def summ(fa):
            from src.model.inputs import fine_target_index
            return np.stack([fa[:, :, fine_target_index()], fa.mean(-1),
                             fa.max(-1), fa.min(-1)], -1)
        Sn = summ(fn)
        sd = Sn.reshape(len(nb), -1).std(0) + 1e-9
        S2 = summ(f2).reshape(2, -1)
        dz = (S2[1] - S2[0]) / sd
        nm = fine_names()
        st = ("타깃", "평균", "최대", "최소")
        order = np.argsort(-np.abs(dz))
        print("    **세밀 57채널 x 4통계** 를 그 동네 σ 로 나눈 변화 (상위 18)")
        print("      %-24s %8s %12s %12s" % ("채널·통계", "Δ/σ", "앞", "뒤"))
        for q in order[:18]:
            c, t_ = q // 4, q % 4
            print("      %-24s %+8.2f %12.4f %12.4f"
                  % ("%s [%s]" % (nm[c], st[t_]), dz[q], S2[0, q], S2[1, q]))

        print(chr(10) + "  [3] **되돌리기** — 뒤 창에서 원시 채널(무리)을 **앞 창의 실제 값**으로")
        gr = raw_groups()
        rn = raw_names()
        items = [(k_, v) for k_, v in gr.items()]
        items += [("ch%02d %s" % (i, rn[i]), [i]) for i in range(45)]
        cand, tags = [], []
        for nm_, idx in items:
            y = w_hi.copy(); y[idx] = w_lo[idx]
            cand.append(y); tags.append(nm_)
        for nm_, idx in items:
            y = w_lo.copy(); y[idx] = w_hi[idx]
            cand.append(y); tags.append("<- " + nm_)
        f3, w3 = build_inputs(np.stack(cand))
        g3, p3 = fwd(m, f3, w3, dev)
        n_it = len(items)
        rows = sorted([(tags[i], g3[i, kk], p3[i, kk]) for i in range(n_it)],
                      key=lambda r: r[1])
        print("    (가) 뒤 창(게이트 %.3f)에서 **되돌렸을 때** — 낮을수록 그 채널이 범인 (상위 12)"
              % g2[1, kk])
        for nm_, g_, p_ in rows[:12]:
            print("      %-26s 게이트 %.3f %s  전력 %5.0fW"
                  % (nm_, g_, "**꺼짐**" if g_ <= 0.5 else "      ", p_))
        rows2 = sorted([(tags[n_it + i], g3[n_it + i, kk], p3[n_it + i, kk])
                        for i in range(n_it)], key=lambda r: -r[1])
        print("    (나) 앞 창(게이트 %.3f)에 **가져왔을 때** — 높을수록 그 채널이 범인 (상위 12)"
              % g2[0, kk])
        for nm_, g_, p_ in rows2[:12]:
            print("      %-26s 게이트 %.3f %s  전력 %5.0fW"
                  % (nm_, g_, "**켜짐**" if g_ > 0.5 else "      ", p_))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
