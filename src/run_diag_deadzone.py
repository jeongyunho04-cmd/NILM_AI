# -*- coding: utf-8 -*-
"""**죽은구역 보존 손실이 성립하나** — δ 를 재고 치환 비용과 나란히 놓는다 (14.161).

사용자: *"이거 어떻게 해결할 수 있어?"* -> 제안: `relu(|잔차| − δ)`.

그 제안은 **두 분포가 안 겹칠 때만** 성립한다.

```
  잔차 바닥   정상 구간의 |Σp̂ + Σŝ + p_noise − P관측|   -> δ 를 여기 p99 위에 둔다
  치환 비용   오븐<->포트를 바꾸면 합이 얼마나 튀나       -> δ 아래로 내려오면 **못 잡는다**
              = V²·|1/R_a − 1/R_b|   (204V 137W · 226V 170W)
```

둘이 겹치면 이 처방은 **여기서 죽는다.** 굽기 전에 그것부터 잰다.

⚠ `L_cons` 를 1단계에 올리는 길은 12.9 가 막았다 — *"합성에서 P관측 ≡ Σ라벨 은 항등식이고
경사가 35~3000배 세서 배분이 미결정인 채 합만 맞추는 해로 끈다"*. 그 논리는 **배분이
자유롭다**는 전제 위에 서 있는데, 12.9 **이후에** 들어온 `state_power_init`(13.84.68)과
`--p-state-cap`(14.121)이 그 전제를 깼다 — 기기별 전력이 거의 고정된 몇 값이라
**합을 맞추는 것이 곧 배분을 고르는 것**이다. 그래서 다시 잰다.

```
  [1] 잔차 바닥    부호있는/절대 잔차의 p50·p90·p99·p99.9·최대
  [2] 치환 비용    창 전압에서 저항쌍마다 V²|1/R_a − 1/R_b|
  [3] 분리         p99(잔차) 대 p5(치환비용). **이 비가 1 보다 한참 커야 한다**
  [4] 반사실       맞춘 창에서 저항 두 기기의 와트를 **실제로 바꿔** 잔차가 얼마나 뛰나
```

    python -X utf8 -m src.run_diag_deadzone --ckpt results/cnn_pcap_s0.pt          # 실측
    python -X utf8 -m src.run_diag_deadzone --holdout processed_data/holdout60_v32h \
        --ckpt results/cnn_pcap_s0.pt results/cnn_pcap_s1.pt results/cnn_pcap_s2.pt
"""
import argparse
import itertools
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.postproc import RESISTIVE_OHM  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

SH = {"oven": "오븐", "electiric_kettle": "포트", "hotplate": "핫플", "hair_dryer": "드라이",
      "beam_projector": "빔", "laptop_charger": "충전", "minipc": "미니PC",
      "fan": "선풍", "air_conditioner": "에어컨"}
Q = (50, 90, 99, 99.9)


def load_holdout(path, apps, n_max):
    import json
    from pathlib import Path
    from src.model.inputs import V_CENTER, V_SPAN, build_inputs
    d = Path(path)
    ha = json.loads((d / "meta.json").read_text(encoding="utf-8"))["appliances"]
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(n_max, len(X))
    fine, wide = build_inputs(np.asarray(X[:n]))
    col = [ha.index(x) for x in apps]

    def L(k):
        return np.asarray(np.load(d / (k + ".npy"), mmap_mode="r"))[:n]

    return dict(fine=fine, wide=wide,
                y_on=L("y_on")[:, col].astype(np.int8),
                y_power=L("y_power")[:, col].astype(np.float64),
                p_obs=L("p_observed").astype(np.float64),
                p_noise=L("p_noise").astype(np.float64),
                v=(fine[:, 25].mean(-1) * V_SPAN + V_CENTER).astype(np.float64),
                base=None)


def load_real(apps, grid_s):
    from src.run_train_seq import real_windows
    c = real_windows(apps, grid_s, "cpu")
    cat = lambda k: np.concatenate([d[k] for d in c.values()])
    base = np.concatenate([np.full(len(d["t"]), float(d["p_base"])) for d in c.values()])
    return dict(fine=cat("fine"), wide=cat("wide"),
                y_on=np.concatenate([d["y"].astype(np.int8) for d in c.values()]),
                y_power=None,
                p_obs=cat("p_obs").astype(np.float64),
                p_noise=cat("p_noise").astype(np.float64),
                v=np.concatenate([np.asarray(d["v_obs"], np.float64) for d in c.values()]),
                base=base)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--grid-s", type=float, default=2.0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    D = (load_holdout(a.holdout, apps, a.n) if a.holdout
         else load_real(apps, a.grid_s))
    src = ("합성 홀드아웃 %s" % a.holdout) if a.holdout else "실측 5파일"
    n = len(D["p_obs"])
    print("%s · 창 %d · 장치 %s" % (src, n, dev))

    # ── [2] 치환 비용 — 창 전압에서 저항쌍마다 ─────────────────────────
    res = [x for x in apps if x in RESISTIVE_OHM]
    v2 = D["v"] ** 2
    print("\n[2] **치환 비용** — V²·|1/R_a − 1/R_b| (W). 그 쌍을 바꾸면 합이 이만큼 튄다")
    print("  %-24s %8s %8s %8s" % ("쌍", "p5", "중앙", "p95"))
    pair_cost = {}
    for x, y in itertools.combinations(res, 2):
        c = v2 * abs(1.0 / RESISTIVE_OHM[x] - 1.0 / RESISTIVE_OHM[y])
        pair_cost[(x, y)] = c
        print("  %-24s %8.0f %8.0f %8.0f"
              % ("%s <-> %s" % (SH.get(x, x), SH.get(y, y)),
                 np.percentile(c, 5), np.median(c), np.percentile(c, 95)))
    key = ("electiric_kettle", "oven")
    key = key if key in pair_cost else ("oven", "electiric_kettle")
    swap_p5 = float(np.percentile(pair_cost[key], 5))

    for ck in a.ckpt:
        m = load_model(ck, dev)[0]
        m.eval()
        PW, SB = [], []
        with torch.no_grad():
            for i in range(0, n, 256):
                o = m(torch.from_numpy(np.ascontiguousarray(D["fine"][i:i + 256])).to(dev),
                      torch.from_numpy(np.ascontiguousarray(D["wide"][i:i + 256])).to(dev))
                PW.append(o["power"].float().cpu().numpy().astype(np.float64))
                SB.append(o["standby"].float().cpu().numpy().astype(np.float64))
        PW, SB = np.concatenate(PW), np.concatenate(SB)
        if a.holdout:
            r = PW.sum(1) + SB.sum(1) + D["p_noise"] - D["p_obs"]
        else:
            r = PW.sum(1) + D["base"] - D["p_obs"]     # `run_diag_rollback` 과 같은 규약
        print("\n" + "=" * 92)
        print("■ %s" % ck.split("/")[-1].replace(".pt", ""))
        print("=" * 92)
        print("[1] **잔차 바닥**  부호있는 평균 %+.2fW · 절대 평균 %.2fW"
              % (r.mean(), np.abs(r).mean()))
        print("    |잔차| " + " · ".join("p%g %.1fW" % (q, np.percentile(np.abs(r), q))
                                        for q in Q) + " · 최대 %.1fW" % np.abs(r).max())
        p99 = float(np.percentile(np.abs(r), 99))
        print("\n[3] **분리** — δ 는 이 사이에 있어야 한다")
        print("    p99(|잔차|) = **%.1fW**   대   p5(포트<->오븐 치환비용) = **%.1fW**"
              % (p99, swap_p5))
        print("    비 = **%.2f 배**  %s" % (swap_p5 / max(p99, 1e-9),
                                          "" if swap_p5 > 2 * p99 else "  <- ⚠ 여유가 없다"))
        for d_ in (40.0, 60.0, 80.0, 100.0):
            print("      δ=%5.0fW ->  정상 창 중 깨어나는 비율 **%5.2f%%** · "
                  "포트<->오븐 치환을 잡는 비율 **%5.1f%%**"
                  % (d_, 100 * (np.abs(r) > d_).mean(),
                     100 * (pair_cost[key] > d_).mean()))

        # ── [4] 반사실 — 저항 두 기기의 와트를 실제로 바꾼다 ─────────────
        print("\n[4] **반사실** — 맞춘 창에서 두 기기의 예측 와트를 서로 바꾸면 잔차가?")
        print("    %-24s %6s %10s %10s %10s"
              % ("쌍", "창", "지금|r|", "바꾼뒤|r|", "증가"))
        for (x, y) in pair_cost:
            kx, ky = apps.index(x), apps.index(y)
            # 한쪽만 큰 와트를 갖고 다른 쪽은 거의 0 인 창 (신원이 걸린 창)
            big = np.maximum(PW[:, kx], PW[:, ky])
            sml = np.minimum(PW[:, kx], PW[:, ky])
            sel = (big > 200) & (sml < 50) & (np.abs(r) < p99)
            if sel.sum() < 20:
                continue
            # 바꾸기: 그 와트를 상대 기기의 컨덕턴스로 다시 낸다
            hi_is_x = PW[sel, kx] >= PW[sel, ky]
            Rfrom = np.where(hi_is_x, RESISTIVE_OHM[x], RESISTIVE_OHM[y])
            Rto = np.where(hi_is_x, RESISTIVE_OHM[y], RESISTIVE_OHM[x])
            dW = v2[sel] * (1.0 / Rto - 1.0 / Rfrom)
            r2 = r[sel] + dW
            print("    %-24s %6d %10.1f %10.1f %+10.1f"
                  % ("%s <-> %s" % (SH.get(x, x), SH.get(y, y)), int(sel.sum()),
                     np.abs(r[sel]).mean(), np.abs(r2).mean(),
                     np.abs(r2).mean() - np.abs(r[sel]).mean()))
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
