# -*- coding: utf-8 -*-
"""**과도 구간에서 `p_raw` 가 폭증하는가** — Chen et al. (TSG 2020) 의 병이 우리에게 있나 (14.155).

사용자: *"전이 구간에서 g 가 0.2~0.6 이 되고 손실이 p̂ 를 2000~4000W 로 폭증시켜
스파이크가 난다 — 이게 우리 증상이랑 정말 비슷한데 왜 해결이 안 되는 거지?"*

그 병의 서명은 **두 개가 동시에** 있어야 한다:

```
  (가) 전이 창에서 게이트가 **중간값**(0.2~0.6)에 머문다
  (나) 같은 창에서 `p_raw` 가 **참값보다 크다** (보상 폭증, log(p_raw/y) > 0)
```

전역으로는 (나)가 없다 (`log(p_raw/참) = −0.0005`). 하지만 **전이 구간만** 보면
있을 수 있고, 전역 평균이 그것을 지웠을 수 있다. 그래서 **창 안 전력 변화폭**으로
갈라서 잰다 — 라벨이 아니라 입력 채널에서 뽑으므로 합성·실측에 똑같이 걸린다.

```
  ΔP(창) = max(P) − min(P)   over 세밀 창(타깃 −4.0 ~ +6.0초),  P = sinh(ch23)·100
```

    python -X utf8 -m src.run_diag_transient --ckpt results/cnn_pcap_s0.pt          # 실측(게이트만)
    python -X utf8 -m src.run_diag_transient --holdout processed_data/holdout60_v32h \
        --ckpt results/cnn_pcap_s0.pt results/cnn_gfp_s0.pt                          # 합성(둘 다)
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.model.net import P_CH_FINE, POWER_SCALE  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

BINS = [0, 50, 200, 500, 1000, 2000, 1e9]
LBL = ["<50W", "50~200", "200~500", "500~1k", "1k~2k", ">2kW"]
ONF = 5.0


def load_real(apps, grid_s=2.0):
    from src.run_train_seq import real_windows
    cache = real_windows(apps, grid_s, "cpu")
    fine = np.concatenate([d["fine"] for d in cache.values()])
    wide = np.concatenate([d["wide"] for d in cache.values()])
    Y = np.concatenate([d["y"].astype(np.int8) for d in cache.values()])
    TRUE = np.concatenate([d["p_obs"].astype(np.float64) - float(d["p_base"])
                           for d in cache.values()])
    n = Y.sum(1)
    YP = np.where(Y > 0, np.maximum(TRUE, 0.0)[:, None], 0.0)
    valid = ((n == 1) & (TRUE > ONF)) | (n == 0)
    return fine, wide, Y, YP, valid


def load_holdout(path, apps, n_max):
    import json
    from pathlib import Path
    from src.model.inputs import build_inputs
    d = Path(path)
    ha = json.loads((d / "meta.json").read_text(encoding="utf-8"))["appliances"]
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(n_max, len(X))
    fine, wide = build_inputs(np.asarray(X[:n]))
    col = [ha.index(x) for x in apps]
    Y = np.asarray(np.load(d / "y_on.npy", mmap_mode="r"))[:n][:, col].astype(np.int8)
    YP = np.asarray(np.load(d / "y_power.npy", mmap_mode="r"))[:n][:, col].astype(np.float64)
    return fine, wide, Y, YP, np.ones(n, bool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_pcap_s0.pt"])
    ap.add_argument("--holdout", default="")
    ap.add_argument("--n", type=int, default=8000)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    if a.holdout:
        fine, wide, Y, YP, valid = load_holdout(a.holdout, apps, a.n)
        src = "합성 홀드아웃 %s" % a.holdout
    else:
        fine, wide, Y, YP, valid = load_real(apps)
        src = "실측 5파일 (참전력은 **혼자켜짐 창만** 유효)"

    # 창 안 전력 변화폭 — 입력 채널에서 (라벨을 안 쓴다)
    pw_t = np.sinh(np.asarray(fine[:, P_CH_FINE], np.float64)) * POWER_SCALE
    dP = pw_t.max(-1) - pw_t.min(-1)
    b = np.digitize(dP, BINS) - 1
    print("%s · 창 %d" % (src, len(Y)))
    print("창 안 ΔP 분포: " + " · ".join(
        "%s %d" % (LBL[i], int((b == i).sum())) for i in range(len(LBL))))

    for ck in a.ckpt:
        m = load_model(ck, dev)[0]
        m.eval()
        gf = bool(getattr(m, "gate_free_power", False))
        G, PR, M0, PWo = [], [], [], []
        with torch.no_grad():
            for i in range(0, len(fine), 256):
                o = m(torch.from_numpy(np.ascontiguousarray(fine[i:i + 256])).to(dev),
                      torch.from_numpy(np.ascontiguousarray(wide[i:i + 256])).to(dev))
                G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
                PR.append(o["power_raw"].float().cpu().numpy())
                M0.append(o["power_mix"].float()[..., 0].cpu().numpy())
                PWo.append(o["power"].float().cpu().numpy())
        G, PR, M0, PWo = (np.concatenate(x) for x in (G, PR, M0, PWo))
        door = (1.0 - M0) if gf else G
        p_on = PWo / np.clip(door, 1e-9, None)
        print("\n" + "=" * 94)
        print("■ %s%s" % (ck.split("/")[-1].replace(".pt", ""),
                          "  (완전분해)" if gf else ""))
        print("=" * 94)
        on = (Y > 0) & (YP > ONF) & valid[:, None]
        print("  %-9s %7s | %8s %8s %8s | %10s %11s %10s"
              % ("창 안 ΔP", "켜짐칸", "문 평균", "문 중앙",
                 "0.2~0.6", "log(p/y)", "log(out/y)", "p/y>1.5"))
        for i in range(len(LBL)):
            sel = on & (b == i)[:, None]
            if sel.sum() < 20:
                continue
            d_ = np.clip(door[sel], 1e-9, 1.0)
            r = np.maximum(p_on[sel], 1e-6) / np.maximum(YP[sel], 1e-6)
            ro = np.maximum(PWo[sel], 1e-6) / np.maximum(YP[sel], 1e-6)
            print("  %-9s %7d | %8.3f %8.3f %7.1f%% | %10.4f %11.4f %9.1f%%"
                  % (LBL[i], int(sel.sum()), d_.mean(), np.median(d_),
                     100 * ((d_ > 0.2) & (d_ < 0.6)).mean(),
                     float(np.mean(np.log(r))), float(np.mean(np.log(ro))),
                     100 * (r > 1.5).mean()))
        print("  ⇒ Chen et al. 의 병이면 ΔP 가 클수록 **문이 중간값**이고 "
              "**log(p/y) > 0** 이어야 한다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
