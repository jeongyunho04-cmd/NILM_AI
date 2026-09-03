"""반파 채널 판정 — 드라이기 헤지가 풀렸는가 (12.155.6)

12.155.5 가 확정한 인과를 다섯 기준으로 잰다. **판정 기준을 결과보다 먼저 적는다**
(규칙 22):

```
① 드라이기 ON 구간의 잔차       236~308W  ->  줄어야 한다
② 저항 4종 게이트 σ합          1.33~1.55 ->  2 에 가까워져야 한다
③ 드라이기 단독 971W 구간 귀속   271W      ->  ~950W 로 가야 한다
④ 합성 홀드아웃 MAE·F1                    ->  나빠지면 안 된다
⑤ 유령·프로젝터W                          ->  12.155.4 판과 같은 수준
```

④가 특히 중요하다 — 12.74 가 짝수차를 되살렸다가 충전기 F1 이 0.937 -> 0.868 로
무너진 전례가 있다. **모양 관문이 그것을 막는지**가 이 실험의 두 번째 질문이다.

    python -X utf8 -m src.run_dryer_probe --ckpt results/adapt_hw58.pt results/adapt_hw59.pt
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

from src.model.inputs import LEGACY_FINE_CHANNELS
from src.model.net import NILMNet, appliance_state_counts
from src.model.realdata import dense_targets
from src.run_adapt import real_targets

RES = ["electiric_kettle", "hair_dryer", "hotplate", "oven"]
FILES = ["test_14", "test_15", "test_16", "test_17", "test_18", "test_5"]


def load(path: str, dev: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    apps = ck["appliances"]
    m = NILMNet(apps, appliance_state_counts(apps), width=ck.get("width", 1.0),
                wide_summary=ck.get("wide_summary", False),
                periodicity=ck.get("periodicity", False),
                fine_dropout=ck.get("fine_dropout", 0.0),
                prior_kappa=ck.get("prior_kappa", 0.0),
                prior_beta=ck.get("prior_beta", 0.5),
                fine_channels=ck.get("fine_channels", LEGACY_FINE_CHANNELS)).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, apps, int(ck.get("fine_channels", LEGACY_FINE_CHANNELS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--events", default="processed_data/real_events_refined.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = json.load(open(a.events, encoding="utf-8"))["files"]

    for path in a.ckpt:
        m, apps, fc = load(path, dev)
        ri = [apps.index(x) for x in RES if x in apps]
        jd = apps.index("hair_dryer")
        print("\n" + "=" * 96)
        print(f"■ {Path(path).stem}   (fine_channels={fc})")
        print("=" * 96)
        print(f"{'파일':<9}{'구간':<20}{'창':>6}{'잔차절대':>9}{'σ합(저항4)':>11}"
              f"{'드라이기 귀속W':>14}{'P_obs':>8}")
        for st in FILES:
            if st not in ev:
                continue
            rw = dense_targets(st, stride=30)
            P, G, Po = [], [], []
            for i in range(0, len(rw), 512):
                f, w, tg = real_targets(rw.batch(np.arange(i, min(i + 512, len(rw)))), dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                     enabled=dev == "cuda"):
                    o = m(f, w)
                P.append(o["power"].float().cpu().numpy())
                G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
                s = (o["power"].sum(1) + o["standby"].sum(1)).float().cpu().numpy()
                Po.append((s + tg["p_noise"].cpu().numpy(),
                           tg["p_observed"].cpu().numpy()))
            P = np.concatenate(P); G = np.concatenate(G)
            pred = np.concatenate([x[0] for x in Po])
            obs = np.concatenate([x[1] for x in Po])
            R = pred - obs
            t = rw.target_cycle / 60.0
            iv = ev[st]["intervals"]

            def on(app: str) -> np.ndarray:
                z = np.zeros(len(t), bool)
                for x, y in iv.get(app, {}).get("on", []):
                    z |= (t >= x) & (t < y)
                return z

            hd = on("hair_dryer")
            for nm, c in (("드라이기 ON", hd),
                          ("드라이기 OFF", ~hd & (obs > 300))):
                if c.sum() < 30:
                    continue
                print(f"{st:<9}{nm:<20}{c.sum():>6}{np.abs(R[c]).mean():>9.1f}"
                      f"{G[c][:, ri].sum(1).mean():>11.2f}{P[c][:, jd].mean():>14.1f}"
                      f"{obs[c].mean():>8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
