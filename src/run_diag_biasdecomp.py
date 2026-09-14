# -*- coding: utf-8 -*-
"""14.49 — **합성 홀드아웃의 기기별 편향을 곱으로 분해한다.** 참값이 정확하다.

    power = σ(on_logit) · Σ_s mix_s · p_states_s

14.38 이 실측 통전 창에서 σ=1.000 · mix_max=1.000 을 쟀지만, **합성 ON 창에서는 안 쟀다.**
합성에는 참 `y_on`·`y_state`·`y_power` 가 다 있으므로 자가 정확하다.
표적: 오븐 편향(on) **−21.3W** (정격 대비 −1.76%) · 핫플 −6.8W.

읽는 법: `power/참` = `σ(on)` × `p_state[참상태]/참` × `혼합 희석` (근사).
  σ 가 낮으면 **게이트**, 상태값이 낮으면 **크기**, 희석이면 **혼합**이다.
"""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.holdout import load_holdout      # noqa: E402
from src.model.inputs import build_inputs            # noqa: E402
from src.run_baseline import S_I                     # noqa: E402
from src.run_gate_check import load_model            # noqa: E402

KO = {"electiric_kettle": "포트", "oven": "오븐", "hotplate": "핫플", "hair_dryer": "드라이",
      "beam_projector": "프로젝", "laptop_charger": "충전기", "minipc": "미니PC",
      "fan": "선풍", "air_conditioner": "에어컨"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--holdout", default="processed_data/holdout60_v32")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--min-w", type=float, default=50.0,
                    help="참 전력이 이보다 큰 창만 (비율을 재려면 분모가 커야 한다)")
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args()
    hs = load_holdout(a.holdout)
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, a.dev); model.eval()
        ON, PR, MX, PS, PW = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(hs), a.batch):
                f, w = build_inputs(np.asarray(hs.X[i:i + a.batch]))
                o = model(torch.from_numpy(f).to(a.dev), torch.from_numpy(w).to(a.dev))
                ON.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
                PR.append(o["power_raw"].float().cpu().numpy())
                MX.append(o["power_mix"].float().cpu().numpy())
                PS.append(o["power_states"].float().cpu().numpy())
                PW.append(o["power"].float().cpu().numpy())
        ON = np.concatenate(ON); PR = np.concatenate(PR); MX = np.concatenate(MX)
        PS = np.concatenate(PS); PW = np.concatenate(PW)
        yp = np.asarray(hs.y_power, np.float64); yo = np.asarray(hs.y_on)
        yst = np.asarray(hs.y_state).astype(int)
        print(f"\n{'=' * 92}\n{ck}   (참 전력 > {a.min_w:.0f}W 인 ON 창)")
        print(f"  {'기기':>7}{'창':>7}{'참 중앙':>9}{'power/참':>10}{'σ(on)':>8}"
              f"{'p_raw/참':>10}{'p_st[참]/참':>12}{'혼합 희석':>10}{'mix[참]':>9}")
        for j, app in enumerate(apps):
            m = (yo[:, j] > 0) & (yp[:, j] > a.min_w)
            if m.sum() < 30:
                continue
            t = yp[m, j]
            sid = np.clip(yst[m, j], 0, PS.shape[-1] - 1)
            ps = PS[m, j][np.arange(m.sum()), sid]
            mx = MX[m, j][np.arange(m.sum()), sid]
            E = PR[m, j]                                   # Σ mix·p_states
            print(f"  {KO.get(app, app):>7}{int(m.sum()):>7}{np.median(t):>8.0f}W"
                  f"{np.median(PW[m, j] / t):>10.4f}{np.median(ON[m, j]):>8.4f}"
                  f"{np.median(E / t):>10.4f}{np.median(ps / t):>12.4f}"
                  f"{np.median(E / np.maximum(ps, 1e-6)):>10.4f}{np.median(mx):>9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
