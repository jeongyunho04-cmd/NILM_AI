# -*- coding: utf-8 -*-
"""**전력 헤드 크기**를 참 라벨로 게이팅해 잰다 — 13.84.70 의 슬롯 고침 채점 (13.84.39).

13.84.68 이 잰 결함: 전력은 `p_raw = Σ_s mix[s]·p_states[s]` 인 **곱**인데 슬롯을 붙잡는
항이 없다. 사슬이 `L_state` 로 분류를 날카롭게 만들면(mix[1]=0.997) 죽은 슬롯으로 전력이
흘러 무너진다 — 드라이기 466 -> 3W · 오븐 상태1 15 -> 1W · 포트 상태1 1456 -> 919W.

여기서 재는 것은 **게이트를 뺀 크기**다. 참 라벨로 켜고 끄므로 게이트 오차가 안 섞인다:
    비 = median(Σ_k p_k·y_k) / median(P_관측)   (관측 상위 60% 단계에서만)
1.00 이 맞는 것이다. 사슬 이전 창별 모델(v25~v37)이 0.99~1.02 이고,
열 캐시로 배운 사슬 판이 0.91(v37 이어)~0.64(처음부터) 였다.

⚠ 게이트 없이 그냥 더하면(문지름) 참 OFF 구간의 헛예측이 과소예측을 상쇄해 **가린다**
  ([[auc-hides-gate-magnitude-collapse]] 의 반대 방향). 반드시 참 라벨로 갈라서 재라.

    python -X utf8 src/run_score_slotpow.py results/hpc4/seq_slot_both_ep12.pt ...
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.run_gate_check import load_model
from src.run_train_seq import FS, FILES, real_windows

BASE = ("results/cnn_v37.pt",)


def head_ratio(model, real, dev, apps):
    """파일별 (참 라벨 게이팅 합) / 관측, 관측 상위 60% 단계에서."""
    from src.preprocessing import load_nilm_npz
    out = []
    for stem in FILES:
        d = real[stem]
        y = d["y"].astype(bool)
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        pobs = np.asarray(r["power_features"])[:, 0]
        po = pobs[np.clip((d["t"] * FS).astype(int), 0, len(pobs) - 1)]
        with torch.no_grad():
            P = []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                P.append(o["power"].float())
            p = torch.cat(P).cpu().numpy()
        s = (p * y).sum(1)
        m = po > np.percentile(po, 40)
        out.append(float(np.median(s[m]) / max(np.median(po[m]), 1e-6)))
    return out


def main():
    cks = sys.argv[1:]
    if not cks:
        cks = sorted(str(p) for p in Path("results/hpc4").glob("*_ep12.pt"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck0 = torch.load(cks[0], map_location="cpu", weights_only=False)
    apps = ck0["appliances"]
    real = real_windows(apps, ck0["meta"]["grid_s"], dev)
    print("**전력 헤드 크기** — 참 라벨로 게이팅, 관측 상위 60% 단계. 1.00 이 맞는 것")
    print("  %-28s %s"
          % ("체크포인트", "".join("%9s" % f for f in FILES) + "     중앙"))
    for b in BASE:
        if Path(b).exists():
            row = head_ratio(load_model(b, dev)[0], real, dev, apps)
            print("  %-28s %s  %7.2f  (사슬 이전 기준선)"
                  % (Path(b).stem, "".join("%9.2f" % x for x in row), np.median(row)))
    for c in cks:
        ck = torch.load(c, map_location=dev, weights_only=False)
        m = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
        m.load_state_dict(ck["model"]); m.eval()
        row = head_ratio(m, real, dev, apps)
        print("  %-28s %s  %7.2f" % (Path(c).stem, "".join("%9.2f" % x for x in row),
                                     np.median(row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
