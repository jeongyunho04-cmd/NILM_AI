# -*- coding: utf-8 -*-
"""`L_harm` 이 쓰는 전압 `vrel` 이 **어느 전압**인가 — 대부하 강하를 놓치고 있나 (14.52).

사용자 질문 (2026-09-14): *"과보정은 왜 발생한거야? 혹시 대부하 켜질때 전압강하는
감안 안한거는 아니지?"*

세 전압이 따로 정의돼 있다:
```
① V_fit   `harmonic_signature_vref` — 그 기기 격리 녹화의 **통전 사이클** V 중앙값
          (자기 강하 포함). `sig = median(I_h/P)` 를 잰 바로 그 사이클들이다.
② v_ref   생성기가 재생에 쓰는 기준전압 (`SegmentPool` 활성화의 v_ref)
③ vrel    `run_train_cnn.py:312` — `fine[:, 25].mean(-1)`
          = 세밀 창 **600사이클(10초) 전체 평균**
```
그런데 `L_harm` 이 맞추는 것은 **타깃 사이클 하나**의 `obs_harm` 이다. 그 사이클의
전압은 그때 켜진 부하가 다 같이 만든 강하를 쓰고 있는데, ③ 은 그 앞뒤 10초를 평균한다.
핫플은 2.00초 주기로 0.47초만 통전하므로(14.48) 10초 평균은 **통전 안 하는 구간의
높은 전압**을 섞는다.

`sig ∝ V^e` 이고 저항은 `e = −1` 이라 **V 를 높게 읽으면 sig 가 작아지고, `L_harm` 은
전력을 그만큼 더 요구한다** — 즉 과대예측 쪽으로 민다. 그리고 그 오차는 **고전력 창에서
가장 크다**. 14.51 의 앵커가 과소를 없애자 이것이 드러났을 수 있다.

    python -X utf8 src/run_diag_vrel.py
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.model.inputs import V_CENTER, V_SPAN  # noqa: E402

FS = 60.0
VCH = 25                       #: 세밀 채널 25 = (V_rms − 222)/10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-s", type=float, default=2.0)
    a = ap.parse_args()

    from src.run_train_seq import real_windows
    import torch
    ck = torch.load("results/cnn_vanch_off_s0.pt", map_location="cpu", weights_only=False)
    apps = list(ck["appliances"])
    cache = real_windows(apps, a.grid_s, "cpu")
    ti = None

    print("세밀 창 전압 — `vrel` 이 쓰는 10초 평균 대 **타깃 사이클** 값\n")
    print("  %-9s%6s%10s%10s%10s%10s%9s" %
          ("자른 곳", "창", "10초평균", "타깃", "차(V)", "차(%)", "sig 오차"))
    CUTS = (("전체", None, 0.0), ("P>1000", None, 1000.0), ("P>1500", None, 1500.0),
            ("핫플통전·P>1500", "hotplate", 1500.0), ("오븐통전·P>1000", "oven", 1000.0))
    for label, app, thr in CUTS:
        ai = apps.index(app) if app else None
        mn, tg = [], []
        for stem, d in cache.items():
            f = d["fine"]
            if ti is None:
                from src.model.inputs import target_index
                ti = target_index(f.shape[-1] * 6) if False else None
            # 세밀 창의 타깃 위치 — 체크포인트가 적어 둔 값을 쓴다
            tpos = int(ck.get("target_pos", 239))
            v = f[:, VCH] * V_SPAN + V_CENTER               # (n, 600) 볼트
            m = d["p_obs"] > thr
            if ai is not None:
                m = m & (d["y"][:, ai] == 1)
            if int(m.sum()) < 5:
                continue
            mn.append(v[m].mean(-1)); tg.append(v[m][:, tpos])
        if not mn:
            continue
        mn = np.concatenate(mn); tg = np.concatenate(tg)
        d_v = float(np.median(mn - tg))
        d_p = 100.0 * d_v / float(np.median(tg))
        # sig ∝ V^(-1) 이므로 V 를 d_p% 높게 읽으면 sig 가 d_p% 작아지고
        # `L_harm` 은 전력을 그만큼 **더** 요구한다 (저항, e=−1).
        print("  %-9s%6d%10.2f%10.2f%+10.2f%+10.2f%%%+8.2f%%" %
              (label, len(mn), float(np.median(mn)), float(np.median(tg)), d_v, d_p, d_p))
    print("\n  ⚠ `sig 오차` 는 **저항(e=−1)** 기준이다 — V 를 높게 읽은 만큼 `L_harm` 이")
    print("     전력을 더 요구한다 (과대예측 쪽). SMPS 는 e=0 이라 안 걸린다.")
    print("  ⚠ 14.51 의 앵커 배수와 견줘라: 오븐 ×0.9479 = **−5.2%**, 핫플 ×0.9659 = −3.4%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
