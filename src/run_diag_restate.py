# -*- coding: utf-8 -*-
"""판정줄 `re_on_median` 을 **상태로 가른다** (14.397, 사용자 질문).

*"충전기 s1 이 판정줄까지 오는지 먼저 따져봐"*

§54 가 (나)의 이득을 **충전기 s1 −44.4%** 하나로 좁혔다. 그 칸이 판정줄 ②
(`laptop_charger.re_on_median`, 지금 **0.2493**) 에 실제로 닿는지 재야 한다.
라벨만으로 이미 안 것:
```
  충전기 통전 창 4,276개 중 **s1 이 2,489개 (58.2%)** · 전력 중앙 **21.2W**
  s2 는 1,787개 (41.8%) · 63.0W
  ⇒ **중앙값이 s1 안에 앉아 있다.** 그런데 그것만으로는 부족하다 —
    그 창의 오차가 실제로 큰지 봐야 한다 ([[split-the-metric-before-sizing-the-disease]])
```
⚠ 자를 **판정이 쓰는 것과 같은 식**으로 낸다 — `score_appliances` 의 `re_on_median`
  이 창마다 `|pred−true|/true` 를 내고 중앙값을 잡는 그 식이다.
⚠ 모델이 낸 것을 그대로 쓴다 ([[dont-rebuild-model-output-from-parts]]) — 학습기의
  `prepare_holdout_inputs`·`evaluate` 를 **그대로 부른다**
  ([[verify-the-input-path-not-just-the-model]]).

    python -X utf8 -m src.run_diag_restate --ckpt results/cnn_comb_s0.pt \
        --holdout processed_data/holdout60_v49
"""
import argparse

import numpy as np
import torch

from src import env_guard  # noqa: F401

from src.evaluation.holdout import load_holdout  # noqa: E402
from src.model.build import build_model  # noqa: E402
from src.run_train_cnn import evaluate, prepare_holdout_inputs  # noqa: E402

WHO = ("laptop_charger", "minipc", "beam_projector")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_comb_s0.pt")
    ap.add_argument("--holdout", default="processed_data/holdout60_v49")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps = build_model(ck, dev, ckpt_path=a.ckpt)
    hs = load_holdout(a.holdout)
    assert list(hs.appliances) == list(apps), "기기 차례가 다르다"
    pred, _ = evaluate(model, prepare_holdout_inputs(hs), dev)

    print("\n판정줄 `re_on_median` 을 상태로 가른다 — %s\n" % a.ckpt)
    for app in WHO:
        j = list(apps).index(app)
        on = np.asarray(hs.y_on)[:, j] > 0.5
        t = np.asarray(hs.y_power)[on, j]
        p = pred[on, j]
        st = np.asarray(hs.y_state)[on, j]
        re = np.abs(p - t) / np.maximum(t, 1e-9)
        print("  %s — 통전 %d창 · **전체 re_on_median %.4f**" % (app, on.sum(), np.median(re)))
        print("      상태   창수  (몫)   참W중앙  예측W중앙  **re중앙**   re평균   편향W")
        for s in sorted(set(st.tolist())):
            k = st == s
            if k.sum() < 10:
                continue
            print("      s%-3d  %5d (%4.1f%%)  %7.1f  %8.1f   **%.4f**   %.4f  %+7.2f"
                  % (s, k.sum(), 100 * k.mean(), np.median(t[k]), np.median(p[k]),
                     np.median(re[k]), np.mean(re[k]), float(np.mean(p[k] - t[k]))))
        #: ⚠ 처음엔 "그 상태를 **완벽히** 맞히면" 을 찍었는데 **쓸모없는 자였다** —
        #:   과반을 0 으로 놓으면 중앙값이 기계적으로 0 이 된다 (충전기 s1 58.2%).
        #:   대신 **그 상태의 오차를 x배로 줄이면** 판정줄이 어디로 가는지 쓸어 본다.
        #: ★ 14.397b — **압축 기울기**. `pred = α·true + β` 를 창 단위로 맞춘다.
        #  α<1 이면 모델이 그 기기의 전력 폭을 **좁히고** 있다 = 수준을 못 읽는다.
        #  ⚠ 상태를 섞어 재면 "두 상태의 전력 차이" 를 재는 것이라, **상태 안**에서도 낸다.
        def slope(tt, pp):
            if len(tt) < 30 or np.std(tt) < 1e-6:
                return float("nan")
            return float(np.polyfit(tt, pp, 1)[0])
        line = ["전체 α=%.3f" % slope(t, p)]
        for s in sorted(set(st.tolist())):
            k = st == s
            if k.sum() >= 30:
                line.append("s%d α=%.3f (참 폭 x%.2f)"
                            % (s, slope(t[k], p[k]),
                               np.percentile(t[k], 90) / max(np.percentile(t[k], 10), 1e-9)))
        print("      ★ **압축 기울기** " + " · ".join(line))
        for s in sorted(set(st.tolist())):
            k = st == s
            if k.sum() < 10 or k.mean() < 0.15:
                continue
            row = []
            for f in (0.9, 0.8, 0.7, 0.5):
                r2 = re.copy()
                r2[k] = r2[k] * f
                row.append("x%.1f -> %.4f" % (f, np.median(r2)))
            print("      ⇒ s%d 오차를  %s   (지금 %.4f)"
                  % (s, " · ".join(row), np.median(re)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
