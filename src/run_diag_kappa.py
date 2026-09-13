# -*- coding: utf-8 -*-
"""왜 헤드가 전압 지수를 못 배우나 — **기울기가 어디 있나** (14.16).

14.16 에서 `--vexp` 가 실패했다. 구조는 정확히 +2.02 를 더했는데 헤드가 여전히 0.66 을
배워 합계가 2.68 이 됐다 (물리 2.0). 즉 **구조를 더해도 헤드가 안 바뀐다.**

내가 처음 댄 이유("합성 창의 전압이 녹화 전압과 같아 α≈1")는 **틀렸다** — 생성기는
`apply_power_voltage_response` 로 전력 라벨을 `P ∝ V^2` (저항)로 정확히 움직인다.

그러면 진짜 이유는 **기울기의 크기**다. 전력 헤드는
    p_raw = Σ_s mix_s · softplus(p_states_s)
로 **상태별 공칭 전력의 혼합**을 낸다. 전압에 의한 상태 **안** 변동이 상태 **사이** 변동보다
훨씬 작으면, 손실은 상태를 맞히는 데 지배당하고 지수는 보이지 않는다.

재는 것 — 저항 기기마다:
  ① kappa = V_창 / V_ref 의 산포 (전압이 얼마나 흔들리나)
  ② 그것이 만드는 **상태 안** 전력 산포
  ③ **상태 사이** 전력 산포와의 비  <- 이 비가 기울기의 몫이다

    python -X utf8 src/run_diag_kappa.py
"""
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

RES = ("electiric_kettle", "hair_dryer", "hotplate", "oven")
N_WIN = 400


def main():
    from src.synthesis.genopts import build_synthesizer, resolve
    from src.synthesis.dataset import NILMBatchGenerator

    syn = build_synthesizer(resolve("v32"), "processed_data/npz", "train")
    gs = syn.grid_sim
    print("저항 기기의 전압 지수 (생성기가 쓰는 값):")
    for a in RES:
        print("   %-18s P ∝ V^%.1f · I ∝ V^%.1f"
              % (a, gs.power_voltage_exponent(a), gs.current_voltage_exponent(a)
                 if hasattr(gs, "current_voltage_exponent") else float("nan")))

    # ── 창을 뽑아 kappa 와 전력을 모은다 ─────────────────────────────────
    gen = NILMBatchGenerator(segment_pool=syn.pool, window_size_cycles=3600,
                             synthesizer=syn, compute_gt_harmonics=False,
                             recipe_mix=None, smps_focus_off_p=0.4)
    kap = defaultdict(list)
    pw = defaultdict(list)
    np.random.seed(0)
    for i in range(N_WIN):
        try:
            w = gen.generate_window()
        except Exception:
            continue
        env = getattr(w, "environment", None) or getattr(w, "env", None)
        vb = float(getattr(env, "base_voltage_v", np.nan)) if env is not None else np.nan
        gp = getattr(w, "gt_target_power_w", None)
        if gp is None or not np.isfinite(vb):
            continue
        for a in RES:
            p = np.asarray(gp.get(a, []), float)
            on = p > 1.0
            if on.sum() < 60:
                continue
            kap[a].append(vb)
            pw[a].append(float(np.median(p[on])))
    if not kap:
        print("\n창을 못 만들었다"); return 1

    print("\n① **창 전압** 산포 (창마다 하나)")
    allv = np.concatenate([np.asarray(v) for v in kap.values()])
    print("   p5~p95 %.1f~%.1f V · 표준편차 %.2f V · 중앙 %.1f V"
          % (np.percentile(allv, 5), np.percentile(allv, 95), allv.std(), np.median(allv)))
    print("   -> kappa 표준편차 ≈ %.4f  ->  **상태 안 전력 산포 ≈ %.1f%%** (P ∝ V²)"
          % (allv.std() / np.median(allv), 200 * allv.std() / np.median(allv)))

    print("\n②③ 기기별 — 상태 안(전압) 대 상태 사이(공칭) 전력 산포")
    print("   %-18s %6s %12s %14s %10s" % ("기기", "창", "전력 CV", "전압몫 CV", "전압몫/전체"))
    for a in RES:
        if len(pw[a]) < 20:
            continue
        p = np.asarray(pw[a]); v = np.asarray(kap[a])
        cv_all = p.std() / max(p.mean(), 1e-9)
        cv_v = 2 * v.std() / max(v.mean(), 1e-9)          # P ∝ V² 이므로 CV_P = 2·CV_V
        print("   %-18s %6d %11.1f%% %13.1f%% %9.1f%%"
              % (a, len(p), 100 * cv_all, 100 * cv_v, 100 * cv_v / max(cv_all, 1e-9)))

    print("\n   => 전압몫이 전체의 몇 %인지가 **기울기의 몫**이다.")
    print("      작으면 손실은 상태를 맞히는 데 지배당하고 지수는 안 배워진다")
    print("      ([[loss-share-is-not-gradient-share]] 의 반대 방향 — 값이 작으면 기울기도 작다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
