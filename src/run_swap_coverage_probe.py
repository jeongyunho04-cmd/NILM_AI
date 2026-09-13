"""장소 B 강풍 창에서 `L_swap` 이 실제로 감독하는가 (12.164.12)."""
import sys, itertools
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from src import env_guard  # noqa: F401
import numpy as np
from src.run_gate_check import forward_file, load_model
from src.model.postproc import RESISTIVE_OHM, HALFWAVE_OHM, HALFWAVE_ABS_MIN

RES = ["electiric_kettle", "oven", "hair_dryer", "hotplate"]
dev, TOL, SLACK = "cuda", 0.02, 1
for ck in sys.argv[1:]:
    model, apps, _ = load_model(ck, dev)
    ji = [apps.index(a) for a in RES]
    jo, jd = apps.index("oven"), apps.index("hair_dryer")
    print(f"\n=== {ck} ===")
    for stem in ("test_16", "test_17", "test_18"):
        d = forward_file(model, stem, dev, stride=30)
        P = d["gate"] * d["p_raw"]; v2 = d["v_rms"] ** 2
        h = d["obs_harm"]
        half = (np.hypot(h[:,1,0],h[:,1,1]) - np.hypot(h[:,3,0],h[:,3,1])) > HALFWAVE_ABS_MIN
        # L_swap 이 보는 저항 잔차
        free = P.sum(1) - P[:, ji].sum(1)
        p_res = d["p_observed"] - free - d["standby"].sum(1) - d["p_noise"]
        g_need = p_res / np.maximum(v2, 1.0)
        m = d["gate"][:, jo] > 0.5
        if m.sum() < 3:
            print(f"  {stem}: 오븐 발화 창 없음"); continue
        combos = np.array(list(itertools.product([0,1], repeat=4)), float)
        rows = []
        for i in np.where(m)[0]:
            g = np.array([1/RESISTIVE_OHM[a] for a in RES])
            if half[i]:
                g[RES.index("hair_dryer")] = 1/HALFWAVE_OHM["hair_dryer"]
            cg = combos @ g
            cur = (d["gate"][i, ji] > 0.5).astype(float)
            k = cur.sum()
            ok = np.abs(combos.sum(1) - k) <= SLACK
            err = np.where(ok, np.abs(cg - g_need[i]), np.inf)
            b = int(err.argmin())
            rel = err[b] * v2[i] / max(abs(p_res[i]), 1.0)
            rows.append((rel, combos[b] @ np.arange(1, 5), p_res[i],
                         float(rel <= TOL), float((combos[b] != cur).any())))
        r = np.array(rows)
        # 드라이기만 켠 조합의 상대오차 (참인 답)
        dry = []
        for i in np.where(m)[0]:
            gd = 1/(HALFWAVE_OHM["hair_dryer"] if half[i] else RESISTIVE_OHM["hair_dryer"])
            dry.append(abs(gd - g_need[i]) * v2[i] / max(abs(p_res[i]), 1.0))
        print(f"  {stem}  오븐 발화 {m.sum():4d}창 | 반파열림 {half[m].mean():.0%}"
              f" | p_res 중앙 {np.median(r[:,2]):6.0f}W"
              f" | 최적조합 상대오차 중앙 {np.median(r[:,0]):.4f}"
              f" | 감독된 창 {r[:,3].mean():.0%} (그중 바꿔야 {r[:,4].mean():.0%})"
              f" | **드라이기 단독 상대오차 중앙 {np.median(dry):.4f}**")
