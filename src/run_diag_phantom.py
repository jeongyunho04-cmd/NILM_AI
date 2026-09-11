# -*- coding: utf-8 -*-
"""유령·합계 보존을 5파일에서 기기별로 집계한다 (13.84.17).

외부 진단 4차의 세 주장을 그대로 검정한다.
  ① 2000W 위에서 합계 보존이 깨진다 (Σ 예측 − 관측 P 를 관측 전력대별로)
  ② 유령은 저항 부하에만 붙는다 (파일에 없는 기기 · 있지만 OFF 인 기기, 초와 와트)
  ③ 프로젝터가 상시 흡수처다 (정답 ON 초 대 예측 ON 초)

    python -X utf8 src/run_diag_phantom.py cnn_v35 [cnn_v31 ...]
"""
import json
import sys

import numpy as np

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.run_gate_check import forward_file, load_model

FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
RESIST = {"oven", "hotplate", "electiric_kettle", "hair_dryer", "fan"}
STRIDE = 15
GATE_ON = 0.5


def main():
    tags = sys.argv[1:] or ["cnn_v35"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    for tag in tags:
        model, apps = load_model("results/%s.pt" % tag, dev)[:2]
        print("\n================ %s  (기기 %d, %s)" % (tag, len(apps), dev))
        tot_abs = {}
        for stem in FILES:
            d = forward_file(model, stem, dev, stride=STRIDE)
            P = d["gate"] * d["p_raw"]                     # (n, K)
            g = d["gate"]
            t = d["targets"] / 60.0                        # 초
            obs = d["p_observed"]
            dt = STRIDE / 60.0                             # 표본당 초
            spec = ev[stem]
            present = set(spec["appliances_present"])
            truth = np.zeros_like(g, dtype=bool)
            for k, a in enumerate(apps):
                for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                    truth[:, k] |= (t >= t0) & (t < t1)
            # ── ① 합계 보존 ─────────────────────────────────────────────
            err = P.sum(1) - obs
            print("\n--- %s  관측 최대 %.0fW · 표본 %d" % (stem, obs.max(), len(obs)))
            print("    관측 전력대     n     Σ예측−관측 중앙   p10~p90        |오차|/관측 중앙")
            for lo, hi in ((0, 200), (200, 800), (800, 1500), (1500, 2200), (2200, 5000)):
                m = (obs >= lo) & (obs < hi)
                if m.sum() < 5:
                    continue
                print("    %5d~%-5d %6d   %+12.1f W  %+7.0f~%+-7.0f   %6.1f%%"
                      % (lo, hi, m.sum(), np.median(err[m]), *np.percentile(err[m], [10, 90]),
                         100 * np.median(np.abs(err[m]) / np.maximum(obs[m], 1))))
            # ── ②③ 기기별 유령·과검출 ──────────────────────────────────
            rows = []
            for k, a in enumerate(apps):
                on_pred = g[:, k] > GATE_ON
                if not (on_pred.any() or truth[:, k].any()):
                    continue
                fp = on_pred & ~truth[:, k]
                miss = truth[:, k] & ~on_pred
                kind = ("없는 기기" if a not in present else "있지만 OFF")
                rows.append((a, kind, truth[:, k].sum() * dt, on_pred.sum() * dt,
                             fp.sum() * dt, float(P[fp, k].mean()) if fp.any() else 0.0,
                             float(P[fp, k].sum() * dt / 3600.0), miss.sum() * dt,
                             float(P[fp, k].max()) if fp.any() else 0.0))
                tot_abs.setdefault(a, [0.0, 0.0, 0.0])
                tot_abs[a][0] += fp.sum() * dt
                tot_abs[a][1] += float(P[fp, k].sum() * dt / 3600.0)
                tot_abs[a][2] += miss.sum() * dt
            rows.sort(key=lambda r: -(r[4] + r[7]))
            print("    기기               구분        정답ON초  예측ON초  유령초 유령평균W 유령최대W  미탐초")
            for a, kind, ton, pon, fps, w, wh, ms, wmax in rows:
                if fps < 5 and ms < 5:
                    continue
                print("    %-18s %-10s %8.0f %9.0f %7.0f %8.1f %9.0f %7.0f"
                      % (a, kind, ton, pon, fps, w, wmax, ms))
        print("\n--- %s 5파일 합계 — 오류 초 순위" % tag)
        print("    기기               부류      유령초   미탐초    합계초   유령Wh")
        for a, (s_, wh, ms) in sorted(tot_abs.items(), key=lambda kv: -(kv[1][0] + kv[1][2])):
            if s_ + ms < 5:
                continue
            print("    %-18s %-8s %8.0f %8.0f %9.0f %8.3f"
                  % (a, "저항" if a in RESIST else "SMPS", s_, ms, s_ + ms, wh))
        rs = sum(v[0] for a, v in tot_abs.items() if a in RESIST)
        ss = sum(v[0] for a, v in tot_abs.items() if a not in RESIST)
        rw = sum(v[1] for a, v in tot_abs.items() if a in RESIST)
        sw = sum(v[1] for a, v in tot_abs.items() if a not in RESIST)
        print("    => 유령 초: 저항 %.0f · SMPS %.0f  (저항 몫 %.0f%%)" % (rs, ss, 100 * rs / max(rs + ss, 1)))
        print("    => 유령 Wh: 저항 %.2f · SMPS %.2f  (저항 몫 %.0f%%) — 와트는 기기 규모지 신원 축퇴가 아니다"
              % (rw, sw, 100 * rw / max(rw + sw, 1e-9)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
