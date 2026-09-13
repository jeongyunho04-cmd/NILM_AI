# -*- coding: utf-8 -*-
"""단독 녹화 사전으로 실측 복합을 직접 푼다 — 모델 없이 (13.84.23).

사용자: "단독 녹화가 있는데 그걸로 재현하면 되는 거 아닌가?"

관측 = 15차 복소 전류. 사전 = 기기 x 상태별 **와트당** 페이저(격리 녹화에서 뽑은 것).
    min_w >= 0  || sum_i w_i * sig_i - obs ||^2     (w_i 는 와트)
를 풀어 배분을 낸다. 12.122 계열의 NNLS 진단을 **새 계측기 자료로** 다시 하는 것이다
(옛 판 결론 "비음수 조합으로 못 간다 = 지문 형상이 틀렸다" 는 옛 계측기 시대다).

세 가지를 낸다.
  1. 실패 ② 구간(test_4 310~371초, 충전기+프로젝터, 미니PC OFF)에서 미니PC 에 몇 W 를 주는가
  2. 미니PC ON 구간에서는 몇 W 를 주는가 (가를 수 있는가)
  3. 잔차 — 관측의 몇 %를 설명 못 하는가 (사전이 맞는지)

    python -X utf8 src/run_diag_nnls.py
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from scipy.optimize import nnls
from scipy.stats import trim_mean

from src.model.net import harmonic_signatures_by_state
from src.preprocessing import load_nilm_npz
from src.synthesis.segment_pool import SegmentPool

FS = 60
FILES = ["test_4", "test_3", "test_2", "test_1"]
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
SEG_S = 10          # 정상 구간 평균 길이(초)
ODD = [0, 2, 4, 6, 8, 10, 12, 14]


def build_dict():
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
    sig, used = harmonic_signatures_by_state(pool, APPS)          # (K,S,H,2), (K,S)
    cols, names = [], []
    for k, a in enumerate(APPS):
        seen = set()
        for s in range(sig.shape[1]):
            if not used[k, s]:
                continue
            v = sig[k, s, :, 0] + 1j * sig[k, s, :, 1]            # 와트당 A
            key = np.round(np.abs(v[:4]), 6).tobytes()
            if key in seen:
                continue
            seen.add(key)
            cols.append(v)
            names.append("%s:s%d" % (a, s))
        if not seen:                                              # 상태별이 없으면 기기 하나
            v = sig[k, 0, :, 0] + 1j * sig[k, 0, :, 1]
            cols.append(v)
            names.append("%s:*" % a)
    return np.stack(cols, axis=1), names                          # (H, C)


def solve(D, obs, orders):
    """비음수 최소제곱. 복소를 Re/Im 로 쌓는다. 관측 크기로 정규화한 잔차도 낸다."""
    A = np.concatenate([D[orders].real, D[orders].imag], axis=0)
    b = np.concatenate([obs[orders].real, obs[orders].imag])
    w, _ = nnls(A, b)
    res = np.linalg.norm(A @ w - b) / (np.linalg.norm(b) + 1e-12)
    return w, float(res)


def main():
    D, names = build_dict()
    print("사전: %d 열 (기기 x 상태), %d 차수" % (D.shape[1], D.shape[0]))
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    for stem in FILES:
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        spec = ev[stem]
        mk = {}
        for a in APPS:
            m = np.zeros(n, bool)
            for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
                m[int(t0 * FS):int(min(t1 * FS, n))] = True
            mk[a] = m
        # 라벨이 바뀌지 않는 정상 구간을 10초씩 쪼갠다
        lab = np.stack([mk[a] for a in APPS])
        chg = np.r_[0, np.nonzero(np.abs(np.diff(lab.astype(np.int8), axis=1)).sum(0))[0] + 1, n]
        segs = []
        for s0, s1 in zip(chg[:-1], chg[1:]):
            s0, s1 = s0 + 5 * FS, s1 - 5 * FS                     # 전이 여유
            for t in range(s0, s1 - SEG_S * FS, SEG_S * FS):
                segs.append((t, t + SEG_S * FS))
        if not segs:
            continue
        print("\n=== %s  정상 구간 %d개" % (stem, len(segs)))
        rows = {}
        for t0, t1 in segs:
            obs = trim_mean(H[t0:t1], 0.2, axis=0)
            w, res = solve(D, obs, ODD)
            per = {a: 0.0 for a in APPS}
            for c, nm in enumerate(names):
                per[nm.split(":")[0]] += w[c]
            on = tuple(a for a in APPS if mk[a][t0])
            rows.setdefault(on, []).append((per, res, float(np.median(P[t0:t1]))))
        for on, lst in sorted(rows.items(), key=lambda kv: -len(kv[1])):
            if len(lst) < 2:
                continue
            med = {a: float(np.median([x[0][a] for x in lst])) for a in APPS}
            res = float(np.median([x[1] for x in lst]))
            pm = float(np.median([x[2] for x in lst]))
            top = sorted(med.items(), key=lambda kv: -kv[1])[:4]
            print("  참값 ON [%s]  n=%d  관측 P %.0fW  잔차 %.1f%%"
                  % (", ".join(x[:8] for x in on) or "없음", len(lst), pm, 100 * res))
            print("      배분 상위: " + " · ".join("%s %.0fW" % (a[:8], v) for a, v in top if v > 0.5))
            print("      미니PC %.1fW · 충전기 %.1fW · 프로젝터 %.1fW"
                  % (med["minipc"], med["laptop_charger"], med["beam_projector"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
