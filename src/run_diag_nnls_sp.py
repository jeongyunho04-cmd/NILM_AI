# -*- coding: utf-8 -*-
"""와트당 선형 사전 대 **전력별** 사전 — SMPS 는 전류가 전력에 비례하지 않는다 (13.84.23 둘째 반).

13.84.23 ①: 단독 녹화의 **와트당** 사전으로 test_4(SMPS 셋, 93W)를 풀면 잔차가 17~27% 이고
미니PC 에 늘 0W 를 준다. 저항이 지배하는 파일에서는 잔차가 1~4% 인데 그건 큰 부하를 맞춘 값이다.

의심: `harmonic_signatures*` 는 **와트당 페이저**라 I_h = w * sig_h 를 가정한다. 그런데 SMPS 는
도통각이 전력에 따라 변해 와트당 지문이 전력의 함수다(생성기는 `sp_curves` 로 그걸 모형화한다).
그래서 사전을 **전력 구간별 절대 전류**로 바꿔 다시 푼다. 잔차가 내려가면 비선형이 범인이다.

    python -X utf8 src/run_diag_nnls_sp.py
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
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
SEG_S = 10
ODD = [0, 2, 4, 6, 8, 10, 12, 14]
N_BIN = 6


def dict_perwatt(pool):
    sig, used = harmonic_signatures_by_state(pool, APPS)
    cols, names = [], []
    for k, a in enumerate(APPS):
        v = sig[k, 0, :, 0] + 1j * sig[k, 0, :, 1]
        cols.append(v)
        names.append(a)
    return np.stack(cols, axis=1), names, None


def dict_power(pool):
    """기기마다 전력 구간별 **절대** 전류 벡터. 가중은 0~1 쯤이어야 맞는다."""
    cols, names, watts = [], [], []
    for a in APPS:
        acts = pool.appliance_activations.get(a, [])
        H, P = [], []
        for act in acts:
            c = np.asarray(act.net_harmonics_complex)
            p = np.asarray(act.net_power_features)[:, 0]
            on = np.asarray(act.is_on).astype(bool)
            if on.sum() < 60:
                continue
            H.append(c[on]); P.append(p[on])
        if not H:
            continue
        H = np.concatenate(H); P = np.concatenate(P)
        good = P > 1.0
        H, P = H[good], P[good]
        if len(P) < 120:
            continue
        qs = np.unique(np.percentile(P, np.linspace(5, 95, N_BIN)))
        for q in qs:
            m = np.abs(P - q) <= max(0.08 * q, 1.0)
            if m.sum() < 30:
                continue
            cols.append(np.median(H[m].real, axis=0) + 1j * np.median(H[m].imag, axis=0))
            names.append("%s@%.0fW" % (a, q))
            watts.append(float(np.median(P[m])))
    return np.stack(cols, axis=1), names, np.asarray(watts)


def solve(D, obs, orders):
    A = np.concatenate([D[orders].real, D[orders].imag], axis=0)
    b = np.concatenate([obs[orders].real, obs[orders].imag])
    w, _ = nnls(A, b)
    res = np.linalg.norm(A @ w - b) / (np.linalg.norm(b) + 1e-12)
    return w, float(res)


def main():
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train", carrier_apps=("oven",))
    Dw, Nw, _ = dict_perwatt(pool)
    Dp, Np_, Wp = dict_power(pool)
    print("와트당 사전 %d 열 · 전력별 사전 %d 열" % (Dw.shape[1], Dp.shape[1]))
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    for stem in ["test_4", "test_2"]:
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
        lab = np.stack([mk[a] for a in APPS])
        chg = np.r_[0, np.nonzero(np.abs(np.diff(lab.astype(np.int8), axis=1)).sum(0))[0] + 1, n]
        segs = []
        for s0, s1 in zip(chg[:-1], chg[1:]):
            s0, s1 = s0 + 5 * FS, s1 - 5 * FS
            for t in range(s0, s1 - SEG_S * FS, SEG_S * FS):
                segs.append((t, t + SEG_S * FS))
        rows = {}
        for t0, t1 in segs:
            obs = trim_mean(H[t0:t1], 0.2, axis=0)
            ww, rw = solve(Dw, obs, ODD)
            wp, rp = solve(Dp, obs, ODD)
            perw = {a: 0.0 for a in APPS}
            for c, nm in enumerate(Nw):
                perw[nm] += ww[c]
            # 통제: 미니PC 열을 빼고 풀었을 때 잔차가 오르는가 = 미니PC 가 정말 필요한가
            keep = [c for c, nm in enumerate(Np_) if not nm.startswith("minipc@")]
            _, rp_no = solve(Dp[:, keep], obs, ODD)
            perp = {a: 0.0 for a in APPS}
            for c, nm in enumerate(Np_):
                perp[nm.split("@")[0]] += wp[c] * Wp[c]
            on = tuple(a for a in APPS if mk[a][t0])
            rows.setdefault(on, []).append((perw, rw, perp, rp, float(np.median(P[t0:t1])), rp_no))
        print("\n=== %s" % stem)
        for on, lst in sorted(rows.items(), key=lambda kv: -len(kv[1])):
            if len(lst) < 3:
                continue
            g = lambda f: float(np.median([f(x) for x in lst]))
            print("  참값 ON [%s]  n=%d  관측 P %.0fW" % (", ".join(x[:8] for x in on) or "없음",
                                                     len(lst), g(lambda x: x[4])))
            print("     와트당  잔차 %5.1f%%   미니PC %5.1fW · 충전기 %5.1fW · 프로젝터 %5.1fW"
                  % (100 * g(lambda x: x[1]), g(lambda x: x[0]["minipc"]),
                     g(lambda x: x[0]["laptop_charger"]), g(lambda x: x[0]["beam_projector"])))
            print("     전력별  잔차 %5.1f%%   미니PC %5.1fW · 충전기 %5.1fW · 프로젝터 %5.1fW"
                  % (100 * g(lambda x: x[3]), g(lambda x: x[2]["minipc"]),
                     g(lambda x: x[2]["laptop_charger"]), g(lambda x: x[2]["beam_projector"])))
            print("     통제    미니PC 열을 빼면 잔차 %5.1f%% (뺀 만큼 오름 %+.1f%%p)"
                  % (100 * g(lambda x: x[5]), 100 * (g(lambda x: x[5]) - g(lambda x: x[3]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
