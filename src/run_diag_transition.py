# -*- coding: utf-8 -*-
"""전이(켜짐·꺼짐) 차분 Δ 가 실측에서 기기를 가르는가 — 정상상태 합 대신 쓸 정보원 검사 (13.84.10).

    python -X utf8 src/run_diag_transition.py

사람 라벨(real_events.json)의 구간 시작=ON, 끝=OFF 사건마다 Δ = mean[t0+3s, t0+13s] − mean[t0−13s, t0−3s]
(15차 복소 페이저 + P) 를 내고(OFF 는 부호 반전), ±15초 안에 다른 사건이 있으면 뺀다. 그런 뒤:
  ① 잡음 바닥 — 사건이 없는 시각의 같은 차분 |Δ| 분포 (이것보다 작은 기기는 전이로도 못 본다)
  ② 제자리 Δ/ΔP 대 풀 와트당 템플릿(상태별) — 차수별 크기 비·각도 차. 정상상태 격차표(13.83.24, |I3| 비 0.36~0.75)와 견줌
  ③ 최근접 템플릿 분류 (풀 서명만, 학습 없음) — 사건별 정답률·혼동
  ④ 파일 하나 빼기 로지스틱 probe (실측 → 실측 일반화)
"""
import json
import sys

import numpy as np
from scipy.stats import trim_mean

sys.path.insert(0, '.')
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.model.net import harmonic_scales, harmonic_signatures_by_state
from src.model.realdata import DEFAULT_DIR as REAL_DIR
from src.preprocessing import load_nilm_npz
from src.synthesis.segment_pool import SegmentPool

FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
PRE, POST, GAP, GUARD = 13, 3, 15, 15          # 초
ODD = [0, 2, 4, 6, 8, 10, 12, 14]
rng = np.random.default_rng(0)


def collect():
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    E, N = [], []
    for stem in FILES:
        spec = ev[stem]
        raw = load_nilm_npz("%s/%s.npz" % (REAL_DIR, stem))
        H = raw["harmonics_ri"][:, :, 0] + 1j * raw["harmonics_ri"][:, :, 1]
        P = np.asarray(raw["power_features"])[:, 0]
        valid = np.asarray(raw["is_valid"]).astype(bool)
        n = len(P)
        P1 = np.array([P[i:i + 60].mean() for i in range(0, n - 60, 60)])     # 1초 평균 (오염 검사용)
        times = []                                  # (t_sec, app, kind)
        for app, d in spec["intervals"].items():
            for a, b in d.get("on", []):
                times.append((float(a), app, "on")); times.append((float(b), app, "off"))
        all_t = np.array([t for t, _, _ in times])

        def delta(t0):
            i0 = int(round(t0 * 60))
            lo, hi = i0 - PRE * 60, i0 - POST * 60
            lo2, hi2 = i0 + POST * 60, i0 + PRE * 60
            if lo < 0 or hi2 > n or not valid[lo:hi].all() or not valid[lo2:hi2].all():
                return None
            # 오염 검사: 전/후 창 안의 1초 평균 전력에 150W 넘는 계단(라벨 없는 듀티·스위칭)이 있으면 뺀다
            s_ = int(round(t0))
            seg = np.concatenate([P1[max(0, s_ - PRE):s_ - POST], P1[s_ + POST:s_ + PRE]])
            if len(seg) < 10 or np.abs(np.diff(seg)).max() > 150:
                return None
            # 절사 평균(20%) — 핫플 듀티·돌입의 영향을 줄인다
            return (trim_mean(H[lo2:hi2], 0.2, axis=0) - trim_mean(H[lo:hi], 0.2, axis=0),
                    float(trim_mean(P[lo2:hi2], 0.2) - trim_mean(P[lo:hi], 0.2)))
        for t0, app, kind in times:
            if np.sum(np.abs(all_t - t0) < GUARD) > 1:
                continue
            d = delta(t0)
            if d is None:
                continue
            dh, dp = d
            if kind == "off":
                dh, dp = -dh, -dp
            E.append((stem, app, kind, dh, dp))
        # 잡음 바닥: 어떤 사건에서도 20초 이상 떨어진 시각 40개
        for _ in range(400):
            t0 = float(rng.uniform(PRE + 1, n / 60 - PRE - 1))
            if np.abs(all_t - t0).min() < 20:
                continue
            d = delta(t0)
            if d is not None:
                N.append((stem, d[0], d[1]))
            if sum(1 for s, _, _ in N if s == stem) >= 40:
                break
    return E, N


def main():
    E, N = collect()
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    apps = sorted(pool.appliance_activations)
    sig, used = harmonic_signatures_by_state(pool, apps)         # (K,S,15,2) 와트당
    hs = harmonic_scales(pool, apps)
    T = {}
    for j, a in enumerate(apps):
        states = [s for s in range(sig.shape[1]) if used[j, s]] or [0]
        T[a] = [sig[j, s, :, 0] + 1j * sig[j, s, :, 1] for s in states]

    print("== 사건 %d개 (파일별: %s) · 잡음 바닥 표본 %d개" % (
        len(E), ", ".join("%s %d" % (f, sum(1 for e in E if e[0] == f)) for f in FILES), len(N)))
    nz = np.array([np.abs(d) for _, d, _ in N])                   # (n,15)
    nzp = np.array([abs(p) for _, _, p in N])
    print("\n① 잡음 바닥 (사건 없음, 같은 차분) |Δ| mA — p50 / p90")
    print("   %-8s" % "" + " ".join("%9s" % ("h%d" % (h + 1)) for h in ODD) + "   |ΔP| W")
    print("   %-8s" % "p50" + " ".join("%9.1f" % (1e3 * np.median(nz[:, h])) for h in ODD) + "   %.1f" % np.median(nzp))
    print("   %-8s" % "p90" + " ".join("%9.1f" % (1e3 * np.percentile(nz[:, h], 90)) for h in ODD) + "   %.1f" % np.percentile(nzp, 90))

    def dist(dh, dp, tmpl):
        v = dh / max(abs(dp), 1e-6)
        best = None
        for t in tmpl:
            d = np.sqrt(np.sum(((np.abs(v[ODD] - t[ODD])) / hs[ODD]) ** 2))
            best = d if best is None or d < best else best
        return best

    print("\n② 제자리 Δ/ΔP 대 풀 템플릿 (가장 가까운 상태) — 차수별 |비| · 각도차(°), 사건 중앙값")
    print("   %-16s %4s %6s | " % ("기기", "n", "ΔP W") + " ".join("%11s" % ("h%d" % (h + 1)) for h in ODD[:6]))
    for a in apps:
        es = [e for e in E if e[1] == a]
        if not es:
            continue
        rows = []
        for _, _, _, dh, dp in es:
            v = dh / max(abs(dp), 1e-6)
            t = min(T[a], key=lambda t: np.sqrt(np.sum(((np.abs(v[ODD] - t[ODD])) / hs[ODD]) ** 2)))
            r = v / np.where(np.abs(t) > 1e-9, t, 1e-9)
            rows.append((np.abs(r), np.degrees(np.angle(r))))
        mag = np.median([r[0] for r in rows], 0); ang = np.median([r[1] for r in rows], 0)
        print("   %-16s %4d %6.0f | " % (a, len(es), np.median([abs(e[4]) for e in es]))
              + " ".join("%5.2f %+5.0f" % (mag[h], ang[h]) for h in ODD[:6]))

    print("\n③ 최근접 템플릿 분류 (학습 없음, 풀 서명 · 차수 정규화 · 홀수차) — 참 기기별 예측")
    conf = {}
    for stem, a, kind, dh, dp in E:
        pred = min(apps, key=lambda b: dist(dh, dp, T[b]))
        conf.setdefault(a, {}); conf[a][pred] = conf[a].get(pred, 0) + 1
    ok = sum(conf[a].get(a, 0) for a in conf); tot = len(E)
    for a in apps:
        if a in conf:
            n_ = sum(conf[a].values())
            print("   %-16s %3d  정답 %3.0f%%  %s" % (a, n_, 100 * conf[a].get(a, 0) / n_,
                  ", ".join("%s %d" % (b[:6], c) for b, c in sorted(conf[a].items(), key=lambda x: -x[1]) if b != a)))
    print("   전체 정답률 %.3f (%d/%d) · 우연 %.3f" % (ok / tot, ok, tot, 1 / len(set(e[1] for e in E))))
    # SMPS 만
    smps = ("laptop_charger", "beam_projector", "minipc")
    es = [e for e in E if e[1] in smps]
    ok2 = 0
    for stem, a, kind, dh, dp in es:
        pred = min(smps, key=lambda b: dist(dh, dp, T[b]))
        ok2 += pred == a
    print("   SMPS 3종 안에서만 %.3f (%d/%d)" % (ok2 / max(len(es), 1), ok2, len(es)))
    # 규칙 셋: 모양만 / 전력 계단만 / 둘 다 (상태별 대표 전력은 S_STATE)
    from src.model.losses import S_STATE
    PW = {a: ([S_STATE.get(a, {}).get(s, pool.get_steady_power_w(a)) for s in range(sig.shape[1]) if used[j, s]]
              or [pool.get_steady_power_w(a)]) for j, a in enumerate(apps)}
    d_pow = lambda dp, a: min(abs(np.log(max(abs(dp), 1.0) / max(p_, 1.0))) for p_ in PW[a])
    for name, rule in (("모양만", lambda dh, dp, a: dist(dh, dp, T[a])), ("ΔP만", lambda dh, dp, a: d_pow(dp, a)),
                       ("모양+ΔP", lambda dh, dp, a: dist(dh, dp, T[a]) / 0.15 + d_pow(dp, a) / 0.5)):
        ok9 = sum(min(apps, key=lambda b: rule(dh, dp, b)) == a for _, a, _, dh, dp in E)
        ok3 = sum(min(smps, key=lambda b: rule(dh, dp, b)) == a for _, a, _, dh, dp in es)
        okm = sum(min(smps, key=lambda b: rule(dh, dp, b)) == a for _, a, _, dh, dp in es if a == "minipc")
        print("   규칙 %-8s 9종 %2d/%2d  SMPS3 %2d/%2d  미니PC %2d/%2d" % (
            name, ok9, len(E), ok3, len(es), okm, sum(1 for e in es if e[1] == "minipc")))
    for stem, a, kind, dh, dp in es:
        if a == "minipc":
            ds = {b: dist(dh, dp, T[b]) for b in smps}
            print("     미니PC 사건 %s %s ΔP %+5.1fW  거리: %s" % (stem, kind, dp, " ".join("%s %.2f" % (b[:6], ds[b]) for b in smps)))

    print("\n④ 파일 하나 빼기 로지스틱 probe (Δ/ΔP 홀수 Re/Im 16차원 + log|ΔP|)")
    X = np.array([np.concatenate([np.real(dh / max(abs(dp), 1e-6))[ODD] / hs[ODD], np.imag(dh / max(abs(dp), 1e-6))[ODD] / hs[ODD],
                                  [np.log(max(abs(dp), 1.0))]]) for _, _, _, dh, dp in E])
    y = np.array([e[1] for e in E]); f = np.array([e[0] for e in E])
    tot_ok, tot_n = 0, 0
    for stem in FILES:
        tr, te = f != stem, f == stem
        if te.sum() == 0:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(X[tr], y[tr])
        pred = clf.predict(X[te]); ok_ = (pred == y[te]).sum()
        tot_ok += ok_; tot_n += te.sum()
        wrong = ["%s→%s" % (a[:6], b[:6]) for a, b in zip(y[te], pred) if a != b]
        print("   %-7s 정답 %2d/%2d  %s" % (stem, ok_, te.sum(), " ".join(wrong)))
    print("   전체 %.3f (%d/%d)" % (tot_ok / tot_n, tot_ok, tot_n))


if __name__ == "__main__":
    main()
