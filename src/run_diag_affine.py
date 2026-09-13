# -*- coding: utf-8 -*-
"""손실의 배분 기울기 `dI/dP` 는 참값인가 — 대기가 만드는 절편을 잰다 (13.84.64).

13.84.51 ④ 가 남긴 처방 후보 1: *"미니PC 의 ON/OFF 는 0 에서 켜지는 것이 아니라
30.8mA 위에 28mA 를 얹는 것이다. 손실·라벨·사슬이 그 구조를 안 쓰고 있다."*

지금 손실은 **원점을 지나는 직선**이다 (`net.harmonic_signatures`):

    I_net(P) = P . sig_k          sig_k = median(I_net / P)

물리는 **절편이 있는 직선**이다 — 통전 중에도 대기 회로는 흐르고, 통전 전력이 0 으로
가도 그 전류는 안 사라진다:  `I_net(P) = a_k + P . b_k`.

⚠ **절편을 직접 재려다 한 번 헛짚었다 (기록).** `I = a + P.b` 를 그냥 최소제곱하면
포트가 |a| = 765mA, 오븐이 1,544mA 로 나온다. 포트는 대기가 없다. 원인은 물리가 아니라
**조건수**다 — 통전 전력이 포트 1.01배 · 오븐 1.05배 · 프로젝터 1.05배 폭밖에 안 움직여
P=0 으로의 외삽이 전부 수치 먼지다 (cond[1,P] = 180,000 / 61,000 / 3,100). 절편을
**물을 수 있는** 기기는 충전기(cond 265) · 미니PC(108) · 팬(145) 셋뿐이고, 그 셋조차
관측 범위 밖으로의 외삽이다. [[check-conditioning-before-believing-a-fit]]

그래서 **잘 정의된 것만 묻는다.** 관측 범위 **안**의 한계 기울기 `b` 는 유한차분으로
바로 잡히고, 배분을 정하는 것이 바로 그 기울기다:

  ① 조건수  누가 이 물음에 답할 수 있나 — 전력 폭과 cond
  ② 기울기  범위 안 유한차분 `b` 대 손실이 믿는 `sig`. 이득비 · 각도 · 녹화별 산포
  ③ 절편    b 에서 되짚은 a = median(I) − P_med.b. 대기 페이저와 견준다 (외삽임을 명시)
  ④ 예측    녹화 하나 빼기. L(지금) / A(절편) / B(13.84.38 전력대) 의 순방향 잔차
  ⑤ 배분    **결정 관문.** 실제 사이클로 합성한 관측을 NNLS 로 풀어 미니PC 오차를 잰다

    python -X utf8 src/run_diag_affine.py [--n 400] [--seed 0]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.synthesis.segment_pool import SegmentPool

ORD = [1, 3, 5, 7, 9, 11, 13, 15]          # 홀수차 — 짝수차 위상은 기기 속성이 아니다 (13.11)
OI = [o - 1 for o in ORD]
SMPS = ("laptop_charger", "beam_projector", "minipc")
ALL = ["laptop_charger", "minipc", "beam_projector", "oven", "hotplate",
       "electiric_kettle", "air_conditioner", "fan", "hair_dryer"]


def gather(pool, app):
    """그 기기의 통전 사이클 (P, I_net) 를 **녹화별로** 모은다.

    포함 규칙은 `net.harmonic_signatures` 와 **같게** 쓴다 (전력 > p90/2). 다른 문턱을
    쓰면 비교가 아니라 다른 실험이 된다 ([[copy-the-inclusion-rule-when-adding-an-axis]]).
    """
    acts = pool.appliance_activations.get(app, [])
    thr = max(0.5 * pool.get_steady_power_w(app), 1.0)
    by = {}
    for a in acts:
        m = a.target_power_w > thr
        if not m.any():
            continue
        f = by.setdefault(a.source_file, [[], []])
        f[0].append(a.net_harmonics_complex[m][:, OI])
        f[1].append(a.target_power_w[m])
    return {k: (np.concatenate(v[1]).astype(float), np.concatenate(v[0]))
            for k, v in by.items()}


def cat(d, keys=None):
    ks = list(keys if keys is not None else d)
    return (np.concatenate([d[k][0] for k in ks]),
            np.concatenate([d[k][1] for k in ks]))


def fit_linear(P, C):
    """지금 손실의 모델 — 원점을 지나는 직선, 와트당 페이저의 **중앙**."""
    pw = C / np.maximum(P, 1e-6)[:, None]
    return np.median(pw.real, 0) + 1j * np.median(pw.imag, 0)


def fit_slope(P, C, q=(0.2, 0.8)):
    """**관측 범위 안**의 한계 기울기. 아래/위 분위 구름의 중앙을 잇는다.

    외삽이 아니라 내삽이라 조건수 문제가 없다. 반환: (a, b) — a 는 그 직선의 절편이니
    **외삽값**이고, b 만 관측이 직접 받치는 양이다.
    """
    lo, hi = np.quantile(P, q)
    ml, mh = P <= lo, P >= hi
    if ml.sum() < 30 or mh.sum() < 30:
        return None, None
    md = lambda m: np.median(C[m].real, 0) + 1j * np.median(C[m].imag, 0)
    pl, ph = float(np.median(P[ml])), float(np.median(P[mh]))
    if ph - pl < 1e-6:
        return None, None
    b = (md(mh) - md(ml)) / (ph - pl)
    a = md(ml) - pl * b
    return a, b


def fit_bands(P, C, nb=3, min_cy=300):
    """13.84.38 — 전력 3분위마다 와트당 중앙. (경계, (nb,H))"""
    e = np.quantile(P, np.linspace(0, 1, nb + 1))[1:-1]
    g = np.zeros((nb, C.shape[1]), complex)
    g[:] = fit_linear(P, C)
    idx = np.digitize(P, e)
    for b in range(nb):
        m = idx == b
        if m.sum() >= min_cy:
            pw = C[m] / np.maximum(P[m], 1e-6)[:, None]
            g[b] = np.median(pw.real, 0) + 1j * np.median(pw.imag, 0)
    return e, g


def ang(u, v):
    """두 복소 벡터를 Re/Im 실수 벡터로 펴서 잰 각도(도)."""
    a = np.concatenate([u.real, u.imag]); b = np.concatenate([v.real, v.imag])
    c = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-18)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def nnls(A, y, iters=300):
    """비음 최소제곱 (사영 경사). A 실수 (M,K), y (M,)."""
    K = A.shape[1]
    x = np.zeros(K)
    L = float(np.linalg.norm(A, 2) ** 2) or 1.0
    AtA, Aty = A.T @ A, A.T @ y
    for _ in range(iters):
        x = np.maximum(0.0, x - (AtA @ x - Aty) / L)
    return x


def ri(c):
    """복소 (H,) -> 실수 (2H,)."""
    return np.concatenate([c.real, c.imag])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="⑤ 합성 관측 수 (뺀 녹화 조합마다)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="all", carrier_apps=("oven",))
    data = {app: gather(pool, app) for app in ALL}
    sb = {}
    for app in ALL:
        try:
            pr = pool.get_standby_profile(app)
            sb[app] = (np.asarray(pr.harmonics_complex)[OI], float(np.mean(pr.power_w)))
        except Exception:
            sb[app] = (None, 0.0)

    # -- ① 누가 이 물음에 답할 수 있나 ---------------------------------------
    print("① **조건수** — 절편을 물으려면 전력이 움직여야 한다. 안 움직이면 외삽은 먼지다")
    print("   %-18s %8s %8s %8s %7s %10s  %s" % ("기기", "p5 W", "p50", "p95", "p95/p5", "cond[1,P]", "판정"))
    askable = []
    for app in ALL:
        d = data[app]
        if not d:
            continue
        P, _ = cat(d)
        q = np.percentile(P, [5, 50, 95])
        cd = float(np.linalg.cond(np.stack([np.ones_like(P), P], 1)))
        ok = cd < 1000
        askable += [app] if ok else []
        print("   %-18s %8.1f %8.1f %8.1f %7.2f %10.0f  %s"
              % (app, q[0], q[1], q[2], q[2] / max(q[0], 1e-9), cd,
                 "물을 수 있다" if ok else "**못 묻는다**"))

    # -- ② 한계 기울기 -------------------------------------------------------
    print("\n② **기울기** — 관측 범위 안 유한차분 `b` 대 손실이 믿는 `sig`. 배분을 정하는 것이 b 다")
    print("   %-18s %7s  %s" % ("기기", "각도", "차수별 |b|/|sig|"))
    fitL, fitS = {}, {}
    for app in ALL:
        d = data[app]
        if not d:
            continue
        P, C = cat(d)
        sg = fit_linear(P, C); aa, bb = fit_slope(P, C)
        if bb is None:
            continue
        fitL[app], fitS[app] = sg, (aa, bb)
        r = np.abs(bb) / np.maximum(np.abs(sg), 1e-15)
        print("   %-18s %6.0f°  %s" % (app, ang(sg, bb),
                                       " ".join("h%d %4.2f" % (o, x) for o, x in zip(ORD, r))))
    print("\n   녹화별 산포 (녹화 하나로만 맞춘 b 들이 서로 몇 도인가 — 크면 b 자체가 못 미덥다)")
    for app in SMPS:
        d = data[app]
        bs = [fit_slope(*d[k])[1] for k in d]
        bs = [x for x in bs if x is not None]
        if len(bs) < 2:
            print("   %-18s 녹화 %d — 잴 수 없다" % (app, len(bs))); continue
        aw = [ang(bs[i], bs[j]) for i in range(len(bs)) for j in range(i + 1, len(bs))]
        av = [ang(fitL[app], fit_linear(*d[k])) for k in d]
        print("   %-18s b 끼리 중앙 %3.0f° (최대 %3.0f°, n=%d)   sig 끼리 중앙 %3.0f°"
              % (app, np.median(aw), np.max(aw), len(bs), np.median(av)))

    # -- ③ 되짚은 절편 -------------------------------------------------------
    print("\n③ **절편** — b 에서 되짚은 a. ⚠ 관측 범위 밖 외삽이다. ① 이 통과한 기기만 본다")
    print("   %-18s %11s %11s %8s %8s  %s" % ("기기", "|a| h1 mA", "대기 h1 mA", "a/대기", "각도", "동작점 몫 |a|/|a+P.b|"))
    for app in askable:
        if app not in fitS or sb[app][0] is None:
            continue
        aa, bb = fitS[app]; s0, pw = sb[app]
        P, _ = cat(data[app]); pm = float(np.median(P))
        sh = np.abs(aa) / np.maximum(np.abs(aa + pm * bb), 1e-15) * 100
        print("   %-18s %11.1f %11.1f %8.2f %7.0f°  %s"
              % (app, 1000 * abs(aa[0]), 1000 * abs(s0[0]),
                 np.linalg.norm(np.abs(aa)) / max(np.linalg.norm(np.abs(s0)), 1e-15),
                 ang(aa, s0), " ".join("h%d %3.0f" % (o, x) for o, x in zip(ORD[:4], sh[:4]))))

    # -- ④ 예측 (LORO) -------------------------------------------------------
    print("\n④ **예측(LORO)** — 남은 녹화로 맞춰 뺀 녹화를 예측. |예측−관측|/|관측| 중앙")
    print("   %-18s %5s %9s %9s %9s   %s" % ("기기", "녹화", "L 지금", "A 절편", "B 전력대", "A 대 L"))
    for app in ALL:
        d = data[app]
        if len(d) < 2:
            continue
        rs = {"L": [], "A": [], "B": []}
        for ho in d:
            tr = [k for k in d if k != ho]
            P, C = cat(d, tr); Pt, Ct = d[ho]
            om = np.maximum(np.abs(Ct), 1e-9)
            rs["L"].append(np.median(np.abs(Pt[:, None] * fit_linear(P, C)[None] - Ct) / om))
            aa, bb = fit_slope(P, C)
            rs["A"].append(np.nan if bb is None else
                           np.median(np.abs(aa[None] + Pt[:, None] * bb[None] - Ct) / om))
            e, g = fit_bands(P, C)
            rs["B"].append(np.median(np.abs(g[np.digitize(Pt, e)] * Pt[:, None] - Ct) / om))
        m = {k: float(np.mean(v)) for k, v in rs.items()}
        print("   %-18s %5d %8.1f%% %8.1f%% %8.1f%%   %+6.1f%%"
              % (app, len(d), 100 * m["L"], 100 * m["A"], 100 * m["B"],
                 100 * (m["A"] / m["L"] - 1)))

    # -- ⑤ 배분 (결정 관문) --------------------------------------------------
    print("\n⑤ **배분** — 뺀 녹화의 **실제 사이클**을 더해 관측을 만들고 NNLS 로 전력을 되찾는다")
    print("   사전만 바꾼다: L = P.sig (지금)  ·  A = a + P.b (절편판, a 는 알려진 오프셋)")
    rng = np.random.default_rng(a.seed)
    trio = [x for x in SMPS if len(data[x]) >= 2]
    errs = {"L": {k: [] for k in trio}, "A": {k: [] for k in trio}}
    nrep = 0
    for ho in range(max(len(data[x]) for x in trio)):
        keys = {x: sorted(data[x])[ho % len(data[x])] for x in trio}
        D_L, D_A, off = [], [], np.zeros(2 * len(ORD))
        src = {}
        bad = False
        for x in trio:
            tr = [k for k in data[x] if k != keys[x]]
            P, C = cat(data[x], tr)
            sg = fit_linear(P, C); aa, bb = fit_slope(P, C)
            if bb is None:
                bad = True; break
            D_L.append(ri(sg)); D_A.append(ri(bb)); off = off + ri(aa)
            src[x] = data[x][keys[x]]
        if bad:
            continue
        nrep += 1
        D_L = np.stack(D_L, 1); D_A = np.stack(D_A, 1)
        for _ in range(a.n):
            y = np.zeros(2 * len(ORD)); tp = []
            for x in trio:
                Px, Cx = src[x]
                j = rng.integers(len(Px))
                y = y + ri(Cx[j]); tp.append(Px[j])
            tp = np.asarray(tp)
            xL = nnls(D_L, y); xA = nnls(D_A, y - off)
            for i, x in enumerate(trio):
                errs["L"][x].append(abs(xL[i] - tp[i]))
                errs["A"][x].append(abs(xA[i] - tp[i]))
    print("   %-18s %12s %12s   %s" % ("기기", "L 오차 W", "A 오차 W", "A 대 L"))
    for x in trio:
        eL, eA = np.median(errs["L"][x]), np.median(errs["A"][x])
        print("   %-18s %11.2f %12.2f   %+6.1f%%" % (x, eL, eA, 100 * (eA / max(eL, 1e-9) - 1)))
    print("   (뺀 녹화 조합 %d 벌 x %d 창. 참 전력은 그 사이클의 실측 전력이다)" % (nrep, a.n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
