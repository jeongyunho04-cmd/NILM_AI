"""교차주파수 어드미턴스 — **전 파일**로 적합한다 (12.148)

왜 다시
------
12.147 이 Norton/교차주파수 보정을 SMPS 3파일(test_7/8/13)로만 적합해서
홀드아웃 전이에 실패했다 (30.5 vs 맹탕 전역상수 24.1). 그런데 **그 3파일이
하필 전압 조건이 가장 평평한 조합**이었다:

    적합에 쓴 3파일     vh1 폭 2.1~3.4V   vh3 4.13~4.41   vh5 1.61~2.80
    전 11파일          vh1 209.6~223.2   vh3 3.35~4.50   vh5 1.53~8.01

저항이 큰 파일이 전압을 끌어내린다 — `test_12` 는 9.2A 에 vh1 209.6V 이고
`test_11` 은 vh5 가 8.01V 다. **어드미턴스를 적합할 동적 범위가 거기 있다.**

문헌의 방법이 원래 그렇다 — 기기를 **여러 전압 왜곡 조건에서** 재서 `Y_h` 를
뽑는다. 우리는 그 조건이 다른 파일에 있었을 뿐이다.

모형
----
창마다 정답 지지 `S_w` 안에서, **프로젝터만 참값 46.9W 로 못 박고** 나머지를
NNLS 로 푼 뒤 남는 것이 잔차다.

    r_w,h  =  y_w,h − Σ_{k∈S_w} P̂_k·sig_k,h

그 잔차를 **홀수차 전압 벡터**로 회귀한다 (교차주파수 결합):

    r_h  ≈  A_h · [1, |V_1|, |V_3|, …, |V_15|]        (A_h 는 2 x 9)

⚠ **반증 조건을 먼저 적는다.**
  - 홀드아웃 파일에서 보정 후 프로젝터 오차가 **전역 상수보다 나쁘면** 전이가
    안 되는 것이고, 전압 폭을 넓힌 것으로도 안 된다는 뜻이다. 그러면 닫는다.
  - 12.147 의 교훈대로 **잔차 설명력이 아니라 배분 오차로 판정한다.**
    오늘 세 처방이 전자에서 이기고 후자에서 졌다.

⚠ 짝수차는 안 쓴다 (12.72 전류 인공물, 그리고 12.147 이 전압 짝수차 2.1% 가
   부하 595배 변화에 불변임을 쟀다 — 상류인지 채널인지 미결).

    python -X utf8 -m src.run_norton_probe
    python -X utf8 -m src.run_norton_probe --fit-stems test.2 test3 test_4 ...
"""
from typing import Dict, List
import argparse
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from src.evaluation.power_ref import REFERENCE_W
from src.evaluation.real_events import build_on_off_truth, load_events
from src.preprocessing import load_nilm_npz

H = 15
ODD = np.arange(0, H, 2)          #: 0-based 인덱스 [0,2,4,…] — 배열 슬라이싱용
ORD = (1, 3, 5, 7, 9, 11, 13, 15)  #: 1-based 차수 — `V_src`/`Z·I` 키로 쓴다
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan",
        "hair_dryer", "hotplate", "laptop_charger", "minipc", "oven"]
RESIST = ("electiric_kettle", "hair_dryer", "hotplate", "oven",
          "air_conditioner", "fan")
#: SMPS 전용 창이 있는 파일 — 배분을 채점할 수 있는 유일한 자리다
EVAL_STEMS = ("test_7", "test_8", "test_13")


def load(stem: str, stride: int, nz: np.ndarray) -> dict:
    """관측 고조파 + **전압 고조파**. 모델을 안 쓴다."""
    from src.model.realdata import dense_targets
    ev = load_events()
    rw = dense_targets(stem, stride=stride)
    OH, POBS, PN = [], [], []
    for i in range(0, len(rw), 512):
        idx = np.arange(i, min(i + 512, len(rw)))
        _, _, pobs, oh, pn = rw.batch(idx)
        OH.append(oh); POBS.append(pobs); PN.append(pn)
    oh = np.concatenate(OH)
    on, _ = build_on_off_truth(stem, APPS, int(ev[stem]["cycles"]), ev)
    t = on[np.clip(rw.target_cycle, 0, len(on) - 1)]

    # 원자료의 vh1~vh15 를 npz 행에 맞춘다. `vrms` 상관으로 시프트를 찾는다 —
    # npz 가 원자료보다 길거나 짧은 파일이 있다 (test_7 은 +330행).
    csv = pd.read_csv(f"data/{stem}.csv",
                      usecols=["vrms"] + [f"vh{h}" for h in range(1, H + 1)])
    z = load_nilm_npz(f"processed_data/composite_eval/{stem}.npz")
    nv = np.asarray(z["power_features"])[:, 4]
    best = (-2.0, 0)
    for sh in range(-400, 401, 10):
        a = csv.vrms.values[max(0, -sh):]; b = nv[max(0, sh):]
        n = min(len(a), len(b))
        c = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if c > best[0]:
            best = (c, sh)
    sh = best[1]
    V = csv[[f"vh{h}" for h in range(1, H + 1)]].values
    j = rw.target_cycle - sh
    ok = (j >= 0) & (j < len(V))
    return {"y": oh[:, :H, :] - nz[None, :H, :],
            "p": np.concatenate(POBS) - np.concatenate(PN),
            "on": t, "V": V[np.clip(j, 0, len(V) - 1)], "ok": ok,
            "cycle": rw.target_cycle, "shift": sh, "align": float(best[0])}


def fit_impedance(stems, orders=(1, 3, 5, 7, 9, 11, 13, 15)) -> tuple:
    """계통 임피던스를 **물리 제약**으로 푼다 — `Z_h = R + j·h·ωL`, 미지수 둘.

    전압 **위상이 기록되지 않는다** (`ihdeg1~15` 는 있는데 `vhdeg` 가 없다).
    그런데 필요 없다:

        V_h  =  V_src,h  −  Z_h·I_h
                ^^^^^^^     ^^^^^^^
                파일 상수     변하는 부분 (I_h 위상은 있다)

    배경 `V_src` 는 그 집·그 시간대의 상수라 회귀 절편이 흡수한다. 회귀에
    필요한 것은 **부하가 만든 왜곡 `Z·I`** 뿐이고 그 위상은 전류 위상이다.

    `Z` 자체는 `|V_h|` 관측과 복소 `I_h` 로 푼다. 차수마다 자유롭게 두면
    h7~h11 에서 **R 이 음수**로 나온다 (수동 임피던스로 불가능. |V_h| 폭이
    1.3~2.8V 뿐이라 잡음을 맞춘다). `R + j·h·ωL` 로 묶으면 h1 의 좋은 조건
    (|V1| 폭 53V)이 전 차수를 정한다.

    ⚠ 전 파일이 **같은 장소**여야 한다 (`Z` 를 공유하므로). `test.csv` 는
       다른 장소라 뺀다 — 사용자 확인.
    """
    from scipy.optimize import least_squares, minimize
    D = {}
    for s in stems:
        c = [f"vh{h}" for h in orders] + [f"ih{h}" for h in orders]             + [f"ihdeg{h}" for h in orders]
        d = pd.read_csv(f"data/{s}.csv", usecols=c)
        n = len(d) // 30 * 30                      # vh 는 30사이클마다 갱신된다
        D[s] = {h: ((d[f"ih{h}"].values
                     * np.exp(1j * np.deg2rad(d[f"ihdeg{h}"].values)))[:n]
                    .reshape(-1, 30).mean(1),
                    d[f"vh{h}"].values[:n].reshape(-1, 30).mean(1)) for h in orders}

    def inner(I, Vm, Z):                            # Z 고정 -> V_src 만 (2변수)
        f = lambda q: np.abs((q[0] + 1j * q[1]) - Z * I) - Vm
        r = least_squares(f, [Vm.mean(), 0.0], method="lm", max_nfev=2000)
        return float(np.sqrt(np.mean(r.fun ** 2)) / max(Vm.std(), 1e-3))

    def outer(q):                                   # 분리 가능 — 바깥은 2변수뿐
        return sum(inner(*D[s][h], q[0] + 1j * h * q[1]) ** 2
                   for s in stems for h in orders)

    b = minimize(outer, [2.0, 0.05], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 60})
    return float(b.x[0]), float(b.x[1])


def zi_design(stem: str, idx: np.ndarray, R: float, X1: float, sh: int,
              orders=(1, 3, 5, 7, 9, 11, 13, 15)) -> np.ndarray:
    """회귀 열 `[1, Re(Z·I), Im(Z·I) …]` — **부하가 만든 전압 왜곡**이다.

    ⚠ 처음에는 복소 `V` 자체를 열로 썼다가 잔차가 20배로 터졌다 — `Re(V_1)≈223`
    인데 다른 열이 1~5 라 조건수가 무너진다. 그리고 그 223 은 `V_src` 라
    **절편이 이미 먹는다.** 변하는 부분만 넣는 것이 맞다.
    """
    csv = pd.read_csv(f"data/{stem}.csv",
                      usecols=[f"ih{h}" for h in orders]
                      + [f"ihdeg{h}" for h in orders])
    j = np.clip(idx - sh, 0, len(csv) - 1)
    cols = []
    for h in orders:
        I = csv[f"ih{h}"].values[j] * np.exp(1j * np.deg2rad(csv[f"ihdeg{h}"].values[j]))
        zi = (R + 1j * h * X1) * I
        cols += [zi.real, zi.imag]
    return np.c_[np.ones(len(j)), np.array(cols).T]


def _cols(sig: np.ndarray, idx) -> np.ndarray:
    return np.array([np.concatenate([sig[j, ODD, 0], sig[j, ODD, 1]])
                     for j in idx]).T


def residuals(d: dict, sig: np.ndarray, mask: np.ndarray, ref: float) -> tuple:
    """프로젝터만 참값에 못 박고 나머지를 NNLS. 남는 것이 잔차다."""
    pj = APPS.index("beam_projector")
    c1 = np.concatenate([sig[pj, ODD, 0], sig[pj, ODD, 1]])
    R, X = [], []
    for w in np.flatnonzero(mask):
        sup = [j for j in np.flatnonzero(d["on"][w]) if j != pj]
        has_pj = bool(d["on"][w, pj])
        b = np.concatenate([d["y"][w][ODD, 0], d["y"][w][ODD, 1]])
        pw = d["p"][w]
        if has_pj:
            b = b - ref * c1; pw = pw - ref
        if not sup:
            R.append(b); X.append(np.c_[1.0, d["V"][w][ODD][None]].ravel()); continue
        A = _cols(sig, sup)
        g = np.linalg.norm(A) / max(np.sqrt(len(sup)) * 50.0, 1e-9)
        x, _ = nnls(np.vstack([A, g * np.ones((1, len(sup)))]),
                    np.concatenate([b, [g * pw]]))
        R.append(b - A @ x)
        X.append(np.concatenate([[1.0], d["V"][w][ODD]]))
    return np.array(R), np.array(X)


def solve3(d: dict, sig: np.ndarray, mask: np.ndarray, corr) -> np.ndarray:
    """SMPS 3종을 나눈다. 프로젝터를 안 박는다 — 그것이 채점 대상이다."""
    j3 = [APPS.index(x) for x in ("beam_projector", "laptop_charger", "minipc")]
    A = _cols(sig, j3)
    out = []
    for k, w in enumerate(np.flatnonzero(mask)):
        b = np.concatenate([d["y"][w][ODD, 0], d["y"][w][ODD, 1]])
        if corr is not None:
            b = b - corr[k]
        g = np.linalg.norm(A) / max(np.sqrt(3) * 50.0, 1e-9)
        out.append(nnls(np.vstack([A, g * np.ones((1, 3))]),
                        np.concatenate([b, [g * d["p"][w]]]))[0])
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--max-fit", type=int, default=6000, help="적합 창 상한 (속도)")
    ap.add_argument("--exclude", default="test_9,test_11,test_12", metavar="STEMS",
                    help="적합에서 뺄 파일 (쉼표). **대조 파일을 빼야 규칙 20 이 산다** — "
                         "대조에서 적합한 계수로 대조 유령을 채점하면 대조가 아니다. "
                         "빈 문자열이면 전 파일")
    ap.add_argument("--vsrc-cols", type=int, default=0, metavar="N",
                    help="**배경 전압 `V_src` 를 회귀자로** 넣는다 (12.149.1). 낮은 "
                         "홀수차부터 N개 (h1, h3, …). 지금 절편은 자유 상수라 "
                         "`−(ΣY)·V_src` 를 뭉뚱그리는데, 그 값이 적합 8파일에서 "
                         "V_src,3 4.12~5.03V 로 좁다. 다른 장소는 10.09V 로 **범위 "
                         "밖**이라(12.148.2 전이 시험) 상수로는 틀린 값을 더하게 된다. "
                         "⚠ `V_src` 는 파일 안에서 상수라 **파일 사이 변동만** 쓴다 — "
                         "적합 파일 수보다 작게 둘 것. 8파일이면 N<=2 다. 0 이면 끔")
    ap.add_argument("--save-coef", default="", metavar="NPZ",
                    help="적합한 계수 A 를 저장한다 (`run_adapt --harm-offset` 이 쓴다)")
    ap.add_argument("--h1-vnorm", type=float, default=0.0, metavar="VREF",
                    help="h1 지문에서 녹화 전압을 나눈다 (12.151.1). 값이 기준 전압 "
                         "(실측 창 중앙 221.5V). 0 이면 끔")
    ap.add_argument("--out", default="results/norton.json")
    a = ap.parse_args()

    from src.model.net import harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, APPS); nz = noise_signature(pool)
    del pool
    if a.h1_vnorm > 0:
        # ── h1 녹화전압 정규화 (12.151.1) — 이 자로 먼저 판정한다 ──────────
        # `Re(I₁)/P = 1/V₁` 은 유효전력의 정의다. 그러면 `1/Re(sig[k,1,0])` 은
        # 기기의 성질이 아니라 **그 격리 녹화의 선전압**이고, 기기 사이 11.8% 의
        # 가짜 판별자가 된다. 여기서 나눠 버리면 전부 `1/V_ref` 로 같아진다.
        vk = 1.0 / np.maximum(sig[:, 0, 0].astype(np.float64), 1e-9)
        f = (vk / a.h1_vnorm).astype(np.float32)
        sig = sig.copy(); sig[:, 0, :] *= f[:, None]
        print(f"  ** h1 녹화전압 정규화 V_ref={a.h1_vnorm:.1f}V: "
              + " ".join(f"{APPS[j][:6]} {vk[j]:.0f}V" for j in range(len(APPS))) + " **")
    pj = APPS.index("beam_projector")
    ref = REFERENCE_W["beam_projector"][0]
    res_j = [APPS.index(x) for x in RESIST]

    ev = load_events()
    stems = [s for s in sorted(ev) if not s.startswith("_")
             and os.path.exists(f"data/{s}.csv")
             and os.path.exists(f"processed_data/composite_eval/{s}.npz")]
    print("=" * 96)
    print("교차주파수 어드미턴스 — **전 파일**로 적합, 파일 홀드아웃 (12.148)")
    print("=" * 96)
    D: Dict[str, dict] = {}
    for s in stems:
        try:
            D[s] = load(s, a.stride, nz)
        except Exception as e:                      # 원자료가 없거나 어긋나면 건너뛴다
            print(f"  ⚠ {s} 건너뜀: {e}")
    excl = {x for x in a.exclude.split(",") if x}
    print(f"  파일 {len(D)}개: {', '.join(D)}")
    if excl:
        print(f"  적합에서 뺀 파일: {sorted(excl & set(D))}  (규칙 20)")
    print(f"  vrms 정렬 상관: " + ", ".join(f"{k} {v['align']:.3f}" for k, v in D.items()))

    # ── 계통 임피던스 (12.148.2) ────────────────────────────────────────
    # `run_fit_impedance` 가 정본이다 — 거기 진단(계단 수·|I| 폭·RMS)이 붙어 있고
    # 새 장소에서 단독으로 돌릴 수 있다. 여기서는 계수와 `V_src` 만 받는다.
    from src.run_fit_impedance import fit_natural, load_blocks
    fit_stems = [s for s in D if s not in excl]
    _blk = {s: load_blocks(s) for s in fit_stems}
    R_s, X1_s, VSRC, _rms = fit_natural(_blk)
    print(f"\n  계통 임피던스 (물리 제약 Z_h = R + j·h·ωL, {len(fit_stems)}파일 공동):")
    if a.vsrc_cols > 0:
        nf = len(fit_stems)
        print(f"  ** V_src 회귀자 {a.vsrc_cols}열 (h{', h'.join(str(h) for h in ORD[:a.vsrc_cols])}) **")
        if a.vsrc_cols + 1 > nf - 1:
            print(f"     ⚠ 적합 파일이 {nf}개인데 절편+V_src 가 {a.vsrc_cols + 1}열이다 — "
                  f"**파일별 더미와 축퇴**한다. 전이가 안 된다.")
        for s_ in fit_stems:
            print("     " + f"{s_:<10s}"
                  + "  ".join(f"h{h} {abs(VSRC[(s_, h)]):7.2f}" for h in ORD[:a.vsrc_cols]))
    print(f"    R = {R_s:.4f} Ω   L = {X1_s / (2 * np.pi * 60) * 1e6:.0f} µH"
          f"   |Z|: h1 {abs(R_s + 1j * X1_s):.2f} -> h15 {abs(R_s + 15j * X1_s):.2f} Ω")

    def _design(stem: str, d: dict, m: np.ndarray) -> np.ndarray:
        """`[1, (V_src…), Re(Z·I), Im(Z·I)…]`. **적합과 평가가 같은 함수를 쓴다.**"""
        X = zi_design(stem, d["cycle"][np.flatnonzero(m)], R_s, X1_s, d["shift"])
        if a.vsrc_cols <= 0:
            return X
        # `V_src` 는 창마다 같다 — 파일 사이 변동만 정보다 (그래서 열 수를 제한한다)
        vs = (np.array([abs(VSRC[(stem, h)]) for h in ORD[:a.vsrc_cols]])
              if (stem, ORD[0]) in VSRC else np.zeros(a.vsrc_cols))
        return np.c_[X[:, :1], np.tile(vs, (len(X), 1)), X[:, 1:]]

    # 파일별 적합 자료
    FIT = {}
    for s, d in D.items():
        m = (d["on"].sum(1) > 0) & d["ok"]
        if m.sum() > a.max_fit:                     # 균등 솎기
            ix = np.flatnonzero(m)
            keep = np.zeros(len(m), bool)
            keep[ix[np.linspace(0, len(ix) - 1, a.max_fit).astype(int)]] = True
            m = keep
        Rr, _Xv = residuals(d, sig, m, ref)
        FIT[s] = (Rr, _design(s, d, m))
    print("\n  적합 창: " + ", ".join(f"{k} {len(v[0])}" for k, v in FIT.items()))

    print(f"\n  {'평가 파일':<10s}{'적합 파일 수':>11s}{'보정 없음':>10s}{'전역 상수':>11s}"
          f"{'**복소 Z·I**':>14s}{'충전기':>8s}{'미니PC':>8s}")
    print("  " + "-" * 74)
    out: dict = {"_config": {"argv": sys.argv}}
    agg = {"none": [], "const": [], "vh": []}
    for tgt in EVAL_STEMS:
        if tgt not in D:
            continue
        use = [s for s in FIT if s != tgt and s not in excl]
        Rf = np.vstack([FIT[s][0] for s in use])
        Xf = np.vstack([FIT[s][1] for s in use])
        d = D[tgt]
        m = ((d["on"].sum(1) > 0) & (~d["on"][:, res_j].any(1))
             & d["on"][:, pj] & d["ok"])
        Xt = _design(tgt, d, m)
        mu, sd = Xf.mean(0), Xf.std(0)
        sd[sd < 1e-12] = 1.0; mu[0] = 0.0; sd[0] = 1.0     # 절편은 그대로 둔다
        B, *_ = np.linalg.lstsq((Xf - mu) / sd, Rf, rcond=None)
        B0, *_ = np.linalg.lstsq(Xf[:, :1], Rf, rcond=None)
        r = {}
        for nm, c in (("none", None), ("const", Xt[:, :1] @ B0),
                      ("vh", ((Xt - mu) / sd) @ B)):
            X = solve3(d, sig, m, c)
            r[nm] = X
            agg[nm].append(X[:, 0].mean() - ref)
        print(f"  {tgt:<10s}{len(use):>11d}{r['none'][:, 0].mean() - ref:>+10.1f}"
              f"{r['const'][:, 0].mean() - ref:>+11.1f}{r['vh'][:, 0].mean() - ref:>+14.1f}"
              f"{r['vh'][:, 1].mean():>8.1f}{r['vh'][:, 2].mean():>8.1f}")
        out[tgt] = {k: {"proj_w": float(v[:, 0].mean()),
                        "lc_w": float(v[:, 1].mean()),
                        "mp_w": float(v[:, 2].mean())} for k, v in r.items()}
    print("  " + "-" * 74)
    print(f"  {'평균 |오차|':<21s}"
          + "".join(f"{np.mean(np.abs(agg[k])):>11.1f}" for k in ("none", "const"))
          + f"{np.mean(np.abs(agg['vh'])):>14.1f}")
    if a.save_coef:
        # 배포용 계수는 **평가 파일도 넣어** 전부로 적합한다 (홀드아웃은 위 표가 낸다).
        use = [s for s in FIT if s not in excl]
        Rf = np.vstack([FIT[s][0] for s in use]); Xf = np.vstack([FIT[s][1] for s in use])
        mu, sd = Xf.mean(0), Xf.std(0)
        sd[sd < 1e-12] = 1.0; mu[0] = 0.0; sd[0] = 1.0
        B, *_ = np.linalg.lstsq((Xf - mu) / sd, Rf, rcond=None)
        np.savez(a.save_coef, coef=B.astype(np.float32), orders=ODD,
                 mu=mu.astype(np.float64), sd=sd.astype(np.float64),
                 R=R_s, X1=X1_s, zi_orders=np.array([1, 3, 5, 7, 9, 11, 13, 15]),
                 stems=np.array(sorted(use)), excluded=np.array(sorted(excl)),
                 n_windows=int(len(Xf)), argv=np.array(sys.argv))
        print(f"\n  계수 저장: {a.save_coef}  (적합 {len(use)}파일 {len(Xf)}창, "
              f"A 는 {B.shape[0]}x{B.shape[1]})")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n  저장: {a.out}")
    print("\n  ⚠ 판정은 **배분 오차**로 한다. 잔차 설명력으로 하지 않는다 (12.147).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
