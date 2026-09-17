# -*- coding: utf-8 -*-
"""부하 위상 법칙을 **주기 단위(60Hz) 그대로** 잰다 (14.398, 사용자 지시).

*"비교할때 어중간하게 집계하지말고 그냥 60Hz 데이터 그대로 해서 비교해줘"*

⚠⚠ 지금까지 잰 것은 전부 **중앙값의 중앙값**이었다:
```
  §51·§52  전력을 분위수 둘(p20~30 대 p70~80)로 묶고, 각 묶음의 **중앙 페이저**끼리
           각도를 재서 k 를 냈다 -> 10만 주기에서 **점 둘**만 쓴 것이다
  §54      a 도 같은 두 점에서 나왔다
  ⇒ 묶음은 (가) 모드 경계(CV 테이퍼, 13.18.2)를 가로질러 평균 내고
    (나) 비선형을 못 보고 (다) 표본 수를 10만 -> 2 로 줄인다
```

여기서는 **묶지 않는다.** 주기마다 `(z_i, P_i)` 를 그대로 쓰고, 목적함수를
**손실이 보는 양 그 자체**로 둔다:
```
  R(a) = Σ_h  중앙_i |z_i,h − s_h(a)·exp(+j·h·a·x_i)|  /  중앙_i |z_i,h|
         단  x_i = ln P_i − ln P_ref
            s_h(a) = 실·허 **따로** 중앙값 of  z_i,h·exp(−j·h·a·x_i)   <- 손실의 그 추정기
  a* = argmin R(a)   (격자 훑기 — 선형화도 근사도 안 한다)
```
`s_h(a)` 를 `a` 마다 **다시 굽는** 것이 핵심이다. 안 그러면 상수 사전에 유리한
비교가 된다 ([[dont-loosen-a-gate-to-make-it-pass]] 의 반대 방향 함정).

★ 차수마다 `a_h` 를 따로도 낸다 — `a_h` 가 h 에 평평하면 **순수 지연**이 주기 수준에서
  선다는 뜻이고, 안 평평하면 §53.2① 의 배경 오염이 주기 수준에도 있다는 뜻이다.

⚠ `P_i` 자체가 잡음이라 **감쇠 편향**(errors-in-variables)이 있다. 주 측정은 **원자료**로
  하고, 참고로 `P` 만 1초(61주기) 중앙값 필터를 먹인 판을 같이 찍는다 — 비교는
  여전히 주기 단위다.

    python -X utf8 -m src.run_diag_cyclelaw
"""
import argparse
from typing import Tuple

import numpy as np

from src import env_guard  # noqa: F401

from src.model.net import harmonic_signatures_by_state, state_power_w  # noqa: E402
from src.model.sigalign import rec_of  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
#: SMPS 셋(상태별) + **대조군 저항** — 저항에서 a≈0 이 안 나오면 자가 틀린 것이다
CELLS = (("laptop_charger", 1), ("laptop_charger", 2), ("minipc", 1), ("minipc", 2),
         ("beam_projector", 1), ("beam_projector", 2), ("oven", 2), ("hotplate", 2),
         ("air_conditioner", 2), ("air_conditioner", 3))
ORD = (3, 5, 7, 9, 11, 13, 15)
DISC = (9, 11, 13)


def med_c(z: np.ndarray) -> np.ndarray:
    """실·허 **따로** 중앙값 — `harmonic_signatures_by_state` 가 쓰는 그 추정기."""
    return np.median(z.real, 0) + 1j * np.median(z.imag, 0)


def gather(pool, app: str, sid: int, smooth: int = 0):
    """그 칸의 **모든 주기**를 그대로 — 묶지 않는다."""
    zs, ps, rs = [], [], []
    for a in pool.appliance_activations.get(app, []):
        st = getattr(a, "state_id", None)
        if st is None:
            continue
        pw = state_power_w(a, False)
        if smooth > 1 and len(pw) >= smooth:
            k = smooth // 2 * 2 + 1
            pad = np.pad(pw, k // 2, mode="edge")
            pw = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), -1)
        m = (pw > 1.0) & (np.asarray(st) == sid)
        if m.any():
            zs.append(np.asarray(a.net_harmonics_complex)[m]
                      / np.maximum(pw[m], 1e-6)[:, None])
            ps.append(pw[m])
            rs.append(np.full(int(m.sum()), rec_of(a)))
    if not zs:
        return None
    return np.concatenate(zs), np.concatenate(ps), np.concatenate(rs)


def resid(z: np.ndarray, x: np.ndarray, a: float, orders) -> Tuple[np.ndarray, np.ndarray]:
    """`a` 를 걸었을 때의 **주기 단위** 상대 잔차 (차수별) 와 그때의 사전."""
    out, sig = [], np.zeros(z.shape[1], complex)
    for h in orders:
        i = h - 1
        rot = np.exp(-1j * np.radians(h * a) * x)
        s = np.median((z[:, i] * rot).real) + 1j * np.median((z[:, i] * rot).imag)
        sig[i] = s
        out.append(np.median(np.abs(z[:, i] - s * np.conj(rot)))
                   / max(np.median(np.abs(z[:, i])), 1e-12))
    return np.asarray(out), sig


def resid2(z, x, t, a, b, orders):
    """전력항과 시간항을 **같이** 건 주기 단위 잔차."""
    out = []
    for h in orders:
        i = h - 1
        rot = np.exp(-1j * np.radians(h) * (a * x + b * t))
        s = np.median((z[:, i] * rot).real) + 1j * np.median((z[:, i] * rot).imag)
        out.append(np.median(np.abs(z[:, i] - s * np.conj(rot)))
                   / max(np.median(np.abs(z[:, i])), 1e-12))
    return float(np.mean(out))


def ls_init(z, x, t, orders, iters: int = 4):
    """가중 최소제곱으로 `(a, b)` 를 **닫힌 꼴**로 낸다 — 격자의 출발점 (14.398d).

    `θ_i,h = ∠(z_i,h · conj(s_h))` 를 `h·(a·x_i + b·t_i)` 에 맞춘다. 2x2 정규방정식이라
    **한 번에 풀린다**. 되풀이하며 `s_h` 를 다시 굽고 남은 각도로 다시 맞춘다 —
    그러면 되감김(wrapping)이 첫 판 뒤에는 작아진다.

    ⚠ 이것으로 **답을 내지 않는다.** 전역 격자와 같은 답인지 확인하려고 그 주변만
    5x5 로 다듬고, 최종 잔차는 **중앙값 추정기로 전 주기**에서 낸다.
    """
    a = b = 0.0
    for _ in range(iters):
        A = np.zeros((2, 2)); r = np.zeros(2)
        for h in orders:
            i = h - 1
            rot = np.exp(-1j * np.radians(h) * (a * x + b * t))
            zz = z[:, i] * rot
            sh = np.median(zz.real) + 1j * np.median(zz.imag)
            if abs(sh) <= 0:
                continue
            th = np.degrees(np.angle(zz * np.conj(sh)))      # 남은 각도 (작다)
            w = abs(sh) * h                                  # 진폭 가중 x 차수
            A[0, 0] += w * h * np.dot(x, x); A[0, 1] += w * h * np.dot(x, t)
            A[1, 0] += w * h * np.dot(t, x); A[1, 1] += w * h * np.dot(t, t)
            r[0] += w * np.dot(x, th);       r[1] += w * np.dot(t, th)
        try:
            d = np.linalg.solve(A, r)
        except np.linalg.LinAlgError:
            break
        a += float(d[0]); b += float(d[1])
        if abs(float(d[0])) < 1e-3 and abs(float(d[1])) < 1e-3:
            break
    return a, b


def sweep2(z, x, t, orders, lim=30.0, n=31, refine=2, fast: bool = True):
    """(a, b). `fast` 면 최소제곱으로 출발해 **그 주변만** 다듬는다.

    ⚠ 전역 격자와 같은 답을 내는지 `--verify` 로 맞대 볼 수 있다. 목적함수는
    **똑같다** (중앙값 추정기 · 전 주기) — 바꾼 것은 **어디를 보느냐**뿐이다.
    """
    if fast:
        a0, b0 = ls_init(z, x, t, orders)
        sa = max(abs(a0), 1.0) * 0.25
        sb = max(abs(b0), 1.0) * 0.25
        alo, ahi, blo, bhi, n = a0 - sa, a0 + sa, b0 - sb, b0 + sb, 5
    else:
        alo, ahi, blo, bhi = -lim, lim, -lim, lim
    best = (0.0, 0.0, None)
    for _ in range(refine + 1):
        ga, gb = np.linspace(alo, ahi, n), np.linspace(blo, bhi, n)
        vals = np.array([[resid2(z, x, t, float(a), float(b), orders) for b in gb]
                         for a in ga])
        ia, ib = np.unravel_index(int(np.argmin(vals)), vals.shape)
        best = (float(ga[ia]), float(gb[ib]), float(vals[ia, ib]))
        sa, sb = ga[1] - ga[0], gb[1] - gb[0]
        alo, ahi, blo, bhi, n = ga[ia] - sa, ga[ia] + sa, gb[ib] - sb, gb[ib] + sb, 5
    return best


def sweep(z, x, orders, lo=-30.0, hi=30.0, n=121, refine=2):
    """격자 훑기 + 정밀화. **선형화 안 한다.**"""
    best = (0.0, None)
    for _ in range(refine + 1):
        grid = np.linspace(lo, hi, n)
        vals = [resid(z, x, float(a), orders)[0].mean() for a in grid]
        k = int(np.argmin(vals))
        best = (float(grid[k]), float(vals[k]))
        step = grid[1] - grid[0]
        lo, hi, n = grid[k] - step, grid[k] + step, 21
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smooth", type=int, default=0,
                    help="참고판 — P 에만 이 주기 수의 중앙값 필터 (0 = 원자료)")
    ap.add_argument("--per-order", action="store_true", help="차수마다 a_h 를 따로 낸다")
    #: ★ 14.398b — **귀무 대조**. 이득이 "전력 의존" 인지 그냥 **자유도 하나**인지 가른다.
    #  shuffle : x 를 무작위 치환 -> 주변분포는 그대로, **짝만 깬다**
    #  time    : x 를 **시간 지표**로 갈아 끼운다 (같은 척도) -> 전력이 아니라 표류인가
    #  ⚠ 오븐 a=+30.55 는 **격자 끝**이었다. 대조 없이 읽으면 없는 물리를 만든다
    ap.add_argument("--null", default="", choices=["", "shuffle", "time", "both"],
                    help="귀무 대조를 같이 찍는다")
    #: ★★ 14.398c — **공선 가르기.** 시간 귀무가 충전기 s2 에서 이득의 **95%** 를 냈다.
    #  충전기는 충전이 진행되면 전력이 **단조 감소**하므로 `ln P` 와 시간이 거의 같은
    #  회귀자다 — "부하 반응" 과 "표류" 를 이 자료로는 못 가른다
    #  ([[distinguish-drift-from-response]]). 둘을 **같이** 넣고 각자의 몫을 본다:
    #      θ_i,h = h·( a·x_i + b·t_i ),   x = lnP − lnP_ref,  t = 표준화 시간
    #  ⚠ 미니PC 는 CPU 부하라 전력이 **단조가 아니다** — 거기서 갈린다
    ap.add_argument("--joint", action="store_true",
                    help="(a, b) 2차원 격자로 전력과 시간을 같이 푼다")
    ap.add_argument("--grid", action="store_true",
                    help="느린 전역 격자를 쓴다 (기본은 최소제곱 출발 + 국소 다듬기)")
    ap.add_argument("--verify", action="store_true",
                    help="빠른 길과 전역 격자를 **맞대 본다** (같은 답이어야 한다)")
    a_ = ap.parse_args()

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig_state, used = harmonic_signatures_by_state(pool, APPS)
    print("부하 위상 법칙 — **주기 단위(60Hz) 그대로** (14.398)")
    print("  묶음 없음 · `s_h(a)` 를 a 마다 다시 구움 · 격자 훑기\n")

    for app, sid in CELLS:
        j = APPS.index(app)
        if not used[j, sid]:
            continue
        g = gather(pool, app, sid, a_.smooth)
        if g is None or len(g[1]) < 2000:
            continue
        z, p, rs = g
        xref = float(np.median(np.log(np.maximum(p, 1e-9))))
        x = np.log(np.maximum(p, 1e-9)) - xref
        a_hat, _ = sweep(z, x, ORD)
        r0 = resid(z, x, 0.0, ORD)[0]
        r1 = resid(z, x, a_hat, ORD)[0]
        d0 = resid(z, x, 0.0, DISC)[0].mean()
        d1 = resid(z, x, a_hat, DISC)[0].mean()
        nulls = []
        if a_.null in ("shuffle", "both"):
            rng = np.random.default_rng(20260917)
            gains = []
            for _ in range(3):
                xs = rng.permutation(x)
                aa, _u = sweep(z, xs, ORD)
                dd = resid(z, xs, aa, DISC)[0].mean()
                gains.append(100 * (dd - d0) / max(d0, 1e-9))
            nulls.append("치환 %+.1f%% (a %+.1f)" % (float(np.median(gains)), aa))
        if a_.null in ("time", "both"):
            #: 시간 지표를 **x 와 같은 척도**로 (표준편차를 맞춘다) — 자유도가 같아야 공정하다
            t = np.arange(len(x), dtype=float)
            t = (t - t.mean()) / max(t.std(), 1e-9) * max(x.std(), 1e-9)
            aa2, _u = sweep(z, t, ORD)
            dd2 = resid(z, t, aa2, DISC)[0].mean()
            nulls.append("시간 %+.1f%% (a %+.1f)" % (100 * (dd2 - d0) / max(d0, 1e-9), aa2))
        print("  %s s%d — 주기 **%d개** · 전력 %.1f~%.1fW (중앙 %.1f · 폭 x%.2f) · 녹화 %d"
              % (app, sid, len(p), p.min(), p.max(), np.exp(xref),
                 np.percentile(p, 95) / max(np.percentile(p, 5), 1e-9), len(set(rs.tolist()))))
        print("    ★ a = **%+.2f** 도/차수/ln W    판별차수 잔차 %.4f -> **%.4f** (%+.1f%%)"
              % (a_hat, d0, d1, 100 * (d1 - d0) / max(d0, 1e-9)))
        print("    차수별 잔차  " + " ".join("h%d %.3f->%.3f" % (h, x0, x1)
                                          for h, x0, x1 in zip(ORD, r0, r1)))
        if nulls:
            print("    ⚠ **귀무 대조**  " + " · ".join(nulls)
                  + "   (실제 %+.1f%%)" % (100 * (d1 - d0) / max(d0, 1e-9)))
        if a_.joint:
            t = np.arange(len(x), dtype=float)
            t = (t - t.mean()) / max(t.std(), 1e-9) * max(x.std(), 1e-9)
            rho = float(np.corrcoef(x, t)[0, 1])
            import time as _tm
            _t0 = _tm.time()
            aj, bj, dj_all = sweep2(z, x, t, ORD, fast=not a_.grid)
            #: ⚠⚠ **자기점검 — 포개진 모형이다.** `(a,b)` 는 `(a,0)` 과 `(0,b)` 를 둘 다
            #:   품으므로 같이 넣은 값이 혼자보다 **나쁠 수가 없다**. 나쁘면 탐색이
            #:   실패한 것이다 (최소제곱 출발점이 달아났다 — 빔 s1 에서 +184% 가 나왔다).
            #:   그때만 전역 격자로 되돌린다. 자를 늘리는 게 아니라 **실패를 잡는 것**이다.
            _a1 = sweep(z, x, ORD)[0]
            _b1 = sweep(z, t, ORD)[0]
            _floor = min(resid2(z, x, t, _a1, 0.0, ORD), resid2(z, x, t, 0.0, _b1, ORD))
            if dj_all > _floor + 1e-9:
                print("    ⚠ 빠른 길이 **달아났다** (%.4f > 단독 %.4f) — 전역 격자로 되돌린다"
                      % (dj_all, _floor))
                aj, bj, dj_all = sweep2(z, x, t, ORD, fast=False)
            _dt = _tm.time() - _t0
            if a_.verify:
                _t1 = _tm.time()
                ag, bg, dg = sweep2(z, x, t, ORD, fast=False)
                print("    ⚠ **맞대보기** 빠른 (a %+.2f b %+.2f %.4f, %.1f초) · "
                      "격자 (a %+.2f b %+.2f %.4f, %.1f초) · **%.0f배**"
                      % (aj, bj, dj_all, _dt, ag, bg, dg, _tm.time() - _t1,
                         (_tm.time() - _t1) / max(_dt, 1e-6)))
            dj = resid2(z, x, t, aj, bj, DISC)
            #: 각자 **혼자** 넣었을 때 (같은 격자·같은 추정기)
            a_only, b_only = _a1, _b1
            da = resid2(z, x, t, a_only, 0.0, DISC)
            db = resid2(z, x, t, 0.0, b_only, DISC)
            print("    ★★ **공선 가르기**  corr(lnP, 시간) = **%+.3f**" % rho)
            print("       전력만 a=%+6.2f -> %.4f (%+.1f%%) · 시간만 b=%+6.2f -> %.4f (%+.1f%%)"
                  % (a_only, da, 100 * (da - d0) / max(d0, 1e-9),
                     b_only, db, 100 * (db - d0) / max(d0, 1e-9)))
            print("       **같이**  a=%+6.2f b=%+6.2f -> %.4f (%+.1f%%)   "
                  "전력의 고유 몫 **%+.1f%%p**"
                  % (aj, bj, dj, 100 * (dj - d0) / max(d0, 1e-9),
                     100 * (dj - db) / max(d0, 1e-9)))
        if a_.per_order:
            per = [sweep(z, x, (h,))[0] for h in ORD]
            sp = float(np.std(per)) / max(abs(float(np.mean(per))), 1e-9)
            print("    차수별 a_h   " + " ".join("h%-2d %+6.2f" % (h, v)
                                              for h, v in zip(ORD, per))
                  + "   **평평함 %.2f**" % sp)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
