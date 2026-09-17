# -*- coding: utf-8 -*-
"""회전을 **녹화 안(부하)** 과 **녹화 간** 으로 가른다 (14.394).

§51 이 잰 "상태 안 회전" 은 녹화를 **섞어서** 쟀다. 그러면 고전력 사이클이 특정 녹화에
몰려 있을 때 **녹화 회전을 부하 회전으로 잘못 읽는다**. §47.5 는 정확히 그 반대를 말한다 —
*"충전기의 '상수가 못 담는 폭' 은 전력 의존이 아니라 **녹화 회전**이었다"*.

둘이 부딪치므로 갈라서 잰다. 이 수가 다음 판을 고른다:
```
  녹화 **안**에서도 k 가 살아 있다   -> `a_k` 가 **적합 가능**하다 (§48.3 을 짓는다)
  녹화 안에서 죽는다                -> 회전은 조각마다 무작위 -> `a_k` 로는 못 고친다
                                      (§47.6 ① 위상 기준 정렬이나 불감대로 간다)
```
`coh`(복소 중앙값이 잃는 크기)도 같은 방식으로 가른다 — 녹화 안에서 이미 잃으면
**재굽기로 되찾을 수 있고**, 녹화 간에서만 잃으면 기준을 옮겨야 한다.

⚠ `run_diag_sigstate`(상태 안 부하) · `run_diag_sigfloor`(녹화 간 재현성) 와 헷갈리지 마라.
  이것은 **그 둘을 같은 표에서 갈라 놓는** 자다.

    python -X utf8 -m src.run_diag_rotsplit
"""
from typing import Dict, List

import numpy as np

from src import env_guard  # noqa: F401

from src.model.net import state_power_w  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

HS = np.array([1, 3, 5, 7, 9, 11, 13, 15], float)
WHO = (("minipc", 2), ("laptop_charger", 2), ("beam_projector", 2),
       ("oven", 2), ("hair_dryer", 2), ("fan", 1))
#: 녹화 하나가 `k` 를 낼 수 있으려면 **전력이 실제로 움직여야** 한다
MIN_RATIO = 1.15
MIN_CYC = 300


def rec_of(a) -> str:
    return str(getattr(a, "source_file", getattr(a, "recording", "?")))


def med_sig(z: np.ndarray) -> np.ndarray:
    return np.median(z.real, 0) + 1j * np.median(z.imag, 0)


def k_fit(s_lo: np.ndarray, s_hi: np.ndarray, skip_h1: bool = True):
    """`Δ∠(h) = k·h` 를 절편 없이 적합 — 순수 지연이면 이 꼴이다.

    ⚠ **h1 은 뺀다** (`skip_h1`). h1 의 녹화 간 변동은 지연이 아니라 다른 기전이고
    (저항 전류가 지배한다), 넣으면 적합이 h1 쪽으로 끌려 **h1 을 잘못 돌린다** —
    정렬이 h1 의 `irr` 을 0.032 -> 0.060 으로 **나쁘게** 만들었다.
    `run_gate_phaselaw.k_of` 도 h3~h13 만 쓴다. 같은 규약이다.
    """
    hh, rr = [], []
    for j, h in enumerate(HS if not skip_h1 else HS[HS > 1]):
        i = int(h) - 1
        if abs(s_lo[i]) > 0 and abs(s_hi[i]) > 0:
            hh.append(h)
            rr.append(np.degrees(np.angle(s_hi[i] / s_lo[i])))
    hh, rr = np.asarray(hh), np.asarray(rr)
    if len(hh) < 4:
        return float("nan"), float("nan")
    k = float((hh @ rr) / (hh @ hh))
    r2 = float(1 - ((rr - k * hh) ** 2).sum() / max(((rr - rr.mean()) ** 2).sum(), 1e-12))
    return k, r2


def main() -> int:
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    print("회전 가르기 (14.394) — **녹화 안(부하)** 대 **녹화 간**\n")

    for app, sid in WHO:
        by: Dict[str, List] = {}
        for a in pool.appliance_activations.get(app, []):
            st = getattr(a, "state_id", None)
            if st is None:
                continue
            pw = state_power_w(a, False)
            m = (pw > 1.0) & (np.asarray(st) == sid)
            if m.any():
                by.setdefault(rec_of(a), [[], []])
                by[rec_of(a)][0].append(np.asarray(a.net_harmonics_complex)[m])
                by[rec_of(a)][1].append(pw[m])
        recs = {r: (np.concatenate(c), np.concatenate(p)) for r, (c, p) in by.items()}
        recs = {r: v for r, v in recs.items() if len(v[1]) >= MIN_CYC}
        if not recs:
            print("  %-16s 표본 부족\n" % app)
            continue

        print("  %s s%d — 녹화 %d개" % (app, sid, len(recs)))
        print("    녹화                     n     전력(W)      폭    k(도/차수)   R²    coh_h13")
        ks, cohs, ref = [], [], {}
        for r, (c, p) in sorted(recs.items()):
            per_w = c / np.maximum(p, 1e-6)[:, None]
            q = np.percentile(p, [25, 35, 65, 75])
            lo = (p >= q[0]) & (p <= q[1])
            hi = (p >= q[2]) & (p <= q[3])
            ratio = np.median(p[hi]) / max(np.median(p[lo]), 1e-9)
            ref[r] = med_sig(per_w)
            coh13 = abs(ref[r][12]) / max(np.median(np.abs(per_w[:, 12])), 1e-12)
            cohs.append(coh13)
            if ratio >= MIN_RATIO and lo.sum() >= 50 and hi.sum() >= 50:
                k, r2 = k_fit(med_sig(per_w[lo]), med_sig(per_w[hi]))
                ks.append((k, r2))
                kt = "%+8.2f %7.3f" % (k, r2)
            else:
                kt = "   폭 좁아 못 잼"
            print("    %-22s %6d  %5.1f~%-5.1f  x%.2f  %s  %6.3f"
                  % (r[-22:], len(p), p.min(), p.max(), ratio, kt, coh13))

        if ks:
            kk = np.array([k for k, _ in ks])
            rr = np.array([r for _, r in ks])
            print("    ⇒ **녹화 안** k 중앙 **%+.2f** (범위 %+.2f~%+.2f · n=%d) · R² 중앙 %.3f"
                  % (np.median(kk), kk.min(), kk.max(), len(kk), np.median(rr)))
        else:
            print("    ⇒ **녹화 안** 은 전력 폭이 좁아 못 잰다")

        if len(ref) >= 2:
            R = np.stack([ref[r] for r in sorted(ref)])
            base = med_sig(R)
            bk = [k_fit(base, R[i])[0] for i in range(len(R))]
            bk = np.array([x for x in bk if np.isfinite(x)])
            print("    ⇒ **녹화 간** k 산포 σ **%.2f** (범위 %+.2f~%+.2f)"
                  % (bk.std(), bk.min(), bk.max()))
        print("    ⇒ coh_h13  녹화 안 중앙 **%.3f** · 통합 %.3f"
              % (np.median(cohs),
                 abs(med_sig(np.concatenate([recs[r][0] / np.maximum(recs[r][1], 1e-6)[:, None]
                                             for r in recs]))[12])
                 / max(np.median(np.abs(np.concatenate(
                     [recs[r][0] / np.maximum(recs[r][1], 1e-6)[:, None]
                      for r in recs])[:, 12])), 1e-12)))
        print()
    decompose(pool)
    return 0


# ── 2부: **바닥의 분해** — 고치면 얼마나 줄어드나 ─────────────────────────────
#: `L_harm` 이 참 배분에서도 못 줄이는 상대 잔차 `irr` 을 세 몫으로 가른다:
#:    지금      기기x상태 상수 하나 (오늘 학습이 쓰는 것)
#:    녹화정렬  녹화마다 **h비례 회전 하나**를 맞춘 뒤 — **사전 재굽기**로 되찾는 몫
#:    +부하     거기서 `a_k·(lnP − lnP_ref)` 까지 뺀 뒤 — **§48.3** 이 되찾는 몫
#:    남은 것   사이클 잡음 = **진짜 바닥**
#: ⚠ 전체 위상은 **안 옮긴다** (녹화별 회전에서 중앙값을 빼므로). 기기 사이 상대
#:   위상이 유지돼야 `Σ_k P_k·sig_k` 가 그대로다.
def decompose(pool):
    print("=" * 78)
    print("  바닥의 분해 — irr = 중앙 |per_w − sig| / 중앙 |per_w| (참 배분에서)")
    print()
    for app, sid in WHO:
        by = {}
        for a in pool.appliance_activations.get(app, []):
            st = getattr(a, "state_id", None)
            if st is None:
                continue
            pw = state_power_w(a, False)
            m = (pw > 1.0) & (np.asarray(st) == sid)
            if m.any():
                by.setdefault(rec_of(a), [[], []])
                by[rec_of(a)][0].append(np.asarray(a.net_harmonics_complex)[m])
                by[rec_of(a)][1].append(pw[m])
        recs = {r: (np.concatenate(c), np.concatenate(p)) for r, (c, p) in by.items()}
        recs = {r: v for r, v in recs.items() if len(v[1]) >= MIN_CYC}
        if len(recs) < 2:
            print("  %-16s 녹화 %d개 — 분해 못 함" % (app, len(recs)))
            print()
            continue
        W = {r: c / np.maximum(p, 1e-6)[:, None] for r, (c, p) in recs.items()}
        P = {r: p for r, (c, p) in recs.items()}
        idx = [int(h) - 1 for h in HS]
        sig_pool = med_sig(np.concatenate([W[r] for r in sorted(W)]))

        rot = {}
        for r in W:
            k, _ = k_fit(sig_pool, med_sig(W[r]))
            rot[r] = 0.0 if not np.isfinite(k) else k
        km = float(np.median(list(rot.values())))
        #: ⚠ **h1 은 안 돌린다.** 순수 지연이면 h1 도 k·1 만큼 돌아야 하는데, 재 보니
        #:   h1 의 녹화 간 변동은 그 회전보다 **훨씬 작다** (충전기 irr 0.032). 돌리면
        #:   없는 회전을 만들어 0.060 으로 **두 배** 나빠진다. 고차만 돌린다.
        _hv = np.arange(1, list(W.values())[0].shape[1] + 1).astype(float)
        _hv[0] = 0.0
        alg = {r: W[r] * np.exp(-1j * np.radians((rot[r] - km) * _hv))[None] for r in W}
        sig_rec = med_sig(np.concatenate([alg[r] for r in sorted(alg)]))

        a_k = []
        for r in W:
            q = np.percentile(P[r], [25, 35, 65, 75])
            lo = (P[r] >= q[0]) & (P[r] <= q[1])
            hi = (P[r] >= q[2]) & (P[r] <= q[3])
            ratio = np.median(P[r][hi]) / max(np.median(P[r][lo]), 1e-9)
            if ratio >= MIN_RATIO and lo.sum() >= 50 and hi.sum() >= 50:
                k, _ = k_fit(med_sig(alg[r][lo]), med_sig(alg[r][hi]))
                if np.isfinite(k):
                    a_k.append(k / np.log(ratio))
        A = float(np.median(a_k)) if a_k else 0.0
        #: ⚠⚠ 기준 전력은 **녹화마다** 잡는다. 통합 중앙값을 쓰면 `a_k·(lnP − lnP_ref)`
        #:   가 녹화의 **평균 전력 차이**를 다시 회전으로 넣어 **정렬을 되돌린다**
        #:   (충전기 판별차수가 0.415 -> 0.425 로 나빠졌다). 녹화 안에서 뺀 뒤 걸어야
        #:   둘이 **직교**한다 — 정렬은 녹화 간, 법칙은 녹화 안.
        hv = _hv                      # 부하 회전도 같은 규약 (h1 제외)
        ldr = {r: alg[r] * np.exp(-1j * np.radians(
                   A * (np.log(np.maximum(P[r], 1e-9))
                        - np.log(max(float(np.median(P[r])), 1e-9)))[:, None] * hv[None]))
               for r in W}
        sig_load = med_sig(np.concatenate([ldr[r] for r in sorted(ldr)]))

        def irr(Z, sg):
            out = []
            for i in idx:
                mag = np.median(np.abs(Z[:, i]))
                out.append(np.median(np.abs(Z[:, i] - sg[i])) / max(mag, 1e-12))
            return np.array(out)

        ip = irr(np.concatenate([W[r] for r in sorted(W)]), sig_pool)
        ir = irr(np.concatenate([alg[r] for r in sorted(alg)]), sig_rec)
        il = irr(np.concatenate([ldr[r] for r in sorted(ldr)]), sig_load)
        print("  %s s%d — 녹화 %d · a_k = **%.2f** (%s)"
              % (app, sid, len(recs), A,
                 ("폭 있는 녹화 %d개" % len(a_k)) if a_k else "**못 잼 -> 0**"))
        print("    차수 |  지금   녹화정렬   +부하   | 정렬이 준 것  부하가 준 것")
        for j, h in enumerate(HS):
            print("    h%-3d | %5.3f   %5.3f    %5.3f  |   %+5.1f%%       %+5.1f%%"
                  % (h, ip[j], ir[j], il[j],
                     100 * (ir[j] - ip[j]) / max(ip[j], 1e-9),
                     100 * (il[j] - ir[j]) / max(ip[j], 1e-9)))
        sel = [j for j, h in enumerate(HS) if h in (9, 11, 13)]
        print("    ⇒ **판별 차수 h9·h11·h13 평균**  %.3f -> %.3f -> **%.3f**  (총 %+.1f%%)"
              % (ip[sel].mean(), ir[sel].mean(), il[sel].mean(),
                 100 * (il[sel].mean() - ip[sel].mean()) / max(ip[sel].mean(), 1e-9)))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
