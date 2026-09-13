"""
차분 특징 3차 제안 — ⑥ σP · ⑦ 다단 강하 · ⑧ 총 역률 (설계 문서 12.57절)
==========================================================================
    ⑥ σP        정상 통전 중 1초 전력 표준편차. 충전기 5.7~8.8W / 미니PC 0.9~1.5 /
                프로젝터 0.2~0.5 라는 주장. **전이가 아니라 상태 특징이다**
    ⑦ 다단 강하  프로젝터 종료가 48.7 -> 5.4(냉각팬) -> 2.3W 로 계단이라는 주장
    ⑧ 총 역률    ΔP/ΔS. 미니PC 0.50 / 충전기 0.54~0.55 / 프로젝터 0.58 이라는 주장

    python -m src.run_steady_feature_probe

[⑥ 은 지금까지와 자가 다르다]
①~⑤ 는 전이 특징이라 "널(전이 없는 시각)보다 큰가" 를 물었다. ⑥ 은 **상태**
특징이므로 물음이 다르다 — **그 기기가 켜져 있을 때와 꺼져 있을 때가 갈리는가.**
그래서 `test_7` 에서 "충전기 ON" 을 σP 하나로 맞히는 AUC 를 잰다.

[⑧ 은 이미 잰 값의 단조 변환일 수 있다]
`PF = P/S = 1/sqrt(1 + (Q/P)²)` 이고 파일의 `Q = sqrt(S²−P²)` 다. 12.55.1 에서
그 `Q/P` 를 쟀다 — 프로젝터 -1.314 / 충전기 -1.316 / 미니PC -1.416, 분리비 0.16.
**단조 변환은 분리비를 안 바꾼다.** 그래도 직접 확인한다.

[⚠ ΔS 는 복소 차분에서 뽑아야 한다]
`I_rms` 는 중첩에서 선형이 아니다 (`rms(a+b) != rms(a)+rms(b)`). 복합 전이에서
`I_rms(after) − I_rms(before)` 는 그 기기의 rms 가 **아니다.** 복소 고조파는
선형이므로 `ΔI_rms = sqrt(Σ_h |ΔI_h|²)` 로 뽑아야 한다. 둘 다 찍어 비교한다.
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.evaluation.real_events import load_events
from src.model.realdata import DEFAULT_DIR
from src.run_delta_feature_probe import (DEV_DIR, DEV_FILES, SMPS, load_cplx,
                                         sepratio, transitions)
from src.run_dq_dphi_probe import RESIST, ALL
from src.run_live import KOR


def rolling_std(p: np.ndarray, win: int) -> np.ndarray:
    """길이 win 의 이동 표준편차 (중심). cumsum 이라 O(T)."""
    n = len(p)
    a = p.astype(np.float64)
    c1 = np.concatenate([[0.0], np.cumsum(a)])
    c2 = np.concatenate([[0.0], np.cumsum(a * a)])
    lo = np.clip(np.arange(n) - win // 2, 0, n)
    hi = np.clip(lo + win, 0, n)
    k = np.maximum(hi - lo, 1)
    m = (c1[hi] - c1[lo]) / k
    v = (c2[hi] - c2[lo]) / k - m * m
    return np.sqrt(np.maximum(v, 0.0))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    a = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(a)
    rk = np.empty(len(a))
    rk[o] = np.arange(1, len(a) + 1)
    return float((rk[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mask_from(pairs, n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for s, e in pairs or []:
        m[int(float(s) * 60):int(float(e) * 60)] = True
    return m


def sigma_isolated(win: int = 60) -> Dict[str, np.ndarray]:
    """⑥ 격리 녹화: 통전 중 1초 σP."""
    out = {}
    for app, files in DEV_FILES.items():
        V = []
        for stem in files:
            f = DEV_DIR / f"{stem}.npz"
            if not f.exists():
                continue
            _, pf = load_cplx(f)
            p = pf[:, 0]
            hi = np.percentile(p, 90)
            on = p > max(0.5 * hi, p.min() + 0.3 * (hi - p.min()))
            if on.sum() < 600:
                continue
            V.append(rolling_std(p, win)[on])
        if V:
            out[app] = np.concatenate(V)
    return out


def sigma_composite(win: int = 60):
    """⑥ 복합: σP 가 '그 기기 ON' 을 맞히는가. test_7 은 SMPS 만이라 깨끗하다."""
    ev = load_events()
    res = {}
    for stem in ("test_7", "test_5", "test_6"):
        f = Path(DEFAULT_DIR) / f"{stem}.npz"
        if not f.exists():
            continue
        _, pf = load_cplx(f)
        p = pf[:, 0]
        n = len(p)
        sd = rolling_std(p, win)
        iv = ev[stem]["intervals"]
        present = ev[stem].get("appliances_present", [])
        for app in SMPS:
            if app not in present or not iv.get(app, {}).get("on"):
                continue
            m = mask_from(iv[app]["on"], n)
            if m.sum() < 300 or (~m).sum() < 300:
                continue
            res.setdefault(stem, {})[app] = {
                "auc": auc(sd[m], sd[~m]),
                "on_med": float(np.median(sd[m])), "off_med": float(np.median(sd[~m])),
                "n_on": int(m.sum()), "n_off": int((~m).sum())}
    return res


def turnoff_tail():
    """⑦ 프로젝터 종료 시퀀스 — 격리와 복합에서 계단이 보이는가."""
    ev = load_events()
    out = {"isolated": [], "composite": []}
    for stem in ("beam_projector", "beam_projector_2"):
        f = DEV_DIR / f"{stem}.npz"
        if not f.exists():
            continue
        _, pf = load_cplx(f)
        p = pf[:, 0]
        n = len(p)
        hi = np.percentile(p, 90)
        on = p > 0.5 * hi
        idx = np.flatnonzero(on)
        if not len(idx):
            continue
        k = idx[-1]
        prof = [(round(d, 1), float(np.median(p[min(n - 1, k + int(d * 60)):
                                                min(n, k + int(d * 60) + 60)])))
                for d in (-2, 1, 5, 15, 30, 60, 120, 240) if k + int(d * 60) + 60 <= n]
        out["isolated"].append({"file": stem, "profile": prof,
                                "tail_s": float((n - k) / 60)})
    for stem in ("test_5", "test_6", "test_7"):
        f = Path(DEFAULT_DIR) / f"{stem}.npz"
        if not f.exists():
            continue
        _, pf = load_cplx(f)
        p = pf[:, 0]
        n = len(p)
        for s0, e0 in (ev[stem]["intervals"].get("beam_projector", {}).get("on") or []):
            k = int(float(e0) * 60)
            if k + 60 * 60 > n:
                continue
            prof = [(d, float(np.median(p[k + int(d * 60):k + int(d * 60) + 60])))
                    for d in (-2, 1, 5, 15, 30, 60)]
            out["composite"].append({"stem": stem, "t_s": float(e0), "profile": prof})
    return out


def true_pf():
    """⑧ ΔP/ΔS. 격리는 상태값, 복합은 복소 차분에서."""
    iso, comp = {}, {}
    for app, files in DEV_FILES.items():
        V = []
        for stem in files:
            f = DEV_DIR / f"{stem}.npz"
            if not f.exists():
                continue
            I, pf = load_cplx(f)
            p, v = pf[:, 0], pf[:, 4]
            hi = np.percentile(p, 90)
            on = p > max(0.5 * hi, p.min() + 0.3 * (hi - p.min()))
            if on.sum() < 300 or (~on).sum() < 100:
                continue
            bgI = np.median(I[~on].real, 0) + 1j * np.median(I[~on].imag, 0)
            bgp = float(np.median(p[~on]))
            for blk in np.array_split(np.flatnonzero(on), max(1, int(on.sum()) // 600)):
                if len(blk) < 200:
                    continue
                dI = (np.median(I[blk].real, 0) + 1j * np.median(I[blk].imag, 0)) - bgI
                dp = float(np.median(p[blk])) - bgp
                vv = float(np.median(v[blk]))
                ds = vv * float(np.sqrt((np.abs(dI) ** 2).sum()))
                if dp <= 1 or ds <= 1e-6:
                    continue
                V.append(dp / ds)
        if V:
            iso[app] = np.array(V)

    ev = load_events()
    for stem in ("test_4", "test_5", "test_6", "test_7"):
        f = Path(DEFAULT_DIR) / f"{stem}.npz"
        if not f.exists():
            continue
        I, pf = load_cplx(f)
        p, v = pf[:, 0], pf[:, 4]
        n = len(p)
        t = np.arange(n) / 60.0
        i3 = np.abs(I[:, 2])
        T = transitions(ev, stem, ALL)
        for s0, e0 in (ev[stem]["intervals"].get("oven", {}).get("_heater_pulses") or []):
            T += [(float(s0), "oven", +1), (float(e0), "oven", -1)]
        T.sort()
        tt = np.array([x[0] for x in T]) if T else np.zeros(0)
        for i, (t0, app, sg) in enumerate(T):
            gap = min(t0 - tt[i - 1] if i > 0 else 99,
                      tt[i + 1] - t0 if i < len(tt) - 1 else 99)
            h = float(np.clip(0.45 * gap, 0.25, 5.0))
            g = min(1.0, 0.25 * h)
            sn = min(6.0, 0.5 * gap)
            if t0 < h + sn + 1 or t0 > t[-1] - h - sn - 1:
                continue
            best, bd = t0, -1.0
            for c in np.arange(t0 - sn, t0 + sn + 1e-9, 0.25):
                pre = (t >= c - h) & (t <= c - g)
                post = (t >= c + g) & (t <= c + h)
                if pre.sum() < 8 or post.sum() < 8:
                    continue
                d = abs(np.median(i3[post]) - np.median(i3[pre]))
                if d > bd:
                    best, bd = c, d
            pre = (t >= best - h) & (t <= best - g)
            post = (t >= best + g) & (t <= best + h)
            if pre.sum() < 8 or post.sum() < 8:
                continue
            dI = ((np.median(I[post].real, 0) + 1j * np.median(I[post].imag, 0))
                  - (np.median(I[pre].real, 0) + 1j * np.median(I[pre].imag, 0)))
            dp = float(np.median(p[post]) - np.median(p[pre]))
            vv = float(np.median(v[post]))
            ds_c = vv * float(np.sqrt((np.abs(dI) ** 2).sum()))       # 복소 차분 (옳다)
            irms = np.sqrt((np.abs(I) ** 2).sum(1))
            ds_n = vv * float(np.median(irms[post]) - np.median(irms[pre]))  # 순진한 차
            if abs(dp) < 5 or ds_c <= 1e-6:
                continue
            comp.setdefault(app, []).append((abs(dp) / ds_c,
                                             abs(dp) / abs(ds_n) if abs(ds_n) > 1e-6 else np.nan))
    return iso, comp


def main() -> int:
    ap = argparse.ArgumentParser(description="⑥σP ⑦다단강하 ⑧총역률 (12.57절)")
    ap.add_argument("--out", default="results/steady_feature_probe.json")
    a = ap.parse_args()

    print("=" * 96)
    print("[⑥] 정상 통전 중 1초 전력 표준편차 σP")
    print("=" * 96)
    iso6 = sigma_isolated()
    print(f"  격리 녹화  {'기기':<12s}{'표본':>9s}{'σP p25':>9s}{'중앙':>9s}{'p75':>9s}{'p95':>9s}")
    for app in ALL:
        if app not in iso6:
            continue
        v = iso6[app]
        print(f"  {'':10s}{KOR.get(app, app):<12s}{len(v):>9,d}"
              + "".join(f"{np.percentile(v, q):>9.2f}" for q in (25, 50, 75, 95)))
    print()
    print("  분리비 (격리)")
    for x, y in (("laptop_charger", "beam_projector"), ("beam_projector", "minipc"),
                 ("laptop_charger", "minipc")):
        if x in iso6 and y in iso6:
            s = sepratio(iso6[x], iso6[y])
            print(f"    {KOR.get(x,x):>8s} vs {KOR.get(y,y):<8s}"
                  f"  {np.median(iso6[x]):7.2f} vs {np.median(iso6[y]):7.2f}"
                  f"   분리비 {s:5.2f}   AUC {auc(iso6[x], iso6[y]):.3f}")
    print()
    print("  복합: σP 하나로 '그 기기 ON' 을 맞히는가")
    c6 = sigma_composite()
    print(f"    {'파일':<9s}{'기기':<10s}{'ON σP중앙':>11s}{'OFF σP중앙':>12s}{'AUC':>9s}{'n(on/off)':>16s}")
    for stem, d in c6.items():
        for app, r in d.items():
            print(f"    {stem:<9s}{KOR.get(app,app):<10s}{r['on_med']:>11.2f}"
                  f"{r['off_med']:>12.2f}{r['auc']:>9.3f}"
                  f"{r['n_on']:>9,d}/{r['n_off']:<7,d}")

    print()
    print("=" * 96)
    print("[⑦] 프로젝터 종료 시퀀스 — 소등 뒤 초 단위 P 중앙 (W)")
    print("=" * 96)
    t7 = turnoff_tail()
    for r in t7["isolated"]:
        print(f"  격리 {r['file']:<18s} 소등 뒤 남은 길이 {r['tail_s']:.1f}s")
        print("       " + "  ".join(f"{d:+.0f}s:{w:6.1f}" for d, w in r["profile"]))
    for r in t7["composite"]:
        print(f"  복합 {r['stem']:<8s} {r['t_s']:7.1f}s  "
              + "  ".join(f"{d:+.0f}s:{w:7.1f}" for d, w in r["profile"]))

    print()
    print("=" * 96)
    print("[⑧] 총 역률 ΔP/ΔS")
    print("=" * 96)
    iso8, comp8 = true_pf()
    print(f"  격리  {'기기':<12s}{'n':>5s}{'PF 중앙':>10s}{'±std':>9s}   (제안값)")
    claim = {"minipc": "0.50", "laptop_charger": "0.54~0.55", "beam_projector": "0.58"}
    for app in ALL:
        if app not in iso8:
            continue
        v = iso8[app]
        print(f"  {'':6s}{KOR.get(app,app):<12s}{len(v):>5d}{np.median(v):>10.3f}"
              f"{v.std():>9.3f}   {claim.get(app,'')}")
    print()
    print("  분리비 (격리)")
    for x, y in (("laptop_charger", "beam_projector"), ("beam_projector", "minipc"),
                 ("laptop_charger", "minipc")):
        if x in iso8 and y in iso8:
            print(f"    {KOR.get(x,x):>8s} vs {KOR.get(y,y):<8s}"
                  f"  {np.median(iso8[x]):6.3f} vs {np.median(iso8[y]):6.3f}"
                  f"   분리비 {sepratio(iso8[x], iso8[y]):5.2f}")
    print()
    print(f"  복합  {'기기':<12s}{'n':>5s}{'PF(복소Δ)':>12s}{'PF(순진Δ)':>12s}")
    for app in ALL:
        v = comp8.get(app)
        if not v or len(v) < 3:
            continue
        A = np.array(v)
        print(f"  {'':6s}{KOR.get(app,app):<12s}{len(A):>5d}{np.median(A[:,0]):>12.3f}"
              f"{np.nanmedian(A[:,1]):>12.3f}")
    print()
    print("  분리비 (복합, 복소Δ 기준)")
    for x, y in (("laptop_charger", "beam_projector"), ("beam_projector", "minipc"),
                 ("laptop_charger", "minipc")):
        if x in comp8 and y in comp8 and len(comp8[x]) > 2 and len(comp8[y]) > 2:
            A = np.array(comp8[x])[:, 0]; B = np.array(comp8[y])[:, 0]
            s = sepratio(A, B)
            print(f"    {KOR.get(x,x):>8s} vs {KOR.get(y,y):<8s}"
                  f"  {np.median(A):6.3f} vs {np.median(B):6.3f}   분리비 {s:5.2f}"
                  + ("  <<< 갈린다" if s >= 2 else ""))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"sigma_isolated": {k: [float(np.percentile(v, q)) for q in (25, 50, 75, 95)]
                            for k, v in iso6.items()},
         "sigma_composite": c6, "turnoff": t7,
         "pf_isolated": {k: [float(np.median(v)), float(v.std())] for k, v in iso8.items()},
         "pf_composite": {k: float(np.median(np.array(v)[:, 0])) for k, v in comp8.items()}},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print()
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
