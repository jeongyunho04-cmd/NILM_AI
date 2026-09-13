# -*- coding: utf-8 -*-
"""회로 모델 적합 재구성과 요소 재검정 (12.185). 기준은 `results/_criteria_circuit.md` 추가분.

    python -X utf8 -m src.run_circuit_model_probe                 # 전부
    python -X utf8 -m src.run_circuit_model_probe --stages 0 1    # 오염 확인 + 기본 재적합
    python -X utf8 -m src.run_circuit_model_probe --devices minipc

왜 다시 하는가 (요약)
---------------------
`_circuit_params_C.json` / `_circuit_elements_C.json` 의 par5 를 **현재 규약**(LOW 0.44)에서
다시 평가하면 기록된 손실과 5~13배 다르다 — 그 표는 2.62 회전이 걸려 있던 동안 만들어졌다.
게다가 충전기 E3 의 "최적" 이 중립값보다 나쁘다 (겹친 모형은 그럴 수 없다 -> 최적화 실패).
그래서 12.184.9 의 재피팅과 12.184.10 의 요소 기각을 둘 다 다시 세운다.

단계
----
    0  오염 확인      저장 par5 를 현 규약에서 재평가 (H1)
    1  기본 재적합    잔차 최소제곱 + 다중 시작 (H2). 손실의 차수별 분해와 전력 정합도 함께
    2  요소 재검정    E1 nvt / E2 Gp / E3 alpha / E4 k_hi 를 겹쳐 시작 + LOO 교차검증 (H3, H7)
    3  계측 전개      RC 위상 오차 −(h·δ1 − δh) 를 I·V 에서 뺀다, 자유 파라미터 0 (H4)
    4  절대 페이저    정규화 서명 대신 절대 전류로 맞춘다 (H5)
    5  식별성        잔차 야코비의 특이값 — 자료가 못 정하는 방향 (H6)
"""
from typing import Dict, List, Tuple
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.raw_csv import read_raw_csv
from src.preprocessing.raw_phasors import steady_signature, voltage_phasors
from src.run_site_voltage_probe import SITE_C_FILES
from src.synthesis import circuit_fit as cf
from src.synthesis.circuit_fit import LossCfg, Model, Point
from src.synthesis.circuit_sim import GUIDE_PARAMS

H = 15
OUT = "results/_circuit_model_C.json"
OUT_NOBG = "results/_circuit_model_C_nobg.json"
DEVICES = ("minipc", "laptop_charger", "beam_projector")
#: 교차검증으로 판정하는 기기 (프로젝터는 동작점이 1개라 참고만)
JUDGED = ("minipc", "laptop_charger")


#: ⚠ **12.185.9 에서 폐기.** "그 기기가 꺼진 창" 을 배경으로 쓰던 표다. 미니PC·프로젝터는 꺼져도
#: 어댑터가 꽂힌 채라 그 창의 30mA ∠+68° 는 배경이 아니라 **그 기기 자신의 X-cap 전류**다
#: (원시로 잰 Cx 로 계산한 28.6·34.1mA 와 맞는다). 그걸 빼면 모델이 Cx 를 못 찾는다 —
#: 실제로 미니PC 2Hz 적합에서 Cx 가 하한 0.010µF 로 갔다 (안 뺀 판은 0.453µF).
#: 진짜 배경은 기기 없이 찍은 noise_noselfpower_C 하나뿐이다: 1.70W, |I1| 7.3mA ∠+10.9°.
BG_WINDOW_DEPRECATED = {"minipc": (1042.0, 1155.0), "laptop_charger": (2.0, 55.0),
                        "beam_projector": (270.0, 327.0)}


def load(dev: str, subtract_bg: bool = True) -> List[Point]:
    stem, bands = SITE_C_FILES[dev]
    cols = ["t_s", "p_w", "vrms", "over_range", "range"] + [f"ih{h}" for h in range(1, H + 1)] \
        + [f"ihdeg{h}" for h in range(1, H + 1)] + [f"vh{h}" for h in range(1, H + 1)] \
        + [f"vhdeg{h}" for h in range(1, H + 1)]
    df, info = read_raw_csv(f"data/{stem}.csv", usecols=cols)
    bg_I = np.zeros(H, complex)
    bg_p = 0.0
    if subtract_bg:
        # 계측계 배경 하나만 뺀다 (기기 꺼짐 창이 아니라 — 12.185.9)
        from src.synthesis.fit_raw import BG_STEM
        bcols = ["p_w", "vrms", "over_range", "range"] + [f"ih{h}" for h in range(1, H + 1)]             + [f"ihdeg{h}" for h in range(1, H + 1)]
        bdf, _ = read_raw_csv(f"data/{BG_STEM}.csv", usecols=bcols)
        bsg = steady_signature(bdf, -1e9, 1e9)
        bg_I, bg_p = bsg["I"], bsg["p_w"]
    pts = []
    for lo, hi in bands:
        sg = steady_signature(df, lo, hi)
        if sg["n"] < 200:
            continue
        V, _ = voltage_phasors(df, sg["mask"])
        I = sg["I"] - bg_I
        pts.append(Point(sg["p_w"] - bg_p, I / I[0], I, V))
    print(f"  {dev}: {stem} {len(pts)}점 (P {pts[0].p_w:.1f}~{pts[-1].p_w:.1f}W, "
          f"위상 복원 {info['phase_fix_deg_per_order']}°/차수"
          + (f", 배경 뺌 {bg_p:.2f}W/{abs(bg_I[0]) * 1000:.1f}mA)" if subtract_bg else ", 배경 그대로)"))
    return pts


def fmt_par(par5, extras=None) -> str:
    C, R, L, Cx, rd = par5
    s = f"C_dc {C * 1e6:6.1f}µF  R {R:5.2f}Ω  L {L * 1e6:8.1f}µH  Cx {Cx * 1e6:5.3f}µF  rd {rd:5.2f}Ω"
    if extras:
        s += "  " + " ".join(f"{k} {v:.4g}" for k, v in extras.items() if v != cf.EXTRA_SPEC[k][3])
    return s


def per_order(par5, extras, pts, cfg) -> np.ndarray:
    """차수별 손실 기여 (h2..h15 또는 h1..h15)."""
    d = cf.residual(par5, extras, pts, cfg).reshape(len(pts), 2, -1)
    return (d ** 2).sum(1).mean(0)


# ── [0] 오염 확인 ────────────────────────────────────────────────────────────
def stage0(data: Dict[str, List[Point]], cfg: LossCfg, rec: Dict) -> None:
    print("\n[0] 저장된 par5 를 현재 규약에서 재평가 (H1)")
    print("    기록은 2.62 회전이 걸려 있던 동안 만들어졌다면 재현되지 않는다.")
    try:
        P5 = json.load(open("results/_circuit_params_C.json", encoding="utf-8"))
    except FileNotFoundError:
        print("    results/_circuit_params_C.json 없음 — 건너뜀")
        return
    rows = {}
    for dev, pts in data.items():
        if dev not in P5:
            continue
        p = cf.prepare(pts, cfg)
        L = cf.loss(tuple(P5[dev]["fit_siteC_wave"]), {}, p, cfg)
        rec0 = P5[dev]["loss_fit_siteC"]
        rows[dev] = {"recorded": rec0, "reeval": L, "ratio": L / rec0}
        print(f"    {dev:15s} 기록 {rec0:.6f}  재평가 {L:.6f}   ×{L / rec0:.1f}"
              + ("   <- 폐기" if abs(L / rec0 - 1) > 0.05 else "   재현됨"))
    rec["stage0"] = rows


# ── [1] 기본 재적합 ──────────────────────────────────────────────────────────
def stage1(data: Dict[str, List[Point]], cfg: LossCfg, rec: Dict) -> Dict[str, cf.FitResult]:
    print("\n[1] 기본 5파라미터 재적합 — 잔차 최소제곱 + 다중 시작 (H2)")
    base: Dict[str, cf.FitResult] = {}
    rows = {}
    try:
        P5 = json.load(open("results/_circuit_params_C.json", encoding="utf-8"))
    except FileNotFoundError:
        P5 = {}
    for dev, pts in data.items():
        p = cf.prepare(pts, cfg)
        starts = [(GUIDE_PARAMS[dev], {})]
        if dev in P5:
            starts.append((tuple(P5[dev]["fit_siteC_wave"]), {}))
        # 물리 범위 안에서 흩뿌린 시작점 (재현되도록 고정 격자)
        for C, R, L, Cx, rd in [(60e-6, 2.0, 100e-6, 0.3e-6, 1.5), (150e-6, 5.0, 1000e-6, 0.5e-6, 4.0),
                                (30e-6, 1.0, 30e-6, 0.15e-6, 0.8), (250e-6, 7.0, 2500e-6, 0.8e-6, 6.0)]:
            starts.append(((C, R, L, Cx, rd), {}))
        t0 = time.time()
        m = Model("base")
        r = cf.fit(p, m, cfg, starts, max_nfev=600)
        base[dev] = r
        old = cf.loss(tuple(P5[dev]["fit_siteC_wave"]), {}, p, cfg) if dev in P5 else np.nan
        pw = cf.power_check(r.par5, r.extras, p, cfg)
        po = per_order(r.par5, r.extras, p, cfg)
        rows[dev] = {**r.as_dict(), "old_reeval": old, "p_ratio": pw.tolist(),
                     "per_order": po.tolist()}
        print(f"    {dev:15s} 손실 {r.loss:.6f} (옛 par5 재평가 {old:.6f}, "
              f"{100 * (1 - r.loss / old):+.0f}%)  {time.time() - t0:.0f}s")
        print(f"        {fmt_par(r.par5)}" + (f"   경계: {r.at_bound}" if r.at_bound else ""))
        print(f"        시뮬 입력전력/측정  {np.min(pw):.3f}~{np.max(pw):.3f}")
        print("        차수별 기여: " + " ".join(f"h{h}={po[h - 2]:.4f}" for h in range(3, 16, 2)))
    rec["stage1"] = rows
    return base


# ── [2] 요소 재검정 ──────────────────────────────────────────────────────────
ELEMENTS = [
    ("E1 nvt", ("nvt",)),
    ("E2 Gp", ("Gp",)),
    ("E3 alpha", ("alpha",)),
    ("E4 k_hi", ("k_hi",)),
    ("E1+E3", ("nvt", "alpha")),
]


def stage2(data: Dict[str, List[Point]], cfg: LossCfg, base: Dict[str, cf.FitResult],
           rec: Dict, do_loo: bool = True) -> None:
    print("\n[2] 요소 재검정 — 겹쳐 시작(단조 보장) + LOO 교차검증 (H3, H7)")
    print("    채택: 훈련 20% 이상 감소 **그리고** LOO 10% 이상 감소. 훈련만이면 과적합.")
    rows: Dict[str, Dict] = {}
    for dev, pts in data.items():
        p = cf.prepare(pts, cfg)
        b = base[dev]
        judged = dev in JUDGED
        loo0 = cf.loo(p, Model("base"), cfg, b)[0] if (do_loo and judged) else np.nan
        print(f"\n    [{dev}]  base 훈련 {b.loss:.6f}" + (f"  LOO {loo0:.6f}" if judged else "  (LOO 없음)"))
        rows[dev] = {"base": {**b.as_dict(), "loo": loo0}}
        for name, ex in ELEMENTS:
            m = Model(name, ex)
            t0 = time.time()
            r = cf.fit_nested(p, m, cfg, b)
            lo_ = cf.loo(p, m, cfg, b)[0] if (do_loo and judged) else np.nan
            dtr = 100 * (1 - r.loss / b.loss)
            dlo = 100 * (1 - lo_ / loo0) if judged else np.nan
            verdict = "기각"
            if judged:
                if dtr >= 20 and dlo >= 10:
                    verdict = "채택"
                elif dtr >= 20:
                    verdict = "과적합"
                elif dtr >= 5:
                    verdict = "부분"
            else:
                verdict = "참고"
            if r.at_bound:
                verdict += f" (경계 {r.at_bound})"
            rows[dev][name] = {**r.as_dict(), "loo": lo_, "d_train_pct": dtr, "d_loo_pct": dlo}
            print(f"      {name:10s} 훈련 {r.loss:.6f} ({dtr:+5.1f}%)  "
                  f"LOO {lo_:.6f} ({dlo:+5.1f}%)  {verdict}   {time.time() - t0:.0f}s")
            if abs(dtr) > 1e-9:
                print(f"          {fmt_par(r.par5, r.extras)}")
    rec["stage2"] = rows


# ── [3] 계측 전개 ────────────────────────────────────────────────────────────
def stage3(data: Dict[str, List[Point]], cfg: LossCfg, base: Dict[str, cf.FitResult],
           rec: Dict, do_loo: bool = True) -> None:
    e = cf.rc_phase_err_deg()
    print("\n[3] 계측 전개 — RC 위상 오차를 I·V 에서 뺀다, 자유 파라미터 0 (H4)")
    print("    남는 오차 h·δ1 − δh [°]: " + " ".join(f"h{h}={e[h - 1]:+.2f}" for h in range(3, 16, 2)))
    cfg2 = LossCfg(mode=cfg.mode, odd_only_loss=cfg.odd_only_loss, vmax=cfg.vmax, deembed_rc=True)
    rows = {}
    for dev, pts in data.items():
        p2 = cf.prepare(pts, cfg2)
        b = base[dev]
        starts = [(b.par5, {}), (GUIDE_PARAMS[dev], {})]
        r = cf.fit(p2, Model("base"), cfg2, starts, max_nfev=600)
        loo_ = cf.loo(p2, Model("base"), cfg2, r)[0] if (do_loo and dev in JUDGED) else np.nan
        d = 100 * (1 - r.loss / b.loss)
        rows[dev] = {**r.as_dict(), "loo": loo_, "d_pct": d}
        print(f"    {dev:15s} 손실 {r.loss:.6f} (전개 전 {b.loss:.6f}, {d:+.1f}%)  LOO {loo_:.6f}")
        print(f"        {fmt_par(r.par5)}")
    rec["stage3"] = rows


# ── [4] 절대 페이저 ──────────────────────────────────────────────────────────
def stage4(data: Dict[str, List[Point]], cfg: LossCfg, base: Dict[str, cf.FitResult],
           rec: Dict) -> None:
    print("\n[4] 절대 페이저로 적합 — |I1|·∠I1 이 정보로 들어온다 (H5)")
    print("    판정: 절대 적합 파라미터를 정규화 손실로 재서 20% 안이면 두 자가 같은 최적을 가리킨다.")
    cfa = LossCfg(mode="abs", odd_only_loss=cfg.odd_only_loss, vmax=cfg.vmax, deembed_rc=cfg.deembed_rc)
    rows = {}
    for dev, pts in data.items():
        pn = cf.prepare(pts, cfg)
        pa = cf.prepare(pts, cfa)
        b = base[dev]
        ra = cf.fit(pa, Model("base"), cfa, [(b.par5, {}), (GUIDE_PARAMS[dev], {})], max_nfev=600)
        # 두 자를 교차로 잰다
        ln_of_abs = cf.loss(ra.par5, {}, pn, cfg)          # 절대 적합 -> 정규화 손실
        la_of_norm = cf.loss(b.par5, {}, pa, cfa)          # 정규화 적합 -> 절대 손실
        gap = 100 * (ln_of_abs / b.loss - 1)
        rows[dev] = {**ra.as_dict(), "norm_loss_of_abs_fit": ln_of_abs, "abs_loss_of_norm_fit": la_of_norm,
                     "gap_pct": gap}
        print(f"    {dev:15s} 절대 손실 {ra.loss:.6f} (정규화 적합의 절대 손실 {la_of_norm:.6f})")
        print(f"        절대 적합의 정규화 손실 {ln_of_abs:.6f} vs 정규화 최적 {b.loss:.6f}  ({gap:+.0f}%)")
        print(f"        {fmt_par(ra.par5)}")
    rec["stage4"] = rows


# ── [5] 식별성 ───────────────────────────────────────────────────────────────
def stage5(data: Dict[str, List[Point]], cfg: LossCfg, base: Dict[str, cf.FitResult],
           rec: Dict) -> None:
    print("\n[5] 식별성 — 잔차 야코비의 특이값 (로그 파라미터) (H6)")
    rows = {}
    for dev, pts in data.items():
        p = cf.prepare(pts, cfg)
        b = base[dev]
        d = cf.sensitivity(b.par5, b.extras, p, cfg, Model("base"))
        S, Vt, names = d["S"], d["Vt"], d["names"]
        rows[dev] = {"S": S.tolist(), "cond": d["cond"], "names": names,
                     "worst_dir": Vt[-1].tolist()}
        print(f"    {dev:15s} 조건수 {d['cond']:.3g}   σ " + " ".join(f"{v:.3g}" for v in S))
        w = Vt[-1]
        print("        가장 무른 방향: " + "  ".join(f"{n} {v:+.2f}" for n, v in zip(names, w)))
        # 파라미터별 유효 정밀도: σ 를 통해 본 기여
        for i, n in enumerate(names):
            contrib = np.sqrt(np.sum((Vt[:, i] * S) ** 2))
            print(f"          {n:5s} 민감도 {contrib:8.3g}", end="")
        print()
    rec["stage5"] = rows


# ── [6] 원시 대조 ────────────────────────────────────────────────────────────
def stage6(data: Dict[str, List[Point]], cfg: LossCfg, base: Dict[str, cf.FitResult],
           rec: Dict) -> None:
    """원시 파형으로 맞춘 파라미터가 2Hz 고조파 자료도 맞추는가 (12.185.11).

    두 자가 같은 최적을 가리키면 계측 두 경로가 일치한다는 뜻이다. 크게 갈리면 둘 중 하나에
    계통 오차가 남아 있다. 2Hz 쪽은 RC 위상 오차를 빼고(자유 파라미터 0), 절대 페이저로도 잰다.
    """
    import json as _json
    try:
        raw = _json.load(open("results/_circuit_raw_C.json", encoding="utf-8"))["devices"]
    except FileNotFoundError:
        print("\n[6] results/_circuit_raw_C.json 없음 — run_raw_fit_probe 를 먼저 돌려라")
        return
    print("\n[6] 원시로 맞춘 파라미터를 2Hz 자료로 잰다 (12.185.11)")
    print("    자 넷: 정규화 / 정규화+RC전개 / 절대 / 절대+RC전개.  2Hz 최적 대비 배수로 읽는다")
    cfgs = {"norm": LossCfg(mode="norm"), "norm+rc": LossCfg(mode="norm", deembed_rc=True),
            "abs": LossCfg(mode="abs"), "abs+rc": LossCfg(mode="abs", deembed_rc=True)}
    rows = {}
    for dev, pts in data.items():
        if dev not in raw:
            continue
        par_raw = tuple(raw[dev]["fit_free"]["par5"])
        b = base.get(dev)
        row = {"par_raw": list(par_raw), "par_2hz": list(b.par5) if b else None}
        print(f"    [{dev}]")
        print(f"      원시 적합 {fmt_par(par_raw)}")
        if b:
            print(f"      2Hz 적합  {fmt_par(b.par5)}")
        for name, c in cfgs.items():
            p2 = cf.prepare(pts, c)
            l_raw = cf.loss(par_raw, {}, p2, c)
            # 그 자에서의 2Hz 최적 (원시값과 2Hz값 둘 다에서 출발)
            starts = [(par_raw, {})] + ([(b.par5, {})] if b else [])
            r = cf.fit(p2, Model("base"), c, starts, max_nfev=500)
            row[name] = {"raw_par": l_raw, "best_2hz": r.loss, "ratio": l_raw / max(r.loss, 1e-12),
                         "par_best": list(r.par5)}
            print(f"      {name:8s} 원시파라미터 {l_raw:.5f}   2Hz 최적 {r.loss:.5f}   "
                  f"×{l_raw / max(r.loss, 1e-12):.2f}")
        rows[dev] = row
    rec["stage6"] = rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*", default=list(DEVICES))
    ap.add_argument("--stages", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--no-loo", action="store_true", help="교차검증 생략 (빠르게)")
    ap.add_argument("--odd-loss", action="store_true", help="짝수차를 손실에서 뺀다")
    ap.add_argument("--no-bg", action="store_true", help="배경(대기) 전류를 빼지 않는다 (옛 방식)")
    a = ap.parse_args()

    cfg = LossCfg(mode="norm", odd_only_loss=a.odd_loss)
    print("자료 (장소 C, LOW 0.44 규약):")
    data = {d: load(d, subtract_bg=not a.no_bg) for d in a.devices}
    rec: Dict = {"cfg": {"mode": cfg.mode, "odd_only_loss": cfg.odd_only_loss,
                         "ncyc": cf.NCYC_FIT, "subtract_bg": not a.no_bg}}

    if 0 in a.stages:
        stage0(data, cfg, rec)
    base = stage1(data, cfg, rec) if 1 in a.stages else {}
    if not base and any(s in a.stages for s in (2, 3, 4, 5, 6)):
        base = stage1(data, cfg, rec)
    if 2 in a.stages:
        stage2(data, cfg, base, rec, do_loo=not a.no_loo)
    if 3 in a.stages:
        stage3(data, cfg, base, rec, do_loo=not a.no_loo)
    if 4 in a.stages:
        stage4(data, cfg, base, rec)
    if 5 in a.stages:
        stage5(data, cfg, base, rec)
    if 6 in a.stages:
        stage6(data, cfg, base, rec)

    out = OUT_NOBG if a.no_bg else OUT
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False, default=float)
    print(f"\n기록: {OUT}")


if __name__ == "__main__":
    main()
