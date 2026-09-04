# -*- coding: utf-8 -*-
"""요소 재검정 — 원시 파형으로, 겹쳐 시작으로 (12.185, 기준 `results/_criteria_circuit.md` [E1]~[E6]).

    python -X utf8 -m src.run_element_raw_probe                 # 전부
    python -X utf8 -m src.run_element_raw_probe --devices minipc
    python -X utf8 -m src.run_element_raw_probe --no-loo        # 훈련만 (빠르게)

**왜 다시 하나.** 12.184.10 이 "§3.3 요소 셋(다이오드 무릎·초크 감쇠·부하 지수) 기각" 이라
적었는데, 그 표에서 충전기 E3 의 "최적" 손실 0.008759 가 중립값(α=1) 0.008725 보다 **컸다**.
겹친 모형은 기본을 특수 케이스로 품으므로 원리적으로 더 나쁠 수 없다 — 그것은 요소의 증거가
아니라 Nelder-Mead 가 못 내려갔다는 기록이다. 판정을 철회했고(12.185.1) 여기서 다시 세운다.

이번에 다른 것 셋:
  1. **원시 파형 위에서.** 12.184.10 은 2Hz 고조파 위에서 판정했는데 그 자료가 세 규약 결함을
     안고 있었다 (배경 정의·PHASE_FIX 과교정·LOW/HIGH 이어붙이기). 원시는 전압을 직접 갖는다.
  2. **겹쳐 시작 + TRF.** `fit_nested` 는 기본 최적점 + 중립 extras 에서 출발하므로 결과가
     기본보다 나쁠 수 없다. 중립에서 야코비가 0 인 방향이 있어 켠 시작점도 함께 넣는다.
  3. **`rd` 를 0.3Ω 에 고정한 위에서.** `nvt` 는 브리지의 비선형 저항, `rd` 는 그 선형 저항이라
     같은 자리에 산다 — `rd` 가 자유로우면 맞바뀌어 어느 쪽도 판정되지 않는다.

단계
----
    1  [E1] 재현      정본 `fit_fixrd` 를 지금 규약에서 다시 재서 기록과 맞는지
    2  [E2][E3] 훈련  요소별 겹쳐 시작 적합 — 손실 변화, 채택값, 단조 보호가 걸렸는지
    3  [E6] 겨냥      점별 RMS 와 par5 변화 (충전기 19W · 미니PC R)
    4  [E4] LOO       폴드마다 기본을 다시 맞춘 뒤 그 위에서 요소를 켠다
    5      판정       [E3]~[E5] 를 표로
"""
from typing import Dict, List, Sequence
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.file_registry import RAW_SNAPSHOT_FILES, raw_snapshots_of
from src.synthesis import fit_raw as fr
from src.synthesis.fit_raw import (PAR_NAMES, RAW_EXTRAS, background_phasors, fit, fit_nested,
                                   in_physical_range, load_raw, loo_folds, loo_nested, rms_of)

OUT_NOTE = "요소 재검정 (12.184.10 철회분) + Vf 대조군"

IN = "results/_circuit_raw_C.json"
OUT = "results/_circuit_elements_raw_C.json"
RD_PHYS = 0.3                       #: 정본 규약 — `run_raw_fit_probe` 와 같은 값
FIX = {4: RD_PHYS}
#: [E3] 채택 문턱 — 훈련 RMS 를 이만큼 줄이지 못하면 "효과 없음"
THR_TRAIN = 0.05
#: 검정할 변형. 마지막은 셋을 동시에 (요소끼리 서로를 가리고 있을 가능성)
VARIANTS: Dict[str, Sequence[str]] = {
    "nvt": ("nvt",), "Gp": ("Gp",), "alpha": ("alpha",), "all3": RAW_EXTRAS,
    # `Vf` 는 12.184.10 의 요소가 아니라 **대조군**이다. nvt·rd·Vf 는 같은 브리지의 세 가지
    # 기술이고 우리는 Vf 1.4V·rd 0.3Ω 을 박아 둔 채 nvt 만 맞췄다. 상수 강하 하나로 같은
    # 이득이 나면 "지수 무릎이 필요하다" 가 아니라 "박아 둔 Vf 가 틀렸다" 가 옳은 결론이다.
    "Vf": ("Vf",), "nvt+Vf": ("nvt", "Vf"),
}
#: 요소별 흩뿌린 시작값. 겹쳐 시작(중립)·켠 시작(EXTRA_SPEC 의 다섯째)에 더해 넣는다 —
#: 중립 근처의 국소 최소에 갇혀 "효과 없음" 이라고 잘못 적는 것을 막는다.
SCATTER: Dict[str, List[float]] = {
    "nvt": [0.005, 0.5], "Gp": [0.005, 0.15], "alpha": [0.5, -0.5], "Vf": [0.7, 2.0],
}
#: [2b] 효과 크기 검정에 쓰는 **물리적으로 그럴듯한** 값. 적합이 안 움직이는 것이
#: "요소가 없어서" 인지 "자료가 못 봐서" 인지를 가른다.
#:   nvt  실리콘 2개 직렬의 n·V_T ≈ 2×0.026×1.5 = 0.08V
#:   Gp   초크 병렬 33Ω = 0.03S (실물 감쇠 저항의 흔한 값)
#:   alpha 0 = 정전류 (DC-DC 루프가 느릴 때의 극단)
PHYSICAL: Dict[str, float] = {"nvt": 0.08, "Gp": 0.03, "alpha": 0.0, "Vf": 0.9}
#: `run_raw_fit_probe.starts_for` 와 같은 격자 (폴드 기본 적합용)
GUIDE_RAW = {"laptop_charger": (66.3e-6, 5.06, 988e-6, 0.164e-6, 0.43),
             "beam_projector": (53.3e-6, 7.18, 711e-6, 0.240e-6, 0.30),
             "minipc": (36.6e-6, 11.70, 2785e-6, 0.337e-6, 0.01)}


def base_starts(dev: str, cx: float) -> List[tuple]:
    c = cx if np.isfinite(cx) and 0.02e-6 < cx < 2e-6 else 0.3e-6
    return [GUIDE_RAW[dev], (70e-6, 4.0, 700e-6, c, 0.5), (50e-6, 8.0, 1500e-6, c, 1.5),
            (110e-6, 2.0, 200e-6, c, 0.2), (35e-6, 12.0, 2500e-6, c, 3.0)]


def scatter_starts(par5, names: Sequence[str]) -> List[tuple]:
    """(par5, extras) 흩뿌린 시작점. 요소 하나씩 켜고 나머지는 중립."""
    out = []
    for n in names:
        for v in SCATTER.get(n, []):
            ex = fr.neutral_extras(names)
            ex[n] = v
            out.append((tuple(par5), ex))
    return out


def fmt5(par5) -> str:
    C, R, L, Cx, rd = par5
    return (f"C_dc {C * 1e6:6.1f}µF  R {R:5.2f}Ω  L {L * 1e6:7.1f}µH  "
            f"Cx {Cx * 1e6:5.3f}µF  rd {rd:4.2f}Ω")


def fmtex(ex: Dict[str, float]) -> str:
    if not ex:
        return "-"
    return " ".join(f"{k}={v:.4g}" for k, v in ex.items())


def combo_check(rec: Dict, canon: Dict, bg, band: int) -> None:
    """조합 6개를 파라미터 집합별로 재현한다. 조합에 맞춘 자유 파라미터는 0개다."""
    from src.preprocessing.file_registry import RAW_COMBO_FILES, RAW_COMBO_SOLO
    from src.synthesis.fit_raw import phasors, sim_current

    sets: Dict[str, Dict] = {
        "fit_fixrd": {d: (tuple(canon["devices"][d]["fit_fixrd"]["par5"]), {}) for d in rec["devices"]},
    }
    for name in VARIANTS:
        sets[name] = {d: (tuple(rec["devices"][d]["variants"][name]["par5"]),
                          rec["devices"][d]["variants"][name]["extras"]) for d in rec["devices"]}

    def rel(x, y):
        e = phasors(x, band) - phasors(y, band)
        return float(np.sqrt(np.sum(np.abs(e) ** 2)) / np.sqrt(np.sum(np.abs(phasors(y, band)) ** 2)))

    print(f"    {'조합':22s} {'구성':22s} {'[A]중첩':>8s}" + "".join(f"{k:>9s}" for k in sets))
    tot: Dict[str, List[float]] = {k: [] for k in sets}
    out: Dict[str, Dict[str, float]] = {}
    for combo, devs in RAW_COMBO_FILES.items():
        parts = [(d, RAW_COMBO_SOLO[combo][d]) for d in devs if d in rec["devices"]]
        if len(parts) != len(devs):
            continue
        c = load_raw(combo, bg=bg)
        solos = {d: load_raw(s, bg=bg) for d, s in parts}
        pw = {d: solos[d].p_w for d, _ in parts}
        if "laptop_charger" in pw:      # 충전기는 배터리 상태로 변한다 — 나머지를 고정하고 잔여를 준다
            pw["laptop_charger"] = c.p_w - sum(v for k, v in pw.items() if k != "laptop_charger")
        else:
            k = c.p_w / sum(pw.values())
            pw = {d: v * k for d, v in pw.items()}
        row = {}
        for key, PAR in sets.items():
            tot_i, ok = np.zeros_like(c.i), True
            for d, _ in parts:
                pt = type(c)(c.stem, c.v, c.i, pw[d], c.vsrc, c.irms, c.n_cyc,
                             c.scatter, c.oob, c.range_mixed)
                x = sim_current(PAR[d][0], pt, match_power=True, extras=PAR[d][1])
                if x is None:
                    ok = False
                    break
                tot_i = tot_i + x
            row[key] = rel(tot_i, c.i) if ok else float("nan")
            tot[key].append(row[key])
        a_sup = rel(sum(solos[d].i for d, _ in parts), c.i)
        out[combo] = {"naive": a_sup, **row}
        print(f"    {combo:22s} {'+'.join(d[:4] for d, _ in parts):22s} {100 * a_sup:7.2f}%"
              + "".join(f"{100 * row[k]:8.2f}%" for k in sets))
    print(f"    {'평균':22s} {'':22s} {'':8s}" + "".join(f"{100 * np.mean(tot[k]):8.2f}%" for k in sets))
    rec["combo"] = out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*", default=list(RAW_SNAPSHOT_FILES))
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--band", type=int, default=fr.BAND)
    a = ap.parse_args()
    canon = json.load(open(IN, encoding="utf-8"))
    if canon.get("vband") is not None or not canon.get("subtract_meter_bg"):
        print(f"⚠ {IN} 이 정본 규약이 아니다 (vband={canon.get('vband')}, "
              f"bg={canon.get('subtract_meter_bg')}) — run_raw_fit_probe 를 먼저 돌려라")
    bg = background_phasors()
    rec: Dict = {"band": a.band, "rd_fixed": RD_PHYS, "thr_train": THR_TRAIN, "devices": {}}

    for dev in a.devices:
        stems = raw_snapshots_of(dev)
        if dev not in canon["devices"] or not stems:
            print(f"[{dev}] 정본 또는 원시 스냅샷 없음 — 건너뜀")
            continue
        print("=" * 100)
        print(f"[{dev}]  원시 스냅샷 {len(stems)}개, rd = {RD_PHYS}Ω 고정")
        print("=" * 100)
        pts = sorted([load_raw(s, bg=bg) for s in stems], key=lambda p: p.p_w)
        cd = canon["devices"][dev]
        base = tuple(cd["fit_fixrd"]["par5"])
        d: Dict = {"base_par5": list(base), "variants": {}}

        # ── 1 [E1] 재현 ────────────────────────────────────────────────────
        l_rec = float(cd["fit_fixrd"]["loss"])
        l_now = fr.loss_at(base, {}, pts, a.band)
        dev_pct = 100 * (l_now / l_rec - 1)
        d["e1"] = {"loss_recorded": l_rec, "loss_now": l_now, "dev_pct": dev_pct}
        ok1 = abs(dev_pct) < 1.0
        print(f"  [1][E1] 정본 재현  기록 {l_rec:.8f}  지금 {l_now:.8f}  ({dev_pct:+.3f}%)  "
              + ("통과" if ok1 else "⚠ 규약이 어긋났다 — 요소를 보지 말고 규약부터 찾아라"))
        print(f"          기본 {fmt5(base)}   훈련 RMS {100 * np.sqrt(l_now):.2f}%")
        if not ok1:
            continue

        # ── 2 [E2][E3] 훈련 ────────────────────────────────────────────────
        print(f"\n  [2][E2][E3] 요소별 겹쳐 시작 적합 (문턱: 훈련 RMS {100 * THR_TRAIN:.0f}% 감소)")
        print(f"    {'변형':6s} {'훈련RMS':>8s} {'기본대비':>9s} {'단조':>5s} {'채택값':38s} {'경계':10s}")
        print(f"    {'기본':6s} {100 * np.sqrt(l_now):7.2f}% {'—':>9s} {'—':>5s} {'-':38s}")
        fits: Dict[str, fr.RawFit] = {}
        for name, names in VARIANTS.items():
            t0 = time.time()
            r = fit_nested(pts, base, names, starts=scatter_starts(base, names),
                           band=a.band, fixed=FIX)
            fits[name] = r
            gain = 1 - np.sqrt(r.loss / l_now)
            bad = in_physical_range(r.extras)
            d["variants"][name] = {"par5": list(r.par5), "extras": r.extras, "loss": r.loss,
                                   "gain_train": float(gain), "guard_fired": r.guard_fired,
                                   "at_bound": r.at_bound, "unphysical": bad,
                                   "rms": {k: list(v) for k, v in r.rms.items()},
                                   "secs": time.time() - t0}
            print(f"    {name:6s} {100 * np.sqrt(r.loss):7.2f}% {100 * gain:+8.2f}% "
                  f"{'보호' if r.guard_fired else 'ok':>5s} {fmtex(r.extras):38s} "
                  f"{','.join(r.at_bound) or '-':10s}"
                  + (f"  ⚠[E5] 물리 밖 {bad}" if bad else ""))

        # ── 2b 효과 크기 — 자료가 이 요소를 볼 수 있기는 한가 ────────────────
        # "적합이 안 움직인다" 는 두 가지로 읽힌다: (i) 요소가 물리적으로 없다,
        # (ii) 있어도 이 자료로는 안 보인다. par5 를 기본에 **얼린 채** 요소를 물리값으로
        # 켜서 파형이 몇 % 움직이는지 재면 갈린다. 재현 바닥(1.3~3.0%) 아래면 (ii) 다.
        print(f"\n  [2b] 효과 크기 — par5 를 얼리고 요소를 물리값으로 켠다 (재현 바닥과 견준다)")
        eff = {}
        for n, v in PHYSICAL.items():
            l = fr.loss_at(base, {n: v}, pts, a.band)
            eff[n] = {"value": v, "loss": l, "rms": float(np.sqrt(l)),
                      "delta_rms_pp": float(100 * (np.sqrt(l) - np.sqrt(l_now)))}
            print(f"    {n}={v:<8g} 훈련 RMS {100 * np.sqrt(l):6.2f}%  "
                  f"(기본 {100 * np.sqrt(l_now):.2f}%, {eff[n]['delta_rms_pp']:+.2f}%p)")
        d["effect_size"] = eff
        # 직류 리플 — alpha 의 지렛대. 부하 전류는 (V0/vc)^alpha 로만 vc 를 보므로
        # 리플이 작으면 alpha 는 원리적으로 식별되지 않는다.
        from src.synthesis.circuit_sim import simulate as _sim
        rip = []
        for p in pts:
            r0 = _sim(p.p_w, *base, vsrc=p.vsrc)
            vdc = float(np.max(np.abs(p.v)))                    # 대략 피크로 잡는다
            dv = (p.p_w / vdc) * (1.0 / (2 * fr.F)) * (1 - r0["cond_deg"] / 180.0) / base[0]
            rip.append(100 * dv / vdc)
        d["dc_ripple_pct"] = rip
        print(f"    직류 리플 추정 {min(rip):.1f}~{max(rip):.1f}% of V_dc "
              f"-> alpha 의 지렛대는 (1±리플)^alpha 뿐이다")

        # ── 3 [E6] 겨냥 ────────────────────────────────────────────────────
        print(f"\n  [3][E6] 점별 RMS 와 par5 변화")
        hdr = "".join(f"{n:>9s}" for n in VARIANTS)
        print(f"    {'스냅샷':26s} {'P[W]':>7s} {'바닥':>7s} {'기본':>8s}{hdr}")
        base_rms = {p.stem: rms_of(base, p, a.band)[0] for p in pts}
        for p in pts:
            fl = 100 * p.scatter / np.sqrt(max(p.n_cyc, 1))
            row = "".join(f"{100 * fits[n].rms[p.stem][0]:8.1f}%" for n in VARIANTS)
            print(f"    {p.stem:26s} {p.p_w:7.2f} {fl:6.2f}% {100 * base_rms[p.stem]:7.1f}%{row}")
        d["base_rms"] = {k: v for k, v in base_rms.items()}
        print(f"    {'par5':26s}")
        print(f"      {'기본':10s} {fmt5(base)}")
        for name in VARIANTS:
            r = fits[name]
            ch = ", ".join(f"{PAR_NAMES[i]} {100 * (r.par5[i] / base[i] - 1):+.0f}%"
                           for i in range(5) if abs(r.par5[i] / base[i] - 1) > 0.05)
            print(f"      {name:10s} {fmt5(r.par5)}" + (f"   [{ch}]" if ch else "   [변화 없음]"))

        # ── 4 [E4] LOO ─────────────────────────────────────────────────────
        if not a.no_loo and len(pts) >= 3:
            print(f"\n  [4][E4] LOO — 폴드마다 기본을 다시 맞춘 뒤 그 위에서 요소를 켠다")
            t0 = time.time()
            st = base_starts(dev, cd.get("cx_measured", np.nan))
            folds = loo_folds(pts, st, band=a.band, fixed=FIX)
            lb = {p.stem: rms_of(b.par5, p, a.band)[0] for p, _, b in folds}
            mb = float(np.mean(list(lb.values())))
            res = {}
            for name, names in VARIANTS.items():
                m, per, exs = loo_nested(folds, names, starts=[], band=a.band, fixed=FIX)
                res[name] = (m, per, exs)
            d["loo"] = {"base": {"mean": mb, "per": lb},
                        **{n: {"mean": res[n][0], "per": res[n][1], "extras": res[n][2]}
                           for n in VARIANTS}}
            print(f"    {'스냅샷':26s} {'기본':>8s}{hdr}")
            for p in pts:
                row = "".join(f"{100 * res[n][1][p.stem]:8.1f}%" for n in VARIANTS)
                print(f"    {p.stem:26s} {100 * lb[p.stem]:7.1f}%{row}")
            row = "".join(f"{100 * res[n][0]:8.1f}%" for n in VARIANTS)
            print(f"    {'평균':26s} {100 * mb:7.1f}%{row}   {time.time() - t0:.0f}s")
            best = min([(mb, "기본")] + [(res[n][0], n) for n in VARIANTS])
            print(f"    -> LOO 최선: {best[1]} ({100 * best[0]:.1f}%)")
        else:
            d["loo"] = None

        # ── 5 판정 ─────────────────────────────────────────────────────────
        print(f"\n  [5] 판정")
        n_pts = len(pts)
        for name in VARIANTS:
            r = fits[name]
            gain = 1 - np.sqrt(r.loss / l_now)
            if r.guard_fired:
                v = "최적화 실패 (겹쳐 시작 보호가 걸렸다) — [E2]"
            elif gain < THR_TRAIN:
                v = f"효과 없음 (훈련 {100 * gain:+.2f}% < 문턱 {100 * THR_TRAIN:.0f}%) — [E3]"
            elif n_pts < 3 or d["loo"] is None:
                v = f"훈련은 {100 * gain:+.1f}% 줄지만 판정 불가 (동작점 {n_pts}개) — [E4]"
            else:
                dl = d["loo"][name]["mean"] - d["loo"]["base"]["mean"]
                v = (f"채택 (LOO {100 * dl:+.2f}%p)" if dl < 0
                     else f"기각 — 훈련은 줄지만 LOO 가 {100 * dl:+.2f}%p 나빠진다 [E4]")
            bad = in_physical_range(r.extras)
            if bad and "효과 없음" not in v:
                v += f"  ⚠ 물리 밖이라 '요소' 가 아니라 자리표 [E5]: {bad}"
            d["variants"][name]["verdict"] = v
            print(f"    {name:6s} {v}")
        rec["devices"][dev] = d
        print()

    # ── 6 유보 자료 — 조합 스냅샷 6개 ──────────────────────────────────────
    # LOO 보다 강한 검정이다. 조합 6개는 어떤 적합에도 들어간 적이 없고(단독 12개로만 맞췄다)
    # 파일도, 기기 구성도, 전력 배분도 다르다. `run_raw_combo_probe` [B] 와 같은 계산이다.
    if len(rec["devices"]) == 3:
        print("=" * 100)
        print("[6] 유보 자료 — 조합 스냅샷 6개 (어떤 적합에도 안 들어갔다)")
        print("=" * 100)
        combo_check(rec, canon, bg, a.band)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False, default=float)
    print(f"기록: {OUT}")


if __name__ == "__main__":
    main()
