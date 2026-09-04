# -*- coding: utf-8 -*-
"""저항 부하는 그 콘센트의 거울인가 (①a, 기준 `results/_criteria_circuit.md` [A1]~[A5]).

    python -X utf8 -m src.run_resistive_site_probe

순저항은 자기 서명이 없다 — `I_h = V_h/R` 이므로 **정규화 서명이 곧 그 콘센트의 정규화 전압
스펙트럼**이다 (`s_h = I_h/I_1 = V_h/V_1`; R 이 상쇄되므로 R 을 몰라도 된다).

지금 생성기(`grid_simulator.apply_cross_appliance_coupling`)는 녹화된 페이저를 재생하고 전압
변화를 **스칼라 배율 하나**로만 넣는다 (`mod = harmonics * kappa`). 그러면 `|I3|/|I1|` 이 녹화
당시 값에 박제된다. 12.184.15(a) 가 그 대가를 보였다 — 장소 A·B 포트 0.3~0.6% 대 장소 C 포트
3.6%, 그래서 운영점이 장소 C 에서 포트를 포트로 못 본다.

단계
----
    1  자료      저항 녹화 목록, 장소, 전력, 전류 크기(레인지)
    2  [A1]      |I_h|/|I_1| 과 |V_h|/|V_1| 을 장소별로 나란히
    3  [A2][A4]  채널 전달 |T_h| — 장소·기기·전류 크기 중 무엇을 따라가는가
    4  [A3]      ∠I_h − ∠V_h (장소 C 만, vhdeg 가 있다)
    5  [A5]      교차 재현 — 한 장소 전압으로 다른 장소 저항 부하를 낼 수 있는가
"""
from typing import Dict, List, Optional, Tuple
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np
import pandas as pd

from src.preprocessing.file_registry import DEVICE_FILES, LoadClass, site_of

H = 15
OUT = "results/_resistive_site.json"
#: 저항 부하 녹화. 드라이기는 모터가 있어 순저항이 아니다 — 대조군으로만 넣는다.
DRYER = ("hair_dryer_1", "hair_dryer_2", "hair_dryer_3", "hair_dryer_4C")


def resistive_stems() -> List[str]:
    """저항 녹화의 **파일 stem** 목록. `APPLIANCE_SPECS` 는 기기 **종류**로 색인돼 있어
    (`_build_appliance_index`) 파일별로 못 돈다 — 파일 색인은 `DEVICE_FILES` 다."""
    import os
    out = [stem for stem, spec in DEVICE_FILES.items()
           if spec.load_class is LoadClass.RESISTIVE and site_of(stem)
           and stem not in DRYER and os.path.exists(f"data/{stem}.csv")]
    return sorted(out, key=lambda s: (site_of(s), s))


def load_point(stem: str, data_dir: str = "data") -> Optional[Dict]:
    """켜진 창의 중앙값 페이저. 반환 I(15,)·V(15,) complex, 위상 관례는 펌웨어 그대로.

    ⚠ `steady_signature` 는 `range==0`(LOW) 사이클만 쓴다 — 포트·오븐은 HIGH 라 전부 버려진다.
    여기서는 레인지를 가리지 않고 대신 `range` 비율을 함께 적는다.
    """
    from src.preprocessing.raw_csv import read_raw_csv
    cols = (["p_w", "vrms", "over_range", "range"]
            + [f"ih{h}" for h in range(1, H + 1)] + [f"ihdeg{h}" for h in range(1, H + 1)]
            + [f"vh{h}" for h in range(1, H + 1)])
    try:
        df, _ = read_raw_csv(f"{data_dir}/{stem}.csv", usecols=cols)
    except Exception as e:
        print(f"    {stem}: 읽기 실패 {e}")
        return None
    has_vdeg = True
    try:
        vd = pd.read_csv(f"{data_dir}/{stem}.csv",
                         usecols=[f"vhdeg{h}" for h in range(1, H + 1)])
    except Exception:
        has_vdeg = False
        vd = None
    p = df["p_w"].to_numpy(float)
    # "켜짐" = 그 파일 최대 전력의 절반 이상 (저항 부하는 계단이 크다)
    on = (p > 0.5 * np.nanmax(p)) & (df["over_range"].to_numpy() == 0)
    if on.sum() < 20:
        return None
    Im = np.stack([df[f"ih{h}"].to_numpy(float) for h in range(1, H + 1)], 1)[on]
    Id = np.stack([df[f"ihdeg{h}"].to_numpy(float) for h in range(1, H + 1)], 1)[on]
    Vm = np.stack([df[f"vh{h}"].to_numpy(float) for h in range(1, H + 1)], 1)[on]
    I = Im * np.exp(1j * np.radians(Id))
    V = Vm.astype(complex)
    if has_vdeg:
        Vd = np.stack([vd[f"vhdeg{h}"].to_numpy(float) for h in range(1, H + 1)], 1)[on]
        V = Vm * np.exp(1j * np.radians(Vd))
    med = lambda X: np.median(X.real, 0) + 1j * np.median(X.imag, 0)
    return {"stem": stem, "site": site_of(stem), "n": int(on.sum()),
            "p_w": float(np.median(p[on])), "vrms": float(np.median(df["vrms"].to_numpy(float)[on])),
            "I": med(I), "V": med(V), "has_vdeg": has_vdeg,
            "range_hi": float(np.mean(df["range"].to_numpy()[on] != 0))}


def main() -> None:
    stems = resistive_stems() + [d for d in DRYER if site_of(d)]
    print("=" * 104)
    print("[1] 저항 녹화 자료")
    print("=" * 104)
    pts = []
    for s in stems:
        q = load_point(s)
        if q is None:
            continue
        q["dryer"] = s in DRYER
        pts.append(q)
    print(f"  {'스냅샷':26s} {'장소':>4s} {'P[W]':>8s} {'Vrms':>7s} {'|I1|[A]':>8s} "
          f"{'HIGH':>6s} {'vhdeg':>6s} {'n':>7s}")
    for q in pts:
        print(f"  {q['stem']:26s} {q['site']:>4s} {q['p_w']:8.1f} {q['vrms']:7.1f} "
              f"{abs(q['I'][0]):8.3f} {100*q['range_hi']:5.0f}% {'있음' if q['has_vdeg'] else '없음':>6s} "
              f"{q['n']:7d}" + ("   ⚠드라이기(모터, 대조군)" if q["dryer"] else ""))

    ODD = [3, 5, 7, 9, 11, 13, 15]
    hdr = "".join(f"{h:>8d}" for h in ODD)

    # ── 2 [A1] 전류 서명과 전압 스펙트럼을 나란히 ──────────────────────────
    print("\n" + "=" * 104)
    print("[2][A1] |I_h|/|I_1| (전류 서명) 과 |V_h|/|V_1| (그 콘센트의 전압) — 둘이 같이 움직이는가 [%]")
    print("=" * 104)
    print(f"  {'스냅샷':26s} {'장소':>4s} {'':>6s}{hdr}")
    for q in pts:
        i = 100 * np.abs(q["I"]) / abs(q["I"][0])
        v = 100 * np.abs(q["V"]) / abs(q["V"][0])
        print(f"  {q['stem']:26s} {q['site']:>4s} {'I':>6s}" + "".join(f"{i[h-1]:8.2f}" for h in ODD)
              + ("   드라이기" if q["dryer"] else ""))
        print(f"  {'':26s} {'':>4s} {'V':>6s}" + "".join(f"{v[h-1]:8.2f}" for h in ODD))

    # 장소별 평균 (드라이기 제외)
    print(f"\n  {'장소 평균 (순저항만)':26s} {'':>4s} {'':>6s}{hdr}")
    for site in ("A", "B", "C"):
        g = [q for q in pts if q["site"] == site and not q["dryer"]]
        if not g:
            continue
        i = np.mean([100 * np.abs(q["I"]) / abs(q["I"][0]) for q in g], 0)
        v = np.mean([100 * np.abs(q["V"]) / abs(q["V"][0]) for q in g], 0)
        print(f"  {'장소 ' + site + f' ({len(g)}개)':26s} {'':>4s} {'I':>6s}"
              + "".join(f"{i[h-1]:8.2f}" for h in ODD))
        print(f"  {'':26s} {'':>4s} {'V':>6s}" + "".join(f"{v[h-1]:8.2f}" for h in ODD))

    # ── 3 [A2][A4] 채널 전달 ───────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("[3][A2][A4] |T_h| = (|I_h|/|I_1|)/(|V_h|/|V_1|) — 순저항이면 1")
    print("=" * 104)
    print(f"  {'스냅샷':26s} {'장소':>4s} {'|I1|':>6s} {'HIGH':>5s}{hdr}")
    rec_t = {}
    for q in pts:
        t = (np.abs(q["I"]) / abs(q["I"][0])) / (np.abs(q["V"]) / abs(q["V"][0]))
        rec_t[q["stem"]] = list(t)
        print(f"  {q['stem']:26s} {q['site']:>4s} {abs(q['I'][0]):6.2f} {100*q['range_hi']:4.0f}%"
              + "".join(f"{t[h-1]:8.2f}" for h in ODD) + ("   드라이기" if q["dryer"] else ""))
    low = [q for q in pts if q["range_hi"] < 0.5 and not q["dryer"]]
    hi = [q for q in pts if q["range_hi"] >= 0.5 and not q["dryer"]]
    for lab, g in (("LOW 에 머무는 저항", low), ("HIGH 로 가는 저항", hi)):
        if not g:
            continue
        t = np.array([[(np.abs(q["I"]) / abs(q["I"][0]) / (np.abs(q["V"]) / abs(q["V"][0])))[h-1]
                       for h in ODD] for q in g])
        print(f"\n  {lab} {len(g)}개  중앙값 " + "".join(f"{v:8.2f}" for v in np.median(t, 0)))
        print(f"  {'':26s}   산포   " + "".join(f"{v:8.2f}" for v in t.std(0)))

    # ── 4 [A3] 위상 (장소 C) ───────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("[4][A3] ∠I_h − ∠V_h [°] — 순저항이면 0 (vhdeg 가 있는 녹화만)")
    print("=" * 104)
    print(f"  {'스냅샷':26s} {'장소':>4s} {'':>6s}{hdr}")
    for q in pts:
        if not q["has_vdeg"]:
            continue
        d = np.degrees(np.angle(q["I"] / q["V"]))
        d = (d - d[0] + 180) % 360 - 180          # 기본파를 0 으로 (채널 지연 교정 상태 무관)
        print(f"  {q['stem']:26s} {q['site']:>4s} {'':>6s}"
              + "".join(f"{d[h-1]:8.1f}" for h in ODD) + ("   드라이기" if q["dryer"] else ""))

    # ── 5 [A5] 교차 재현 ───────────────────────────────────────────────────
    print("\n" + "=" * 104)
    print("[5][A5] 교차 재현 — 다른 장소 전압으로 이 저항 부하의 서명을 낼 수 있는가")
    print("=" * 104)
    print("  예측 s_h = V_h/V_1 (그 장소 전압). 관측 = 그 녹화의 I_h/I_1. h3 [%] 로 본다.")
    print(f"  {'저항 녹화':26s} {'장소':>4s} {'관측 h3':>8s} {'자기 장소 V':>11s} "
          f"{'장소A V':>9s}{'장소B V':>9s}{'장소C V':>9s}")
    vsite = {}
    for site in ("A", "B", "C"):
        g = [q for q in pts if q["site"] == site and not q["dryer"]]
        if g:
            vsite[site] = np.mean([np.abs(q["V"]) / abs(q["V"][0]) for q in g], 0)
    for q in pts:
        if q["dryer"]:
            continue
        i3 = 100 * abs(q["I"][2]) / abs(q["I"][0])
        own = 100 * (np.abs(q["V"]) / abs(q["V"][0]))[2]
        oth = "".join(f"{100 * vsite[k][2]:8.2f}%" if k in vsite else f"{'-':>9s}"
                      for k in ("A", "B", "C"))
        print(f"  {q['stem']:26s} {q['site']:>4s} {i3:7.2f}% {own:10.2f}% " + oth)

    # ── 6 절대값으로 — 덧셈 인공물인가, 참 전압인가 ────────────────────────
    # 상대값(|I_h|/|I_1|)만 보면 같은 장소 안에서도 기기마다 3~4배 갈린다. 그런데 기기마다
    # |I_1| 이 2.1~6.4A 로 3배 다르다. **절대 |I_h| [mA]** 로 보면 무엇이 무엇에 붙어 있는지
    # 갈린다: 참 전압에서 온 것이면 I_h = (V_h/V_1)·I_1 이라 |I_1| 에 비례하고,
    # 계측 경로의 덧셈 인공물이면 |I_1| 과 무관하게 일정하다.
    print()
    print("=" * 104)
    print("[6] 절대 |I_h| [mA] — |I_1| 에 비례하는가(참 전압) 일정한가(덧셈 인공물)")
    print("=" * 104)
    print(f"  {'스냅샷':26s} {'장소':>4s} {'|I1|[A]':>8s}{hdr}")
    for q in pts:
        if q["dryer"]:
            continue
        a = 1000 * np.abs(q["I"])
        print(f"  {q['stem']:26s} {q['site']:>4s} {abs(q['I'][0]):8.2f}"
              + "".join(f"{a[h-1]:8.1f}" for h in ODD))
    for site in ("A", "B", "C"):
        g = [q for q in pts if q["site"] == site and not q["dryer"]]
        if len(g) < 2:
            continue
        a = np.array([1000 * np.abs(q["I"]) for q in g])
        i1 = np.array([abs(q["I"][0]) for q in g])
        cv_abs = a.std(0) / np.maximum(a.mean(0), 1e-9)          # 절대값의 변동계수
        r = a / i1[:, None]
        cv_rel = r.std(0) / np.maximum(r.mean(0), 1e-9)          # 상대값의 변동계수
        print(f"  {'장소 ' + site + ' 절대 변동계수':26s} {'':>4s} {'':>8s}"
              + "".join(f"{cv_abs[h-1]:8.2f}" for h in ODD))
        print(f"  {'  같은 것 상대':26s} {'':>4s} {'':>8s}"
              + "".join(f"{cv_rel[h-1]:8.2f}" for h in ODD)
              + "    <- 작은 쪽이 그 차수의 실체")

    # ── 7 장소 C 에서 전압 채널 인공물을 직접 푼다 ──────────────────────────
    # 순저항이면 V_true,h = I_h·(V_1/I_1). vhdeg 가 있는 장소 C 에서는 복소로 풀 수 있다:
    #     A_h = V_meas,h − I_h·(V_1/I_1)
    # 이것이 같은 장소의 서로 다른 기기(포트·드라이기)에서 같으면 계측 경로의 성질이다.
    print()
    print("=" * 104)
    print("[7] 장소 C: 전압 채널 인공물 A_h = V_meas,h − I_h·(V_1/I_1)  [V_1 의 %]")
    print("=" * 104)
    print(f"  {'스냅샷':26s} {'':>10s}{hdr}")
    for q in pts:
        if not q["has_vdeg"]:
            continue
        Z = q["V"][0] / q["I"][0]                    # R (기본파에서. 복소지만 저항이면 실수)
        vt = q["I"] * Z                              # 참 전압 (저항이 본 것)
        A = (q["V"] - vt) / abs(q["V"][0])
        print(f"  {q['stem']:26s} {'|V_true|':>10s}"
              + "".join(f"{100 * abs(vt[h-1]) / abs(q['V'][0]):8.2f}" for h in ODD)
              + ("   드라이기" if q["dryer"] else ""))
        print(f"  {'':26s} {'|A_h|':>10s}" + "".join(f"{100 * abs(A[h-1]):8.2f}" for h in ODD))
        print(f"  {'':26s} {'∠A_h':>10s}"
              + "".join(f"{np.degrees(np.angle(A[h-1])):8.0f}" for h in ODD))

    # ── 8 [A5] 두 성분 분해와 홀드아웃 ──────────────────────────────────────
    # [6] 이 말한 것: h3 은 |I_1| 과 무관한 **절대** 양(장소 A 12~32mA)이고 h5·h9·h15 는
    # |I_1| 에 **비례**한다. 그러면 저항 부하의 차수별 전류는 두 항이다
    #
    #     |I_h| = a_h          (계측 경로의 덧셈 바닥 [A], 장소 무관 가설)
    #           + d_h · |I_1|  (그 콘센트의 전압 왜곡 [무차원])
    #
    # 장소 A 8개로 두 항을 최소제곱으로 가른다. 장소 B 는 2개(정확히 결정), 장소 C 는 1개라
    # a_h 를 장소 A 값으로 고정하고 d_h 만 푼다. 검정은 홀드아웃 — 한 녹화를 빼고 맞춘 뒤
    # 그 녹화의 |I_h| 를 예측한다. 지금 생성기(고정 서명 재생 + 스칼라 배율)와 견준다.
    print()
    print("=" * 104)
    print("[8][A5] |I_h| = a_h + d_h·|I_1| 분해와 홀드아웃 예측")
    print("=" * 104)
    fit_sites = {}
    for site in ("A", "B"):
        g = [q for q in pts if q["site"] == site and not q["dryer"]]
        if len(g) < 2:
            continue
        i1 = np.array([abs(q["I"][0]) for q in g])
        M = np.stack([np.ones_like(i1), i1], 1)
        a, d = [], []
        for h in ODD:
            y = np.array([abs(q["I"][h - 1]) for q in g])
            c, *_ = np.linalg.lstsq(M, y, rcond=None)
            a.append(max(c[0], 0.0))
            d.append(max(c[1], 0.0))
        fit_sites[site] = (np.array(a), np.array(d))
        print(f"  장소 {site} ({len(g)}개)  a_h [mA] " + "".join(f"{1000*v:8.1f}" for v in a))
        print(f"  {'':16s}  d_h [%]   " + "".join(f"{100*v:8.2f}" for v in d))
    # 장소 C: a_h 를 장소 A 값으로 고정하고 d_h 만
    gc = [q for q in pts if q["site"] == "C" and not q["dryer"]]
    if gc and "A" in fit_sites:
        aA = fit_sites["A"][0]
        q = gc[0]
        d = np.array([max(abs(q["I"][h - 1]) - aA[j], 0.0) / abs(q["I"][0])
                      for j, h in enumerate(ODD)])
        fit_sites["C"] = (aA, d)
        print(f"  장소 C (1개)   a_h [mA] " + "".join(f"{1000*v:8.1f}" for v in aA) + "   <- A 에서 고정")
        print(f"  {'':16s}  d_h [%]   " + "".join(f"{100*v:8.2f}" for v in d))
    print()
    print("  왜곡 d_3: 장소 A " + f"{100*fit_sites['A'][1][0]:.2f}%" +
          ("  B " + f"{100*fit_sites['B'][1][0]:.2f}%" if "B" in fit_sites else "") +
          ("  C " + f"{100*fit_sites['C'][1][0]:.2f}%" if "C" in fit_sites else "") +
          "   <- 12.184.15(a) 가 말한 장소축")

    # 홀드아웃 (장소 A 만 — 8개라 뺄 수 있다)
    gA = [q for q in pts if q["site"] == "A" and not q["dryer"]]
    print()
    print("  홀드아웃 (장소 A, 하나 빼고 맞춘 뒤 예측). 상대 오차 = |예측−관측|/관측 의 차수 평균")
    print(f"  {'뺀 녹화':26s} {'|I1|':>6s} {'두 성분':>9s} {'비례만':>9s} {'절대만':>9s} {'고정서명(지금)':>14s}")
    errs = {"two": [], "prop": [], "abs": [], "fixed": []}
    for k, qk in enumerate(gA):
        tr = [q for j, q in enumerate(gA) if j != k]
        i1 = np.array([abs(q["I"][0]) for q in tr])
        M = np.stack([np.ones_like(i1), i1], 1)
        e = {"two": [], "prop": [], "abs": [], "fixed": []}
        for j, h in enumerate(ODD):
            y = np.array([abs(q["I"][h - 1]) for q in tr])
            obs = abs(qk["I"][h - 1])
            c, *_ = np.linalg.lstsq(M, y, rcond=None)
            e["two"].append(abs(max(c[0], 0) + max(c[1], 0) * abs(qk["I"][0]) - obs) / obs)
            e["prop"].append(abs(np.mean(y / i1) * abs(qk["I"][0]) - obs) / obs)
            e["abs"].append(abs(np.mean(y) - obs) / obs)
            # 지금 생성기: 훈련 녹화 하나의 서명을 그대로 재생 (전압 배율은 h 에 공통이라 비 불변)
            e["fixed"].append(abs(abs(tr[0]["I"][h - 1]) / abs(tr[0]["I"][0]) * abs(qk["I"][0]) - obs) / obs)
        for kk in errs:
            errs[kk].append(np.mean(e[kk]))
        print(f"  {qk['stem']:26s} {abs(qk['I'][0]):6.2f} {100*np.mean(e['two']):8.1f}% "
              f"{100*np.mean(e['prop']):8.1f}% {100*np.mean(e['abs']):8.1f}% "
              f"{100*np.mean(e['fixed']):13.1f}%")
    print(f"  {'평균':26s} {'':>6s} {100*np.mean(errs['two']):8.1f}% "
          f"{100*np.mean(errs['prop']):8.1f}% {100*np.mean(errs['abs']):8.1f}% "
          f"{100*np.mean(errs['fixed']):13.1f}%")
    rec_sites = {k: {"a_h": list(v[0]), "d_h": list(v[1]), "orders": ODD}
                 for k, v in fit_sites.items()}

    json.dump({"points": [{k: (list(np.asarray(v).astype(complex).view(float))
                               if k in ("I", "V") else v) for k, v in q.items()} for q in pts],
               "T": rec_t, "sites": rec_sites}, open(OUT, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=float)
    print(f"\n기록: {OUT}")


if __name__ == "__main__":
    main()
