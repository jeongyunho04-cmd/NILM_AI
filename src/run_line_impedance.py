"""대전력 계단에서 선로 임피던스를 추정한다 (12.167).

    Z1 = -(V1_after - V1_before) / (|I1|_after - |I1|_before)

전압의 **차분**을 쓰므로 스펙트럼 누설과 계통 기준 전압의 느린 표류가 상당 부분
상쇄된다. 부록 A 가 제안한 방법을 우리 데이터 형식으로 다시 구현한 것이다
(원본은 외부 CSV 경로에 묶여 있었다).

    python -m src.run_line_impedance
"""
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

NPZ = "processed_data/composite_eval/{}.npz"
#: 장소 구분 (12.161). test_14 는 이사 당일이라 어느 쪽도 아니다.
SITE = {**{s: "A" for s in ("test.2", "test3", "test_4", "test_5", "test_6",
                            "test_7", "test_8", "test_9", "test_10",
                            "test_11", "test_12", "test_13")},
        **{s: "B" for s in ("test_15", "test_16", "test_17", "test_18")}}


def measure(stem, events, min_dp=300.0, min_di=0.5, pre=(3, 12), post=(3, 12)):
    z = np.load(NPZ.format(stem))
    C = z["harmonics_complex"]
    pf = z["power_features"]
    V = pf[:, 4].astype(float)                 # vrms
    T = z["t_rel_s"].astype(float)
    good = z["is_valid"].astype(bool)
    out = []
    for e in events:
        if abs(e.get("delta_p_w", 0.0)) < min_dp:
            continue
        t0 = float(e["t_s"])
        a = (T > t0 - pre[1]) & (T < t0 - pre[0]) & good
        b = (T > t0 + post[0]) & (T < t0 + post[1]) & good
        if a.sum() < 60 or b.sum() < 60:
            continue
        dI = float(np.abs(C[b, 0]).mean() - np.abs(C[a, 0]).mean())
        dV = float(V[b].mean() - V[a].mean())
        if abs(dI) < min_di:
            continue
        out.append(dict(t=t0, dev=e.get("appliance", "?"), kind=e.get("kind", "?"),
                        dP=float(e["delta_p_w"]), dI=dI, dV=dV, Z=-dV / dI))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="선로 임피던스 추정")
    ap.add_argument("--events", nargs="+",
                    default=["processed_data/real_events_refined.json",
                             "processed_data/real_events.json"])
    ap.add_argument("--min-dp", type=float, default=300.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    files = {}
    for f in a.events:                                  # 앞의 파일이 우선
        try:
            for k, v in json.load(open(f, encoding="utf-8"))["files"].items():
                files.setdefault(k, v)
        except FileNotFoundError:
            pass

    by, rows = {}, []
    for stem in sorted(files, key=lambda s: (SITE.get(s, "?"), s)):
        ev = files[stem].get("events")
        if not ev:
            continue
        try:
            r = measure(stem, ev, min_dp=a.min_dp)
        except FileNotFoundError:
            continue
        if r:
            by[stem] = [x["Z"] for x in r]
            rows += [(stem, x) for x in r]

    if a.verbose:
        print(f"{'파일':9s}{'t':>8s}{'기기':>18s}{'kind':>6s}{'ΔP(W)':>9s}"
              f"{'ΔI(A)':>9s}{'ΔV(V)':>9s}{'Z1(Ω)':>9s}")
        for stem, x in rows:
            print(f"{stem:9s}{x['t']:8.1f}{x['dev']:>18s}{x['kind']:>6s}"
                  f"{x['dP']:+9.1f}{x['dI']:+9.3f}{x['dV']:+9.3f}{x['Z']:9.3f}")
        print()

    print(f"{'파일':9s}{'장소':>5s}{'N':>5s}{'Z 중앙':>9s}{'Z 표준편차':>11s}")
    for stem, zs in by.items():
        print(f"{stem:9s}{SITE.get(stem, '?'):>5s}{len(zs):5d}"
              f"{np.median(zs):9.3f}{np.std(zs):11.3f}")
    print()
    for site in ("A", "B"):
        zs = [z for s, v in by.items() if SITE.get(s) == site for z in v]
        med = [np.median(v) for s, v in by.items() if SITE.get(s) == site]
        if zs:
            print(f"  장소 {site}: 이벤트 {len(zs):3d}개  Z 중앙 {np.median(zs):.3f}Ω"
                  f"  (파일 중앙들 {', '.join(f'{m:.2f}' for m in med)})")
    za = [np.median(v) for s, v in by.items() if SITE.get(s) == "A"]
    zb = [np.median(v) for s, v in by.items() if SITE.get(s) == "B"]
    if za and zb:
        print(f"\n  장소 A / 장소 B = {np.median(za) / np.median(zb):.2f}배")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
