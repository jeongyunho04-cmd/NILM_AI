# -*- coding: utf-8 -*-
"""녹화가 어느 LOW 교정값으로 찍혔는지 자료로 판정한다 (12.185.x).

    python -X utf8 -m src.run_cal_epoch_probe

왜 필요한가
-----------
2026-09-04 저녁에 펌웨어 LOW 위상 교정을 0.44 -> 2.62 로 바꿨다가 되돌렸다. 그 사이에 찍힌
파일은 모든 차수가 −2.18°×h 만큼 돌아 있다. 등록부(`LOW_CAL_OVERRIDE`)는 `beam_projector_4C` /
`test_19` / `test_20` 을 "옛 교정(0.44)" 으로 적어 두었지만 그 근거는 **대기전류 h1 위상 하나**
였다 (12.184.15). h1 하나로는 h 선형 회전을 못 재고, 대기 부하는 파일마다 다르다.

방법
----
교정값 차이는 **h 에 선형인 순수 회전 (크기비 1)** 이다. 같은 부하를 담은 두 창의 위상차를
h 로 회귀하면 기울기가 곧 교정값 차이다.
  [A] 상태 비교   같은 상태를 담은 두 창을 직접 뺀다 (파일 안 자기 일관성 / 파일 사이).
  [B] 계단 델타   기기 하나가 켜지고 꺼진 앞뒤 창의 차 ΔI 는 그 기기의 전류다 (배경 상쇄).
                  단독 녹화의 같은 전력 델타와 맞댄다.
  [E] 교란 배제   ⚠ 장소 전압 파형이 달라져도 펄스가 밀려 **같은 모양의 회전**이 난다 (12.184.2).
                  두 창의 전압 페이저를 각각 회로 모델에 넣어 모델이 예측하는 회전을 빼야
                  남는 몫이 교정값 차이다. 이 단계 없이는 [A][B] 만으로 단정하면 안 된다.

결론 (12.185.2, 2026-09-05)
---------------------------
  · test_19 <-> test_20 : 기울기 −0.4°/차수, 잔차 0.6° -> **같은 판**
  · minipc_4C -> test_19 (같은 기기·같은 동작점 6.5W, 크기비 1.01~1.03):
      −2.36 / −2.56 / −2.57 / −2.70 / −2.90 °/차수 (창 조합 다섯)
      전압 차가 내는 몫은 모델로 +0.33°/차수 -> 남는 몫 약 −2.8°/차수
  · 자기 잡음 바닥: 같은 파일·같은 기기 +0.15~+0.68°/차수 (프로젝터는 예열로 흐른다)
  -> test_19/20 은 다른 교정 판에서 찍혔다. 0.44->2.62 가 예측하는 −2.18 보다 0.6° 더 크다
     (플래시된 값이 정확히 무엇이었는지는 모른다). **사용자가 두 파일을 삭제하기로 했다.**
  · beam_projector_4C 는 이 방법으로 판정 불가 — 프로젝터 서명이 같은 파일 안에서 예열로
    +0.7°/차수 흐르고, 대기 부하가 미니PC 파일과 다르다 (h3/h1 0.348 vs 0.278).
"""
from typing import Dict, List, Optional, Tuple
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

from src.preprocessing.raw_csv import read_raw_csv
from src.preprocessing.file_registry import phase_fix_of

H = 15
COLS = ["t_s", "p_w", "range", "over_range"] + [f"ih{h}" for h in range(1, H + 1)] \
    + [f"ihdeg{h}" for h in range(1, H + 1)]
#: 판정에 쓰는 차수 (크기가 충분한 홀수차)
ORD = np.array([1, 3, 5, 7, 9, 11])


def load(stem: str):
    """(t, p, range, 절대 전류 페이저 (N,15)). 없는 파일이면 None. 읽기 경로 회전을 건다 (5C 의 +10.8°×h)."""
    import os
    if not os.path.exists(f"data/{stem}.csv"):
        return None
    df, info = read_raw_csv(f"data/{stem}.csv", usecols=COLS)
    t = df["t_s"].to_numpy(float)
    p = df["p_w"].to_numpy(float)
    rg = df["range"].to_numpy(int)
    ov = df["over_range"].to_numpy(int)
    C = np.stack([df[f"ih{h}"].to_numpy(float) * np.exp(1j * np.deg2rad(df[f"ihdeg{h}"].to_numpy(float)))
                  for h in range(1, H + 1)], 1)
    C[ov != 0] = np.nan
    return t, p, rg, C


def window(t, p, rg, C, t0: float, t1: float) -> Optional[Dict]:
    """[t0,t1) 창의 대표 페이저 (크기 중앙값 · 위상 원형평균) 와 통계."""
    m = (t >= t0) & (t < t1) & np.isfinite(C[:, 0])
    if m.sum() < 100:
        return None
    X = C[m]
    ph = np.angle(np.exp(1j * np.angle(X)).mean(0))
    mag = np.median(np.abs(X), 0)
    # 위상 산포 (원형 표준편차) — 잡음 판정에 쓴다
    r = np.abs(np.exp(1j * np.angle(X)).mean(0))
    sd = np.degrees(np.sqrt(np.maximum(-2 * np.log(np.maximum(r, 1e-12)), 0)))
    return {"I": mag * np.exp(1j * ph), "n": int(m.sum()), "p_w": float(np.median(p[m])),
            "high": float((rg[m] == 1).mean()), "sd": sd, "t": (t0, t1)}


def slope(a: np.ndarray, b: np.ndarray, orders: np.ndarray = ORD) -> Tuple[float, float, float]:
    """a 를 기준으로 b 의 위상차를 h 로 회귀. (기울기 [°/차수], 절편, 가중 잔차 [°])"""
    k = orders - 1
    d = np.degrees(np.angle(b[k] / a[k]))
    # 절편이 0 에 가깝도록 h1 을 먼저 맞춘 뒤 감는다 (2π 넘김 방지)
    d = (d - d[0] * orders / orders[0] + 180) % 360 - 180 + d[0] * orders / orders[0]
    w = np.minimum(np.abs(a[k]), np.abs(b[k]))
    w = w / w.sum()
    A = np.c_[orders.astype(float), np.ones(len(orders))]
    sol, *_ = np.linalg.lstsq(A * np.sqrt(w)[:, None], d * np.sqrt(w), rcond=None)
    res = float(np.sqrt(np.sum(w * (d - A @ sol) ** 2)))
    return float(sol[0]), float(sol[1]), res


def verdict(s: float, res: float) -> str:
    if res > 4.0:
        return f"판정 불가 (잔차 {res:.1f}°)"
    if abs(s) < 0.7:
        return "같은 교정값"
    if abs(s + 2.18) < 0.7:
        return "상대가 2.62 로 찍혔다"
    if abs(s - 2.18) < 0.7:
        return "기준이 2.62 로 찍혔다"
    return f"어느 쪽도 아님 ({s:+.2f}°/차수)"


#: 판정에 쓸 창 — 전력 계단표에서 고른 안정 구간 (전부 LOW, over_range 없음).
#: test_19 꼬리는 3종 -> 프로젝터 off -> 충전기 off -> 미니PC off 순이라 기기별 델타가 셋 다 나온다.
WINDOWS: Dict[str, List[Tuple[str, float, float]]] = {
    "test_19":           [("3종 126W", 250, 278), ("3종 126W b", 332, 382), ("프로젝터off 82W", 388, 405),
                          ("충전기off 11W", 410, 419), ("미니PCoff 4.8W", 424, 442)],
    "test_20":           [("3종 125W", 62, 218), ("3종 125W b", 516, 570)],
    "minipc_4C":         [("대기", 1042, 1155), ("IDLE 9W", 64, 230), ("12W", 918, 1030), ("22W", 798, 914)],
    "laptop_charger_5C": [("대기", 2, 55), ("72W", 100, 400)],
    "beam_projector_4C": [("대기", 270, 327), ("대기 b", 918, 954), ("ON 49W", 67, 262), ("ON 49W b", 575, 910)],
}


def main() -> None:
    print("창 요약 (읽기 경로 회전 적용: 5C 만 +10.8°×h)")
    W: Dict[str, Dict[str, Dict]] = {}
    for stem, wins in WINDOWS.items():
        got = load(stem)
        if got is None:
            print(f"\n  {stem}  — 파일 없음 (삭제됨), 건너뜀")
            continue
        t, p, rg, C = got
        W[stem] = {}
        fx = phase_fix_of(stem)
        print(f"\n  {stem}  (PHASE_FIX {fx:+.1f}°/차수)")
        for name, t0, t1 in wins:
            w = window(t, p, rg, C, t0, t1)
            if w is None:
                print(f"    {name:16s} 표본 부족")
                continue
            W[stem][name] = w
            print(f"    {name:16s} {t0:5.0f}~{t1:5.0f}s n={w['n']:6d}  P {w['p_w']:7.1f}W  "
                  f"HIGH {100 * w['high']:3.0f}%  |I1| {abs(w['I'][0]) * 1000:6.1f}mA  "
                  f"∠I1 {np.degrees(np.angle(w['I'][0])):+6.1f}°  ∠산포 h3 {w['sd'][2]:4.1f}°")

    print("\n" + "=" * 78)
    print("[A] 같은 상태 직접 비교 — SMPS 3종 켠 125W 창")
    print("=" * 78)
    pairs_a = [
        ("test_19", "3종 126W", "test_19", "3종 126W b"),      # 같은 파일 안 (자기 일관성)
        ("test_20", "3종 125W", "test_20", "3종 125W b"),      # 같은 파일 안
        ("test_19", "3종 126W", "test_20", "3종 125W"),        # 파일 사이 (7분 차이)
        ("test_19", "3종 126W b", "test_20", "3종 125W b"),
    ]
    for f1, w1, f2, w2 in pairs_a:
        if w1 not in W.get(f1, {}) or w2 not in W.get(f2, {}):
            continue
        s, b, r = slope(W[f1][w1]["I"], W[f2][w2]["I"])
        print(f"  {f1}:{w1:12s} -> {f2}:{w2:12s}  기울기 {s:+6.2f}°/차수  절편 {b:+5.1f}°  "
              f"잔차 {r:4.1f}°   {verdict(s, r)}")

    print("\n" + "=" * 78)
    print("[B] 계단 델타 — 기기 하나가 꺼진 앞뒤 창의 차 (배경 상쇄)")
    print("=" * 78)
    deltas: Dict[str, np.ndarray] = {}
    for stem, a, b, label in [
        ("test_19", "프로젝터off 82W", "3종 126W b", "T19 Δ프로젝터"),
        ("test_19", "충전기off 11W", "프로젝터off 82W", "T19 Δ충전기"),
        ("test_19", "미니PCoff 4.8W", "충전기off 11W", "T19 Δ미니PC"),
        ("minipc_4C", "대기", "IDLE 9W", "단독 Δ미니PC"),
        ("laptop_charger_5C", "대기", "72W", "단독 Δ충전기"),
        ("beam_projector_4C", "대기", "ON 49W", "단독 Δ프로젝터"),
        ("beam_projector_4C", "대기 b", "ON 49W b", "단독 Δ프로젝터 b"),
    ]:
        if a not in W.get(stem, {}) or b not in W.get(stem, {}):
            continue
        d = W[stem][b]["I"] - W[stem][a]["I"]
        deltas[label] = d
        print(f"  {label:20s} ΔP {W[stem][b]['p_w'] - W[stem][a]['p_w']:7.1f}W  "
              f"|ΔI1| {abs(d[0]) * 1000:6.1f}mA  ∠ΔI1 {np.degrees(np.angle(d[0])):+6.1f}°  "
              f"h3/h1 {abs(d[2]) / abs(d[0]):.3f}  h5/h1 {abs(d[4]) / abs(d[0]):.3f}")

    print("\n  같은 기기의 델타끼리 — 단독 녹화(기준) -> test_19 (대상):")
    for ka, kb in [("단독 Δ미니PC", "T19 Δ미니PC"),
                   ("단독 Δ충전기", "T19 Δ충전기"),
                   ("단독 Δ프로젝터", "T19 Δ프로젝터"),
                   ("단독 Δ프로젝터 b", "T19 Δ프로젝터")]:
        if ka not in deltas or kb not in deltas:
            continue
        s, b, r = slope(deltas[ka], deltas[kb])
        print(f"    {ka:18s} -> {kb:16s} 기울기 {s:+6.2f}°/차수  절편 {b:+5.1f}°  잔차 {r:4.1f}°   {verdict(s, r)}")

    print("\n  대조군 (같은 파일 안 · 같은 기기 — 0 이 나와야 자가 잡음을 안다):")
    for ka, kb in [("단독 Δ프로젝터", "단독 Δ프로젝터 b")]:
        if ka not in deltas or kb not in deltas:
            continue
        s, b, r = slope(deltas[ka], deltas[kb])
        print(f"    {ka:18s} -> {kb:16s} 기울기 {s:+6.2f}°/차수  절편 {b:+5.1f}°  잔차 {r:4.1f}°")

    print("\n" + "=" * 78)
    print("[C] 가장 잘 맞는 쌍의 차수별 표 — 미니PC (규칙 74: 크기 1 · 위상 h 선형 = 교정/채널)")
    print("=" * 78)
    for ka, kb in [("단독 Δ미니PC", "T19 Δ미니PC"), ("단독 Δ프로젝터", "T19 Δ프로젝터"),
                   ("단독 Δ프로젝터", "단독 Δ프로젝터 b")]:
        if ka not in deltas or kb not in deltas:
            continue
        a, b_ = deltas[ka], deltas[kb]
        print(f"  {ka} -> {kb}")
        print("    차수      " + "".join(f"h{h:<6d}" for h in range(1, 12, 2)))
        print("    크기비    " + "".join(f"{abs(b_[h - 1] / a[h - 1]):<7.3f}" for h in range(1, 12, 2)))
        print("    위상차°   " + "".join(f"{np.degrees(np.angle(b_[h - 1] / a[h - 1])):<+7.1f}" for h in range(1, 12, 2)))
        print("    /h        " + "".join(f"{np.degrees(np.angle(b_[h - 1] / a[h - 1])) / h:<+7.2f}" for h in range(1, 12, 2)))

    print("\n[D] 창을 바꿔가며 미니PC 기울기의 흔들림 (자가 잡음 대비)")
    g1, g9 = load("minipc_4C"), load("test_19")
    if g1 is None or g9 is None:
        print("    자료 없음 — 건너뜀")
        return
    t, p, rg, C = g1
    t9, p9, rg9, C9 = g9
    ref_pairs = [((1042, 1155), (64, 230)), ((1042, 1155), (268, 296)), ((232, 245), (64, 230)),
                 ((1, 8), (64, 230)), ((232, 245), (268, 296))]
    tgt = window(t9, p9, rg9, C9, 424, 442)["I"], window(t9, p9, rg9, C9, 410, 419)["I"]
    dt = tgt[1] - tgt[0]
    for (s0, s1), (o0, o1) in ref_pairs:
        wa, wb = window(t, p, rg, C, s0, s1), window(t, p, rg, C, o0, o1)
        if wa is None or wb is None:
            continue
        s, b, r = slope(wb["I"] - wa["I"], dt)
        print(f"    대기 {s0:4.0f}~{s1:4.0f}s / IDLE {o0:3.0f}~{o1:4.0f}s   "
              f"ΔP {wb['p_w'] - wa['p_w']:5.1f}W  기울기 {s:+6.2f}  잔차 {r:4.1f}°")


def stage_e() -> None:
    """[E] 교란 배제 — 전압 파형이 달라도 h 선형 회전이 난다 (12.184.2). 전압은 교정과 무관하다.

    전압 페이저를 직접 비교하고, 회로 모델에 두 전압을 각각 넣어 **모델이 예측하는 회전**을 뺀다.
    남는 회전이 곧 교정값 차이다.
    """
    import numpy as np
    from src.preprocessing.raw_phasors import voltage_phasors
    from src.synthesis import circuit_fit as cf
    from src.synthesis.circuit_sim import simulate
    from src.preprocessing.raw_csv import read_raw_csv

    print("\n" + "=" * 78)
    print("[E] 전압 교란 배제 — 같은 창의 전압 페이저와 모델이 예측하는 회전")
    print("=" * 78)
    vcols = ["t_s", "p_w", "range", "over_range", "vrms"] + [f"vh{h}" for h in range(1, H + 1)] \
        + [f"vhdeg{h}" for h in range(1, H + 1)]
    got = {}
    for stem, t0, t1, label in [("minipc_4C", 64, 230, "minipc_4C IDLE"),
                                ("minipc_4C", 1042, 1155, "minipc_4C 대기"),
                                ("test_19", 410, 419, "test_19 11W"),
                                ("test_19", 424, 442, "test_19 4.8W"),
                                ("beam_projector_4C", 67, 262, "projector ON"),
                                ("test_19", 332, 382, "test_19 126W")]:
        df, _ = read_raw_csv(f"data/{stem}.csv", usecols=vcols)
        t = df["t_s"].to_numpy(float)
        m = (t >= t0) & (t < t1)
        V, st = voltage_phasors(df, m)
        got[label] = V
        print(f"  {label:18s} Vrms {abs(V[0]):6.2f}  " +
              "  ".join(f"h{h}:{abs(V[h - 1]):5.2f}V∠{np.degrees(np.angle(V[h - 1])):+6.1f}°"
                        for h in (3, 5, 7, 9)))

    print("\n  전압 차 (test_19 − minipc_4C, 같은 부하 창):")
    a, b = got["minipc_4C IDLE"], got["test_19 11W"]
    for h in (3, 5, 7, 9, 11):
        print(f"    h{h:<2d} 크기 {abs(a[h-1]):5.2f} -> {abs(b[h-1]):5.2f} V   "
              f"위상 {np.degrees(np.angle(a[h-1])):+7.1f}° -> {np.degrees(np.angle(b[h-1])):+7.1f}°  "
              f"(Δ {np.degrees(np.angle(b[h-1] / a[h-1])):+6.1f}°)")

    print("\n  모델이 예측하는 전류 회전 (미니PC par5 를 두 전압에 넣는다):")
    import json
    try:
        J = json.load(open("results/_circuit_model_C.json", encoding="utf-8"))
        par = tuple(J["stage1"]["minipc"][k] for k in ("C_dc", "R", "L", "Cx", "rd"))
    except Exception:
        print("    results/_circuit_model_C.json 없음 — run_circuit_model_probe 를 먼저 돌려라")
        return
    P = 9.2 - 2.6
    out = {}
    for label in ("minipc_4C IDLE", "test_19 11W"):
        V = cf.odd_only(got[label])
        r = simulate(P, *par, vsrc=cf.to_wave(V, n=cf.NCYC_FIT * 3072))
        out[label] = r["I"]
    s, b_, res = slope(out["minipc_4C IDLE"], out["test_19 11W"])
    print(f"    모델 예측 회전 {s:+.2f}°/차수 (잔차 {res:.1f}°)  <- 전압 차이만으로 생기는 몫")
    print(f"    실측 회전     −2.36 ~ −2.90°/차수")
    print(f"    남는 몫 (교정값 차이) 약 {-2.5 - s:+.2f}°/차수")


if __name__ == "__main__":
    main()
    stage_e()
