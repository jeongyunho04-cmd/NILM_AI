"""프로젝터 델타 서명의 THD 가 무엇과 함께 움직이는가 (12.167.7).

    python -m src.run_delta_thd_probe

부록 A §4.5 의 "부호 문제"를 검증한다. 부록은 장소 A 2파일 / 장소 B 4파일로
THD 1.367 vs 1.268 (7.9% 차이) 을 얻고 "Z 가 큰 장소 A 의 THD 가 더 높다 —
물리와 반대다" 라고 적었다. 장소 A 를 8파일로 늘려 재면 그 차이가 사라진다.

이벤트 델타(`ΔI(h) = I_after − I_before`)는 KCL 로 다른 기기 기여가 정확히
소거되므로 혼합 중에도 순수 서명을 준다. 다만 **파일당 이벤트가 1~4개뿐이라
잡음이 크다** — 부록도 그렇게 적었고, 이 스크립트가 보이는 것은 그 잡음이
장소 간 비교까지 먹는다는 것이다.
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

NPZ = "processed_data/composite_eval/{}.npz"
SITE = {**{s:"A" for s in ("test.2","test3","test_4","test_5","test_6","test_7",
                           "test_8","test_9","test_10","test_11","test_12","test_13")},
        **{s:"B" for s in ("test_15","test_16","test_17","test_18")}}
files = {}
for f in ("processed_data/real_events_refined.json", "processed_data/real_events.json"):
    for k,v in json.load(open(f, encoding="utf-8"))["files"].items(): files.setdefault(k,v)

rows = []
for stem, ev in files.items():
    try: z = np.load(NPZ.format(stem))
    except FileNotFoundError: continue
    C = z["harmonics_complex"]; pf = z["power_features"]
    V = pf[:,4].astype(float); P = pf[:,0].astype(float); T = z["t_rel_s"].astype(float)
    good = z["is_valid"].astype(bool)
    evs = ev.get("events") or []
    for e in evs:
        if e.get("appliance") != "beam_projector" or e.get("kind") == "mode": continue
        if abs(e.get("delta_p_w",0)) < 25: continue
        t0 = float(e["t_s"])
        a = (T>t0-14)&(T<t0-4)&good; b = (T>t0+5)&(T<t0+15)&good
        if a.sum()<120 or b.sum()<120: continue
        busy = any(e2 is not e and t0-16 < float(e2["t_s"]) < t0+17 for e2 in evs)
        if busy: continue
        dC = C[b].mean(0) - C[a].mean(0)
        if e["kind"] != "on": dC = -dC
        if abs(dC[0]) < 0.05: continue
        s = dC/dC[0]
        # 그 순간의 배경: 프로젝터를 뺀 나머지 부하와 전압
        rows.append(dict(stem=stem, site=SITE.get(stem,"?"), dP=abs(e["delta_p_w"]),
                         thd=float(np.sqrt((np.abs(s[1:])**2).sum())),
                         h3=float(abs(s[2])), V=float(V[b].mean()),
                         Pbg=float(P[a].mean())))
print(f"{'파일':9s}{'장소':>4s}{'N':>4s}{'ΔP':>7s}{'THD':>8s}{'h3/h1':>8s}"
      f"{'그때 V':>9s}{'동시부하W':>10s}")
by = {}
for stem in sorted({r["stem"] for r in rows}, key=lambda s:(SITE.get(s,"?"),s)):
    g = [r for r in rows if r["stem"]==stem]
    by.setdefault(g[0]["site"], []).extend(g)
    print(f"{stem:9s}{g[0]['site']:>4s}{len(g):4d}{np.mean([x['dP'] for x in g]):7.1f}"
          f"{np.mean([x['thd'] for x in g]):8.3f}{np.mean([x['h3'] for x in g]):8.4f}"
          f"{np.mean([x['V'] for x in g]):9.1f}{np.mean([x['Pbg'] for x in g]):10.0f}")
print()
for s, g in by.items():
    print(f"  장소 {s}: N={len(g):2d}  THD {np.mean([x['thd'] for x in g]):.3f}"
          f" ± {np.std([x['thd'] for x in g]):.3f}   V {np.mean([x['V'] for x in g]):.1f}"
          f"   동시부하 {np.mean([x['Pbg'] for x in g]):.0f}W")
# 견고성: N=1 파일 제외, 파일 단위 평균으로도 본다
print("
  [견고성] 파일 단위 평균으로 (이벤트 가중 아님)")
for s_, g in by.items():
    per = {}
    for r in g: per.setdefault(r["stem"], []).append(r["thd"])
    mm = {k: float(np.mean(v)) for k, v in per.items()}
    big = {k: v for k, v in mm.items() if len(per[k]) >= 2}
    print(f"    장소 {s_}: 전체 {len(mm)}파일 중앙 {np.median(list(mm.values())):.3f} "
          f"(범위 {min(mm.values()):.3f}~{max(mm.values()):.3f})")
    if big:
        print(f"           N>=2 인 {len(big)}파일 중앙 {np.median(list(big.values())):.3f} "
              f"(범위 {min(big.values()):.3f}~{max(big.values()):.3f})")
print("
  [부록이 쓴 두 파일이 장소 A 안에서 어디인가]")
gA = by.get("A", [])
perA = {}
for r in gA: perA.setdefault(r["stem"], []).append(r["thd"])
mmA = sorted(((float(np.mean(v)), k, len(v)) for k, v in perA.items()), reverse=True)
for i, (t, k, n) in enumerate(mmA, 1):
    mark = "  <- 부록이 쓴 파일" if k in ("test_8", "test_13") else ""
    print(f"    {i}위  {k:9s} THD {t:.3f}  (N={n}){mark}")

allr = [r for g in by.values() for r in g]
if len(allr) >= 6:
    t = np.array([r["thd"] for r in allr]); v = np.array([r["V"] for r in allr])
    pb = np.array([r["Pbg"] for r in allr])
    print(f"\n  상관 (이벤트 {len(allr)}개)")
    print(f"    THD ↔ 그때 전압       r = {np.corrcoef(t,v)[0,1]:+.3f}")
    print(f"    THD ↔ 동시 부하       r = {np.corrcoef(t,pb)[0,1]:+.3f}")
