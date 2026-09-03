"""사람 타임라인 x 신호 — 라벨 정밀화 (12.155)

무엇을 하나
----------
사람이 적은 것은 **무엇이 언제쯤 바뀌었나** 이고, 신호가 아는 것은 **언제 정확히
무엇이 얼마나 바뀌었나** 다. 둘을 순서를 지키며 맞춘다.

```
사람    seq 254  "드라이기 강 켜짐"          ±3초, 빠뜨림 있음, 오타 있음
신호    블록 512  ΔP +1003W, u₃ 0.002       시각 0.5초, 정체는 지문으로
```

자 (`u_h`)
---------
전이의 **모양**은 `u_h = ΔI_h / Re(ΔI₁)` 이다. 분모는 항등식상 `ΔP/V` 라(12.151)
크기와 장소 전압이 나눠진다. 단독녹화 800개 전이에서 잰 값:

```
laptop_charger   |u₃| 0.786    air_conditioner 0.543    beam_projector 0.042
minipc           0.015         저항 4종          0.00~0.01
```

기기 간 거리가 전부 0.75 이상이라 **SMPS 3종과 에어컨이 깨끗하게 갈린다.** 저항
4종은 서로 안 갈리므로 **전력 준위**로 가른다 (포트 1262/1377, 오븐 1148,
핫플 465/210/70, 드라이기 1003/517 — 저항이라 `(V/V_녹화)²` 로 옮긴다).

정렬
----
Needleman-Wunsch. 사람 항목을 건너뛰는 벌점은 크고(기록한 것은 일어났다), 신호
계단을 건너뛰는 벌점은 작다(오븐 히터 듀티처럼 **안 적은 전이가 원래 많다**).

쓰는 법
------
    python -X utf8 -m src.run_refine_labels --stems test_5          # 자 검증
    python -X utf8 -m src.run_refine_labels --all --out results/refined_labels.json
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.labeling.timeline_parser import parse as parse_timeline
from src.preprocessing.raw_csv import read_raw_csv
from src.run_switch_sig import (ORDERS, BLOCK, to_blocks, to_cycles, transitions,
                                steps_multi, edges_p, edges_i3, measure)

ODD = [3, 5, 7, 9, 11, 13, 15]
OI = [h - 1 for h in ODD]
#: 전압 응답 지수 — 저항 P∝V², SMPS 정전력, 전동기 ~V^0.7 (file_registry 와 같다)
V_EXP = {"electiric_kettle": 2.0, "hotplate": 2.0, "oven": 2.0, "hair_dryer": 2.0,
         "beam_projector": 0.0, "laptop_charger": 0.0, "minipc": 0.0,
         "fan": 0.7, "air_conditioner": 0.7}
#: `u` 거리의 바닥. 저항은 u≈0 이라 이게 없으면 0 나누기가 된다.
U_FLOOR = 0.05


def u_of(di_re, di_im):
    """전이의 **모양 불변량** 두 벌.

    ``h1``  (크기비, 위상)  = ΔI_h / ΔI₁   — 저항 부류를 가른다
    ``h3``  (크기비, 위상)  = ΔI_h / ΔI₃   — **SMPS 부류를 가른다**

    ⚠ h1 정규화는 **중첩에서 깨진다.** 핫플이 −400W 로 꺼지는 순간 충전기가 켜지면
      분모(ΔI₁)는 핫플 것이고 분자(ΔI₃)는 충전기 것이라 `u₃` 가 0.13 이라는
      무의미한 값이 된다. 저항은 h3 에 거의 아무것도 안 내므로(|u₃| 0.004~0.012)
      **h3 로 정규화하면 저항 오염이 원천적으로 없다** (12.155).

    ⚠ 복소 비를 그냥 평균하면 위상 때문에 상쇄된다. 크기와 위상 불변량
      `φ_h = arg(ΔI_h) − h·arg(ΔI_ref)` 로 나눠 다룬다 (12.34).
    """
    z = np.asarray(di_re) + 1j * np.asarray(di_im)
    out = {}
    for key, ref_h in (("h1", 1), ("h3", 3)):
        r = z[ref_h - 1]
        if abs(r) < (0.02 if ref_h == 1 else 0.01):
            out[key] = None
            continue
        mag = np.abs(z[OI]) / abs(r)
        ph = np.angle(z[OI]) - np.array(ODD) / ref_h * np.angle(r)
        out[key] = (mag, (np.rad2deg(ph) + 180.0) % 360.0 - 180.0)
    return out


def build_ref(path: str = "results/switch_sig.json") -> Dict[str, Dict]:
    """기기별 전이 지문. **|ΔP| 로 갈래를 나눈다.**

    ⚠ 한 기기를 지문 하나로 두면 안 된다. 오븐은 제어보드(SMPS, ~14W)와
    히터(저항, ~1,148W)가 따로 있고 히터 전이가 200개라 통째로 평균하면 오븐이
    "저항" 하나가 된다. 그러면 `오븐 켜짐`(= 제어보드 통전, |u₃| 0.97)을 거부한다.
    핫플의 다이얼 단계(465/210/70W)와 드라이기 강/약도 같은 문제다.

    그래서 `log|ΔP|` 로 군집해 갈래마다 지문을 따로 둔다. 채점은 갈래 중 최소값.
    """
    d = json.load(open(path, encoding="utf-8"))
    acc: Dict[str, List[tuple]] = {}
    for st, v in d.items():
        app = v["appliance"]
        for t in v["transitions"]:
            u = u_of(t["di_re"], t["di_im"])
            if u["h1"] is None or abs(t["dp_w"]) < 3.0:
                continue
            # 저항 부하는 전압으로 옮겨 비교해야 같은 갈래로 모인다 (P∝V²)
            e = V_EXP.get(app, 0.0)
            p_ref = abs(t["dp_w"]) * (220.0 / max(t["v_rms"], 1.0)) ** e
            z = np.asarray(t["di_re"]) + 1j * np.asarray(t["di_im"])
            acc.setdefault(app, []).append(
                (u["h1"][0], u["h1"][1], p_ref, t["v_rms"], u["h3"],
                 abs(z[2]) / max(abs(z[0]), 1e-9)))
    ref = {}
    for app, L in acc.items():
        P = np.array([x[2] for x in L])
        o = np.argsort(P); P = P[o]
        L = [L[i] for i in o]
        # log 전력에서 0.35 (=1.42배) 넘게 벌어지면 다른 갈래로 본다
        cuts = np.flatnonzero(np.diff(np.log(np.maximum(P, 1.0))) > 0.35) + 1
        branches = []
        for gi in np.split(np.arange(len(L)), cuts):
            if len(gi) < 2:
                continue
            M = np.array([L[i][0] for i in gi]); Ph = np.array([L[i][1] for i in gi])
            H3 = [L[i][4] for i in gi if L[i][4] is not None]
            br = {"mag": np.median(M, 0),
                  "ph": np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(Ph)), 0))),
                  "p_w": float(np.median([L[i][2] for i in gi])),
                  "u3": float(np.median([L[i][5] for i in gi])),
                  "n": int(len(gi))}
            if H3:
                br["mag3"] = np.median(np.array([h[0] for h in H3]), 0)
                br["ph3"] = np.rad2deg(np.angle(np.mean(
                    np.exp(1j * np.deg2rad(np.array([h[1] for h in H3]))), 0)))
            branches.append(br)
        if branches:
            ref[app] = {"branches": branches,
                        # 준위는 군집 중심이 아니라 **관측값 전부**를 쓴다.
                        # 드라이기 강(890)/약(460)처럼 다단 기기는 군집이 뭉개고
                        # 핫플은 다이얼이 20~470W 로 연속에 가깝다 (12.155).
                        "p_obs": np.sort(P),
                        "mag": branches[0]["mag"], "ph": branches[0]["ph"]}
    return ref


def cost_device(u, dp: float, v: float, app: str, ref: Dict,
                z=None) -> Tuple[float, float]:
    """(부류 비용, 준위 비용). 기기의 **갈래마다** 재고 최소를 쓴다.

    SMPS·에어컨은 **h3 정규화**로 잰다 — 저항이 h3 에 아무것도 안 내므로 중첩에
    안 깨진다. 준위도 `|ΔI₃|/u₃ × V` 로 낸 **기기 몫 전력**으로 본다 (12.155).
    저항 부류는 h1 정규화와 총전력 계단을 그대로 쓴다.
    """
    r = ref.get(app)
    if r is None:
        return 9.9, 9.9
    smps = CLASS.get(app, "res") in ("smps", "ac")
    e = V_EXP.get(app, 0.0)

    # 준위로 쓸 전력 — SMPS 는 h3 에서 되돌린 기기 몫
    p_use = abs(dp)
    if smps and z is not None:
        m3 = r["branches"][0].get("u3", 0.9)
        if m3 > 0.05:
            p_use = abs(z[2]) / m3 * v
    p_obs = p_use * (220.0 / max(v, 1.0)) ** e

    best = 9.9
    for b in r["branches"]:
        key_m, key_p, uu = ("mag3", "ph3", u.get("h3")) if smps else ("mag", "ph", u.get("h1"))
        if uu is None or key_m not in b:
            c_u = 3.0
        else:
            dm = float(np.abs(np.log((uu[0][:4] + U_FLOOR) / (b[key_m][:4] + U_FLOOR))).mean())
            dph = float(np.abs((uu[1][:3] - b[key_p][:3] + 180.0) % 360.0 - 180.0).mean()) / 60.0
            c_u = dm + dph
        best = min(best, c_u)
    po = r["p_obs"]
    c_p = float(np.min(np.abs(np.log(max(p_obs, 1.0) / np.maximum(po, 1.0)))))
    return best, c_p


#: 부류 — 고조파로 갈리는 단위. SMPS 3종은 이 안에서 안 갈린다.
CLASS = {"beam_projector": "smps", "laptop_charger": "smps", "minipc": "smps",
         "air_conditioner": "ac", "fan": "fan",
         "electiric_kettle": "res", "hotplate": "res", "oven": "res",
         "hair_dryer": "res"}


def dp_device(s: Dict, app: str, ref: Dict) -> Tuple[float, str]:
    """그 기기 몫의 ΔP. **총전력 계단을 그대로 쓰면 안 된다.**

    오븐 히터가 ±1,100W 로 듀티를 도는 동안 SMPS 가 켜지면 총전력 계단은 오븐
    것이다. 기존 라벨이 정확히 그 함정에 빠져 있었다 — `충전기 off` 에
    `ΔP +1002.6W`, `미니PC off` 에 `+1007.2W` 가 적혀 있다 (12.155).

    SMPS 는 h3 로 뗄 수 있다. 저항은 h3 에 거의 아무것도 안 내므로(|u₃| 0.004~0.012)
    `|ΔI₃|` 는 SMPS 몫이고, `|ΔI₃| = u₃·|ΔI₁|` 과 `Re(ΔI₁) = ΔP/V` 에서

        ΔP(기기) ≈ |ΔI₃| / u₃(기기) × V

    저항 부하는 반대로 총전력 계단이 곧 제 몫이라 그대로 쓴다.
    """
    cls = CLASS.get(app, "res")
    if cls == "res" or s["u"] is None or s["u"].get("h3") is None:
        return s["dp_w"], "total"
    # 갈래 중 |ΔP| 가 가장 가까운 것의 u₃ 를 쓴다
    m3 = float(ref[app]["branches"][0].get("u3", 0.9)) if app in ref else 0.9
    z = np.asarray(s["di_re"]) + 1j * np.asarray(s["di_im"])
    d3 = abs(z[2])
    if m3 < 0.05 or d3 < 1e-6:
        return s["dp_w"], "total"
    mag = d3 / m3 * s["v_rms"]
    # **방향은 h3 크기 변화가 정한다.** SMPS 3종의 φ₃ 가 −5° 부근으로 모여 있어
    # 켜지면 |I₃| 가 늘고 꺼지면 준다. 총전력 부호를 쓰면 핫플 잡음(+5.6W)에
    # 끌려가 `미니PC 꺼짐` 이 `켜짐` 으로 뒤집힌다 (12.155).
    d_i3 = s.get("i3_after", 0.0) - s.get("i3_before", 0.0)
    sgn = np.sign(d_i3) if abs(d_i3) > 1e-6 else np.sign(s["dp_w"] or 1.0)
    return float(sgn * mag), "h3"


def detect(stem: str, min_dp: float, K: int, g: int,
           min_di3: float = 0.02) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """**2Hz 로 후보를 찾고 그 자리만 60Hz 로 확대한다** (12.155).

    * 0.5초 블록만 쓰면 1초 안의 두 사건이 하나의 **섞인 계단**이 된다.
    * 전부 60Hz 로 가면 계단이 6배로 늘고 측정 창이 이웃에 잘려, 1,600W 위에
      얹힌 45W 의 h3 를 못 잰다.

    그래서 후보는 2Hz 가 정하고(넓은 창 = 좋은 SNR) 확대는 그 자리에서만 한다.
    측정 창은 **채널마다 제 이웃**으로 자르고, 고조파는 **h3 계단 자리**에 맞춘다.

    ⚠ 지속성 게이트(리플 제거)는 넣었다가 뺐다. 핫플 듀티처럼 **진짜지만 주기적인**
      전이를 같이 지워서 `핫플 꺼짐` 의 부호가 뒤집혔다. 리플과 듀티를 지속성만으로
      가르지 못한다 — 미결로 남긴다.
    """
    Pb, Vb, Ib = to_blocks(stem)
    cand = steps_multi(Pb, Ib, min_dp, min_di3, K, g)
    Pc, Vc, Ic = to_cycles(stem)
    n = len(Pc)

    # ⚠ **두 채널의 후보를 합쳐서 병합하면 안 된다.** 0.1초 차이로 핫플(P 사건)과
    #   충전기(h3 사건)가 일어나는데, 합친 뒤 0.15초 안을 같은 사건으로 묶으면
    #   둘 중 하나가 사라진다 (12.155). 채널 안에서만 병합한다.
    def _fine(chan: str) -> List[int]:
        got: List[int] = []
        for b in cand:
            lo = max(0, (b - K) * BLOCK)
            hi = min(n, (b + g + K + 1) * BLOCK)
            if hi - lo < 4 * BLOCK:
                got.append(min(max(int((b + g / 2) * BLOCK), 0), n - 1))
                continue
            e = (edges_p(Pc[lo:hi], min_dp, 12, 6) if chan == "p"
                 else edges_i3(Ic[lo:hi], min_di3, 12, 6)) + lo
            e = e[(e > lo) & (e < hi)]
            got.extend(int(x) for x in e) if len(e) else got.append(int((b + g / 2) * BLOCK))
        out: List[int] = []
        for x in sorted(set(got)):
            if not out or x - out[-1] >= 9:
                out.append(int(x))
        return out

    ed = np.array(sorted(set(_fine("p")) | set(_fine("i3"))), np.int64)

    ep = edges_p(Pc, min_dp, 12, 6)
    e3 = edges_i3(Ic, min_di3, 12, 6)
    tr = measure(Pc, Vc, Ic, ed, W=12, G=6, wmax=60, ed_p=ep, ed_3=e3)
    for t in tr:
        t["u"] = u_of(t["di_re"], t["di_im"])
    return tr, Pc, Vc


def pair_cost(e: Dict, s: Dict, seq_lo: int, ref: Dict,
              w_time: float = 0.10) -> Tuple[float, float, float, float]:
    """(총비용, 부류비용, 준위비용, Δt초). 극성은 **기기 몫**의 부호로 본다."""
    dt = abs(s["t_block"] - (e["seq"] - seq_lo)) * 0.5
    z = np.asarray(s["di_re"]) + 1j * np.asarray(s["di_im"])
    cu, cp = cost_device(s["u"], s["dp_w"], s["v_rms"], e["appliance"], ref, z)
    dpd, _ = dp_device(s, e["appliance"], ref)
    pol = 0.0
    if e["kind"] == "on" and dpd < 0:
        pol = 8.0
    elif e["kind"] == "off" and dpd > 0:
        pol = 8.0
    return 1.5 * cu + 2.0 * cp + w_time * dt + pol, cu, cp, dt


def assign(tl: List[Dict], det: List[Dict], seq_lo: int, ref: Dict,
           max_dt_s: float = 7.0, max_cost: float = 9.0):
    """**사람 항목마다** 그 자리 근방에서 그 기기다운 계단을 고른다.

    ⚠ 창은 **±7초**다. 사람 기록의 시각 오차가 중앙 1초 / p90 4~6초라 그 안에 다
    들어온다. 더 넓히면 오븐 듀티 계단 170여 개 중 엉뚱한 것을 집기 시작한다
    (±15초에서 seq 875 가 13.1초 떨어진 계단을 가져갔다).

    전역 Needleman-Wunsch 를 안 쓰는 이유: 사람 기록이 거의 완전하고 시각 오차도
    중앙 1초 수준이다. 그런데 검출된 계단은 오븐 듀티 때문에 170개가 넘어서,
    전역 정렬은 그 사이에서 엉뚱한 것을 집는다. 사람의 정보(정체는 확실, 시각은
    대략)를 그대로 쓰는 편이 곧고 튼튼하다 (12.155).

    계단 하나는 항목 하나만 가져간다. 비용 낮은 짝부터 확정한다.
    """
    cand = []
    for i, e in enumerate(tl):
        if e["kind"] in ("already_on", "work_start", "work_end"):
            continue
        for j, s in enumerate(det):
            if abs(s["t_block"] - (e["seq"] - seq_lo)) * 0.5 > max_dt_s:
                continue
            c, cu, cp, dt = pair_cost(e, s, seq_lo, ref)
            if c <= max_cost:
                cand.append((c, i, j, cu, cp, dt))
    cand.sort(key=lambda x: x[0])
    used_i, used_j, out = set(), set(), {}
    for c, i, j, cu, cp, dt in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j); out[i] = (j, c, cu, cp, dt)
    # 순서 위반 — seq 는 오름차순인데 고른 시각이 거꾸로면 표시한다
    order = sorted(out, key=lambda i: tl[i]["seq"])
    bad = set()
    prev = -1e9
    for i in order:
        t = det[out[i][0]]["t_block"]
        if t < prev:
            bad.add(i)
        prev = max(prev, t)
    return out, bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--min-dp", type=float, default=8.0)
    ap.add_argument("--win", type=int, default=4)
    ap.add_argument("--guard", type=int, default=2)
    ap.add_argument("--min-di3", type=float, default=0.02,
                    help="h3 채널 계단 문턱 (A). 미니PC 켜짐이 |ΔI₃| 0.08A 다")
    ap.add_argument("--sig", default="results/switch_sig.json")
    ap.add_argument("--from-labels", action="store_true",
                    help="사람 타임라인 대신 **기존 라벨의 시각**을 앵커로 쓴다. "
                         "test_11/12/13 은 seq 기록이 남아 있지 않지만 라벨 자체가 "
                         "사람 기록에서 정밀화된 것이라(오차 중앙 0.3~0.4초) 앵커로 "
                         "쓸 수 있다. 시각은 다시 잡고 **ΔP 를 기기 몫으로 고친다**")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tl_all, bad = parse_timeline()
    if bad:
        print(f"⚠ 타임라인에서 못 읽은 줄 {len(bad)}개")
    if a.from_labels:
        smap0 = json.load(open("results/seq_time_map.json", encoding="utf-8"))
        ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
        for st, v in ev.items():
            # ⚠ 기존 라벨의 시각은 **이미 상대 시간**이라 seq_lo 를 더해 둬야
            #   아래의 `seq - seq_lo` 가 되돌린다 (test_11 은 seq_lo=507 이다).
            lo0 = smap0.get(st, {}).get("seq_lo", 0)
            rows = [{"seq": int(round(e["t_s"] / 0.5)) + lo0, "appliance": e["appliance"],
                     "kind": e["kind"], "mode": None, "raw": f"기존 라벨 {e['t_s']}s"}
                    for e in v.get("events", []) if e["kind"] in ("on", "off")]
            if rows:
                tl_all[st] = sorted(rows, key=lambda r: r["seq"])
    ref = build_ref(a.sig)
    smap = json.load(open("results/seq_time_map.json", encoding="utf-8"))

    stems = a.stems or (sorted(tl_all) if a.all else ["test_5"])
    out_doc = {}
    for st in stems:
        if st not in tl_all:
            print(f"{st}: 타임라인 없음"); continue
        tl = tl_all[st]
        seq_lo = smap.get(st, {}).get("seq_lo", 0)
        det, P, V = detect(st, a.min_dp, a.win, a.guard, a.min_di3)
        got, bad = assign(tl, det, seq_lo, ref)
        n_m, n_tl = len(got), sum(1 for i, e in enumerate(tl)
                                  if e["kind"] in ("on", "off", "mode") and i not in got)
        n_dt = len(det) - n_m
        print("\n" + "=" * 104)
        print(f"■ {st}   사람 {len(tl)}  신호계단 {len(det)}  "
              f"맞춤 {n_m}  사람만 {n_tl}  신호만 {n_dt}   (seq_lo={seq_lo})")
        print("=" * 104)
        print(f"{'사람 seq':>8} {'기기':<16}{'동작':<11}{'신호 t_s':>9}{'ΔP총':>9}"
              f"{'ΔP기기':>9} {'출처':<5}{'u3':>6}{'Δt':>7}{'비용':>7}  비고")
        rows = []
        for i, e in enumerate(tl):
            if i in got:
                j, c, cu, cp, dt = got[i]
                s_ = det[j]
                t = s_["t_block"] * 0.5
                dts = t - (e["seq"] - seq_lo) * 0.5
                zz = np.asarray(s_["di_re"]) + 1j * np.asarray(s_["di_im"])
                u3 = float(abs(zz[2]) / max(abs(zz[0]), 1e-9))
                dpd, how = dp_device(s_, e["appliance"], ref)
                flag = "  ⚠ 순서 위반" if i in bad else ""
                print(f"{e['seq']:>8} {e['appliance'][:15]:<16}"
                      f"{(e['kind'] + ('/' + e['mode'] if e['mode'] else '')):<11}"
                      f"{t:>9.1f}{s_['dp_w']:>9.1f}{dpd:>9.1f} {how:<5}"
                      f"{u3:>6.2f}{dts:>+7.1f}{c:>7.2f}{flag}")
                rows.append({"seq": e["seq"], "appliance": e["appliance"],
                             "kind": e["kind"], "mode": e["mode"], "t_s": t,
                             "dp_total_w": s_["dp_w"], "dp_device_w": dpd,
                             "dp_from": how, "cost": c, "cu": cu, "cp": cp,
                             "dt_s": dts, "order_violation": i in bad,
                             "matched": True})
            else:
                why = "시작부터 켜짐" if e["kind"] == "already_on" else (
                    "부하 변동(전원 아님)" if e["kind"].startswith("work") else "❌ 계단 없음")
                print(f"{e['seq']:>8} {e['appliance'][:15]:<16}"
                      f"{(e['kind'] + ('/' + e['mode'] if e['mode'] else '')):<11}"
                      f"{'—':>9}{'—':>9}{'—':>9} {'':<5}{'—':>6}{'—':>7}{'':>7}  {why}")
                rows.append({"seq": e["seq"], "appliance": e["appliance"],
                             "kind": e["kind"], "mode": e["mode"], "matched": False})
        out_doc[st] = {"seq_lo": seq_lo, "n_detected": len(det), "rows": rows}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out_doc, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
