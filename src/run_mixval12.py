# -*- coding: utf-8 -*-
"""혼합 검증 (mixval, v12) — 복합 녹화의 SMPS 창에서 생성기의 전류 재현을 실측과 견준다 (13.2 · 13.24.14).

무엇을 재나
----------
라벨 있는 복합 녹화 **다섯 개 전부**에서 켜진 기기 집합이 일정한 2초 창(사건 앞뒤 3초 제외)을 골라,
실측 총전류 페이저 I_meas(h1..h15) 와 다음을 비교한다:

    [A] 녹화 중첩      Σ_i I_rec,i(p_i) · (V_rec/V_meas)                      지금까지의 생성기 = 단순 중첩
    [B] A + 텍스처 델타 [A] + Σ_i [ I_sim,i(p_i, V15_meas) − I_sim,i(p_i, V15_rec,i) ]     새 생성기 (13.2)
    [D] 모델 단독       Σ_i I_sim,i(p_i, V15_meas)                                교체안 — 델타보다 나은지 본다
    [C] **결합 경로**   되돌린 개방 전압에서 생성기의 고정점을 돌려 낸 전류. 기준은 **잠근 풀개**다

V15_meas 는 그 창의 **측정 단자 전압**이라 공유 임피던스 결합이 이미 들어 있다 — 그래서 [A][B][D] 에는
결합 델타를 더하지 않는다 (더하면 이중 계상).

⚠ **그래서 [A][B][D] 만 보면 결합 코드가 한 줄도 안 돈다.** 13.22·13.23 이 그 상태로 "[B] 가
소수점까지 같다" 며 변경을 통과시켰고, 결합의 26~50% 오차를 학습 한 판을 돌릴 때까지 아무도 못 봤다
(13.24.14). [C] 는 그 경로를 **일부러** 태우려고 있다. 실행 끝의 "[C] 결합 경로를 탄 창" 이 0 이면
이 판은 결합 변경을 검증하지 못한 것이다.

⚠ [C] 의 한계: 평가 창은 SMPS 가 한 대뿐인 경우가 많아(test_1 은 단독 42창) 풀개 오차가 작게 나온다.
**학습 자료 분포**에서의 오차는 `src/run_coupling_check.py` 로 따로 재라 — 거기서는 27% 가 10% 넘게 틀린다.

기기 전력 p_i: 저항·선풍기는 라벨의 ΔP(기기 몫), 미니PC·프로젝터도 라벨 ΔP. 충전기는 나머지
(P_total − 나머지 − 계측 바닥) — 창 안에서 크게 변하는 유일한 SMPS 라 그렇게 둔다.
켜져 있지 않은 SMPS 는 풀의 대기 지문을, 계측계는 noise 기준을 더한다.

쓰는 법
------
    python -X utf8 -m src.run_mixval12 --out results/_mixval12.json    # 기본이 다섯 파일 전부다
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.preprocessing import load_nilm_npz
from src.synthesis.segment_pool import SegmentPool
from src.synthesis.vtexture import VoltageTextureLibrary, default_library
from src.preprocessing.file_registry import SITE_SESSIONS, LoadClass, get_load_class
from src.synthesis.vtexture import load_vtail, splice_tail

#: 세션별 전압 꼬리 h17~h31. **기본은 비어 있다** — 생성기 기본(꺼짐)과 맞춘다.
#: `--vtail` 을 주면 채워지고, 그때만 [B]/[D] 가 꼬리를 본다 (13.73).
_TAILS: dict = {}
from src.synthesis.coupling import SmpsCircuit, SMPS_DEVICES

H = 15
SMPS = set(SMPS_DEVICES)
#: 창 길이와 사건 앞뒤 가드 (초). 사람 시각 ±3초 + 정착.
WIN_S = 2.0
GUARD_S = 3.0
#: 이 기기들이 켜진 창은 뺀다 — 1kW 급 저항이 h1 을 지배해 SMPS 재현을 못 본다.
EXCLUDE_ON = ("electiric_kettle", "hair_dryer", "oven", "hotplate", "air_conditioner")
#: 라벨 있는 복합 녹화 전부. **기본을 한 파일로 두면 그 자리만 검증된다** (13.24.14).
ALL_STEMS = ("test_1", "test_2", "test_3", "test_4", "test_5")
#: [C] 의 되돌리기에 쓰는 선로 인덕턴스 — 텍스처 되돌리기와 같은 값 (13.22).
DEEMBED_L_H = 225e-6


def rel_err(pred: np.ndarray, meas: np.ndarray, orders: slice = slice(0, H)) -> float:
    a = np.asarray(pred)[orders]; b = np.asarray(meas)[orders]
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2) / max(np.sum(np.abs(b) ** 2), 1e-18)))


class RecordedSig:
    """단독 녹화(풀)에서 전력이 p 인 사이클의 중앙 페이저 + 그 파일의 텍스처."""

    def __init__(self, pool: SegmentPool, lib: VoltageTextureLibrary):
        self.pool = pool
        self.lib = lib
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]] = {}

    def _table(self, dev: str):
        if dev not in self._cache:
            acts = self.pool.appliance_activations.get(dev, [])
            P = np.concatenate([a.net_power_features[:, 0] for a in acts]) if acts else np.zeros(0)
            V = np.concatenate([a.net_power_features[:, 4] for a in acts]) if acts else np.zeros(0)
            C = np.concatenate([a.net_harmonics_complex for a in acts]) if acts else np.zeros((0, H), complex)
            S = sum(([a.source_file] * a.duration_cycles for a in acts), [])
            self._cache[dev] = (P, V, C, S)
        return self._cache[dev]

    def at(self, dev: str, p: float, v_meas: float) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], str]]:
        """(전압 환산한 녹화 페이저, 그 녹화의 상대 텍스처, stem). 표본이 없으면 None."""
        P, V, C, S = self._table(dev)
        if len(P) == 0:
            return None
        tol = max(2.0, 0.06 * p)
        m = np.abs(P - p) <= tol
        if int(m.sum()) < 30:
            idx = np.argsort(np.abs(P - p))[:200]
            m = np.zeros(len(P), bool); m[idx] = True
        stems, counts = np.unique(np.asarray(S)[m], return_counts=True)
        stem = str(stems[np.argmax(counts)])
        mm = m & (np.asarray(S) == stem)
        I = np.median(C[mm].real, 0) + 1j * np.median(C[mm].imag, 0)
        v_rec = float(np.median(V[mm]))
        scale = (v_rec / v_meas)                      # SMPS: I ∝ 1/V (선풍기는 ~V^0.7 이지만 작다)
        # 13.73: 녹화 텍스처도 **h1..h31** 로 (그 녹화가 찍힌 세션의 꼬리가 붙는다).
        # 환경 쪽에만 얹으면 "녹화에는 꼬리가 없었다" 가 되어 델타에 가짜 항이 생긴다.
        return I * scale, self.lib.file_rel_full(stem), stem


def windows_of(z: dict, spec: dict, apps: List[str]) -> List[Tuple[int, int, Dict[str, bool]]]:
    """켜진 집합이 일정하고 사건에서 GUARD_S 떨어진 WIN_S 창들 [(a, b, {app: on})]."""
    n = len(z["t_rel_s"])
    on = np.zeros((n, len(apps)), bool)
    for j, a in enumerate(apps):
        for t0, t1 in spec["intervals"].get(a, {}).get("on", []):
            on[int(t0 * 60):int(t1 * 60), j] = True
    ev_t = sorted(e["t_s"] for e in spec["events"])
    w = int(WIN_S * 60)
    out = []
    for a0 in range(0, n - w, w):
        b0 = a0 + w
        if any(abs(a0 / 60.0 - t) < GUARD_S or abs(b0 / 60.0 - t) < GUARD_S or (a0 / 60.0 < t < b0 / 60.0) for t in ev_t):
            continue
        state = on[a0:b0]
        if not (state == state[0]).all():
            continue
        out.append((a0, b0, {a: bool(state[0, j]) for j, a in enumerate(apps)}))
    return out


def device_powers(spec: dict, t_mid: float, on: Dict[str, bool]) -> Dict[str, float]:
    """라벨 사건에서 각 기기의 최근 켬 ΔP(기기 몫) 을 그 기기 전력으로. 충전기는 밖에서 나머지로 채운다."""
    P: Dict[str, float] = {}
    for a, is_on in on.items():
        if not is_on:
            continue
        cand = [e for e in spec["events"] if e["appliance"] == a and e["kind"] == "on" and e["t_s"] <= t_mid
                and e.get("delta_p_w") is not None]
        if cand:
            P[a] = float(abs(cand[-1]["delta_p_w"]))
        else:                                                      # 시작부터 켜짐 -> 첫 끔의 |ΔP|
            cand = [e for e in spec["events"] if e["appliance"] == a and e["kind"] == "off" and e.get("delta_p_w") is not None]
            P[a] = float(abs(cand[0]["delta_p_w"])) if cand else 0.0
    return P


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=ALL_STEMS,
                    help="기본은 라벨 있는 **다섯 파일 전부**다. 옛 기본값은 test_2 하나였고, "
                         "그것이 E 파일이라 D 의 후퇴를 아무도 못 봤다 (13.24.14)")
    ap.add_argument("--events", default="processed_data/real_events.json")
    ap.add_argument("--npz-dir", default="processed_data/composite_eval")
    ap.add_argument("--out", default="results/_mixval12.json")
    ap.add_argument("--relax", type=float, default=1.0,
                    help="[C] 결합 풀개의 감쇠. 1.0 은 지금 생성기 거동, 0.5 는 감쇠 (13.24.13)")
    ap.add_argument("--n-iter", type=int, default=3, help="[C] 결합 풀개의 반복 수")
    ap.add_argument("--vtail", action="store_true",
                    help="전압 꼬리(h17~h31)를 켠다 (13.73). 기본은 생성기와 같이 꺼짐 — "
                         "켜면 [D] 는 좋아지고 [B] 는 나빠진다")
    a = ap.parse_args()
    if a.vtail:
        _TAILS.update(load_vtail())
        print(f"전압 꼬리 켬 — 키 {sorted(_TAILS)} (비면 꼬리 없이 돈다)")

    ev = json.load(open(a.events, encoding="utf-8"))["files"]
    pool = SegmentPool(npz_dir="processed_data/npz")
    lib = default_library()
    circ = SmpsCircuit(n_iter=a.n_iter, relax=a.relax)
    #: [C] 의 기준 — 잠근 풀개. 감쇠 0.5 로 200회면 잔차가 ~1e-20 이다 (13.24.13).
    circ_ref = SmpsCircuit(models=circ.models, n_iter=200, relax=0.5)
    rec = RecordedSig(pool, lib)
    noise = pool._pick_noise_reference(1.6)
    doc: Dict[str, dict] = {}
    print("=" * 110)
    print("혼합 검증 v12 — 실측 총전류 대 [A] 녹화중첩 / [B] +텍스처델타 / [C] **결합 경로** / [D] 모델단독")
    print(f"   파일 {len(a.stems)}개 · 결합 풀개 n_iter={a.n_iter} relax={a.relax:g}   (13.2 · 13.24.14)")
    print("=" * 110)
    for stem in a.stems:
        if stem not in ev:
            print(f"{stem}: 라벨 없음"); continue
        sess = next((k for k, v in SITE_SESSIONS.items() if stem in v["stems"]), None)
        z_ohm = SITE_SESSIONS[sess]["z_ohm"] if sess else None
        # 그 세션의 **단자** 꼬리 (rel_meas 가 단자라 짝이 맞는다). 세션을 모르면 자리로.
        _t = _TAILS.get(sess) or _TAILS.get(SITE_SESSIONS.get(sess, {}).get("site", ""))
        tail_env = None if _t is None else _t[0]
        z = load_nilm_npz(f"{a.npz_dir}/{stem}.npz")
        spec = ev[stem]
        apps = list(spec["appliances_present"])
        hc = z["harmonics_complex"]; V15 = z["voltage_harmonics_complex"]; pf = z["power_features"]; valid = z["is_valid"] == 1
        rows = []
        for a0, b0, on in windows_of(z, spec, apps):
            if any(on.get(x, False) for x in EXCLUDE_ON):
                continue
            if not any(on.get(d, False) for d in SMPS):
                continue
            m = np.zeros(len(hc), bool); m[a0:b0] = True; m &= valid
            if m.sum() < 60:
                continue
            I_meas = np.median(hc[m].real, 0) + 1j * np.median(hc[m].imag, 0)
            Vm = np.median(V15[m].real, 0) + 1j * np.median(V15[m].imag, 0)
            v1 = float(abs(Vm[0])); rel_meas = Vm / v1
            # 이 창의 실측 전압에도 **그 세션의 꼬리**를 얹는다 (13.73). npz 가 h15 까지만
            # 적을 뿐 실제 전압에는 h17+ 가 있었다. 꼬리를 모르는 세션이면 그대로 h15 다.
            rel_env = splice_tail(rel_meas, tail_env)
            P_tot = float(np.median(pf[m, 0]))
            t_mid = (a0 + b0) / 120.0
            pw = device_powers(spec, t_mid, on)
            # 꺼져 있지만 꽂힌 기기의 대기 지문 (오븐 꽂힘 14mA 처럼 미니PC h1 의 1/3 이 되는 것도 있다)
            A = np.array(noise.median_phasor, complex).copy()
            P_stby = 0.0
            for d in apps:
                if not on.get(d):
                    sp = pool.standby_profiles.get(d)
                    if sp is not None:
                        A += sp.harmonics_complex; P_stby += float(sp.power_w)
            # 나머지 전력은 켜진 SMPS 중 하나에 (충전기 > 미니PC > 프로젝터 순 — 창 안에서 크게 변하는 것부터)
            for d in ("laptop_charger", "minipc", "beam_projector"):
                if on.get(d):
                    others = sum(v for k, v in pw.items() if k != d)
                    pw[d] = max(P_tot - others - float(noise.noise_floor_w) - P_stby, 3.0)
                    break
            B = A.copy(); D = A.copy(); C = A.copy()
            # [B2] = [B] + **저항 부하의 전압 텍스처 델타** (13.69). 같은 실행 안에서 짝지어 보려고
            # 따로 쌓는다 — I_h += I_1·(rel_meas[h] − rel_rec[h]). 생성기의 `apply_site_distortion`
            # 과 같은 식이고, 여기서는 rel_env 가 라이브러리 표본이 아니라 **이 창의 실측 전압**이다.
            d_res = np.zeros(H, complex)
            ok = True
            for d, p in pw.items():
                got = rec.at(d, p, v1)
                if got is None:
                    ok = False; break
                I_rec, rel_rec, rstem = got
                A += I_rec; B += I_rec
                if d in SMPS and circ.has(d):
                    I_env = circ.current(d, p, rel_env, v1)
                    I_own = circ.current(d, p, rel_rec, v1) if rel_rec is not None else None
                    if I_env is not None and I_own is not None:
                        B += (I_env - I_own)
                        D += I_env
                    else:
                        ok = False; break
                else:
                    D += I_rec; C += I_rec
                    if get_load_class(d) is LoadClass.RESISTIVE and rel_rec is not None:
                        # ⚠ 저항 출력은 h15 까지라 **꼬리를 자른다.** 안 자르면 (31,) 이
                        #    (15,) 에 더해져 터진다. 꼬리는 저항 서명에 안 들어간다.
                        dd = rel_meas - np.asarray(rel_rec, complex)[:H]
                        dd[0] = 0.0
                        d_res += I_rec[0] * dd
            if not ok:
                continue

            # ── [C] **결합 경로** — 생성기가 실제로 타는 길을 잰다 (13.24.14) ──────────────
            # 이 창의 실측 단자 전압에서 개방 소스를 되돌려(V_src = V_term + Z·I) **실제 동작점**을
            # 만들고, 거기서 생성기의 고정점을 돌린다. 기준은 실측이 아니라 **잠근 풀개**
            # (relax 0.5 x 200회, 잔차 ~1e-20) 다 — 실측을 기준으로 삼으면 SMPS 아닌 전류(선풍기·
            # 대기)까지 되돌리게 되어 왕복이 안 닫히고, 풀개 오차와 모형 오차가 섞인다.
            # 그래서 [C] 대 [C*] 는 **풀개 오차만** 가른다.
            C_all = C_odd = dV = float("nan")
            p_smps = {d: p for d, p in pw.items() if d in SMPS and circ.has(d)}
            if z_ohm is not None and p_smps:
                Zh = z_ohm + 1j * 2 * np.pi * 60.0 * np.arange(1, H + 1) * DEEMBED_L_H
                # 생성기와 같은 부기: 강하는 h1~h15 에만 걸고 꼬리는 그대로 통과시킨다
                rel_src = splice_tail(rel_meas + Zh * I_meas / v1, tail_env)
                V_now = circ.solve_terminal(p_smps, rel_src, v1, z_ohm, DEEMBED_L_H)
                V_ref = circ_ref.solve_terminal(p_smps, rel_src, v1, z_ohm, DEEMBED_L_H)
                if V_now is not None and V_ref is not None:
                    Cs = C.copy(); Cr = C.copy(); good = True
                    for d, p in p_smps.items():
                        I_n = circ.current(d, p, V_now / v1, v1)
                        I_r = circ.current(d, p, V_ref / v1, v1)
                        if I_n is None or I_r is None:
                            good = False; break
                        Cs += I_n; Cr += I_r
                    if good:
                        C = Cs
                        C_all = rel_err(Cs, Cr); C_odd = rel_err(Cs, Cr, slice(2, H, 2))
                        dV = float(np.max(np.abs(V_now[2:] - V_ref[2:])
                                          / np.maximum(np.abs(V_ref[2:]), 1e-9)))
            odd = slice(2, H, 2)          # h3..h15 (SMPS 판별 대역)
            rows.append({"t0": a0 / 60.0, "on": [d for d, v in on.items() if v], "pw": {k: round(v, 1) for k, v in pw.items()},
                         "vh3_pct": round(100 * abs(rel_meas[2]), 2), "I1_mA": round(1e3 * abs(I_meas[0]), 1),
                         "stems": {}, "P_tot": round(P_tot, 1),
                         "C_all": C_all, "C_odd": C_odd, "dV_term": dV,
                         "A_all": rel_err(A, I_meas), "B_all": rel_err(B, I_meas), "D_all": rel_err(D, I_meas),
                         "B2_all": rel_err(B + d_res, I_meas), "B2_odd": rel_err(B + d_res, I_meas, odd),
                         "res_on": [d for d, v in on.items() if v and get_load_class(d) is LoadClass.RESISTIVE],
                         "A_odd": rel_err(A, I_meas, odd), "B_odd": rel_err(B, I_meas, odd), "D_odd": rel_err(D, I_meas, odd),
                         "per_order": {k: [float(abs(X[h - 1] - I_meas[h - 1]) / max(abs(I_meas[h - 1]), 1e-9))
                                          for X in (A, B, D, B + d_res)] for k, h in (("h1", 1), ("h3", 3), ("h5", 5), ("h9", 9), ("h13", 13))}})
        if not rows:
            print(f"{stem}: SMPS 창이 없다"); continue
        doc[stem] = rows
        def med(key): return float(np.median([r[key] for r in rows]))
        print(f"\n■ {stem}  창 {len(rows)}개 (2초, SMPS 만·선풍기 허용)   vh3 {min(r['vh3_pct'] for r in rows):.2f}~{max(r['vh3_pct'] for r in rows):.2f}%")
        print(f"   상대 RMS 오차  h1~h15 전체:  [A] 녹화중첩 {med('A_all'):.3f}   "
              f"[B] +텍스처델타 {med('B_all'):.3f}   [D] 모델단독 {med('D_all'):.3f}")
        print(f"                 h3~h15 홀수:  [A] {med('A_odd'):.3f}   [B] {med('B_odd'):.3f}   [D] {med('D_odd'):.3f}")
        # 풀개가 잠기면 [C] 는 [D] 와 같아진다 — 되돌린 소스에서 다시 풀면 V_meas 로 돌아오므로.
        # 벌어진 만큼이 **풀개 오차**다 (13.24.14).
        if not np.isnan(med("C_all")):
            print(f"   **[C] 결합 풀개 오차** (기준: 잠근 풀개)  전류 {med('C_all'):.4f} / 홀수 {med('C_odd'):.4f}   "
                  f"단자전압 최대 어긋남 {100 * med('dV_term'):.2f}%"
                  f"   (세션 {sess}, Z={z_ohm}Ω, n_iter={a.n_iter} relax={a.relax:g})")
        print("   차수별 |오차|/|실측| 중앙:   " + "  ".join(
            f"{k} A {np.median([r['per_order'][k][0] for r in rows]):.2f}/B {np.median([r['per_order'][k][1] for r in rows]):.2f}/D {np.median([r['per_order'][k][2] for r in rows]):.2f}"
            for k in ("h1", "h3", "h5", "h9", "h13")))
        # 조합별
        combos = {}
        for r in rows:
            combos.setdefault("+".join(x[:6] for x in r["on"]), []).append(r)
        for k, rs in sorted(combos.items(), key=lambda kv: -len(kv[1])):
            print(f"   {k:32s} n={len(rs):3d}  A {np.median([r['A_all'] for r in rs]):.3f}  B {np.median([r['B_all'] for r in rs]):.3f}  D {np.median([r['D_all'] for r in rs]):.3f}   홀수 A {np.median([r['A_odd'] for r in rs]):.3f} B {np.median([r['B_odd'] for r in rs]):.3f} D {np.median([r['D_odd'] for r in rs]):.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    st = circ.stats()
    n_c = sum(1 for rs in doc.values() for r in rs
              if not np.isnan(r.get("C_all", float("nan"))))
    print(f"\n저장: {a.out}   circuit {st}   [C] 결합 경로를 탄 창 {n_c}개")
    if n_c == 0:
        print("  ⚠⚠ **결합 경로가 한 번도 안 불렸다.** 이 판은 결합 변경을 검증하지 못한다 — "
              "세션 Z 가 없거나 SMPS 창이 없다. 13.24.14 의 함정 그 자체다")
    if st["failures"]:
        print(f"  ⚠ 회로 모델이 {st['failures']}회 조용히 실패했다 — "
              "circuit_model/__pycache__ 를 지워라 (13.23.5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
