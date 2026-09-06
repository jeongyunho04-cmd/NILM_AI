# -*- coding: utf-8 -*-
"""혼합 검증 (mixval, v12) — 복합 녹화의 SMPS 창에서 생성기의 세 가지 전류 재현을 실측과 견준다 (13.2, 2026-09-06).

무엇을 재나
----------
복합 녹화(`test_2`)에서 켜진 기기 집합이 일정한 2초 창(사건 앞뒤 3초 제외)을 골라, 실측 총전류 페이저
I_meas(h1..h15) 와 다음을 비교한다:

    [A] 녹화 중첩      Σ_i I_rec,i(p_i) · (V_rec/V_meas)                      지금까지의 생성기 = 단순 중첩
    [B] A + 텍스처 델타 [A] + Σ_i [ I_sim,i(p_i, V15_meas) − I_sim,i(p_i, V15_rec,i) ]     새 생성기 (13.2)
    [D] 모델 단독       Σ_i I_sim,i(p_i, V15_meas)                                교체안 — 델타보다 나은지 본다

V15_meas 는 그 창의 **측정 단자 전압**이라 공유 임피던스 결합이 이미 들어 있다 — 그래서 여기서는 결합 델타를
따로 더하지 않는다 (더하면 이중 계상). 결합 항의 크기는 합성 소스 전압에서 V_term 을 만들 때만 뜻이 있다.

기기 전력 p_i: 저항·선풍기는 라벨의 ΔP(기기 몫), 미니PC·프로젝터도 라벨 ΔP. 충전기는 나머지
(P_total − 나머지 − 계측 바닥) — 창 안에서 크게 변하는 유일한 SMPS 라 그렇게 둔다.
켜져 있지 않은 SMPS 는 풀의 대기 지문을, 계측계는 noise 기준을 더한다.

쓰는 법
------
    python -X utf8 -m src.run_mixval12 --stems test_2 --out results/_mixval12.json
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
from src.synthesis.coupling import SmpsCircuit, SMPS_DEVICES

H = 15
SMPS = set(SMPS_DEVICES)
#: 창 길이와 사건 앞뒤 가드 (초). 사람 시각 ±3초 + 정착.
WIN_S = 2.0
GUARD_S = 3.0
#: 이 기기들이 켜진 창은 뺀다 — 1kW 급 저항이 h1 을 지배해 SMPS 재현을 못 본다.
EXCLUDE_ON = ("electiric_kettle", "hair_dryer", "oven", "hotplate", "air_conditioner")


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
        return I * scale, self.lib.file_rel(stem), stem


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
    ap.add_argument("--stems", nargs="*", default=["test_2"])
    ap.add_argument("--events", default="processed_data/real_events.json")
    ap.add_argument("--npz-dir", default="processed_data/composite_eval")
    ap.add_argument("--out", default="results/_mixval12.json")
    a = ap.parse_args()

    ev = json.load(open(a.events, encoding="utf-8"))["files"]
    pool = SegmentPool(npz_dir="processed_data/npz")
    lib = default_library()
    circ = SmpsCircuit()
    rec = RecordedSig(pool, lib)
    noise = pool._pick_noise_reference(1.6)
    doc: Dict[str, dict] = {}
    print("=" * 110)
    print("혼합 검증 v12 — SMPS 창의 실측 총전류 대 [A] 녹화 중첩 / [B] +텍스처 델타 / [D] 모델 단독   (13.2)")
    print("=" * 110)
    for stem in a.stems:
        if stem not in ev:
            print(f"{stem}: 라벨 없음"); continue
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
            B = A.copy(); D = A.copy()
            ok = True
            for d, p in pw.items():
                got = rec.at(d, p, v1)
                if got is None:
                    ok = False; break
                I_rec, rel_rec, rstem = got
                A += I_rec; B += I_rec
                if d in SMPS and circ.has(d):
                    I_env = circ.current(d, p, rel_meas, v1)
                    I_own = circ.current(d, p, rel_rec, v1) if rel_rec is not None else None
                    if I_env is not None and I_own is not None:
                        B += (I_env - I_own)
                        D += I_env
                    else:
                        ok = False; break
                else:
                    D += I_rec
            if not ok:
                continue
            odd = slice(2, H, 2)          # h3..h15 (SMPS 판별 대역)
            rows.append({"t0": a0 / 60.0, "on": [d for d, v in on.items() if v], "pw": {k: round(v, 1) for k, v in pw.items()},
                         "vh3_pct": round(100 * abs(rel_meas[2]), 2), "I1_mA": round(1e3 * abs(I_meas[0]), 1),
                         "stems": {}, "P_tot": round(P_tot, 1),
                         "A_all": rel_err(A, I_meas), "B_all": rel_err(B, I_meas), "D_all": rel_err(D, I_meas),
                         "A_odd": rel_err(A, I_meas, odd), "B_odd": rel_err(B, I_meas, odd), "D_odd": rel_err(D, I_meas, odd),
                         "per_order": {k: [float(abs(X[h - 1] - I_meas[h - 1]) / max(abs(I_meas[h - 1]), 1e-9))
                                          for X in (A, B, D)] for k, h in (("h1", 1), ("h3", 3), ("h5", 5), ("h9", 9), ("h13", 13))}})
        if not rows:
            print(f"{stem}: SMPS 창이 없다"); continue
        doc[stem] = rows
        def med(key): return float(np.median([r[key] for r in rows]))
        print(f"\n■ {stem}  창 {len(rows)}개 (2초, SMPS 만·선풍기 허용)   vh3 {min(r['vh3_pct'] for r in rows):.2f}~{max(r['vh3_pct'] for r in rows):.2f}%")
        print(f"   상대 RMS 오차  h1~h15 전체:  [A] 녹화중첩 {med('A_all'):.3f}   [B] +텍스처델타 {med('B_all'):.3f}   [D] 모델단독 {med('D_all'):.3f}")
        print(f"                 h3~h15 홀수:  [A] {med('A_odd'):.3f}   [B] {med('B_odd'):.3f}   [D] {med('D_odd'):.3f}")
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
    print(f"\n저장: {a.out}   circuit {circ.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
