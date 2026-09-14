# -*- coding: utf-8 -*-
"""14.33 ⓑ 검사 — **v32t 가 의도대로 됐나.** 학습 전에 자료만 본다.

14.12 의 주장: 텍스처 표집 간격 60 -> 20초면
    텍스처 280 -> 819개 · h3~h15 **산포/원시 0.73 -> 0.95** · 표류비 0.75 -> **0.80**
그리고 0.80 이 정답이다 — 14.11 이 표류 **분산의 65.4%** 를 회로로 설명했으므로
진폭비 √0.654 = **0.809** 다.

자: **기기 하나만 켜진 창**에서 와트당 고조파 `|c_h|/P` 의 **변동계수(CV)** 를 차수별로 잰 뒤,
합성 CV / 실측 CV 를 낸다. 1.0 이면 합성이 실측만큼 흔들린다는 뜻이다.
⚠ 모델을 안 쓴다. `obs_harm`(n,15,2)·`y_power`·`y_on` 만 읽으므로 36MB 면 된다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ODD = [2, 4, 6, 8, 10, 12, 14]          # h=3,5,...,15 (0 기반)
KO = {"beam_projector": "프로젝", "laptop_charger": "충전기", "minipc": "미니PC",
      "electiric_kettle": "포트", "oven": "오븐", "hotplate": "핫플", "hair_dryer": "드라이"}
SM = ("beam_projector", "laptop_charger", "minipc")


def cv_per_order(c, p, min_w=3.0):
    """와트당 `|c_h|/P` 의 차수별 변동계수. c (n,15) complex, p (n,)."""
    m = p > min_w
    if m.sum() < 40:
        return None, int(m.sum())
    r = np.abs(c[m]) / p[m][:, None]                    # (n,15)
    med = np.median(r, axis=0)
    mad = np.median(np.abs(r - med), axis=0) * 1.4826   # 로버스트 std
    return mad / np.maximum(med, 1e-12), int(m.sum())


def from_cache(d: Path, apps):
    oh = np.load(d / "obs_harm.npy", mmap_mode="r")
    yp = np.load(d / "y_power.npy", mmap_mode="r")
    yo = np.load(d / "y_on.npy", mmap_mode="r")
    yo = np.asarray(yo); yp = np.asarray(yp)
    only = yo.sum(1) == 1
    out = {}
    for j, a in enumerate(apps):
        m = only & (yo[:, j] > 0)
        if m.sum() < 40:
            continue
        c = np.asarray(oh[m]); c = c[:, :, 0] + 1j * c[:, :, 1]
        out[a] = cv_per_order(c, yp[m, j].astype(np.float64))
    return out


def from_real(apps):
    from src.evaluation.sealing import is_sealed
    from src.preprocessing import load_nilm_npz
    from src.run_plot_real import load_events
    from src.evaluation.power_ref import REFERENCE_W
    ev = load_events(); acc = {}
    for s in sorted(ev):
        if is_sealed(s):
            continue
        d = load_nilm_npz(f"processed_data/composite_eval/{s}.npz")
        P = np.asarray(d["power_features"])[:, 0].astype(np.float64)
        H = np.asarray(d["harmonics_complex"])
        n = len(P); t = np.arange(n) / 60.0
        lab = {}
        for a in apps:
            mm = np.zeros(n, bool)
            for x, y in ev[s]["intervals"].get(a, {}).get("on", []):
                mm |= (t >= x) & (t < y)
            lab[a] = mm
        tot = np.sum([lab[a] for a in apps], axis=0)
        for a in apps:
            m = lab[a] & (tot == 1)
            if m.sum() < 40:
                continue
            acc.setdefault(a, [[], []])
            acc[a][0].append(H[m]); acc[a][1].append(P[m])
    out = {}
    for a, (cs, ps) in acc.items():
        out[a] = cv_per_order(np.concatenate(cs), np.concatenate(ps))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", default=["cache/train60_v32", "cache/train60_v32t"])
    ap.add_argument("--apps", nargs="+", default=list(SM))
    a = ap.parse_args()
    import json
    apps = json.load(open(Path(a.caches[0]) / "meta.json", encoding="utf-8"))["appliances"]
    print("기기 하나만 켜진 창의 **와트당 고조파 변동계수** (h3~h15 평균)\n")
    real = from_real(apps)
    cache = {c: from_cache(Path(c), apps) for c in a.caches}
    hdr = f"{'기기':>7}{'실측 CV':>9}{'창(실측)':>9}"
    for c in a.caches:
        hdr += f"{Path(c).name[-6:]:>11}{'비':>7}"
    print(hdr)
    for app in a.apps:
        if app not in real or real[app][0] is None:
            continue
        rv = float(np.mean(real[app][0][ODD])); rn = real[app][1]
        line = f"{KO.get(app, app):>7}{rv:>9.4f}{rn:>9}"
        for c in a.caches:
            x = cache[c].get(app)
            if x is None or x[0] is None:
                line += f"{'-':>11}{'-':>7}"; continue
            sv = float(np.mean(x[0][ODD]))
            line += f"{sv:>11.4f}{sv / max(rv, 1e-12):>7.2f}"
        print(line)
    print("\n'비' = 합성 CV / 실측 CV. **1.0 이면 합성이 실측만큼 흔들린다.**")
    print("14.12 의 주장: 0.73 -> 0.95 (표류비로는 0.75 -> 0.80, 정답 √0.654=0.809)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
