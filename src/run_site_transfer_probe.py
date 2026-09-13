# -*- coding: utf-8 -*-
"""장소별 고조파 **전달비** `T_h` — 실측/합성 간극의 정체를 1분에 판정한다 (12.179).

    python -m src.run_site_transfer_probe                          # T_h 표 + 파일별 안정성
    python -m src.run_site_transfer_probe --apply results/cnn_ac.pt  # 실측을 1/T 로 보정, 합성에 T 를 곱해 인과를 닫는다

[무엇을 재나]
같은 기기가 같은 전력으로 켜진 창에서 **실측 페이저 / 합성 페이저** 를 차수마다
나눈다. 단일 SMPS 창(충전기만, 프로젝터만)으로 재면 배분과 무관한 순수 간극이다.

    장소 A  h3~h15 전부 |T| 0.9~1.3, ∠T 30° 안        <- 합성이 그 장소다 (격리 녹화 장소)
    장소 B  h7 0.8/+34°  h9 0.7/+66°  h11 1.25/+105°  h13 2.9/+89°  h15 3.2/+48°

충전기와 프로젝터가 **같은 T** 를 준다 — 기기가 아니라 **장소의 성질**이다. 원자료
전압 고조파가 그것을 설명한다: 장소 B 는 vh9~vh13 이 2~4배 크고 vh15 는 1/7 이다
(격리 녹화의 전압 고조파는 장소 A 와 같다).

[왜 그것이 문제인가]
`--apply` 가 보인다. 실측을 1/T 로 나누면 1단계(cnn_ac) 가 장소 B 드라이기 강풍을
0.12 -> 1.000 으로 잡고(잔차 +780 -> +36W), 포트+드라 겹침 창의 에어컨 0.44 -> 0.00,
미니PC p_raw 3.0 -> 10.5W(참 ~10) 가 된다. 반대로 합성 창에 T 를 곱하면 합성에서
맞히던 미니PC 가 17.0 -> 2.8W 로 **실측과 똑같이** 무너진다 — 그리고 **위상만
곱해도** 무너진다(1.8W). 크기만 곱하면 멀쩡하다(19.5W). (12.179.1 의 91짝에서는
11.7 -> 0.8 / ∠ 0.11 / |T| 13.6 — 같은 방향.)

⚠ 적응된 모델(adapt_ac_*)은 보정 안 된 분포에 맞춰져 있어 보정 입력을 그대로
넣으면 게이트가 움직인다(미니PC 0.73 -> 0.48, 포트 -> 오븐). **보정을 쓰려면
그 입력으로 다시 적응해야 한다.**
"""
from typing import Dict, Tuple
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401
import numpy as np

NPZ = "processed_data/composite_eval/{}.npz"
RES = ("electiric_kettle", "oven", "hair_dryer", "hotplate", "air_conditioner", "fan")
SMPS = ("minipc", "laptop_charger", "beam_projector")
SITE = {"A": ("test_5", "test_6", "test_7", "test_8", "test_13"),
        "B": ("test_15", "test_16", "test_17", "test_18")}
ORDERS = (3, 5, 7, 9, 11, 13, 15)


def _mask(pairs, n):
    m = np.zeros(n, bool)
    for s, e in pairs:
        m[int(s * 60):min(int(e * 60), n)] = True
    return m


def real_windows(ev, stems, key: Tuple[str, ...], guard_s: float = 5.0):
    """저항이 없고 SMPS ON 집합이 정확히 `key` 인 사이클의 (페이저, 전력)."""
    Cs, Ps = [], []
    for stem in stems:
        if stem not in ev:
            continue
        n = int(ev[stem]["cycles"]); iv = ev[stem]["intervals"]
        big = np.zeros(n, bool)
        for a in RES:
            if a in iv:
                big |= _mask(iv[a].get("on", []) + iv[a].get("uncertain", []), n)
        z = np.load(NPZ.format(stem)); ok = z["is_valid"].astype(bool) & ~big
        for a in SMPS:
            ms = _mask(iv[a].get("on", []), n) if a in iv else np.zeros(n, bool)
            ok &= (ms == (a in key))
        g = int(guard_s * 60)
        for e in ev[stem]["events"]:
            c = int(e["t_s"] * 60); ok[max(0, c - g):c + g] = False
        Cs.append(z["harmonics_complex"][ok, :15]); Ps.append(z["power_features"][ok, 0])
    if not Cs:
        return np.zeros((0, 15), complex), np.zeros(0)
    return np.concatenate(Cs), np.concatenate(Ps)


class SynthCache:
    def __init__(self, cache: str):
        m = json.load(open(cache + "/meta.json", encoding="utf-8"))
        self.apps = m["appliances"]
        self.yo = np.asarray(np.load(cache + "/y_on.npy", mmap_mode="r"))
        self.po = np.asarray(np.load(cache + "/p_observed.npy", mmap_mode="r"))
        self.oh = np.load(cache + "/obs_harm.npy", mmap_mode="r")
        self.jr = [self.apps.index(a) for a in RES if a in self.apps]

    def phasors(self, key, p_med: float, tol: float = 15.0, limit: int = 20000):
        sel = (self.yo[:, self.jr].sum(1) == 0) & (np.abs(self.po - p_med) < tol)
        for a in SMPS:
            sel &= ((self.yo[:, self.apps.index(a)] > 0) == (a in key))
        idx = np.where(sel)[0][:limit]
        h = np.asarray(self.oh[idx]).astype(float)
        return h[:, :, 0] + 1j * h[:, :, 1]


def cmed(C):
    return np.median(C.real, 0) + 1j * np.median(C.imag, 0)


def fmt(t):
    return "".join(f"{abs(t[h - 1]):6.2f}/{np.degrees(np.angle(t[h - 1])):+5.0f}" for h in ORDERS)


def transfer(ev, sc, stems, key):
    C, P = real_windows(ev, stems, key)
    if len(P) < 100:
        return None, len(P), np.nan
    pm = float(np.median(P)); S = sc.phasors(key, pm)
    if len(S) < 30:
        return None, len(P), pm
    return cmed(C) / cmed(S), len(P), pm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default="cache/train60_ac")
    ap.add_argument("--events", default="processed_data/real_events_refined.json")
    ap.add_argument("--apply", default="", metavar="CKPT",
                    help="이 체크포인트로 실측(1/T)·합성(xT) 반사실을 돌린다")
    ap.add_argument("--save", default="results/_site_transfer_TB.npy")
    a = ap.parse_args()
    ev = json.load(open(a.events, encoding="utf-8"))["files"]
    sc = SynthCache(a.cache)

    print("전달비 T_h = median(실측)/median(합성), 단일 SMPS 창  (|T| / ∠T°)")
    print(f"{'기기·장소':14s}{'n':>7s}{'P':>5s}" + "".join(f"{'h%d' % h:>12s}" for h in ORDERS))
    T: Dict[Tuple[str, str], np.ndarray] = {}
    for key, lab in ((("laptop_charger",), "충전기"), (("beam_projector",), "프로젝터"), (("minipc",), "미니PC")):
        for site in ("B", "A"):
            t, n, pm = transfer(ev, sc, SITE[site], key)
            if t is None:
                print(f"{lab + ' ' + site:14s}{n:7d}  (표본 부족)"); continue
            T[(lab, site)] = t
            print(f"{lab + ' ' + site:14s}{n:7d}{pm:5.0f}" + fmt(t))
    print("\n파일별 (충전기 단독) — 장소의 성질이면 파일 간에 같아야 한다")
    for stem in SITE["B"]:
        t, n, pm = transfer(ev, sc, (stem,), ("laptop_charger",))
        print(f"{stem:14s}{n:7d}{pm:5.0f}" + (fmt(t) if t is not None else "  (표본 부족)"))
    TB = np.ones(15, complex)
    for h in range(1, 15):
        v = [T[k][h] for k in (("충전기", "B"), ("프로젝터", "B")) if k in T]
        if v:
            TB[h] = np.mean(v)
    np.save(a.save, TB)
    print(f"\nT_B (충전기·프로젝터 평균, h2~h15) -> {a.save}\n{'':26s}" + fmt(TB))

    if not a.apply:
        return 0
    import torch
    from src.run_gate_check import load_model
    from src.model.inputs import build_inputs, CURRENT_SCALE
    from src.model.realdata import RealWindows, load_nilm_npz, target_index, WINDOW_CYCLES, DEFAULT_DIR
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, _ = load_model(a.apply, dev); J = {x: apps.index(x) for x in apps}

    @torch.no_grad()
    def run(f, w):
        ft = torch.from_numpy(np.ascontiguousarray(f)).to(dev)
        wt = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(ft, wt)
        return torch.sigmoid(o["on_logit"]).float().cpu().numpy(), o["power_raw"].float().cpu().numpy()

    def windows(stem, Tdiv, select):
        raw = load_nilm_npz(f"{DEFAULT_DIR}/{stem}.npz"); x = RealWindows._to_33ch(raw).copy()
        if Tdiv is not None:
            C = (x[0:15] + 1j * x[15:30]) / Tdiv[:, None]; x[0:15], x[15:30] = C.real, C.imag
        n = x.shape[1]; off = target_index(WINDOW_CYCLES)
        t = np.arange(off, n - (WINDOW_CYCLES - 1 - off), 30); t = t[select(t, n)]
        f, w = build_inputs(np.stack([x[:, i - off:i - off + WINDOW_CYCLES] for i in t]))
        return f, w, raw["power_features"][t, 0]

    def sel_minipc(stem):
        iv = ev[stem]["intervals"]

        def f(t, n):
            on = _mask(iv["minipc"]["on"], n); big = np.zeros(n, bool)
            for x_ in RES:
                if x_ in iv:
                    big |= _mask(iv[x_].get("on", []), n)
            return np.array([on[max(0, i - 360):i + 360].all() and (~big[max(0, i - 360):i + 360]).all() for i in t])
        return f

    print(f"\n[실측을 1/T_B 로 보정 -> {a.apply}]  미니PC 창(저항 없음)")
    for stem in ("test_15", "test_18"):
        for lab, Td in (("원본", None), ("보정", TB)):
            f, w, P = windows(stem, Td, sel_minipc(stem)); g, r = run(f, w); j = J["minipc"]
            print(f"   {stem} {lab:4s} 창 {len(P):4d} 관측 {P.mean():5.1f}W  미니PC 게이트 {g[:, j].mean():.3f}"
                  f" p_raw {r[:, j].mean():5.2f} 전력 {(g[:, j] * r[:, j]).mean():5.2f}"
                  f" | 충전기 {(g[:, J['laptop_charger']] * r[:, J['laptop_charger']]).mean():5.1f}"
                  f" 프로젝터 {(g[:, J['beam_projector']] * r[:, J['beam_projector']]).mean():5.1f}")
    print("  드라이기 강풍 / 포트+드라 겹침 구간")
    for stem, t0, t1 in (("test_15", 355, 384), ("test_15", 420, 445), ("test_18", 340, 376), ("test_15", 388, 414)):
        for lab, Td in (("원본", None), ("보정", TB)):
            f, w, P = windows(stem, Td, lambda t, n: (t / 60 >= t0) & (t / 60 < t1)); g, r = run(f, w)
            print(f"   {stem} {t0}-{t1}s {lab:4s} 관측 {P.mean():5.0f}W  드라 {g[:, J['hair_dryer']].mean():.3f}"
                  f" 포트 {g[:, J['electiric_kettle']].mean():.3f} AC {g[:, J['air_conditioner']].mean():.3f}"
                  f" 오븐 {g[:, J['oven']].mean():.3f} 핫플 {g[:, J['hotplate']].mean():.3f}"
                  f" | 잔차 {(P - (g * r).sum(1)).mean():+5.0f}W")

    def mul_T(F, Tm, orders):
        F = F.copy()
        for h in orders:
            z = (np.sinh(F[:, h - 1]) + 1j * np.sinh(F[:, 15 + h - 1])) / CURRENT_SCALE * Tm[h - 1]
            F[:, h - 1] = np.arcsinh(z.real * CURRENT_SCALE); F[:, 15 + h - 1] = np.arcsinh(z.imag * CURRENT_SCALE)
        return F

    fine_mm = np.load(a.cache + "/fine.npy", mmap_mode="r"); wide_mm = np.load(a.cache + "/wide.npy", mmap_mode="r")
    yp = np.asarray(np.load(a.cache + "/y_power.npy", mmap_mode="r"))
    rng = np.random.default_rng(0); ca = sc.apps
    cj = ca.index("minipc")
    sel = (sc.yo[:, cj] > 0) & (yp[:, cj] > 6) & (yp[:, cj] < 16) & (sc.yo[:, sc.jr].sum(1) == 0)
    idx = rng.choice(np.where(sel)[0], size=200, replace=False)
    F = np.stack([np.asarray(fine_mm[i]) for i in idx]).astype(np.float32)
    W = np.stack([np.asarray(wide_mm[i]) for i in idx]).astype(np.float32)
    print("\n[합성 미니PC 6~16W 창(저항 없음) 200개에 T_B 를 곱한다 — 인과 닫기]")
    for lab, Tm, orders in (("원본", TB, ()), ("T_B h2~h15", TB, range(2, 16)), ("h11~h15 만", TB, range(11, 16)),
                            ("h5~h9 만", TB, range(5, 10)), ("|T_B| 만", np.abs(TB).astype(complex), range(2, 16)),
                            ("∠T_B 만", np.exp(1j * np.angle(TB)), range(2, 16))):
        g, r = run(mul_T(F, Tm, orders), W); j = J["minipc"]
        print(f"   {lab:12s} 미니PC 게이트 {g[:, j].mean():.3f} p_raw {r[:, j].mean():5.2f} 전력 {(g[:, j] * r[:, j]).mean():5.2f}"
              f"  (참 {yp[idx, cj].mean():.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
