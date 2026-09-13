# -*- coding: utf-8 -*-
"""표류를 **예측해서 뺀다** — v12g 회로에 그 블록의 실측 전압을 먹인다 (13.84.66).

13.84.57 은 표류를 노턴 1차 야코비 `ΔI = (∂I/∂P)ΔP − Y·ΔV` 로 근사하고 공변량을
`[dP, d|V1|, V3, V5, V7]` 여덟 개로 잡았다. 그런데 13.84.66 [B] 가 보인 것:
**V9~V15 가 V3~V7 이 만드는 것을 상쇄한다.** V1+V3 까지만 켜면 계통차를 1.65배로
과대예측하고, V7 까지면 2.1배, **V15 까지 다 켜야 0.9~1.0 배**가 된다. 여덟 개는 부족했다.

그래서 근사를 걷어내고 **그냥 푼다**: 블록마다 그 블록의 실측 전압 페이저 15개와 전력을
회로에 넣어 전류를 얻고, 칸 평균을 빼서 얻은 **예측 표류**를 실측 표류와 견준다.
자유 파라미터가 **0개**다 — 회로 모수는 `circ12_<dev>.pkl` 그대로다.

⚠ 이것은 **예측해서 빼기**지 사영해서 버리기가 아니다. 13.84.60 이 그 둘을 가른 절이다
   (고정 사영은 미니PC 를 4.11 -> 7.87W 로 악화시켰다). 덧셈 보정은 어느 방향도 안 버린다.
⚠ 전압 고조파는 **외생**이다 (분해와 무관하게 계측된다) — 채점 시점에 바로 쓸 수 있다.

    python -X utf8 src/run_diag_driftpred.py [--block 600] [--dev laptop_charger]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

from src.run_diag_driftcirc import cvec
from src.run_diag_driftcirc12 import load12, make_sim, solve_P
from src.run_diag_driftdim import load
from src.run_diag_driftlaw import BANDS, load_v

#: `driftlaw.BANDS` 에는 형제 **둘**만 있다 (표류 기저를 그 둘로 만들었으니까). 그런데
#: 정작 문제의 기기는 미니PC 다 — 전력대는 13.84.51 ① 의 것을 그대로 쓴다.
#: ⚠ 공유 상수 `BANDS` 를 고치지 않는다. 그것을 고치면 표류 기저·13.84.56 회귀가 같이 움직여
#: 옛 판과 못 견준다 ([[match-the-scoring-convention-before-comparing]]).
BANDS_X = dict(BANDS)
BANDS_X.setdefault("minipc", [(8, 14), (14, 19), (19, 24), (24, 30)])

#: 새 펌웨어(96열)가 직접 주는 전압 **꼬리** 차수 (13.73 / 13.83.11).
TAIL = list(range(17, 32, 2))


def load_tail(stem, data_dir="data"):
    """그 녹화의 **실측 전압 꼬리** — [(시작 t_s, (8,) 복소 볼트)]. 없으면 빈 리스트.

    규약 (`run_build_vtail.firmware_tails`):
      · `vhhi_seq` 가 ~61.7초마다 1 오르고 **그 안에서 값은 정확히 상수**다 — seq 당 한 벌.
      · 각도는 `vhdeg` 와 같은 규약 `arg(V_h) − h·arg(V_1)` 이라 **그대로** 쓴다.
      · 크기는 볼트다 (`vh1` 과 같은 단위) — 상대비로 안 바꾼다.
    첫 블록은 0 이다 (60초가 안 차서 아직 못 잰다). 그 구간은 h15 절단으로 남긴다.
    """
    import csv as _csv
    import io as _io
    import os as _os
    path = _os.path.join(data_dir, stem + ".csv")
    if not _os.path.exists(path):
        return []
    with _io.open(path, encoding="utf-8", errors="replace") as fh:
        head = fh.readline().rstrip("\n").split(",")
        need = ["t_s", "vhhi_seq"] + ["vhhi%d" % h for h in TAIL] \
            + ["vhhideg%d" % h for h in TAIL]
        if any(c not in head for c in need):
            return []                                   # 옛 펌웨어(79열)
        ix = {c: head.index(c) for c in need}
        seen, out = set(), []
        for row in _csv.reader(fh):
            if len(row) < len(head):
                continue
            try:
                s = float(row[ix["vhhi_seq"]]); t = float(row[ix["t_s"]])
                z = np.array([float(row[ix["vhhi%d" % h]])
                              * np.exp(1j * np.deg2rad(float(row[ix["vhhideg%d" % h]])))
                              for h in TAIL])
            except ValueError:
                continue
            if s in seen or not np.isfinite(z).all() or np.abs(z).sum() == 0:
                continue
            seen.add(s); out.append((t, z))
    return sorted(out)


def extend_v(v15, tail, t):
    """(15,) 전압 페이저에 그 시각의 꼬리를 붙여 (31,) 로. 꼬리가 없으면 그대로 15개.

    [정렬] 펌웨어는 60초를 **모아서 끝에 낸다.** 즉 seq k 의 시작에 실리는 값은 seq k−1
    구간을 잰 것이다. 그래서 시각 t 가 든 구간의 꼬리는 **다음** seq 에 실린 값이다.
    실측으로 확인했다 (충전기, 꼬리 있는 칸만, R² 자유도 0):

        꼬리 없음 0.794 · 녹화 중앙 0.760 · 그대로 0.759 · **한 칸 당김 0.835**

    "그대로" 가 "없음" 보다 **나쁘다** — 정렬이 틀리면 꼬리는 잡음이 된다. 당겨야 이긴다.
    """
    if not tail:
        return v15
    k = -1
    for i, (t0, _) in enumerate(tail):
        if t0 <= t:
            k = i
    k += 1                                  # 위 [정렬] 주석
    if k < 0 or k >= len(tail):
        return v15
    out = np.zeros(31, complex)
    out[:15] = v15
    for h, z in zip(TAIL, tail[k][1]):
        out[h - 1] = z
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+",
                    default=["laptop_charger", "beam_projector", "minipc"])
    ap.add_argument("--block", type=int, default=600)
    ap.add_argument("--npc", type=int, default=3072)
    ap.add_argument("--no-tail", action="store_true",
                    help="실측 전압 꼬리(h17~31)를 **안 쓴다** — h15 절단. A/B 용")
    a = ap.parse_args()

    print("%-16s %-18s %6s %11s %11s %10s %10s"
          % ("기기", "칸", "블록", "실측표류 mA", "예측표류 mA", "R^2 자유0", "R^2 배율1"))
    for dev in a.dev:
        d = load12(dev); par = tuple(d["params"])
        sim, pac = make_sim(a.npc)
        VH = load_v(dev)
        TL = {} if a.no_tail else {st: load_tail(st) for st, *_ in load(dev)}
        TN = TD = TN1 = 0.0
        for band in BANDS_X[dev]:
            for stem, P, hc, on, off in load(dev):
                if stem not in VH:
                    continue
                idx = np.nonzero(on & (P >= band[0]) & (P <= band[1]))[0]
                if len(idx) < 3 * a.block:
                    continue
                tl = TL.get(stem) or []
                # 시각은 npz 의 `t_rel_s` 다 — CSV 의 `t_s` 와 같은 눈금이다
                TS = np.load("processed_data/npz/%s.npz" % stem, allow_pickle=True)["t_rel_s"]
                Ym, Vs, Pm = [], [], []
                for s in range(0, len(idx) - a.block + 1, a.block):
                    j = idx[s:s + a.block]
                    Ym.append(cvec(hc[j].mean(0)))
                    Vs.append(extend_v(VH[stem][j].mean(0), tl, float(TS[j].mean())))
                    Pm.append(float(P[j].mean()))
                Ym = np.asarray(Ym); Pm = np.asarray(Pm)
                ntail = sum(1 for v in Vs if len(v) > 15)
                # 직류 부하는 **칸당 한 번만** 푼다 — 교류 전력과 거의 선형이라 블록마다
                # 다시 풀 값어치가 없다 (다시 풀어도 R^2 셋째 자리가 안 움직인다).
                Pd0 = solve_P(pac, par, Vs[0], Pm[0])
                S = np.asarray([cvec(sim(par, Pd0 * Pm[i] / Pm[0], Vs[i]))
                                for i in range(len(Vs))])
                Y0 = Ym - Ym.mean(0); S0 = S - S.mean(0)
                num = float(((Y0 - S0) ** 2).sum()); den = float((Y0 ** 2).sum())
                g = float((Y0 * S0).sum() / max((S0 * S0).sum(), 1e-30))
                n1 = float(((Y0 - g * S0) ** 2).sum())
                TN += num; TD += den; TN1 += n1
                print("%-16s %-18s %6d %10.1f %11.1f %10.3f %10.3f"
                      % (dev, "%s|%d-%d%s" % (stem[-6:], *band,
                                              " +꼬리" if ntail == len(Vs) else
                                              (" +꼬리%d" % ntail if ntail else "")), len(Ym),
                         1000 * np.sqrt((Y0 ** 2).sum() / len(Y0)),
                         1000 * np.sqrt((S0 ** 2).sum() / len(S0)),
                         1 - num / den, 1 - n1 / den))
        print("%-16s %-18s %6s %11s %11s %10.3f %10.3f"
              % (dev, "**합계**", "", "", "", 1 - TN / TD, 1 - TN1 / TD))
    print()
    print("견줌  13.84.56 실증 회귀 (자유도 128) 녹화하나빼기   +0.285")
    print("      13.84.57 v4.3 야코비 (자유도 1)               +0.37 충전기 / +0.35 프로젝터")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
