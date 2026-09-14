# -*- coding: utf-8 -*-
"""14.33 ⓑ 검사 ② — **전압 채널**이 v32t 에서 실제로 더 흔들리나.

14.12 의 기전: 텍스처는 `step_s` 구간의 **중앙값**이라 그보다 짧은 전압 변동이 뭉개진다.
그러면 `step_s` 를 60 -> 20 으로 줄일 때 **창 안(10초) 전압 변동이 커져야** 한다.

14.33 ⓑ 검사 ① 은 **전류**(`obs_harm`)를 봤고 `v32t/v32 = 1.000~1.005` 였다 — 안 늘었다.
여기서 가른다: **전압에서는 늘었는데 전류로 안 넘어간 것**인가, **전압도 안 늘었나.**

채널 (13.12 배치 v2): 25 = `(V_rms−222)/10` · 45~50 = Re(V_h) · 51~56 = Im(V_h), h=1,3,5,7,9,11.
⚠ h=1 은 `(vr−222)/10`, h>=3 은 `arcsinh(vr·2)` 로 스케일이 다르다 — **차수끼리 더하지 마라.**
⚠ 두 캐시는 같은 시드·같은 청크라 창 `i` 가 **같은 시나리오**다 (검사 ① 에서 `y_on` 완전 일치,
  `y_power` 최대 차 0.09W 로 확인). 그래서 **짝지어** 볼 수 있다.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VCH = [25] + list(range(45, 57))
NAME = {25: "V_rms"}
for i, h in enumerate((1, 3, 5, 7, 9, 11)):
    NAME[45 + i] = f"Re(V{h})"
    NAME[51 + i] = f"Im(V{h})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+",
                    default=["cache/train60_v32", "cache/train60_v32t"])
    ap.add_argument("--n", type=int, default=20000, help="표본 창 수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    n_tot = json.load(open(Path(a.caches[0]) / "meta.json", encoding="utf-8"))["n_windows"]
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(n_tot, size=min(a.n, n_tot), replace=False))
    print(f"표본 {len(idx):,}창 / {n_tot:,}  (두 캐시에서 **같은 색인**)\n")

    W = {}
    for c in a.caches:
        f = np.load(Path(c) / "fine.npy", mmap_mode="r")
        W[c] = np.asarray(f[idx][:, VCH, :]).astype(np.float32)   # (n, 13, 600)
        print(f"  {c}: {W[c].shape} 읽음")

    A, B = a.caches[0], a.caches[1]
    print(f"\n{'채널':>9}{'창 안 std':>22}{'비':>8}{'창 사이 std':>22}{'비':>8}{'짝 |Δ| 중앙':>13}")
    print(f"{'':>9}{Path(A).name[-4:]:>11}{Path(B).name[-5:]:>11}{'':>8}"
          f"{Path(A).name[-4:]:>11}{Path(B).name[-5:]:>11}")
    for k, ch in enumerate(VCH):
        ia = W[A][:, k, :]; ib = W[B][:, k, :]
        wa = float(np.median(ia.std(-1))); wb = float(np.median(ib.std(-1)))
        ba = float(ia.mean(-1).std());     bb = float(ib.mean(-1).std())
        d = float(np.median(np.abs(ib.mean(-1) - ia.mean(-1))))
        mark = " ★" if wb > wa * 1.03 else ""
        print(f"{NAME[ch]:>9}{wa:>11.5f}{wb:>11.5f}{wb/max(wa,1e-12):>8.3f}"
              f"{ba:>11.5f}{bb:>11.5f}{bb/max(ba,1e-12):>8.3f}{d:>13.5f}{mark}")
    print("\n**창 안 std** 가 14.12 의 겨냥이다 — step_s 를 줄이면 짧은 변동이 살아 **커져야** 한다.")
    print("★ = v32t 가 3% 넘게 크다. 안 붙으면 전압에서도 안 늘어난 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
