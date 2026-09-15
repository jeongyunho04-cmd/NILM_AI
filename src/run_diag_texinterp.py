# -*- coding: utf-8 -*-
"""연속 텍스처 토막을 **양 끝만 풀고 보간**해도 되나 (14.101).

왜 묻나
-------
`vtex_seg_s=3` 이면 창 하나를 20토막으로 나눠 토막마다 회로를 다시 푼다 — 그것이 굽기
비용의 거의 전부다 (14.96: seg_s=0 의 0.52 -> seg_s=3 의 ~30 core-s/window).

그런데 실측에서 잰 것 (14.101 앞부분):
```
  홀수차 텍스처가 창 하나(120블록)에서 움직이는 양
     h3  0.5%(무작위보행) ~ 5.5%(한 방향)   ·  h5  0.4% ~ 4.3%
  토막 하나(6블록=3초)면 h3 이 **0.1~0.3%** 움직인다
```
**연속 토막 사이가 아주 매끄럽다.** 그러면 양 끝 둘만 풀고 가운데를 **복소 선형보간**
해도 될지 모른다 — 회로 해 20번이 2번이 된다 (**~10배**).

⚠ 그런데 그냥 되리라고 믿으면 안 된다. 14.51 이 *"연속 텍스처가 만드는 창-안 전류
  변동이 텍스처 델타 **자체의 48~73%**"* 라고 쟀다 — **작은 입력이 큰 출력을 낸다**는
  뜻이고 그것이 SMPS 의 초선형성이다. 초선형이면 보간 오차도 클 수 있다. **재야 안다.**

무엇을 재나
-----------
같은 녹화의 **연속 텍스처 n장**을 뽑아, 기기·전력마다
```
  정확     d_k = I(rel_k·v1) − I(rel_rec·v1)          <- 지금 하는 것 (n번 푼다)
  보간     양 끝 d_0, d_{n-1} 만 풀고 가운데를 복소 선형보간 (2번 푼다)
  한 장    가운데 텍스처 하나로 전부            <- seg_s=0 (1번 푼다)
```
그리고 **셋을 견준다**. 기준은 두 가지다:
```
  ① |보간 − 정확| / |정확|                 절대적으로 얼마나 틀리나
  ② |보간 − 정확| / (정확의 창-안 산포)     **이것이 진짜 자다** —
     우리가 seg_s 로 사려는 것이 바로 그 '창-안 산포' 이므로,
     오차가 산포만 하면 보간은 아무것도 안 사는 것이다
```
`한 장` 열은 **음성 대조**다 — 거기서는 ②가 1.0 근처여야 한다 (산포를 통째로 잃으니까).

    python -X utf8 src/run_diag_texinterp.py
"""
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

ORD = (1, 3, 5, 7, 9, 11, 13, 15)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseg", type=int, default=20, help="토막 수 (seg_s=3 이면 20)")
    ap.add_argument("--runs", type=int, default=40, help="몇 창어치를 볼까")
    ap.add_argument("--step-s", type=float, default=3.0, help="텍스처 표집 간격 (seg_s 와 같아야)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from src.synthesis.coupling import SmpsCircuit, SMPS_DEVICES, _pbin, P_BIN_W
    from src.synthesis.vtexture import (VoltageTextureLibrary, set_default_step_s,
                                        set_default_vtail, DEFAULT_VTAIL_NPZ)
    set_default_step_s(float(a.step_s))
    set_default_vtail(DEFAULT_VTAIL_NPZ if Path(DEFAULT_VTAIL_NPZ).exists() else None)
    lib = VoltageTextureLibrary.from_npz_dir()
    circ = SmpsCircuit()

    # 세션별로 텍스처를 시간순으로 모은다 — `_sample_texture_run` 이 뽑는 것과 같은 꼴.
    by_stem = {}
    for t in lib.textures:
        by_stem.setdefault(t.stem, []).append(t)
    for k in by_stem:
        by_stem[k].sort(key=lambda x: getattr(x, "t_rel_s", 0.0))
    usable = [k for k, v in by_stem.items() if len(v) >= a.nseg]
    print("텍스처 %d장 · 세션 %d개 · 토막 %d장을 담는 세션 %d개 (표집 간격 %.1f초)"
          % (len(lib.textures), len(by_stem), a.nseg, len(usable), a.step_s))
    if not usable:
        raise SystemExit("연속 %d장을 담는 세션이 없다 — --step-s 를 줄여라" % a.nseg)

    rng = np.random.default_rng(a.seed)
    devs = [d for d in SMPS_DEVICES if circ.has(d)]
    print("기기 %s · 창 %d개\n" % (devs, a.runs))
    acc = {d: {"n": 0, "e_int": [], "e_one": [], "spread": [], "exact": []} for d in devs}

    for _ in range(a.runs):
        stem = usable[rng.integers(len(usable))]
        seq = by_stem[stem]
        i0 = int(rng.integers(0, len(seq) - a.nseg + 1))
        run = seq[i0:i0 + a.nseg]
        rel_rec = lib.file_rel_full(stem)
        if rel_rec is None:
            continue
        v1 = float(rng.uniform(210.0, 232.0))
        for d in devs:
            p = float(max(_pbin(rng.uniform(10.0, 80.0)), 1) * P_BIN_W)
            b = circ.current(d, p, rel_rec, v1, None)
            if b is None:
                continue
            ex = []
            for t in run:
                cur = circ.current(d, p, t.source_rel_full(), v1, None)
                if cur is None:
                    ex = []
                    break
                ex.append(cur - b)
            if len(ex) != a.nseg:
                continue
            ex = np.asarray(ex)                                   # (nseg, 15)
            w = np.linspace(0.0, 1.0, a.nseg)[:, None]
            itp = ex[0][None] * (1 - w) + ex[-1][None] * w        # 양 끝만 알고 보간
            one = np.repeat(ex[a.nseg // 2][None], a.nseg, 0)     # 가운데 한 장 (seg_s=0)
            sp = np.abs(ex - ex.mean(0, keepdims=True))           # 창-안 산포
            A = acc[d]
            A["n"] += 1
            A["e_int"].append(np.abs(itp - ex))
            A["e_one"].append(np.abs(one - ex))
            A["spread"].append(sp)
            A["exact"].append(np.abs(ex))

    print("  %-14s %5s | %-24s | %-24s" % ("", "창",
                                           "**보간** (양끝 2번)", "한 장 (seg_s=0, 음성대조)"))
    print("  %-14s %5s | %10s %13s | %10s %13s"
          % ("기기·차수", "", "|Δ|/|정확|", "|Δ|/창안산포", "|Δ|/|정확|", "|Δ|/창안산포"))
    for d in devs:
        A = acc[d]
        if not A["n"]:
            print("  %-14s %5d  (표본 없음)" % (d, 0))
            continue
        E = np.concatenate(A["e_int"]); O = np.concatenate(A["e_one"])
        S = np.concatenate(A["spread"]); X = np.concatenate(A["exact"])
        for h in ORD:
            j = h - 1
            ref = np.median(S[:, j]) + 1e-15      # 창-안 산포 — seg_s 로 사려는 바로 그것
            base = np.median(X[:, j]) + 1e-15     # |정확 델타|
            print("  %-10s h%-2d %5d | %10.4f %13.3f | %10.4f %13.3f"
                  % (d if h == 1 else "", h, A["n"] if h == 1 else 0,
                     np.median(E[:, j]) / base, np.median(E[:, j]) / ref,
                     np.median(O[:, j]) / base, np.median(O[:, j]) / ref))
    print("")
    print("  읽는 법 — **|Δ|/창안산포** 가 본다. `한 장` 열이 1.0 근처여야 자가 제대로 선 것이고,")
    print("           `보간` 열이 그보다 **훨씬 작아야** 보간이 산포를 실제로 사는 것이다.")
    print("           0.2 밑이면 보간으로 회로 해를 20 -> 2 로 줄일 수 있다 (~10배).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
