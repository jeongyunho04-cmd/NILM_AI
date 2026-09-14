# -*- coding: utf-8 -*-
"""창-안 전압 변동을 **실측 복합 녹화**와 맞대 본다 — 과보정 여부의 자 (14.51).

왜 이 자인가
------------
`run_diag_vtexseg.py` 는 **격리 기기 녹화**에서 천장을 쟀다 (n=6 이면 78% 회수).
그런데 합성 창은 복합이라 `V_h = rel_h·v_open − Z_h·I_h` 의 **둘째 항**이 크게 산다 —
여러 기기가 켜지고 꺼지면서 만드는 강하다. 그래서 "텍스처를 얼마나 넣으면 실측과 같은가"
는 **복합 대 복합**으로 재야 한다. 그 자가 `processed_data/composite_eval/test_*.npz` 다.

14.50 이 남긴 수치와 같은 양을 잰다: 창 안 `Re/Im(V_h)` 의 표준편차.
여기서는 `V_h/|V_1|` 로도 같이 낸다 — `v_open` 표류(모든 차수 공통)를 걷어낸,
**텍스처가 소유한 축**이다.

    python -X utf8 src/run_diag_vtexreal.py --segs 0 10 --n 200

`--segs` 는 `--vtex-seg-s` 값들이다 (0 = 끔). `--step-s` 는 텍스처 표집 간격.
"""
from pathlib import Path
import argparse
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.model.inputs import VOLT_ORDERS                                # noqa: E402
from src.synthesis import genopts                                       # noqa: E402

ORD = [h for h in VOLT_ORDERS if h > 1]
NPZ = "processed_data/npz"
REAL = "processed_data/composite_eval"


def _stats(V: np.ndarray) -> np.ndarray:
    """(W, 15) 복소 전압 고조파 -> (K, 4): Re(V_h)·Im(V_h)·Re(rel)·Im(rel) 의 창-안 std."""
    v1 = np.abs(V[:, 0])
    v1 = np.where(v1 > 1.0, v1, 1.0)
    idx = [h - 1 for h in ORD]
    A = V[:, idx]
    R = A / v1[:, None]
    return np.stack([A.real.std(0), A.imag.std(0), R.real.std(0), R.imag.std(0)], 1)


def real_windows(w: int, stride: int) -> np.ndarray:
    out = []
    for f in sorted(Path(REAL).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        V = np.asarray(z["voltage_harmonics_complex"], dtype=np.complex128)
        ok = (np.asarray(z["is_valid"]) == 1) if "is_valid" in z.files else np.ones(len(V), bool)
        for a in range(0, len(V) - w + 1, stride):
            if ok[a:a + w].mean() < 0.99:
                continue
            out.append(_stats(V[a:a + w]))
    return np.asarray(out)


def syn_windows(seg_s: float, step_s: float, n: int, w: int, seed: int) -> np.ndarray:
    from src.synthesis.dataset import NILMBatchGenerator
    opts = dict(genopts.V32, vtex_step_s=step_s)
    if seg_s:
        opts["vtex_seg_s"] = seg_s
    gen = genopts.build_synthesizer(opts, NPZ, "train", compute_gt_harmonics=False)
    bad = genopts.check(opts, gen)
    if bad:
        raise SystemExit("[vtexreal] 조립 관문 실패: " + " | ".join(bad))
    g = NILMBatchGenerator(segment_pool=gen.pool, window_size_cycles=w,
                           synthesizer=gen, compute_gt_harmonics=False)
    np.random.seed(seed)
    out = []
    for _ in range(n):
        smp, _ = g._synthesize_window()
        out.append(_stats(np.asarray(smp.voltage_harmonics_complex, dtype=np.complex128)))
    return np.asarray(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segs", nargs="+", type=float, default=[0.0, 10.0],
                    help="견줄 --vtex-seg-s 값들 (0 = 끔)")
    ap.add_argument("--step-s", type=float, default=10.0, help="텍스처 표집 간격 (초)")
    ap.add_argument("--n", type=int, default=200, help="합성 창 수")
    ap.add_argument("--window-cycles", type=int, default=3600)
    ap.add_argument("--stride", type=int, default=1800, help="실측 창 보폭 (사이클)")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    w = int(a.window_cycles)
    R = real_windows(w, int(a.stride))
    if not len(R):
        print("실측 복합 창이 없다")
        return 1
    S = {s: syn_windows(s, a.step_s, a.n, w, a.seed) for s in a.segs}
    print(f"\n실측 복합 {len(R)}창 · 합성 {a.n}창 · 창 {w / 60:.0f}초 · step_s {a.step_s}\n")

    cols = ("Re(V_h) [V]", "Im(V_h) [V]", "Re(V_h/|V1|)", "Im(V_h/|V1|)")
    for c in range(4):
        scale = 1.0 if c < 2 else 1e4
        unit = "" if c < 2 else " ×1e-4"
        print(f"  {cols[c]}{unit} — 창-안 표준편차 (창 중앙값)")
        hdr = "    차수     실측  " + "".join(f"{('seg=%g' % s):>16}" for s in a.segs)
        print(hdr)
        for i, h in enumerate(ORD):
            r0 = float(np.median(R[:, i, c])) * scale
            row = f"    h{h:<4d}{r0:9.3f}  "
            for s in a.segs:
                v = float(np.median(S[s][:, i, c])) * scale
                row += f"{v:9.3f}({100 * v / max(r0, 1e-12):4.0f}%)"
            print(row)
        print()
    print("  전 차수 중앙 — 합성/실측")
    for c in range(4):
        line = f"    {cols[c]:<14s}"
        for s in a.segs:
            rr = np.median(S[s][:, :, c], 0) / np.maximum(np.median(R[:, :, c], 0), 1e-15)
            line += f"  seg={s:g}: {100 * np.median(rr):5.0f}%"
        print(line)
    print("\n  ⚠ 100%% 를 크게 넘으면 **과보정**이다 — 실측에 없는 변동을 넣는 것이다.")
    print("  ⚠ `Re(V_h)` 는 `v_open` 표류(모든 차수 공통)를 포함한다. 텍스처가 소유한 축은")
    print("     `V_h/|V1|` 쪽이다 — 판정은 그 두 줄로 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
