"""`inv_h²` 를 **대조 파일을 넣고** 다시 잰다 (12.142)

왜 다시 재는가
-------------
12.135/12.136 이 차수 가중을 채택한 근거는 `모델오차/판별신호` 가 균등 2.23 에서
`1/h²` 1.57 로 단조로 준다는 것이었다. 그 자를 만든 창은

    HUMAN_ON_DEFAULT_STEMS = ("test_5","test_6","test_7","test_8","test_13")

**겨냥 5파일뿐이고 대조 파일이 하나도 없다.** 규칙 20 이 재학습 실험에 대조를
요구하는 이유가 그대로 여기에도 걸린다 — 그리고 12.137.1 에서 실제로 나빠진 곳이
**대조**였다 (유령대조 7.09 -> 11.51).

무엇이 다른가
------------
대조 파일(test_9/11/12)은 정답상 SMPS 가 0종이다. 그래서 거기서는

    자유해 = 9종 전부를 쓸 수 있는 NNLS
    정답해 = **저항만** 쓰는 NNLS

이고, 자유해가 SMPS 에 얹는 와트는 **전부 유령**이다. 모델도 손실도 학습도 없이
지문과 관측만으로 나오는 값이라, 유령의 하한을 지문 수준에서 직접 읽는다.

    ⚠ 겨냥 파일에서는 이것이 안 된다 — 거기서는 SMPS 가 실제로 켜져 있어서
      자유해가 SMPS 를 쓰는 것이 옳을 수도 있다. **대조에서만 깨끗하다.**

⚠ **반증 조건을 먼저 적는다.**
  - 대조에서도 `1/h²` 의 비가 균등보다 **낮으면** 12.136 의 채택은 그대로 산다.
    그러면 유령대조 회귀의 원인은 차수 가중이 아니다.
  - 대조에서 `1/h²` 의 **유령 와트가 균등보다 크면** 12.136 의 자가 대조를
    안 봐서 놓친 것이고, 채택 근거가 반쪽이었다는 뜻이다.

    python -X utf8 -m src.run_harm_weight_recheck
"""
from pathlib import Path
from typing import Dict, List, Sequence
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch
from scipy.optimize import nnls

from src.evaluation.real_events import build_on_off_truth, load_events
from src.evaluation.sealing import is_sealed
from src.model.realdata import HUMAN_ON_DEFAULT_STEMS, SMPS_APPLIANCES
from src.preprocessing import load_nilm_npz
from src.run_fit_insitu_sig import EVAL_DIR, WINDOW

H = 15
CTRL = ("test_9", "test_11", "test_12")


def load_windows(apps: Sequence[str], nz: np.ndarray, stems: Sequence[str]) -> List[dict]:
    """12.135 와 **같은 창 구성**. 파일 목록만 넓힌다 (규칙 37 — 기준을 양쪽에 똑같이)."""
    ev = load_events()
    NZ = nz[:H, 0] + 1j * nz[:H, 1]
    out = []
    for stem in stems:
        f = Path(EVAL_DIR) / f"{stem}.npz"
        if stem not in ev or not f.exists():
            continue
        z = load_nilm_npz(str(f))
        hc = np.asarray(z["harmonics_complex"])
        ok = np.asarray(z["is_valid"]).astype(bool)
        on, _ = build_on_off_truth(stem, apps, len(hc), ev)
        i = np.flatnonzero(ok)
        for k in range(0, len(i) - WINDOW, WINDOW):
            sl = i[k:k + WINDOW]
            sup = np.flatnonzero(on[sl].mean(0) > 0.5)
            if len(sup):
                out.append({"stem": stem, "sup": sup, "y": hc[sl].mean(0)[:H] - NZ})
    return out


def solve_group(sig: np.ndarray, wins: List[dict], allj: List[int],
                w: np.ndarray, smps: List[int]) -> dict:
    """가중 `w` 좌표계에서 자유해/정답해. **자유해가 SMPS 에 얹는 와트**도 낸다."""
    s = np.sqrt(np.maximum(w, 0))
    fr, tr, ghost = [], [], []
    for win in wins:
        y = win["y"] * s
        b = np.concatenate([y.real, y.imag])
        A_all = np.array([np.concatenate([sig[j, :H, 0] * s, sig[j, :H, 1] * s])
                          for j in allj]).T
        x, r = nnls(A_all, b)
        fr.append(r)
        # 자유해가 **정답에 없는** SMPS 에 얹은 전력 = 유령 (W). sig 가 A/W 라 x 는 W 다.
        ghost.append(sum(x[j] for j in smps if j not in set(win["sup"])))
        idx = list(win["sup"])
        A_t = np.array([np.concatenate([sig[j, :H, 0] * s, sig[j, :H, 1] * s])
                        for j in idx]).T
        tr.append(nnls(A_t, b)[1])
    f, t = float(np.mean(fr)), float(np.mean(tr))
    return {"free": f, "true": t, "signal": t - f, "ratio": t / max(t - f, 1e-12),
            "ghost_w": float(np.mean(ghost)), "n": len(wins)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/cnn_ovh.pt")
    ap.add_argument("--sig", default="", metavar="NPZ",
                    help="지문을 갈아끼운다 (`--sig-insitu` 와 같은 npz). "
                         "비우면 합성 지문. **학습 전에 그 지문의 유령을 미리 잰다**")
    ap.add_argument("--out", default="results/harm_weight_recheck.json")
    a = ap.parse_args()

    apps = list(torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"])
    from src.model.losses import HARM_DEADZONE_PROFILE
    from src.model.net import harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    SIG = harmonic_signatures(pool, apps); NZ = noise_signature(pool)
    del pool
    if a.sig:
        z = np.load(a.sig, allow_pickle=True)
        if list(z["appliances"]) != apps:
            raise SystemExit(f"{a.sig} 의 기기 목록이 다릅니다")
        SIG = np.asarray(z["sig"], np.float32)
        print(f"  지문 교체: {a.sig}")
    allj = list(range(len(apps)))
    smps = [apps.index(x) for x in SMPS_APPLIANCES if x in apps]

    ev = load_events()
    every = sorted(s for s in ev if not s.startswith("_") and not is_sealed(s))
    GROUPS = [
        ("12.136 원본 (겨냥 5)", list(HUMAN_ON_DEFAULT_STEMS)),
        ("겨냥 전부", [s for s in every if s not in CTRL]),
        ("**대조 3**", [s for s in every if s in CTRL]),
    ]
    tau = np.array(HARM_DEADZONE_PROFILE[:H])
    CAND = [
        ("균등 (현행 1단계)", np.ones(H)),
        ("홀수차만", np.array([1.0 if h % 2 == 0 else 0.0 for h in range(H)])),
        ("1/tau_h", 1.0 / np.maximum(tau, 1e-6)),
        ("1/h", np.array([1.0 / (h + 1) for h in range(H)])),
        ("**1/h² (채택된 것)**", np.array([1.0 / (h + 1) ** 2 for h in range(H)])),
        ("h1,h3 만", np.array([1.0, 0, 1.0] + [0.0] * 12)),
    ]

    print("=" * 96)
    print("차수 가중의 자를 **대조 파일을 넣고** 다시 푼다 (모델도 손실도 학습도 없다)")
    print("=" * 96)

    out: dict = {"_config": {"argv": sys.argv}}
    for gname, stems in GROUPS:
        wins = load_windows(apps, NZ, stems)
        if not wins:
            print(f"\n  {gname}: 창 0개 — 건너뜀 (규칙 28)")
            continue
        is_ctrl = "대조" in gname
        print(f"\n  ── {gname}  ({len(stems)}파일 {len(wins)}창: {', '.join(stems)})")
        print(f"     {'가중':<22s}{'유효차원':>9s}{'모델오차':>9s}{'판별신호':>10s}"
              f"{'모델오차/판별신호':>19s}"
              + (f"{'**유령 SMPS W**':>17s}" if is_ctrl else f"{'자유해 SMPS W':>15s}"))
        print("     " + "-" * (60 + 19 + 17))
        for name, w in CAND:
            w = w / max(w.max(), 1e-12)
            r = solve_group(SIG, wins, allj, w, smps)
            eff = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12)) * 2
            out.setdefault(gname, {})[name] = dict(r, eff_dim=eff)
            print(f"     {name:<22s}{eff:>9.1f}{r['true']:>9.4f}{r['signal']:>10.4f}"
                  f"{r['ratio']:>17.2f}배{r['ghost_w']:>16.1f}")
        if is_ctrl:
            print("     `유령 SMPS W` = 자유해가 **정답에 없는** SMPS 에 얹은 전력. 전부 유령이다.")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"\n  저장: {a.out}")
    print("\n  [읽는 법] 12.136 은 위 첫 줄만 보고 `1/h²` 를 골랐다. 대조 줄이 같은")
    print("  방향이면 그 선택은 그대로 산다. 반대면 자가 반쪽이었다는 뜻이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
