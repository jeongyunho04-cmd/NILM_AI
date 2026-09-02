"""등가저항을 **손실에** 넣으면 유령이 죽나 — 재학습 없이 (12.143)

왜 이것인가
----------
12.132 가 근본 원인을 **판별자가 하나뿐인 것**으로 짚었다. 저항은 등가저항
`R = V²/P` 라는 둘째 판별자가 있어서 조건수 21.7 로 더 나쁜데도 배분이 되고,
SMPS 는 그것이 없어서 `L_harm` 하나에 걸려 있다.

그런데 **그 둘째 판별자가 후처리에만 있다** (`postproc.resistive_match`,
12.112). 2단계 적응 손실은

    L_adapt = w_cons·|Σ P̂ + Σ Ŝ + P_noise − P관측|  +  w_harm·‖Σ P̂·sig + … − 관측‖

둘뿐이고, 여기 어디에도 `R` 이 없다. 그래서 **저항 전력이 자유롭다.**

12.139/12.142 가 잰 유령은 정확히 그 자유를 쓴다 — 대조 파일에서 프로젝터에
12W 를 얹고 저항에서 그만큼 빼면 `L_cons` 는 합이 같아 모르고, `inv_h²` 의
`L_harm` 은 h1 에서 아홉 기기 기울기가 −0.008 로 같아 못 가른다.

무엇을 재는가
------------
같은 NNLS 자(12.142)를 두 번 푼다.

    ① 자유해     아홉 기기 전력이 전부 연속 자유       <- 지금의 손실
    ② 저항 못박음 저항 4종은 `V²/R` 의 **켜짐/꺼짐**만, SMPS 만 자유
                 (드라이기는 전열 54.3Ω / 반파 108.6Ω 둘 다 후보. 12.109.2)

②는 **라벨을 안 쓴다** — 24개 조합을 다 풀고 잔차가 가장 작은 것을 고른다.
`resistive_match` 가 후처리에서 하는 것과 같고, 그것을 손실이 하면 어떻게 되는지
미리 보는 것이다.

⚠ **반증 조건을 먼저 적는다.**
  - ② 의 유령 SMPS W 가 ① 과 비슷하면 **등가저항은 유령을 못 막는다.**
    그러면 이 축은 닫고 학습을 안 돌린다.
  - ② 가 저항 조합을 자주 틀리면(정답 대비) 못박음이 오히려 해롭다. 같이 찍는다.

⚠ 이것은 **NNLS 자의 진술**이지 학습된 모델의 진술이 아니다. 12.142 가 이 자를
   차수 가중에서는 검증했고(예측 7.2->9.4 vs 실측 7.09->8.22) **지문 교체에서는
   틀렸다.** 이번 것은 가중도 지문도 아닌 세 번째 종류라, 자의 신뢰도는 미지다.

    python -X utf8 -m src.run_resmatch_loss_probe
"""
from pathlib import Path
from typing import Dict, List, Sequence
import argparse
import itertools
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
from src.model.postproc import HALFWAVE_OHM, RESISTIVE_OHM
from src.model.realdata import SMPS_APPLIANCES
from src.preprocessing import load_nilm_npz
from src.run_fit_insitu_sig import EVAL_DIR, WINDOW

H = 15
CTRL = ("test_9", "test_11", "test_12")
#: 2단계 실측 손실의 가중 (`run_adapt` 기본값, `_wi_s0.log:17`).
W_CONS, W_HARM = 0.1, 4.0


def load_windows(apps: Sequence[str], nz: np.ndarray, stems: Sequence[str],
                 stride: int = 30) -> List[dict]:
    """**`resistive_match` 와 같은 입자**로 창을 만든다 (stride 30 사이클 = 0.5초).

    ⚠ 처음에는 12.135 의 3600사이클(60초) 창을 그대로 썼다가 조합 정답률이 6% 로
    나왔다. 60초 안에서 핫플·오븐의 **온도조절기가 껐다 켠다** — 그 창에서는
    `V²/R` 이 상수가 아니고 '켜진 조합' 자체가 정의되지 않는다. 후처리가 도는
    입자와 맞춰야 그 처방을 재는 자가 된다 (규칙 25).
    """
    from src.model.realdata import dense_targets
    ev = load_events()
    NZ = nz[:H, 0] + 1j * nz[:H, 1]
    out = []
    for stem in stems:
        if stem not in ev:
            continue
        rw = dense_targets(stem, stride=stride)
        OH, POBS, PN = [], [], []
        for i in range(0, len(rw), 512):
            idx = np.arange(i, min(i + 512, len(rw)))
            _, _, pobs, oh, pn = rw.batch(idx)
            OH.append(oh); POBS.append(pobs); PN.append(pn)
        oh = np.concatenate(OH); pobs = np.concatenate(POBS); pn = np.concatenate(PN)
        v = rw.v_observed.astype(np.float64)
        on, _ = build_on_off_truth(stem, apps, int(ev[stem]["cycles"]), ev)
        t = on[np.clip(rw.target_cycle, 0, len(on) - 1)]
        for k in range(len(oh)):
            sup = np.flatnonzero(t[k])
            if len(sup):
                out.append({"stem": stem, "sup": sup, "v": float(v[k]),
                            "p": float(pobs[k] - pn[k]),
                            "y": (oh[k, :H, 0] + 1j * oh[k, :H, 1]) - NZ})
    return out


def combos(apps: Sequence[str]) -> List[Dict[int, float]]:
    """저항 4종의 on/off 조합. 드라이기는 전열/반파 둘 다 후보 (12.109.2)."""
    opts = []
    for a, r in RESISTIVE_OHM.items():
        if a not in apps:
            continue
        j = apps.index(a)
        cand = [None, r] + ([HALFWAVE_OHM[a]] if a in HALFWAVE_OHM else [])
        opts.append([(j, c) for c in cand])
    return [{j: r for j, r in c if r is not None} for c in itertools.product(*opts)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/cnn_ovh.pt")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--max-win", type=int, default=400,
                    help="파일군당 창 상한 (NNLS x 24조합이라 느리다)")
    ap.add_argument("--out", default="results/resmatch_loss.json")
    a = ap.parse_args()

    apps = list(torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"])
    from src.model.losses import HARM_DEADZONE_PROFILE
    from src.model.net import harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    SIG = harmonic_signatures(pool, apps); NZ = noise_signature(pool)
    del pool
    allj = list(range(len(apps)))
    smps = [apps.index(x) for x in SMPS_APPLIANCES if x in apps]
    res_j = {apps.index(x) for x in RESISTIVE_OHM if x in apps}
    COMBOS = combos(apps)

    ev = load_events()
    every = sorted(s for s in ev if not s.startswith("_") and not is_sealed(s))
    tau = np.array(HARM_DEADZONE_PROFILE[:H])
    CAND = [("균등 (1단계)", np.ones(H)),
            ("1/tau_h", 1.0 / np.maximum(tau, 1e-6)),
            ("**1/h² (현행 2단계)**", np.array([1.0 / (h + 1) ** 2 for h in range(H)]))]

    print("=" * 98)
    print("등가저항을 손실에 넣으면 유령이 죽나 — 저항을 못 박고 SMPS 만 자유롭게 푼다")
    print(f"조합 {len(COMBOS)}개 (드라이기 전열 54.3Ω / 반파 108.6Ω 포함), 라벨 안 씀")
    print("=" * 98)

    out: dict = {"_config": {"argv": sys.argv}}
    for gname, stems in [("**대조 3**", [s for s in every if s in CTRL]),
                         ("겨냥 전부", [s for s in every if s not in CTRL])]:
        wins = load_windows(apps, NZ, stems, a.stride)
        if not wins:
            continue
        if len(wins) > a.max_win:      # 균등 솎기. 파일 비율은 보존된다
            wins = [wins[i] for i in
                    np.linspace(0, len(wins) - 1, a.max_win).astype(int)]
        print(f"\n  ── {gname}  ({len(wins)}창)")
        print(f"     {'가중':<22s}{'유령 SMPS W':>13s}{'저항 못박음 뒤':>15s}"
              f"{'감소':>9s}{'조합 정답률':>12s}")
        print("     " + "-" * 74)
        for name, w in CAND:
            w = w / max(w.max(), 1e-12)
            s = np.sqrt(np.maximum(w, 0))
            A_all = np.array([np.concatenate([SIG[j, :H, 0] * s, SIG[j, :H, 1] * s])
                              for j in allj]).T
            A_smps = np.array([np.concatenate([SIG[j, :H, 0] * s, SIG[j, :H, 1] * s])
                               for j in smps]).T
            g_free, g_pin, hit = [], [], []
            for win in wins:
                b = np.concatenate([(win["y"] * s).real, (win["y"] * s).imag])
                sup = set(win["sup"])
                x, _ = nnls(A_all, b)
                g_free.append(sum(x[j] for j in smps if j not in sup))
                # 저항을 V²/R 로 못 박고 SMPS 만 자유. 조합은 잔차로 고른다.
                best = (np.inf, None, None)
                for cb in COMBOS:
                    pr = sum(win["v"] ** 2 / r for r in cb.values())     # 저항 전력 W
                    fixed = np.zeros(len(b))
                    for j, r in cb.items():
                        fixed = fixed + (win["v"] ** 2 / r) * A_all[:, j]
                    xs, rr = nnls(A_smps, b - fixed)
                    # **2단계 손실 그대로 점수를 매긴다** — 고조파만 보면 저항끼리
                    # 구분이 안 된다 (전부 h1 4.6 mA/W). 조합을 정하는 것은 전력이고,
                    # 그것이 `resistive_match` 가 컨덕턴스를 맞추는 이유다 (12.112).
                    sc = W_CONS * abs(pr + xs.sum() - win["p"]) + W_HARM * rr
                    if sc < best[0]:
                        best = (sc, cb, xs)
                _, cb, xs = best
                g_pin.append(sum(v for k, v in zip(smps, xs)
                                 if k not in sup))
                hit.append(float(set(cb) == (sup & res_j)))
            f, p = float(np.mean(g_free)), float(np.mean(g_pin))
            out.setdefault(gname, {})[name] = {
                "free_w": f, "pinned_w": p, "combo_acc": float(np.mean(hit))}
            print(f"     {name:<22s}{f:>13.1f}{p:>15.1f}"
                  f"{(1 - p / max(f, 1e-9)) * 100:>8.0f}%{np.mean(hit) * 100:>11.0f}%")
        if "대조" in gname:
            print("     대조는 정답상 SMPS 0종이라 두 열 다 **전부 유령**이다.")

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  저장: {a.out}")
    print("\n  `조합 정답률` 이 낮으면 못박음이 저항을 틀리게 고르는 것이라 해롭다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
