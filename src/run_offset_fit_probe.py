"""파일별 **계통 오프셋**을 공동 추정한다 — 모델도 학습도 없이 (12.147)

왜
--
12.146 이 계통 잔차를 분해했다. **66%가 상수다** (전역 48.8% + 파일별 17.3%p).
부하 의존이 7.7%p, 전압 고조파(Norton/감쇠)가 2.5%p, 나머지 24%가 미설명.

그리고 홀드아웃이 상한을 보였다 — `test_8` 에서 계통 잔차를 (다른 파일에서 뽑아)
빼니 프로젝터 배분 오차가 **+38.6 -> +0.9W** 였다. **상수만 알면 배분이 풀린다.**

상수는 라벨이 싸다. 파일 하나당 복소 벡터 15개(=30개 숫자)이고, ON 조합이
다양하면 자료만으로 식별된다 (test_5~13 이 조합 5~22개, 프로젝터 OFF 창 133~786개).

모형
----
파일 `f` 의 창 `w` 마다

    y_w  =  Σ_{k∈S_w} P_{k,w}·sig_k  +  c_f  +  ε
    Σ_k P_{k,w}  ≈  P관측_w                       (총전력 식 한 줄)

`c_f` 가 그 파일의 계통 오프셋이다. `P` 와 `c` 둘 다에 선형이므로 ALS 로 푼다.

    ① c 고정 -> 창마다 NNLS 로 P
    ② P 고정 -> c = median_w (y_w − A_w P_w)

⚠ **식별성** — 어떤 기기가 그 파일의 **모든** 창에서 켜져 있으면 그 기기의 전력과
   `c` 가 구분되지 않는다. 그래서 `--fit-on off` 로 **프로젝터가 꺼진 창에서만**
   적합하는 판을 같이 낸다 (평가는 켜진 창에서 하므로 순환이 아니다).

⚠ **짝수차 (12.72)** — 레인지 전환 DC 오프셋 단차가 만드는 인공물이다. 영점 교차
   0°/180° 대칭 펄스라 주기가 T/2 이고 **짝수차만 만든다**. 그리고
   **상수가 아니다** — 크기가 파형의 '영점 근처 체류 시간'(파고율 대리)을 따르고,
   같은 기기의 녹화 사이에서 **1.3~1.8배 흔들린다**. 근본 처방은 펌웨어 쪽
   (`NILM_SENS_LOW/HIGH` DC 오프셋 개별 캘리브레이션)이고 아직 안 됐다.

   그래서 여기서는:
     - **배분(NNLS)은 홀수차로만 푼다** — 짝수차는 기기 정보가 아니다
     - **오프셋은 전 차수에서 적합한다** — 짝수차 인공물에 갈 자리를 준다
     - 오프셋의 홀수/짝수 부분을 **따로 찍는다**. 짝수 쪽이 크면 인공물을
       흡수하고 있다는 뜻이고, 그것은 이 추정의 부수 효과이지 목적이 아니다

⚠ **반증 조건을 먼저 적는다.**
  - `--fit-on off` 로 적합한 오프셋이 프로젝터 배분을 **안 고치면** 오프셋은
    파일 상수가 아니라 창마다 다른 것이고, 이 축은 닫는다.
  - 짝수차를 빼고 적합한 오프셋이 더 잘 들면 인공물이 방해하고 있는 것이다.

    python -X utf8 -m src.run_offset_fit_probe
    python -X utf8 -m src.run_offset_fit_probe --fit-on off --orders odd
"""
from typing import Dict, List, Sequence
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
from scipy.optimize import nnls

from src.evaluation.power_ref import REFERENCE_W
from src.evaluation.real_events import build_on_off_truth, load_events

H = 15
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan",
        "hair_dryer", "hotplate", "laptop_charger", "minipc", "oven"]
RESIST = ("electiric_kettle", "hair_dryer", "hotplate", "oven",
          "air_conditioner", "fan")
ODD = np.arange(0, H, 2)
EVEN = np.arange(1, H, 2)


def load_file(stem: str, stride: int, nz: np.ndarray) -> dict:
    """창 단위 관측. **모델을 안 쓴다.**"""
    from src.model.realdata import dense_targets
    ev = load_events()
    rw = dense_targets(stem, stride=stride)
    OH, POBS, PN = [], [], []
    for i in range(0, len(rw), 512):
        idx = np.arange(i, min(i + 512, len(rw)))
        _, _, pobs, oh, pn = rw.batch(idx)
        OH.append(oh); POBS.append(pobs); PN.append(pn)
    oh = np.concatenate(OH)
    on, _ = build_on_off_truth(stem, APPS, int(ev[stem]["cycles"]), ev)
    t = on[np.clip(rw.target_cycle, 0, len(on) - 1)]
    y = oh[:, :H, :] - nz[None, :H, :]                      # (n,H,2)
    return {"y": y, "p": np.concatenate(POBS) - np.concatenate(PN), "on": t}


def _cols(sig: np.ndarray, idx: Sequence[int], rows: np.ndarray) -> np.ndarray:
    return np.array([np.concatenate([sig[j, rows, 0], sig[j, rows, 1]])
                     for j in idx]).T


def fit_offset(d: dict, sig: np.ndarray, rows: np.ndarray, mask: np.ndarray,
               iters: int = 6) -> np.ndarray:
    """ALS. `mask` 인 창만 쓴다. 오프셋은 **전 차수**로 돌려준다 (H,2)."""
    ix = np.flatnonzero(mask)
    c = np.zeros((H, 2))
    allrows = np.arange(H)
    for _ in range(iters):
        resid = []
        for w in ix:
            sup = np.flatnonzero(d["on"][w])
            if not len(sup):
                continue
            yy = d["y"][w] - c
            b = np.concatenate([yy[rows, 0], yy[rows, 1]])
            A = _cols(sig, sup, rows)
            g = np.linalg.norm(A) / max(np.sqrt(len(sup)) * 50.0, 1e-9)
            x, _ = nnls(np.vstack([A, g * np.ones((1, len(sup)))]),
                        np.concatenate([b, [g * d["p"][w]]]))
            full = _cols(sig, sup, allrows) @ x            # (2H,)
            resid.append(d["y"][w] - np.stack([full[:H], full[H:]], 1))
        c = np.median(np.array(resid), 0)
    return c


def solve_smps(d: dict, sig: np.ndarray, rows: np.ndarray, c: np.ndarray,
               mask: np.ndarray, ref: float) -> np.ndarray:
    """SMPS 전용 창에서 3종을 나눈다. 프로젝터를 못 박지 않는다 — 그것이 평가 대상이다."""
    j3 = [APPS.index(x) for x in ("beam_projector", "laptop_charger", "minipc")]
    A = _cols(sig, j3, rows)
    out = []
    for w in np.flatnonzero(mask):
        yy = d["y"][w] - c
        b = np.concatenate([yy[rows, 0], yy[rows, 1]])
        g = np.linalg.norm(A) / max(np.sqrt(3) * 50.0, 1e-9)
        x, _ = nnls(np.vstack([A, g * np.ones((1, 3))]),
                    np.concatenate([b, [g * d["p"][w]]]))
        out.append(x)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="+", default=["test_7", "test_8", "test_13"])
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default="results/offset_fit.json")
    a = ap.parse_args()

    from src.model.net import harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, APPS); nz = noise_signature(pool)
    del pool
    pj = APPS.index("beam_projector")
    ref = REFERENCE_W["beam_projector"][0]
    res_j = [APPS.index(x) for x in RESIST]

    print("=" * 100)
    print("파일별 계통 오프셋 공동 추정 — 모델도 학습도 없다 (ALS, 짝수차는 12.72)")
    print("=" * 100)
    out: dict = {"_config": {"argv": sys.argv}}
    tot = {}
    for stem in a.stems:
        d = load_file(stem, a.stride, nz)
        ne = d["on"].sum(1) > 0
        ev_ = ne & (~d["on"][:, res_j].any(1)) & d["on"][:, pj]       # 평가: SMPS 전용
        print(f"\n  ── {stem}  (지지 {int(ne.sum())}창, 평가 {int(ev_.sum())}창)")
        print(f"     {'적합 창':<26s}{'배분 차수':<10s}{'프로젝터W':>10s}{'오차':>8s}"
              f"{'충전기':>8s}{'미니PC':>8s}{'오프셋 홀수':>11s}{'짝수':>9s}")
        print("     " + "-" * 92)
        for fname, fmask in (("(오프셋 없음)", None),
                             ("전체 창", ne),
                             ("**프로젝터 OFF 창만**", ne & ~d["on"][:, pj])):
            for oname, rows in (("홀수차", ODD), ("전 15차", np.arange(H))):
                c = (np.zeros((H, 2)) if fmask is None
                     else fit_offset(d, sig, rows, fmask, a.iters))
                X = solve_smps(d, sig, ODD, c, ev_, ref)   # **배분은 늘 홀수차** (12.72)
                co = np.linalg.norm(c[ODD]) * 1000
                ce = np.linalg.norm(c[EVEN]) * 1000
                print(f"     {fname:<26s}{oname:<10s}{X[:, 0].mean():>10.1f}"
                      f"{X[:, 0].mean() - ref:>+8.1f}{X[:, 1].mean():>8.1f}"
                      f"{X[:, 2].mean():>8.1f}{co:>11.1f}{ce:>9.1f}")
                tot.setdefault(f"{fname}|{oname}", []).append(X[:, 0].mean() - ref)
                out.setdefault(stem, {})[f"{fname}|{oname}"] = {
                    "proj_w": float(X[:, 0].mean()), "lc_w": float(X[:, 1].mean()),
                    "mp_w": float(X[:, 2].mean()), "offset": c.tolist()}
                if fmask is None:
                    break                                   # 오프셋 없음은 한 줄이면 된다
        print(f"     {'참값':<26s}{'':<10s}{ref:>10.1f}{0.0:>+8.1f}")

    print(f"\n  {'적합 창 | 배분 차수':<40s}{'평균 |오차|':>12s}{'파일별':>28s}")
    print("  " + "-" * 82)
    for k, v in tot.items():
        print(f"  {k:<40s}{np.mean(np.abs(v)):>12.1f}"
              + "".join(f"{x:>+9.1f}" for x in v))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n  저장: {a.out}")
    print("\n  ⚠ `프로젝터 OFF 창만` 이 순환이 없는 판이다 — 평가는 켜진 창에서 한다.")
    print("  ⚠ 짝수차 오프셋이 크면 12.72 의 레인지 인공물을 흡수하는 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
