"""지문의 sim2real 오차를 **모델 없이** 잰다 — 단일기기 창으로 (12.140)

왜 이것인가
----------
12.139 가 대조 파일 유령의 원인을 손실까지 좁혔다. 유령 프로젝터는 L_harm 을
**줄인다** — 모든 판에서, 두 대조 파일 전부에서. 그리고 그 이득의 **84%가 h1**
에서 나온다 (`inv_h2` 일 때. 균등 가중일 때는 26%).

h1 결손이 실재한다:

    test_11  |I1|관측 4996 mA  vs  지문 예측 4692 mA   결손 303 mA (6.1%)
    test_12          7616                7289                327 mA (4.3%)
    test_9           5795                5668                127 mA (2.2%)

그리고 **무효가 아니다** — PF1 이 0.94~0.97 이고 |I1|·V − P 가 1.6 VA 다.
결손의 98%가 유효분이다. 즉 **지문이 와트당 기본파 전류를 덜 만든다.**

`sig[k,1]` 이 6% 작으면 모델은 총전력을 맞추면서도 h1 전류가 모자란다. 그 구멍을
**와트당 h1 이 큰 기기**로 메우는 것이 손실상 이득이고, 그게 유령이다.
`inv_h2` 는 h1 무게를 1/15 에서 사실상 전부로 올려 그 이득을 3배로 키운다.

이 스크립트가 재는 것
-------------------
**단일기기 창** — 정답 구간상 딱 한 기기만 켜져 있는 창. 거기서는 배분이 없으므로

    실측 지문[h] = (|I_h 관측| − |계측 지문[h]|) / (P관측 − p_noise)      [A/W]

를 **모델 없이** 뽑을 수 있다. 그것을 합성 지문(`harmonic_signatures`)과 견준다.

⚠ **규칙 37** — sim2real 비교는 선택 기준을 양쪽에 똑같이 건다. 여기서는 양쪽이
   같은 양(와트당 전류)이고 실측 쪽만 창을 고른다. 합성 쪽은 상수라 고를 것이
   없다. 12.138 이 실패한 자리는 **SMPS 전용 창이 0~2개**여서였는데(규칙 28),
   저항 전용 창은 대조 파일에만 100개 넘게 있다. **저항으로 먼저 잰다.**

⚠ 이것은 **저항 5종만** 결론 낼 수 있다. SMPS 3종은 표본이 없다 — 없는 것을
   있다고 쓰지 않는다.

    python -X utf8 -m src.run_sig_calib_probe
    python -X utf8 -m src.run_sig_calib_probe --min-w 50 --min-win 8
"""
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.evaluation.real_events import build_on_off_truth, load_events
from src.evaluation.sealing import is_sealed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--min-w", type=float, default=30.0,
                    help="이보다 작은 창은 버린다 — 나눗셈이 폭발한다")
    ap.add_argument("--min-win", type=int, default=8, help="기기당 최소 창 수")
    ap.add_argument("--out", default="results/sig_calib.json")
    a = ap.parse_args()

    from src.model.net import harmonic_signatures, noise_signature
    from src.model.realdata import dense_targets
    from src.synthesis.segment_pool import SegmentPool

    ev = load_events()
    stems = [s for s in ev if not s.startswith("_") and not is_sealed(s)]
    apps = ["air_conditioner", "beam_projector", "electiric_kettle", "fan",
            "hair_dryer", "hotplate", "laptop_charger", "minipc", "oven"]
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, apps)          # (K,15,2) A/W  합성/격리
    nz = noise_signature(pool)                     # (15,2) A      계측계
    try:
        z = np.load("results/sig_insitu.npz", allow_pickle=True)
        sig_i = np.asarray(z["sig"], np.float32) if list(z["appliances"]) == apps else None
    except FileNotFoundError:
        sig_i = None

    # ── 단일기기 창을 모은다 ──────────────────────────────────────────────
    bag: Dict[str, List[np.ndarray]] = {}
    bagp: Dict[str, List[float]] = {}
    for stem in sorted(stems):
        rw = dense_targets(stem, stride=a.stride)
        OH, POBS, PN = [], [], []
        for i in range(0, len(rw), 512):
            idx = np.arange(i, min(i + 512, len(rw)))
            _, _, pobs, oh, pn = rw.batch(idx)
            OH.append(oh); POBS.append(pobs); PN.append(pn)
        oh = np.concatenate(OH); pobs = np.concatenate(POBS); pn = np.concatenate(PN)
        on, _ = build_on_off_truth(stem, apps, int(ev[stem]["cycles"]), ev)
        t = on[np.clip(rw.target_cycle, 0, len(on) - 1)]          # (n,K)
        n_on = t.sum(1)
        for j, app in enumerate(apps):
            m = (n_on == 1) & t[:, j] & (pobs - pn > a.min_w)
            if not m.any():
                continue
            bag.setdefault(app, []).append(oh[m])
            bagp.setdefault(app, []).append((pobs - pn)[m])

    print("=" * 100)
    print("실측 지문 vs 합성 지문 — 단일기기 창에서 (모델을 안 쓴다)")
    print(f"stride {a.stride} | 창당 최소 {a.min_w:g}W | 기기당 최소 {a.min_win}창")
    print("=" * 100)

    out: dict = {"_config": {"argv": sys.argv, "args": vars(a)}}
    rows = []
    for app in apps:
        if app not in bag:
            continue
        O = np.concatenate(bag[app]); P = np.concatenate(bagp[app])
        if len(P) < a.min_win:
            continue
        # 계측계 전류를 뺀 뒤 와트로 나눈다. 창마다 따로 풀고 중앙값을 쓴다 —
        # 평균은 고전력 창이 지배한다.
        # 복소로 남긴다 — 크기만 재면 위상이 사라져 지문으로 되돌릴 수 없다.
        cx = (O - nz[None]) / P[:, None, None]                     # (n,15,2) A/W
        cmed = np.median(cx, 0)                                    # (15,2)
        meas = np.linalg.norm(O - nz[None], axis=2) / P[:, None]
        med = np.median(meas, 0)
        syn = np.linalg.norm(sig[apps.index(app)], axis=1)
        rows.append((app, len(P), float(P.mean()), med, syn,
                     np.linalg.norm(sig_i[apps.index(app)], axis=1)
                     if sig_i is not None else None))
        out[app] = {"n_win": int(len(P)), "mean_w": float(P.mean()),
                    "measured": med.tolist(), "synth": syn.tolist(),
                    "measured_cx": cmed.tolist()}

    print(f"\n  [1] h1 — 와트당 기본파 전류 (mA/W). 1/V 가 이론값이다\n")
    print(f"  {'기기':<18s}{'창':>5s}{'평균W':>8s}{'실측':>9s}{'합성':>9s}"
          f"{'in-situ':>10s}{'합성 오차':>10s}{'1/V 등가':>10s}")
    print("  " + "-" * 82)
    for app, n, pw, med, syn, ins in rows:
        err = (syn[0] - med[0]) / med[0] * 100
        print(f"  {app:<18s}{n:>5d}{pw:>8.0f}{med[0] * 1000:>9.3f}{syn[0] * 1000:>9.3f}"
              f"{(ins[0] * 1000 if ins is not None else float('nan')):>10.3f}"
              f"{err:>+9.1f}%{1000 / med[0] / 1000:>10.1f}V")

    print(f"\n  [2] 차수별 실측/합성 비 — 1.0 이면 맞다, >1 이면 합성이 모자란다\n")
    print(f"  {'기기':<18s}" + "".join(f"{'h' + str(h + 1):>8s}" for h in range(9)))
    print("  " + "-" * 90)
    for app, n, pw, med, syn, ins in rows:
        print(f"  {app:<18s}" + "".join(
            f"{(med[h] / syn[h] if syn[h] > 1e-9 else float('nan')):>8.2f}"
            for h in range(9)))
    if any(r[5] is not None for r in rows):
        print(f"\n  같은 비, in-situ 지문 기준\n")
        print(f"  {'기기':<18s}" + "".join(f"{'h' + str(h + 1):>8s}" for h in range(9)))
        print("  " + "-" * 90)
        for app, n, pw, med, syn, ins in rows:
            if ins is None:
                continue
            print(f"  {app:<18s}" + "".join(
                f"{(med[h] / ins[h] if ins[h] > 1e-9 else float('nan')):>8.2f}"
                for h in range(9)))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n  저장: {a.out}")
    print("\n  ⚠ 표본이 없는 기기는 줄이 없다. 없는 것을 있다고 쓰지 않는다 (규칙 28).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
