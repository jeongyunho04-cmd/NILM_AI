"""단일기기 창에서 지문을 적합한다 — sim2real 보정 (12.140)

무엇을 고치나
------------
`harmonic_signatures` 는 **격리 녹화/합성**에서 나온 와트당 전류 상수다. 그것이
실측과 얼마나 다른지는 여태 잰 적이 없다 — 12.122.10 이 `sig(P,V)` 전이 실패를
확인했고 12.122.11 이 `--sig-insitu` 로 **복합 파일 합동 적합**을 만들었지만,
합동 적합은 배분을 모형이 풀어야 해서 SMPS 3종만 건드린다.

이쪽은 **배분이 없는 창만 골라** 모델도 최적화도 없이 상수를 읽는다.

    단일기기 창 = 정답 구간상 딱 한 기기만 켜져 있는 창
    실측 지문[h] = median_창( (I_h관측 − 계측지문[h]) / (P관측 − p_noise) )   [A/W]

**저항 5종에도 표본이 있다** — 12.138 이 막힌 자리(SMPS 전용 창 0~2개, 규칙 28)를
비켜간다. 오븐 718창, 프로젝터 172창, 핫플 125창, 충전기 69창.

왜 홀수차만인가 (`--orders odd`, 기본값)
--------------------------------------
12.72 가 짝수 차수를 **계측 인공물**로 확정했고 입력에서 지운다
(`run_gate_check.EVEN_CHANNELS`). 그런데 `L_harm` 은 짝수차를 아직 본다.
`harm_scale` 이 짝수차에서 0.003 대라 그 차수의 오차는 16~40배 증폭된다.

실측 지문을 **전 차수**에 넣으면 짝수차의 실측값(헤어드라이어 h2 가 합성의
450배)이 그대로 들어와 `L_harm` 이 3.70 -> 15.88 로 뛴다. 홀수차만 갈아 끼우면
3.70 -> **3.51** 로 오히려 내려간다. 그래서 기본이 `odd` 다.

무엇이 확인됐나 (재학습 없이, 12.140)
-----------------------------------
유령 프로젝터가 `L_harm` 에서 얻는 이득 (`가중+지문` 판, `inv_h2`):

    파일        합성 지문     전부 적합    **LOFO**
    test_11    -0.1345    -0.0059    -0.0166  (12%)
    test_12    -0.0494    +0.0049    +0.0036  (부호가 뒤집힌다)
    test_9     -0.0024    -0.0009    -0.0005  (22%)

**LOFO 로도 산다** — 그 파일을 빼고 적합해도 유령의 이득이 88% 사라진다.
짝수차 마스크만 걸고 지문을 안 고치면 **오히려 나빠진다**(131%) — 고칠 곳은
마스크가 아니라 지문이다.

⚠ **이것은 손실의 유인만 잰 것이다.** 학습된 모델이 실제로 나아지는지는
   재학습해서 채점해야 안다 (규칙 22 — 순서가 맞다고 기제가 아니다).

⚠ 표본이 없는 기기(에어컨·선풍기)는 **합성 그대로 둔다.** 없는 것을 있다고
   쓰지 않는다 (규칙 28).

쓰는 법
------
    python -X utf8 -m src.run_fit_single_sig --out results/sig_single.npz
    python -X utf8 -m src.run_fit_single_sig --lofo          # 파일별 검증
    python -X utf8 -m src.run_fit_single_sig --orders all    # 짝수차까지 (권장 안 함)

    # 적응에 넣기 — `--sig-insitu` 와 같은 자리에 들어간다
    python -X utf8 -m src.run_adapt --init results/cnn_ovh.pt \
        --cache cache/train60_ovh --harm-weight inv_h2 \
        --sig-insitu results/sig_single.npz --tag adapt_ss_s0 --out results
"""
from typing import Dict, List, Optional, Set
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.evaluation.real_events import build_on_off_truth, load_events
from src.evaluation.sealing import is_sealed

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan",
        "hair_dryer", "hotplate", "laptop_charger", "minipc", "oven"]


def collect(stride: int) -> Dict[str, tuple]:
    """파일별로 (관측 고조파, 기기 전력, 정답 on) 을 모은다."""
    from src.model.realdata import dense_targets
    ev = load_events()
    out: Dict[str, tuple] = {}
    for stem in sorted(s for s in ev if not s.startswith("_") and not is_sealed(s)):
        rw = dense_targets(stem, stride=stride)
        OH, POBS, PN = [], [], []
        for i in range(0, len(rw), 512):
            idx = np.arange(i, min(i + 512, len(rw)))
            _, _, pobs, oh, pn = rw.batch(idx)
            OH.append(oh); POBS.append(pobs); PN.append(pn)
        on, _ = build_on_off_truth(stem, APPS, int(ev[stem]["cycles"]), ev)
        out[stem] = (np.concatenate(OH), np.concatenate(POBS) - np.concatenate(PN),
                     on[np.clip(rw.target_cycle, 0, len(on) - 1)])
    return out


def baselines(data: Dict[str, tuple], nz: np.ndarray, min_bg: int) -> Dict[str, np.ndarray]:
    """파일별 **배경 부하**의 고조파. 정답상 전부 꺼진 창에서 읽는다.

    라벨은 아홉 기기만 적는다. 그 집의 나머지(냉장고·조명·공유기…)는 라벨에 없고,
    단일기기 창에도 **그대로 들어 있다.** 빼지 않으면 그 전류가 전부 그 기기의
    지문으로 들어간다 — 저항의 3차처럼 **원래 0 근처인 차수에서 치명적**이다
    (오븐 h3 이 합성의 19.2배로 나왔고, 그것이 12.37.2 의 3차 판별자를 죽인다).

    ⚠ 배경이 파일 안에서 일정하다고 가정한다. 창 수가 모자란 파일은 추정 못 한다.
    """
    out: Dict[str, tuple] = {}
    for stem, (oh, p, t) in data.items():
        m = t.sum(1) == 0
        if m.sum() >= min_bg:
            # 전류와 **전력을 같이** 낸다. 분자만 빼면 분모가 부풀어 저전력 기기가
            # 망가진다 (미니PC 51W 에 배경 13W 면 지문이 −44% 로 나온다).
            out[stem] = (np.median(oh[m] - nz[None], 0), float(np.median(p[m])))
    return out


def fit(data: Dict[str, tuple], sig0: np.ndarray, nz: np.ndarray, orders: str,
        min_w: float, min_win: int, exclude: Optional[Set[str]] = None,
        bg: Optional[Dict[str, np.ndarray]] = None,
        credible: float = 0.0) -> tuple:
    """단일기기 창에서 지문을 읽는다. 표본이 모자란 기기는 `sig0` 그대로."""
    exclude = exclude or set()
    out = sig0.copy()
    n_used: Dict[str, int] = {}
    skipped: Dict[str, List[int]] = {}
    sl = slice(0, None, 2) if orders == "odd" else slice(None)
    for j, app in enumerate(APPS):
        O, P, B = [], [], []
        for stem, (oh, p, t) in data.items():
            if stem in exclude or (bg is not None and stem not in bg):
                continue
            bi, bw = bg[stem] if bg else (np.zeros_like(nz), 0.0)
            # 배경의 전력까지 뺀 것이 그 기기의 전력이다. 기기가 배경의 2배는
            # 돼야 나눗셈이 의미가 있다 — 아니면 그 창은 단일기기가 아니다.
            m = (t.sum(1) == 1) & t[:, j] & (p - bw > max(min_w, 2.0 * bw))
            if m.any():
                O.append(oh[m]); P.append(p[m] - bw)
                B.append(np.repeat(bi[None], int(m.sum()), 0))
        if not O:
            continue
        O = np.concatenate(O); P = np.concatenate(P); B = np.concatenate(B)
        n_used[app] = int(len(P))
        if len(P) < min_win:
            continue
        # 창마다 따로 풀고 **중앙값**을 쓴다. 평균은 고전력 창이 지배한다.
        cx = np.median((O - nz[None] - B) / P[:, None, None], 0)   # (H,2) A/W
        idx = list(range(0, len(nz), 2)) if orders == "odd" else list(range(len(nz)))
        for h in idx:
            # **볼 수 없는 차수는 안 고친다** (규칙 28). 뺀 배경이 그 기기가 내는
            # 것보다 크면 그 차수의 측정은 배경의 것이지 기기의 것이 아니다.
            own = float(np.linalg.norm(sig0[j, h])) * float(np.median(P))
            bgh = float(np.median(np.linalg.norm(B[:, h], axis=1)))
            if credible > 0 and own < credible * bgh:
                skipped.setdefault(app, []).append(h + 1)
                continue
            out[j, h] = cx[h].astype(np.float32)
    return out, n_used, skipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--min-w", type=float, default=30.0)
    ap.add_argument("--min-win", type=int, default=8)
    ap.add_argument("--orders", choices=("odd", "all"), default="odd",
                    help="짝수차는 계측 인공물이다 (12.72). 기본은 홀수차만 간다")
    ap.add_argument("--exclude", default="", metavar="STEMS",
                    help="적합에서 뺄 파일 (쉼표). **대조 파일을 빼야 규칙 20 이 산다** — 대조에서 적합한 지문으로 대조 유령을 채점하면 대조가 아니다")
    ap.add_argument("--subtract-baseline", action="store_true",
                    help="파일별 배경 부하(정답상 전부 꺼진 창)를 빼고 적합한다. "
                         "**안 빼면 라벨에 없는 부하가 그 기기의 지문이 된다**")
    ap.add_argument("--min-bg", type=int, default=5, help="배경 추정에 필요한 최소 창 수")
    ap.add_argument("--credible", type=float, default=0.0, metavar="R",
                    help="그 기기가 그 차수에 내는 양이 배경의 R배 미만이면 **안 고친다** "
                         "(규칙 28). 1.0 이면 '배경보다 크면 고친다'. 0 이면 끔")
    ap.add_argument("--lofo", action="store_true", help="파일별 leave-one-out 검증")
    ap.add_argument("--out", default="results/sig_single.npz")
    a = ap.parse_args()

    from src.model.net import harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig0 = harmonic_signatures(pool, APPS)
    nz = noise_signature(pool)
    del pool

    print("=" * 92)
    print(f"단일기기 창 지문 적합 | stride {a.stride} | 창당 >{a.min_w:g}W | "
          f"기기당 >={a.min_win}창 | 차수 {a.orders}")
    print("=" * 92)
    data = collect(a.stride)
    excl = {x for x in a.exclude.split(",") if x}
    if excl:
        miss = excl - set(data)
        if miss:
            raise SystemExit(f"없는 파일: {sorted(miss)}")
        print(f"\n  적합에서 뺀 파일: {sorted(excl)}  (규칙 20)")
    bg = baselines(data, nz, a.min_bg) if a.subtract_baseline else None
    if bg is not None:
        no = sorted(set(data) - set(bg))
        print(f"\n  배경을 뺐다 — 추정된 파일 {len(bg)}개"
              + (f", **창 부족으로 통째로 뺀 파일** {no}" if no else ""))
        for stem in sorted(bg):
            bi, bw = bg[stem]
            print(f"     {stem:<10s}{bw:>8.1f}W   h1 {np.linalg.norm(bi[0]) * 1000:7.1f}mA"
                  f"  h3 {np.linalg.norm(bi[2]) * 1000:6.1f}mA"
                  f"  h5 {np.linalg.norm(bi[4]) * 1000:6.1f}mA")
    sig, n_used, skipped = fit(data, sig0, nz, a.orders, a.min_w, a.min_win, excl,
                               bg, a.credible)
    if skipped:
        print("\n  **안 고친 차수** (배경이 기기보다 크다 — 규칙 28):")
        for app, hs in skipped.items():
            print(f"     {app:<18s}h{', h'.join(map(str, hs))}")

    print(f"\n  {'기기':<18s}{'창':>6s}{'h1 합성':>10s}{'h1 실측':>10s}{'Δ':>8s}"
          f"{'h3 비':>8s}{'h5 비':>8s}{'상태':>12s}")
    print("  " + "-" * 82)
    for j, app in enumerate(APPS):
        n = n_used.get(app, 0)
        s0 = np.linalg.norm(sig0[j], axis=1); s1 = np.linalg.norm(sig[j], axis=1)
        st = "적합" if n >= a.min_win else ("표본 부족" if n else "표본 없음")
        print(f"  {app:<18s}{n:>6d}{s0[0] * 1000:>10.3f}{s1[0] * 1000:>10.3f}"
              f"{(s1[0] - s0[0]) / s0[0] * 100:>+7.1f}%"
              f"{s1[2] / max(s0[2], 1e-9):>8.2f}{s1[4] / max(s0[4], 1e-9):>8.2f}{st:>12s}")
    print("\n  `h3 비`/`h5 비` = 실측/합성. 1.0 이면 합성이 맞다.")

    np.savez(a.out, sig=sig.astype(np.float32), appliances=np.array(APPS),
             stems=np.array(sorted(data)), n_windows=np.array(
                 [n_used.get(x, 0) for x in APPS]),
             orders=np.array(a.orders), excluded=np.array(sorted(excl)),
             argv=np.array(sys.argv))
    print(f"\n  저장: {a.out}  (규칙 33 — 만든 명령을 npz 안에 같이 넣었다)")
    print(f"  적응에 넣으려면:  --sig-insitu {a.out}")

    if a.lofo:
        print("\n  [LOFO] 그 파일을 빼고 적합한 지문이 그 파일에서 얼마나 다른가")
        print(f"  {'뺀 파일':<10s}{'h1 최대차':>12s}{'전차수 최대차':>14s}{'남은 창':>10s}")
        print("  " + "-" * 48)
        for stem in sorted(data):
            s_o, n_o, _ = fit(data, sig0, nz, a.orders, a.min_w, a.min_win,
                              excl | {stem}, bg, a.credible)
            d1 = np.abs(np.linalg.norm(s_o[:, 0], axis=1)
                        - np.linalg.norm(sig[:, 0], axis=1)).max()
            da = np.abs(s_o - sig).max()
            print(f"  {stem:<10s}{d1 * 1000:>11.3f}{da * 1000:>14.3f}"
                  f"{sum(n_o.values()):>10d}")
        print("\n  단위는 mA/W. h1 이 4.5 대이므로 0.1 이면 2% 다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
