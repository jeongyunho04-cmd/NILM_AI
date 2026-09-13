"""기기 하나에 지문이 **둘**인가 — 단일기기 창을 전력으로 층화한다 (12.141)

왜 이것인가
----------
12.140 이 배경 부하를 제대로 빼고 나니 합성 지문이 거의 다 맞았다. **하나만
빼고** — 헤어드라이어가 +7.5% 다. 그리고 12.139 가 유령이 붙는 자리로 지목한
기기가 바로 그것이다 (test_12 에서 총전력을 맞춘 뒤에도 ON/OFF 87.6배).

가설: **헤어드라이어는 반파 모드가 있다.** 열선에 다이오드를 물려 반주기만
통전하면 와트당 전류가 달라지고 짝수차·DC 가 생긴다. 그런데 `harmonic_signatures`
는 **기기당 지문이 하나**다. 두 모드의 평균을 쓰면 어느 모드에서도 틀린다.

무엇을 재는가
------------
단일기기 창(배경 뺀 것)을 **전력으로 층화**해서 와트당 전류가 층마다 다른지 본다.
모드가 둘이면 전력이 두 무리로 갈리고 **무리마다 와트당 값이 다르다.**

⚠ **반증 조건을 먼저 적는다.**
  - 와트당 전류가 층에 걸쳐 평평하면 **모드는 하나**다. 가설이 죽는다.
  - 헤어드라이어만이 아니라 **모든 기기가 같은 기울기**를 보이면 그것은 모드가
    아니라 계측 비선형이거나 배경 추정의 잔여다 (규칙 5 — 비교하려는 변수 하나만
    다른지 확인한다). 그래서 **표본이 있는 기기를 전부 같이 찍는다.**
  - 전력 무리가 하나뿐이면 (연속 분포) 모드가 아니라 조광/온도조절이다.

⚠ 짝수차는 계측 인공물로 확정돼 있다 (12.72). 그런데 **반파의 증거는 짝수차에
   있다.** 인공물이 차수 전체를 설명하는지, 반파 창에서만 커지는지가 갈린다 —
   그것도 같이 찍는다.

    python -X utf8 -m src.run_sig_mode_probe
    python -X utf8 -m src.run_sig_mode_probe --app hair_dryer --bins 6
"""
from typing import Dict, List
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.run_fit_single_sig import APPS, baselines, collect


def gather(data, nz, bg, app: str, min_w: float) -> tuple:
    """그 기기의 단일기기 창 — 배경(전류·전력)을 뺀 것."""
    j = APPS.index(app)
    O, P, S = [], [], []
    for stem, (oh, p, t) in data.items():
        if stem not in bg:
            continue
        bi, bw = bg[stem]
        m = (t.sum(1) == 1) & t[:, j] & (p - bw > max(min_w, 2.0 * bw))
        if m.any():
            O.append(oh[m] - nz[None] - bi[None]); P.append(p[m] - bw)
            S += [stem] * int(m.sum())
    if not O:
        return None, None, None
    return np.concatenate(O), np.concatenate(P), np.array(S)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default="", help="비우면 표본 있는 기기 전부 (대조)")
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--min-w", type=float, default=30.0)
    ap.add_argument("--min-bin", type=int, default=6, help="층당 최소 창 수 (규칙 28)")
    a = ap.parse_args()

    from src.model.net import harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool

    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig0 = harmonic_signatures(pool, APPS)
    nz = noise_signature(pool)
    del pool
    data = collect(a.stride)
    bg = baselines(data, nz, 5)

    apps = [a.app] if a.app else APPS
    print("=" * 104)
    print(f"단일기기 창을 **전력으로 층화** — 와트당 전류가 층마다 다른가"
          f"  (배경 뺀 뒤, 층당 >={a.min_bin}창)")
    print("=" * 104)

    for app in apps:
        O, P, S = gather(data, nz, bg, app, a.min_w)
        if O is None or len(P) < a.bins * a.min_bin:
            if not a.app:
                continue
            print(f"\n  {app}: 표본 부족 ({0 if P is None else len(P)}창) — 규칙 28")
            continue
        j = APPS.index(app)
        print(f"\n  ── {app}  ({len(P)}창, {len(set(S))}파일: {', '.join(sorted(set(S)))})")
        # 전력 분포가 무리를 이루는가 — 정렬해서 가장 큰 틈을 본다
        ps = np.sort(P)
        gaps = np.diff(ps)
        k = int(np.argmax(gaps))
        print(f"     전력 {ps.min():.0f}~{ps.max():.0f}W  중앙 {np.median(P):.0f}W"
              f" | 가장 큰 틈 {gaps[k]:.0f}W @ {ps[k]:.0f}W"
              f"  (전체폭의 {gaps[k] / (ps.max() - ps.min()) * 100:.0f}%)")

        edges = np.quantile(P, np.linspace(0, 1, a.bins + 1))
        print(f"     {'전력층':<16s}{'창':>5s}{'평균W':>8s}"
              + "".join(f"{'h' + str(h):>9s}" for h in (1, 2, 3, 4, 5, 7)))
        print("     " + "-" * 82)
        rows = []
        for b in range(a.bins):
            m = (P >= edges[b]) & (P <= edges[b + 1] if b == a.bins - 1
                                   else P < edges[b + 1])
            if m.sum() < a.min_bin:
                continue
            perw = np.linalg.norm(O[m], axis=2) / P[m][:, None]      # (n,15) A/W
            med = np.median(perw, 0) * 1000
            rows.append(med)
            print(f"     {f'{edges[b]:.0f}~{edges[b + 1]:.0f}W':<16s}{int(m.sum()):>5d}"
                  f"{P[m].mean():>8.0f}"
                  + "".join(f"{med[h - 1]:>9.3f}" for h in (1, 2, 3, 4, 5, 7)))
        syn = np.linalg.norm(sig0[j], axis=1) * 1000
        print(f"     {'합성 지문':<16s}{'':>5s}{'':>8s}"
              + "".join(f"{syn[h - 1]:>9.3f}" for h in (1, 2, 3, 4, 5, 7)))
        if len(rows) >= 2:
            r = np.array(rows)
            sp = [(r[:, h - 1].max() - r[:, h - 1].min()) / max(np.median(r[:, h - 1]), 1e-9)
                  for h in (1, 2, 3, 4, 5, 7)]
            print(f"     {'층간 상대폭':<16s}{'':>5s}{'':>8s}"
                  + "".join(f"{x * 100:>8.0f}%" for x in sp))
    print("\n  `층간 상대폭` = (층 최대 − 최소) / 중앙. h1 이 평평하면 모드는 하나다.")
    print("  ⚠ 모든 기기가 같은 기울기를 보이면 그것은 모드가 아니다 (규칙 5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
