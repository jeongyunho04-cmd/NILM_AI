# -*- coding: utf-8 -*-
"""텍스처 토막 경계를 **창마다 다른 자리**에 두는지 (14.102).

왜 필요했나
-----------
사용자: *"seg_s 가 일어나는 지점을 랜덤화 못하나? 혹시 전압 텍스처 변화 위치를 외울 수도
있잖아."* 맞다. 고치기 전 경계는

    [(i * n // k, (i+1) * n // k, seq[i]) for i in range(k)]

라 **창마다 정확히 같은 자리**였다 — `n=3600, k=20` 이면 늘 사이클 180·360·540… 이다.
세밀 창은 마지막 600사이클이라 그 안에도 **늘 같은 오프셋**(60·240·420)으로 들어간다.
실측에는 그런 규칙이 없으니 **합성에만 있는 단서**이고, 모델이 그것을 외우면
`seg_s` 가 사려던 것과 정반대로 **새 가짜 판별자**를 만든다.
(13.84.8 의 `sibling_rotate` 가 없앤 것과 같은 부류다 — 원천에서 지운다.)

무엇을 확인하나
---------------
```
  [1] `seg_s` 를 안 쓰면(토막 1장) **옛 경로와 완전히 같다** — 경계도 난수 흐름도
  [2] 토막을 쓰면 경계가 **창마다 흩어진다** (같은 자리에 몰리면 실패)
  [3] 토막 길이의 분포가 온전하다 — 빈틈·겹침 없이 창을 정확히 덮는다
  [4] **세밀 창 안**(마지막 600사이클)에 들어오는 경계 위치가 흩어진다  <- 진짜 겨냥
```

    python -X utf8 src/run_gate_texoffset.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

from src.synthesis.grid_simulator import texture_segments  # noqa: E402

N = 3600          # 창 = 60초 @60Hz
FINE0 = N - 600   # 세밀 창은 마지막 600사이클


class _Tex:
    def __init__(self, i):
        self.id = i
        self.stem = "s%d" % i


class _Env:
    def __init__(self, k, off, edges=()):
        self.texture_seq = tuple(_Tex(i) for i in range(k))
        self.texture_offset = int(off)
        self.texture_edges = tuple(edges)
        self.texture = self.texture_seq[0] if k else None


def main() -> int:
    print("텍스처 토막 경계 무작위화 관문 (14.102)")
    print("  창 %d사이클(60초) · 세밀 창 = [%d, %d)" % (N, FINE0, N))
    print("")
    ok = True

    seg1 = texture_segments(_Env(1, 0), N)
    good = seg1 == [(0, N, seg1[0][2])]
    ok &= good
    print("  [1] 토막 1장 -> 창 전체 한 덩이 %s" % ("OK" if good else "** 다르다 **"))
    old = texture_segments(_Env(20, 0), N)
    want = [(i * N // 20, (i + 1) * N // 20) for i in range(20)]
    good = [(a, b) for a, b, _ in old] == want
    ok &= good
    print("      오프셋 0 이면 **옛 경계 그대로** (i·n//k)   %s" % ("OK" if good else "** 다르다 **"))

    rng = np.random.default_rng(0)
    K = 21                     # 14.102 는 토막을 쓸 때 한 장 더 뽑는다 (nseg+1)
    L = N // (K - 1)
    firsts, fine_edges, lens = [], [], []
    for _ in range(400):
        off = int(rng.integers(0, L))
        seg = texture_segments(_Env(K, off), N)
        e = [a for a, _b, _t in seg] + [N]
        cov = all(e[i + 1] > e[i] for i in range(len(e) - 1)) and e[0] == 0 and e[-1] == N
        if not cov:
            ok = False
            print("      ** 빈틈/겹침 (off=%d) **" % off)
            break
        firsts.append(seg[1][0] if len(seg) > 1 else 0)
        lens.extend([b - a for a, b, _ in seg])
        fine_edges.extend([a for a, _b, _t in seg if FINE0 < a < N])

    fa = np.asarray(firsts)
    spread = len(np.unique(fa))
    good = spread >= 100
    ok &= good
    print("  [2] 첫 경계가 흩어지나 — 서로 다른 값 **%d개** / 400창 (토막 길이 %d)   %s"
          % (spread, L, "OK" if good else "** 몰려 있다 **"))
    print("      범위 %d~%d · 중앙 %d" % (fa.min(), fa.max(), int(np.median(fa))))

    la = np.asarray(lens)
    good = la.min() >= 1 and la.sum() == 400 * N
    ok &= good
    print("  [3] 창을 정확히 덮는다 — 토막 길이 합 %d (기대 %d) · 최소 %d   %s"
          % (la.sum(), 400 * N, la.min(), "OK" if good else "** 안 맞다 **"))

    fe = np.asarray(fine_edges)
    uniq = len(np.unique(fe))
    good = uniq >= 100
    ok &= good
    print("  [4] ★ **세밀 창 안**에 들어온 경계 %d개 · 서로 다른 자리 **%d개**   %s"
          % (len(fe), uniq, "OK" if good else "** 같은 자리에 몰린다 **"))
    if len(fe):
        h, _ = np.histogram(fe - FINE0, bins=10, range=(0, 600))
        print("      세밀 창 안 위치 분포(60사이클 칸): %s" % " ".join("%d" % x for x in h))
        print("      -> 고치기 전에는 **60·240·420 세 자리에만** 몰렸다")

    # ── 14.103 비균일: 세밀 창은 촘촘히, 그 앞은 성기게 ───────────────────
    print("")
    print("  [5] ★ **비균일** (14.103) — 세밀 1.25초 + 바깥 25초")
    from src.synthesis.grid_simulator import GridSimulator
    g = GridSimulator(vtex_seg_s=1.25, vtex_coarse_s=25.0)
    r2 = np.random.default_rng(1)
    ns, fine_in, edge_in_fine, lens2 = [], [], [], []
    for _ in range(300):
        k, ed = g._texture_plan(N, r2)
        seg = texture_segments(_Env(k, 0, ed), N)
        e = [a for a, _b, _t in seg] + [N]
        if not (e[0] == 0 and e[-1] == N and all(e[i + 1] > e[i] for i in range(len(e) - 1))):
            ok = False
            print("      ** 빈틈/겹침 **")
            break
        ns.append(len(seg))
        fine_in.append(sum(1 for a, b, _ in seg if b > FINE0))
        edge_in_fine.extend([a for a, _b, _t in seg if FINE0 < a < N])
        lens2.extend([b - a for a, b, _ in seg])
    if ns:
        na, fa2 = np.asarray(ns), np.asarray(fine_in)
        good = na.mean() < 14 and fa2.mean() >= 7
        ok &= good
        print("      총 토막 중앙 **%d** (균일 3초면 20) · **세밀 창 안 토막 중앙 %d** (균일이면 3)   %s"
              % (int(np.median(na)), int(np.median(fa2)), "OK" if good else "** 기대와 다르다 **"))
        print("      비용비 20/%.1f = **%.2f배** · 세밀 해상도 %.2f초 (균일 3.33초)"
              % (na.mean(), 20.0 / na.mean(), 600.0 / 60.0 / max(fa2.mean(), 1)))
        fe2 = np.asarray(edge_in_fine)
        uq = len(np.unique(fe2))
        good2 = uq >= 100
        ok &= good2
        print("      세밀 창 안 경계 %d개 · 서로 다른 자리 **%d개**   %s"
              % (len(fe2), uq, "OK" if good2 else "** 몰려 있다 **"))
        s2 = np.asarray(lens2).sum()
        good3 = s2 == 300 * N
        ok &= good3
        print("      창을 정확히 덮는다 — 길이 합 %d (기대 %d)   %s"
              % (s2, 300 * N, "OK" if good3 else "** 안 맞다 **"))
        g0 = GridSimulator(vtex_seg_s=3.0, vtex_coarse_s=0.0)
        k0, ed0 = g0._texture_plan(N, np.random.default_rng(1))
        good4 = (k0 == 20 and ed0 == ())
        ok &= good4
        print("  [6] `--vtex-coarse-s 0` 이면 **옛 균일 경로** (토막 %d · 비율경계 없음)   %s"
              % (k0, "OK" if good4 else "** 다르다 **"))

    print("")
    print("관문 전부 통과" if ok else "** 관문 실패 **")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
