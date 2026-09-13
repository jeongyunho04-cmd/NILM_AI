# -*- coding: utf-8 -*-
"""비대칭 가림 행렬 rho_ij — **누가 누구를 덮는가** (13.89 [1], 2026-09-12).

Entropy 28(3) 334 (2026-03) 가 고주파 NILM 에서 **가림은 방향성이고 비대칭**임을
조건부 상호정보로 보였다 (충전기를 조건으로 걸면 램프가 0.09 만 남고 반대는 0.92).

우리는 SMPS 배분을 줄곧 **대칭** 문제로 봐 왔다 — "형제 셋이 서로 안 갈린다",
13.85 [2] 의 사이각도 대칭량이다. **누가 누구를 덮는지는 한 번도 안 쟀다.**

[식을 논문 그대로 안 쓰는 이유 — 우리 자료에서는 교락이 생긴다]
논문은 rho = I(X_i;Y|Z_j)/I(X_i;Y) 를 쓴다. 그런데 `Z_j` 로 조건을 걸면 **i 와 j 가
같이 켜지는 습관**까지 설명돼 버려서, 값이 떨어지는 것이 *신호 가림*인지 *사용 상관*인지
안 갈린다. 실측 5파일은 사람이 켠 순서가 정해져 있어 특히 그렇다. 그래서:

    rho_ij = I(X_i ; Y | Z_j = 1) / I(X_i ; Y | Z_j = 0)

**j 가 켜져 있을 때 대 꺼져 있을 때**, 관측이 i 에 대해 말해 주는 양의 비다. 양쪽 다
Z_j 값이 고정이므로 상관 항이 상쇄된다. 1 이면 j 가 있어도 i 가 그대로 보이고,
0 에 가까우면 j 가 i 를 덮는다.

[자료 — 직접 합성한다. 전체 합성기가 아니다]
기기 on/off 를 **독립 베르누이**로 뽑아 사용 상관을 구조적으로 없앤다. 켜진 기기마다
격리 녹화에서 **사이클 하나를 무작위로** 뽑아 그 페이저를 더한다 — 그 기기의 실제
산포가 그대로 들어온다 (규칙 14 의 격리 통계지만, 여기서는 "모델이 무엇을 써야 한다"
가 아니라 **정보가 있는가**를 묻는 것이라 쓸 수 있다).

⚠ **한계**: 선로 임피던스 결합도 드리프트도 안 넣는다. 가림의 1차 효과는 "j 의 전류가
  합에서 i 를 삼키는가" 이고 그것은 가산으로 잡힌다. 2차 효과는 못 본다.
  전체 합성기로 다시 재면 값이 바뀔 수 있다 — 방향(비대칭 여부)이 결론이다.

[추정량 — 분류기 하한]
X_i 가 이진이므로 `I(X_i;Y) = H(X_i) - H(X_i|Y)` 이고, 분류기의 교차엔트로피가
`H(X_i|Y)` 의 상한이다. 따라서 `I >= H(X_i) - CE` 는 **변분 하한**이다.
양쪽 부분집합에서 `P(X_i=1)=0.5` 로 균형을 맞춰 `H(X_i)=1비트` 로 고정한다 —
안 그러면 기저율 차이가 비에 섞인다.

    python -X utf8 src/run_diag_masking.py
    python -X utf8 src/run_diag_masking.py -n 40000 --pair laptop_charger minipc
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
SHORT = {"electiric_kettle": "포트", "oven": "오븐", "hair_dryer": "드라이기",
         "hotplate": "핫플", "fan": "선풍", "laptop_charger": "충전기",
         "beam_projector": "프로젝터", "minipc": "미니PC", "air_conditioner": "에어컨"}
RESISTIVE = {"electiric_kettle", "oven", "hair_dryer", "hotplate", "fan"}
SMPS = {"laptop_charger", "beam_projector", "minipc"}
NOISE_A = 1.0e-3        # 계측 바닥 잡음 진폭 (A). file_registry 1.4~2.4W / ~215V


def draw_pool(pool, n_draw, rng):
    """기기별로 통전 사이클 `n_draw` 개를 뽑아 (K, n_draw, 15) 복소로 쌓는다."""
    from src.model.net import harmonic_signatures  # noqa: F401  (같은 규약 확인용)
    out = np.zeros((len(APPS), n_draw, 15), np.complex128)
    have = np.zeros(len(APPS), bool)
    for j, a in enumerate(APPS):
        acts = pool.appliance_activations.get(a, [])
        thr = 0.5 * pool.get_steady_power_w(a)
        cs = []
        for x in acts:
            m = x.target_power_w > max(thr, 1.0)
            if m.any():
                cs.append(x.net_harmonics_complex[m])
        if not cs:
            continue
        c = np.concatenate(cs)
        out[j] = c[rng.integers(0, len(c), n_draw)]
        have[j] = True
    return out, have


def features(Y):
    """(N,15) 복소 -> (N,45) 실수.

    ⚠ **asinh 만으로는 모자란다.** 첫 판에서 asinh(Re/1mA) 만 썼더니 핫플(2.1A)이
    포트(6.4A)와 함께 켜진 창에서 정보 0.000 이 나왔다 — asinh 가 큰 부하 대역을
    9.2~9.5 로 눌러 버려 차이가 표준화 뒤 잡음에 묻힌다. 0.2절이 "판별 정보는
    **비(ratio)** 에 있다" 고 적은 그대로다. 그래서 **비를 같이 넣는다**:

        asinh(Re/1mA) · asinh(Im/1mA)   h1..h15   크기와 위상       30
        |I_h| / |I_1|                   h2..h15   **크기 무관**     14
        asinh(|I1|/1mA)                            전체 크기         1
    """
    s = 1e-3
    re = np.arcsinh(Y.real / s)
    im = np.arcsinh(Y.imag / s)
    m = np.abs(Y)
    ratio = m[:, 1:] / np.maximum(m[:, 0:1], 1e-9)
    mag = np.arcsinh(m[:, 0:1] / s)
    return np.concatenate([re, im, ratio, mag], 1).astype(np.float32)


def mi_bits(F, x, rng, folds=3, dev=None):
    """I(x ; F) 의 변분 하한 (비트). x 는 이진이고 **균형이 맞춰져 있어야** 한다.

    분류기는 은닉 64 MLP 둘이다. 선형으로는 안 된다 — 0.3절이 충전기에서
    로지스틱 69.8% 대 MLP 100% 를 쟀다. 하한이므로 **분류기가 약하면 가림을
    과대평가**한다. 그래서 양쪽 부분집합에 **같은 분류기**를 쓴다 — 비에서 상쇄된다.
    """
    import torch
    import torch.nn as nn
    n = len(x)
    if n < 400 or x.sum() < 100 or (1 - x).sum() < 100:
        return float("nan"), n
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g).numpy()
    Ft = torch.from_numpy(np.ascontiguousarray(F[perm])).float().to(dev)
    xt = torch.from_numpy(x[perm].astype(np.float32)).to(dev)
    ce = []
    for f in range(folds):
        te = torch.zeros(n, dtype=torch.bool, device=dev)
        te[f::folds] = True
        tr = ~te
        mu, sd = Ft[tr].mean(0, keepdim=True), Ft[tr].std(0, keepdim=True).clamp(min=1e-6)
        A, B = (Ft[tr] - mu) / sd, (Ft[te] - mu) / sd
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(F.shape[1], 64), nn.GELU(),
                          nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)
        lo = nn.BCEWithLogitsLoss()
        ya = xt[tr]
        for _ in range(400):
            opt.zero_grad(set_to_none=True)
            lo(m(A).squeeze(-1), ya).backward()
            opt.step()
        with torch.no_grad():
            lg = m(B).squeeze(-1)
            # 비트 단위 교차엔트로피
            ce.append(float(nn.functional.binary_cross_entropy_with_logits(
                lg, xt[te]).item()) / np.log(2.0))
    return float(max(0.0, 1.0 - float(np.mean(ce)))), n


def balanced(idx_on, idx_off, rng, cap):
    """켜짐/꺼짐을 같은 수로 맞춰 섞는다 -> (인덱스, 라벨)."""
    k = min(len(idx_on), len(idx_off), cap)
    if k < 200:
        return None, None
    a = rng.choice(idx_on, k, replace=False)
    b = rng.choice(idx_off, k, replace=False)
    return np.concatenate([a, b]), np.concatenate([np.ones(k, int), np.zeros(k, int)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=30000, help="합성 창 수")
    ap.add_argument("-p", type=float, default=0.4, help="기기별 켜질 확률 (독립)")
    ap.add_argument("--cap", type=int, default=2500, help="부분집합당 클래스별 표본 상한")
    ap.add_argument("--pair", nargs=2, default=None, metavar=("I", "J"),
                    help="한 쌍만 자세히 본다")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    print("비대칭 가림 행렬 rho_ij = I(X_i;Y|Z_j=1) / I(X_i;Y|Z_j=0)", flush=True)
    print("=" * 78)
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool()
    draws, have = draw_pool(pool, a.n, rng)
    live = [j for j in range(len(APPS)) if have[j]]
    print("격리 녹화에서 뽑음 · 창 %d · 기기별 켜질 확률 %.2f (독립) · 살아 있는 기기 %d"
          % (a.n, a.p, len(live)))

    # ── 쌍마다 **그 둘만** 켜서 잰다 ────────────────────────────────────
    # ⚠ 첫 판은 9종을 p=0.4 로 한꺼번에 켰다. 그러면 평균 3.6대가 켜져 있어
    #   **전부가 전부를 덮고**, rho 의 분모(j 가 꺼진 조건)에도 나머지 7종의 가림이
    #   그대로 남는다 — 쌍의 가림을 잰 것이 아니라 주변 소음을 잰 것이다.
    #   여기서는 **i 와 j 만** 독립 베르누이 0.5 로 켠다. 그러면
    #       분모 = i 혼자 (j 꺼짐) · 분자 = i + j
    #   가 되어 "j 하나가 i 를 얼마나 덮는가" 가 그대로 나온다.
    def compose(cols, flags):
        Y = np.zeros((a.n, 15), np.complex128)
        for c in cols:
            Y += flags[:, cols.index(c), None] * draws[c]
        return Y + (rng.normal(0, NOISE_A, (a.n, 15))
                    + 1j * rng.normal(0, NOISE_A, (a.n, 15)))

    print("쌍마다 **그 둘만** 켠다 (독립 0.5). 분모 = i 혼자 · 분자 = i + j")
    print("⚠ 선로 결합·드리프트는 안 넣었다 — 가림의 1차 효과만 본다 (독스트링)")
    print()

    pairs = None
    if a.pair:
        pairs = [(APPS.index(a.pair[0]), APPS.index(a.pair[1]))]

    # 기준 — 혼자 켜졌을 때 (아무도 안 덮을 때) i 가 얼마나 보이나
    base = {}
    for i in live:
        fl = (rng.random((a.n, 1)) < 0.5)
        Y = compose([i], fl)
        idx, lab = balanced(np.flatnonzero(fl[:, 0]), np.flatnonzero(~fl[:, 0]), rng, a.cap)
        if idx is None:
            continue
        base[i], _ = mi_bits(features(Y)[idx], lab, rng)
    print("기준 — **혼자 켜졌을 때** I(X_i;Y) (비트, 최대 1). 1 이면 완전히 보인다")
    for i in live:
        if i in base:
            print("   %-9s %.3f" % (SHORT[APPS[i]], base[i]))
    print()

    R = np.full((len(APPS), len(APPS)), np.nan)
    todo = pairs if pairs else [(i, j) for i in live for j in live if i != j]
    for i, j in todo:
        fl = (rng.random((a.n, 2)) < 0.5)
        Y = compose([i, j], fl)
        F = features(Y)
        vals = {}
        for zv in (1, 0):
            sel = fl[:, 1] if zv else ~fl[:, 1]
            idx, lab = balanced(np.flatnonzero(sel & fl[:, 0]),
                                np.flatnonzero(sel & ~fl[:, 0]), rng, a.cap)
            if idx is None:
                vals[zv] = float("nan")
                continue
            vals[zv], _ = mi_bits(F[idx], lab, rng)
        if np.isfinite(vals.get(0, np.nan)) and vals[0] > 0.05:
            R[i, j] = vals[1] / vals[0]
        if pairs:
            print("  I(%s;Y | %s 켜짐) = %.3f   ·   %s 꺼짐 = %.3f   ->  rho = %.3f"
                  % (SHORT[APPS[i]], SHORT[APPS[j]], vals[1],
                     SHORT[APPS[j]], vals[0], R[i, j]))
    if pairs:
        return

    # ── 표 ──────────────────────────────────────────────────────────────
    print("rho_ij — **세로 i 가 가로 j 때문에 얼마나 남는가** (1 = 안 덮인다)")
    hdr = "  %-9s" % "i \\ j" + "".join("%8s" % SHORT[APPS[j]] for j in live)
    print(hdr)
    for i in live:
        row = "  %-9s" % SHORT[APPS[i]]
        for j in live:
            row += "       ·" if i == j else ("%8.2f" % R[i, j] if np.isfinite(R[i, j]) else "       -")
        print(row)

    print()
    print("가장 비대칭인 쌍 (|rho_ij - rho_ji| 큰 순, 위 10)")
    rows = []
    for i in live:
        for j in live:
            if i < j and np.isfinite(R[i, j]) and np.isfinite(R[j, i]):
                rows.append((abs(R[i, j] - R[j, i]), i, j))
    rows.sort(reverse=True)
    for d, i, j in rows[:10]:
        lo, hi = (i, j) if R[i, j] < R[j, i] else (j, i)
        print("   %-9s 가 %-9s 을 덮는다   rho(%s|%s)=%.2f  대  반대 %.2f   차 %.2f"
              % (SHORT[APPS[hi]], SHORT[APPS[lo]], SHORT[APPS[lo]][:4],
                 SHORT[APPS[hi]][:4], R[lo, hi], R[hi, lo], d))

    print()
    print("무리별 평균 rho (덮이는 쪽 i / 덮는 쪽 j)")
    grp = {"저항성": RESISTIVE, "SMPS": SMPS}
    print("  %-12s %10s %10s" % ("", "j=저항성", "j=SMPS"))
    for gi, si in grp.items():
        row = "  i=%-10s" % gi
        for gj, sj in grp.items():
            v = [R[i, j] for i in live for j in live
                 if APPS[i] in si and APPS[j] in sj and i != j and np.isfinite(R[i, j])]
            row += "%10s" % ("%.2f" % np.mean(v) if v else "-")
        print(row)


if __name__ == "__main__":
    main()
