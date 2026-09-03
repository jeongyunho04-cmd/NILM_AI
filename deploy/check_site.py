"""현장 점검 — **이 체크포인트를 여기서 써도 되나** (2026-09-02)

왜 필요한가
----------
`models/adapt_zi_s0.pt` 은 학습 때 **계통 임피던스 보정**을 손실에 넣고 만들었다
(학습 저장소 12.148). 그 보정의 계수는 **학습 장소의 배선값**을 담고 있다.

추론에는 `Z` 가 안 들어간다 — 보정은 학습 시 손실 항이었고 이 체크포인트는
평범한 CNN 이다. 그래서 새 현장에서 **인자를 줄 것이 없고**, 대신
**그 현장이 학습 장소와 비슷한지 확인**해야 한다.

그리고 다르다. 실측으로 확인했다:

```
                     R (Ω)   L (µH)   |Z₁|(계단법)   V_src,1   V_src,3
학습 장소             1.63     455        2.04       222.7      4.6
다른 장소 (test.csv)  0.51     144        (계단 부족)  234.7     10.1
                     3.2배    3.2배                   +12V     2.2배
```

**임피던스가 3.2배, 배경 3차 전압이 2.2배 다르다.** 이 정도면 학습 때 모델이
본 계통과 다른 계통이고, 배분 성능이 그대로 나온다는 보장이 없다.

무엇을 재나
----------
**① |Z₁| — 계단법.** 큰 부하가 켜지고 꺼질 때 전압이 얼마나 끌려 내려가나.

    Δ|V₁|  ≈  −Re( conj(u)·Z·ΔI₁ )
    -> Δ|V₁| 를 [ReΔI₁, ImΔI₁] 로 회귀하면 |Z₁| 이 나온다

전역 위상 `u` 는 미지지만 크기는 나오고, 크기만 있으면 현장 비교에는 충분하다.
**numpy 최소제곱 하나로 끝난다** — 이 폴더는 numpy 와 torch 만 쓴다.

**② V_src — 배경 전압.** 부하가 가장 작은 창의 `|V_h|` 다. 그때는 `Z·I` 가 작아
`V ≈ V_src` 이기 때문이다.

⚠ **계단이 있어야 한다.** 부하가 크다고 되는 게 아니라 **자주 바뀌어야** 한다.
   학습 저장소의 `test.csv` 는 전류가 최대 4.8A 인데 `ΔI>0.5A` 계단이 16개뿐이라
   이 방법을 못 썼다. 필요량은 실측했다:

```
계단 수     |Z₁| 중앙   p5~p95        판정
     5      2.930    1.69 ~ 7.39   못 쓴다
    20      1.989    1.49 ~ 5.63
   100      2.351    1.53 ~ 3.54   ±40%
   300      2.146    1.52 ~ 2.85   ±30%
 1,699      2.037    (기준)
```

**수백 개는 있어야 한다.** 전기포트나 드라이기를 100~300번 켰다 껐다 하는
30분 녹화면 충분하다.

쓰는 법
------
```bash
# ① 현장에서 수신기를 돌려 CSV 를 쌓는다 (30분, 큰 부하를 자주 껐다 켜면서)
python -m nilm_runtime.receiver

# ② 점검
python check_site.py --csv data/live.csv
```

판정이 `주의` 나 `다름` 이면 학습 저장소에서 **그 현장 자료로 다시 학습**해야 한다
(`run_fit_impedance` -> `run_norton_probe --save-coef` -> `run_adapt --harm-offset`).

⚠ 이 도구는 **|Z₁| 만** 낸다. 차수별 `Z_h = R + j·h·ωL` 의 두 상수를 다 뽑으려면
   학습 저장소의 `src/run_fit_impedance.py` 를 쓸 것 (scipy 가 필요하다).
"""
from pathlib import Path
import argparse
import csv
import sys

import numpy as np

#: 학습 장소의 값 (학습 저장소 12.148.2 / 12.149). 비교 기준이다.
REF = {"z1_step": 2.04, "z1_fit": 1.64, "R": 1.63, "L_uh": 455,
       "vsrc1": 222.7, "vsrc3": 4.60}
#: `vh*` 는 30사이클(0.5초)마다 갱신된다. `ih*` 는 매 사이클이다.
BLOCK = 30
#: `Z` 추정 자체의 폭이 ±25% 다 (자 둘의 차이, 그리고 적합 파일 집합을 바꾸면 20%).
#: 그보다 작은 차이는 같다고 봐야 한다.
SAME, DIFFERENT = 1.5, 2.0


def read_csv(path: Path, cols) -> dict:
    """수신기 CSV 에서 필요한 열만. 표준 라이브러리 + numpy 뿐이다."""
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        head = next(r)
        idx = {}
        for c in cols:
            if c not in head:
                raise SystemExit(f"CSV 에 '{c}' 열이 없습니다. 수신기 CSV 가 맞습니까?")
            idx[c] = head.index(c)
        out = {c: [] for c in cols}
        for row in r:
            if len(row) <= max(idx.values()):
                continue
            try:
                for c in cols:
                    out[c].append(float(row[idx[c]]))
            except ValueError:
                continue
    d = {c: np.asarray(v, float) for c, v in out.items()}
    return _canonical(d)


def _canonical(d: dict) -> dict:
    """**정본 순서로 되돌린다.** 수신기는 Wi-Fi 재전송 때문에 패킷을 순서가
    뒤바뀐 채 기록한다 — 학습 자료 44파일에서 3,226곳이었다 (최대 7.5초).
    계단법은 연속한 두 창의 차를 보므로 정렬 안 하면 없는 계단을 만든다.

    `seq`/`cycle` 이 없으면(옛 펌웨어) 그대로 둔다.
    """
    if "seq" not in d or "cycle" not in d:
        return d
    key = d["seq"] * BLOCK + d["cycle"]
    # 보드 리셋으로 seq 가 되감기면 뒤 세션만 남긴다 (겹치는 seq 가 섞이면 못 쓴다).
    reset = np.flatnonzero(np.diff(d["seq"]) < -32)
    lo = int(reset[-1]) + 1 if len(reset) else 0
    if lo:
        print(f"  ⚠ 녹화가 {len(reset)+1}개 이어져 있습니다. 마지막 것만 씁니다 "
              f"({len(d['seq']) - lo:,}/{len(d['seq']):,}행).")
        d = {c: v[lo:] for c, v in d.items()}
        key = key[lo:]
    o = np.argsort(key, kind="stable")
    _, first = np.unique(key[o], return_index=True)      # 중복 패킷 제거
    o = o[first]
    return {c: v[o] for c, v in d.items()}


def blocks(x: np.ndarray) -> np.ndarray:
    n = len(x) // BLOCK * BLOCK
    return x[:n].reshape(-1, BLOCK).mean(1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/live.csv", help="수신기가 쓴 CSV")
    ap.add_argument("--min-di", type=float, default=0.5, help="계단으로 셀 최소 ΔI (A)")
    ap.add_argument("--quiet-frac", type=float, default=0.05,
                    help="V_src 를 읽을 저부하 창의 비율 (하위 5%%)")
    a = ap.parse_args()

    p = Path(a.csv)
    if not p.exists():
        raise SystemExit(f"파일이 없습니다: {p}")
    hs = [1, 3, 5, 7, 9]
    # `seq`/`cycle` 은 정본 순서 복원에 쓴다 (`_canonical`). 없어도 돈다.
    d = read_csv(p, ["seq", "cycle", "irms", "vrms"] + [f"ih{h}" for h in hs]
                 + [f"ihdeg{h}" for h in hs] + [f"vh{h}" for h in hs])

    I1 = blocks(d["ih1"] * np.cos(np.deg2rad(d["ihdeg1"]))) \
        + 1j * blocks(d["ih1"] * np.sin(np.deg2rad(d["ihdeg1"])))
    V1 = blocks(d["vh1"])
    if len(I1) < 60:
        raise SystemExit(f"블록이 {len(I1)}개뿐입니다 (30초 미만). 더 길게 녹화하세요.")

    print("=" * 78)
    print(f"현장 점검 — {p}")
    print("=" * 78)
    print(f"\n  녹화 {len(I1) * BLOCK / 60 / 60:.1f}분  |  0.5초 블록 {len(I1)}개")

    # ── 진단: 잴 수 있는 자료인가 ────────────────────────────────────────
    dI, dV = np.diff(I1), np.diff(V1)
    m = np.abs(dI) > a.min_di
    print(f"\n  [진단]")
    print(f"    |I₁| 중앙 {np.median(np.abs(I1)):.2f} A"
          f"   폭 {np.abs(I1).max() - np.abs(I1).min():.2f} A")
    print(f"    |V₁| 중앙 {np.median(V1):.1f} V   폭 {V1.max() - V1.min():.2f} V")
    print(f"    ΔI>{a.min_di:g}A 계단 **{int(m.sum())}개**")
    if m.sum() < 100:
        print(f"\n    ⚠ 계단이 부족합니다. 수백 개는 있어야 ±30% 안에 듭니다.")
        print(f"      전기포트·드라이기 같은 큰 부하를 100~300번 껐다 켜면서")
        print(f"      30분쯤 다시 녹화하세요. **부하가 크기만 해서는 안 됩니다.**")

    # ── ① |Z₁| ──────────────────────────────────────────────────────────
    print(f"\n  [① 계통 임피던스]")
    if m.sum() < 20:
        print(f"    계단 {int(m.sum())}개 — **못 잽니다.**")
        z1 = float("nan")
    else:
        X = np.c_[dI[m].real, dI[m].imag]
        b, *_ = np.linalg.lstsq(X, dV[m], rcond=None)
        z1 = float(abs(b[0] + 1j * b[1]))
        ss = ((dV[m] - X @ b) ** 2).sum()
        st = ((dV[m] - dV[m].mean()) ** 2).sum()
        r2 = 1 - ss / max(st, 1e-12)
        ratio = z1 / REF["z1_step"]
        print(f"    |Z₁| = **{z1:.3f} Ω**   R² {r2:.3f}   (계단 {int(m.sum())}개)")
        print(f"    학습 장소 {REF['z1_step']:.2f} Ω  ->  **{ratio:.2f}배**")
        if r2 < 0.5:
            print(f"    ⚠ R² 가 {r2:.2f} 로 낮습니다. 이 값을 믿지 마세요.")

    # ── ② 배경 전압 ─────────────────────────────────────────────────────
    lo = np.abs(I1) <= np.quantile(np.abs(I1), a.quiet_frac)
    print(f"\n  [② 배경 전압]  저부하 하위 {a.quiet_frac * 100:.0f}% 창 {int(lo.sum())}개")
    vs = {}
    for h in hs:
        vs[h] = float(np.median(blocks(d[f"vh{h}"])[lo]))
    print(f"    " + "   ".join(f"V_src,{h} {vs[h]:6.2f}V" for h in hs))
    print(f"    학습 장소       V_src,1 {REF['vsrc1']:6.2f}V   V_src,3 {REF['vsrc3']:6.2f}V")
    r3 = vs[3] / max(REF["vsrc3"], 1e-9)
    print(f"    3차 배경 비 **{r3:.2f}배**")

    # ── 판정 ────────────────────────────────────────────────────────────
    print(f"\n  [판정]")
    worst, why = 1.0, []
    if not np.isnan(z1):
        rr = max(z1 / REF["z1_step"], REF["z1_step"] / z1)
        worst = max(worst, rr); why.append(f"|Z₁| {rr:.2f}배")
    rr3 = max(r3, 1 / max(r3, 1e-9))
    worst = max(worst, rr3); why.append(f"3차 배경 {rr3:.2f}배")
    if np.isnan(z1):
        # 임피던스를 못 재도 배경 전압은 잰다 — 그것만으로도 판정이 설 수 있다
        if rr3 >= DIFFERENT:
            print(f"    **다르다** (3차 배경 {rr3:.2f}배) — 임피던스는 못 쟀지만")
            print("    배경 전압만으로도 다른 계통입니다. 아래 재적응을 권합니다.")
        elif rr3 >= SAME:
            print(f"    **주의** (3차 배경 {rr3:.2f}배) — 임피던스는 못 쟀습니다.")
            print("    계단이 있는 녹화를 받아 다시 점검하세요.")
        else:
            print(f"    **보류** — 배경 전압은 비슷한데({rr3:.2f}배) 임피던스를 못 쟀습니다.")
            print("    계단이 있는 녹화를 다시 받으세요.")
    elif worst < SAME:
        print(f"    **같다** ({', '.join(why)}) — 측정 폭(±25%) 안입니다.")
        print("    `models/adapt_zi_s0.pt` 을 그대로 쓰면 됩니다.")
    elif worst < DIFFERENT:
        print(f"    **주의** ({', '.join(why)}) — 측정 폭보다는 크고 뚜렷하지는 않습니다.")
        print("    써 보되 배분(특히 SMPS 3종)을 실제 값과 대조해 보세요.")
    else:
        print(f"    **다르다** ({', '.join(why)}) — 학습 계통과 다른 계통입니다.")
        print("    배분 성능이 그대로 나온다는 보장이 없습니다.")
    if np.isnan(z1) and rr3 < SAME:
        pass
    elif worst >= SAME or (np.isnan(z1) and rr3 >= SAME):
        # **재학습이 아니라 재적응이다.** 2단계는 원래 라벨 없이 도는 준지도
        # 적응이고(`--w-real-on` 기본 0), 계수의 기울기는 물리적으로 `ΣY`
        # (기기들의 어드미턴스 합)라 기기 구성이 같으면 그대로 쓴다.
        print("\n    학습 저장소에서 **2단계 적응만** 다시 돌리면 됩니다 — 라벨 불필요, 5분:")
        print("      # ① 현장 임피던스 (11초, 라벨 0)")
        print("      python -X utf8 -m src.run_fit_impedance --stems <현장파일> \\")
        print("          --out results/z_site.npz")
        print("      # ② 계수는 그대로, 임피던스만 현장 값으로 (3분, 라벨 0)")
        print("      python -X utf8 -m src.run_adapt --init results/cnn_ovh.pt \\")
        print("          --cache cache/train60_ovh_30k --steps 1000 --seed 0 --harm-weight inv_h2 \\")
        print("          --harm-offset results/norton_coef.npz \\")
        print("          --harm-offset-z results/z_site.npz --tag adapt_zi_site --out results")
        print("      cp results/adapt_zi_site.pt deploy/models/")
        print("\n    ⚠ 계수 기울기가 정말 전이되는지는 **확인 못 했습니다** (다른 집 라벨이 없어서).")
        print("    ⚠ 상수항은 학습 장소 것이 남습니다 (배경 V_src 효과, 보정의 47%).")
        print("    ⚠ 기기 구성이 다르면 계수도 다시 적합해야 하고 **그때는 라벨이 필요합니다**")
        print("      (`run_norton_probe --save-coef`).")
    print("\n  ⚠ 이 판정은 **계통 조건만** 봅니다. 기기 구성이 다르면 그것대로 별개 문제입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
