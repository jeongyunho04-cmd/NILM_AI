"""
차분 고조파 특징 검증 — 사용자 제안 3종 (설계 문서 12.53절)
==============================================================
사용자가 원데이터를 직접 분석해 세 가지를 제안했다.

    ① |ΔI2| (또는 |ΔI2|/|ΔI1|)     충전기 vs 프로젝터
    ② |ΔI13|/|ΔI3|                 프로젝터 vs 미니PC
    ③ |ΔI3|/ΔP  (mA/W)            핫플 vs 오븐·포트

그리고 **절대값이 아니라 차분을 써야 한다**고 했다 — 건물마다 배경 고조파가
달라 절대값을 외우면 도메인이 바뀔 때 무너진다는 것이다.

절대값 형태는 이미 있다: `ch35 = |I2|/|I1|`, `ch47 = |I9|/|I3|`.
**새로운 것은 차분이고, 13차까지 올린 것이다.**

    python -m src.run_delta_feature_probe

[두 단계로 잰다 — 12.36 의 교훈]
12.36 이 격리 녹화에서 d' 를 1.97 -> 3.79 로 벌리는 특징 셋을 찾았는데
**복합에서는 겨냥한 쌍이 전혀 안 움직였다.** 그래서 둘 다 본다.

    A. 격리 녹화   물리 주장이 맞는가.  그리고 **기기내 변동 대비** 갈리는가
                   (12.35.1 의 분리비 — 기기간 차이 / 기기내 변동. 2 미만이면 겹침)
    B. 복합 실측   test_5/6/7 의 라벨된 전이에서 실제로 스위칭한 기기를 짚어내는가

B 가 본 시험이다. A 만 좋고 B 가 안 되면 12.36 과 같은 결말이다.

[⚠ 차분은 복소로 잰다]
합계는 복소 고조파에서 선형이다 (12.35, 인수인계 2.3 이 ±10% 안에서 확인했다).
`|median(after)| − |median(before)|` 가 아니라 **`|median_complex(after) −
median_complex(before)|`** 를 써야 스위칭한 기기의 벡터가 나온다.
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np

from src.evaluation.real_events import load_events
from src.model.inputs import CURRENT_SCALE, TRANSIENT_BLOCK, TRANSIENT_LOOKBACK
from src.model.realdata import DEFAULT_DIR
from src.run_live import KOR

DEV_DIR = Path("processed_data/npz")
SMPS = ("beam_projector", "laptop_charger", "minipc")
DEV_FILES = {
    "beam_projector": ["beam_projector", "beam_projector_2"],
    "laptop_charger": ["laptop_charger_1", "laptop_charger_2"],
    "minipc": ["minipc_1", "minipc_2", "minipc_3"],
    "oven": ["oven", "oven_2"],
    "hotplate": ["hotplate_1", "hotplate_2"],
    "electiric_kettle": ["electiric_kettle"],
}


def load_cplx(path: Path):
    raw = np.load(path)
    hr = np.asarray(raw["harmonics_ri"], np.float32)      # (N,15,2)
    return hr[:, :, 0] + 1j * hr[:, :, 1], np.asarray(raw["power_features"], np.float32)


def isolated(min_win: int = 300) -> Dict[str, List[dict]]:
    """격리 녹화: ON 구간마다 배경(OFF)을 복소로 뺀 순수 기기 벡터."""
    out: Dict[str, List[dict]] = {}
    for app, files in DEV_FILES.items():
        for stem in files:
            f = DEV_DIR / f"{stem}.npz"
            if not f.exists():
                continue
            I, pf = load_cplx(f)
            p = pf[:, 0]
            hi = np.percentile(p, 90)
            on = p > max(0.5 * hi, p.min() + 0.3 * (hi - p.min()))
            off = p < np.percentile(p[~on], 60) if (~on).sum() > 100 else ~on
            if on.sum() < min_win or off.sum() < 100:
                continue
            bg = np.median(I[off].real, 0) + 1j * np.median(I[off].imag, 0)
            bgp = float(np.median(p[off]))
            idx = np.flatnonzero(on)
            for blk in np.array_split(idx, max(1, len(idx) // 600)):
                if len(blk) < 200:
                    continue
                v = (np.median(I[blk].real, 0) + 1j * np.median(I[blk].imag, 0)) - bg
                dp = float(np.median(p[blk])) - bgp
                m = np.abs(v)
                if m[0] < 1e-4 or dp <= 0:
                    continue
                out.setdefault(app, []).append({
                    "file": stem, "dP": dp,
                    "I2_I1": float(m[1] / m[0]),
                    "I13_I3": float(m[12] / max(m[2], 1e-6)),
                    "I9_I3": float(m[8] / max(m[2], 1e-6)),
                    "I3_per_W": float(m[2] * 1000.0 / dp)})
    return out


def sepratio(a: np.ndarray, b: np.ndarray) -> float:
    """12.35.1 의 분리비: |평균차| / (기기내 변동 평균). 2 미만이면 겹친다."""
    w = 0.5 * (a.std() + b.std())
    return float(abs(a.mean() - b.mean()) / max(w, 1e-9))


def transitions(ev, stem, apps):
    out = []
    info = ev[stem]
    for app in apps:
        if app not in info.get("appliances_present", []):
            continue
        for s, e in info["intervals"].get(app, {}).get("on", []):
            out.append((float(s), app, +1)); out.append((float(e), app, -1))
    return sorted(out)


def composite(half=5.0, guard=1.0, snap=6.0) -> List[dict]:
    """복합 실측의 라벨된 전이마다 복소 차분 특징."""
    ev = load_events()
    rows = []
    for stem in ("test_5", "test_6", "test_7"):
        f = Path(DEFAULT_DIR) / f"{stem}.npz"
        if not f.exists():
            continue
        I, pf = load_cplx(f)
        p = pf[:, 0]
        n = len(p)
        t = np.arange(n) / 60.0
        i3 = np.abs(I[:, 2])
        for t0, app, sign in transitions(ev, stem, SMPS):
            if t0 < half + snap + 1 or t0 > t[-1] - half - snap - 1:
                continue
            # |I3| 계단으로 스냅 (12.40 과 같은 이유)
            best, bd = t0, -1.0
            for c in np.arange(t0 - snap, t0 + snap, 0.25):
                pre = (t >= c - half) & (t <= c - guard)
                post = (t >= c + guard) & (t <= c + half)
                if pre.sum() < 30 or post.sum() < 30:
                    continue
                d = abs(np.median(i3[post]) - np.median(i3[pre]))
                if d > bd:
                    best, bd = c, d
            pre = (t >= best - half) & (t <= best - guard)
            post = (t >= best + guard) & (t <= best + half)
            if pre.sum() < 30 or post.sum() < 30:
                continue
            dv = ((np.median(I[post].real, 0) + 1j * np.median(I[post].imag, 0))
                  - (np.median(I[pre].real, 0) + 1j * np.median(I[pre].imag, 0)))
            m = np.abs(dv)
            dp = float(np.median(p[post]) - np.median(p[pre]))
            if m[0] < 1e-3 or abs(dp) < 5:
                continue
            rows.append({"stem": stem, "t_s": t0, "app": app, "sign": sign,
                         "dP": dp, "dI1": float(m[0]),
                         "dI2": float(m[1]), "dI2_dI1": float(m[1] / m[0]),
                         "dI13_dI3": float(m[12] / max(m[2], 1e-6)),
                         "dI9_dI3": float(m[8] / max(m[2], 1e-6)),
                         "dI3_per_W": float(m[2] * 1000.0 / max(abs(dp), 1e-6))})
    return rows


def _trail_1d(a, lookback, block):
    nb = lookback // block
    T = len(a)
    pad = (block - T % block) % block
    b = np.concatenate([a, np.repeat(a[-1:], pad)]).reshape(-1, block).max(1)
    out = np.empty_like(b)
    for i in range(len(b)):
        out[i] = np.median(b[max(0, i - nb + 1):i + 1])
    return np.repeat(out, block)[:T]


def _shift(a, n):
    return np.concatenate([np.repeat(a[:1], n), a[:-n]]) if n > 0 else a


def _auc(pos, neg):
    a = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(a)
    rk = np.empty(len(a))
    rk[o] = np.arange(1, len(a) + 1)
    return float((rk[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _dev_signature():
    """격리 녹화의 기기별 **W당 복소 지문**."""
    sig = {}
    for app in SMPS:
        V = []
        for stem in DEV_FILES[app]:
            f = DEV_DIR / (stem + ".npz")
            if not f.exists():
                continue
            I, pf = load_cplx(f)
            p = pf[:, 0]
            hi = np.percentile(p, 90)
            on = p > max(0.5 * hi, p.min() + 0.3 * (hi - p.min()))
            if on.sum() < 300 or (~on).sum() < 100:
                continue
            bg = np.median(I[~on].real, 0) + 1j * np.median(I[~on].imag, 0)
            v = (np.median(I[on].real, 0) + 1j * np.median(I[on].imag, 0)) - bg
            dp = float(np.median(p[on]) - np.median(p[~on]))
            V.append(v / max(dp, 1e-6))
        if V:
            sig[app] = np.mean(V, 0)
    return sig


def audit(half=5.0, guard=1.0, n_null=150):
    """[C] 특징 심사 기준 (12.53.6 / 12.54).

    ① 차분이 배경을 지우는가 — 복합 전이의 복소 차분이 격리 지문과 같은가 (코사인)
    ② 신호가 잡음보다 큰가 — 라벨 전이에서 12초 이상 떨어진 시각의 **널 분포**
       (계측 잡음뿐 아니라 **다른 기기의 움직임**도 들어 있다. 그래서 꼬리가 길다)
    ③ 과도 채널 48~50 을 같은 자로 심사
    """
    ev = load_events()
    sig = _dev_signature()
    rng = np.random.default_rng(0)
    rec, nulls, tr_sig = {}, [], {}
    tr_null = {"c48": [], "c49": [], "c50": []}
    for stem in ("test_5", "test_6", "test_7"):
        f = Path(DEFAULT_DIR) / (stem + ".npz")
        if not f.exists():
            continue
        I, pf = load_cplx(f)
        p = pf[:, 0]
        n = len(p)
        t = np.arange(n) / 60.0
        i3 = np.abs(I[:, 2])
        ch = {"c48": np.arcsinh((_trail_1d(i3, TRANSIENT_LOOKBACK, TRANSIENT_BLOCK) - i3) * CURRENT_SCALE),
              "c49": np.arcsinh((i3 - _shift(i3, TRANSIENT_LOOKBACK)) * CURRENT_SCALE),
              "c50": np.arcsinh((i3 - _shift(i3, 60)) * CURRENT_SCALE)}
        T = transitions(ev, stem, SMPS)
        tt = np.array([x[0] for x in T]) if T else np.zeros(0)
        for t0, app, sg in T:
            k = int((t0 + 1.5) * 60)          # 12.37.3 의 관측 시점
            if TRANSIENT_LOOKBACK + 60 <= k < n:
                for c in ch:
                    tr_sig.setdefault(app, {}).setdefault(c, []).append(float(ch[c][k]))
            if t0 < half + 7 or t0 > t[-1] - half - 7 or app not in sig:
                continue
            best, bd = t0, -1.0
            for c in np.arange(t0 - 6, t0 + 6, 0.25):
                pre = (t >= c - half) & (t <= c - guard)
                post = (t >= c + guard) & (t <= c + half)
                if pre.sum() < 30 or post.sum() < 30:
                    continue
                d = abs(np.median(i3[post]) - np.median(i3[pre]))
                if d > bd:
                    best, bd = c, d
            pre = (t >= best - half) & (t <= best - guard)
            post = (t >= best + guard) & (t <= best + half)
            dv = ((np.median(I[post].real, 0) + 1j * np.median(I[post].imag, 0))
                  - (np.median(I[pre].real, 0) + 1j * np.median(I[pre].imag, 0)))
            dp = float(np.median(p[post]) - np.median(p[pre]))
            if abs(dp) < 5:
                continue
            ref = np.sign(dp) * sig[app]
            rec.setdefault(app, []).append({
                "cos": float(np.real(np.vdot(dv, ref))
                             / (np.linalg.norm(dv) * np.linalg.norm(ref) + 1e-12)),
                "gain": float(np.linalg.norm(dv)
                              / (np.linalg.norm(sig[app]) * abs(dp) + 1e-12))})
        cand = [c for c in np.arange(half + 7, t[-1] - half - 7, 2.0)
                if len(tt) == 0 or np.min(np.abs(tt - c)) > 12]
        for c in rng.choice(cand, min(n_null, len(cand)), replace=False):
            pre = (t >= c - half) & (t <= c - guard)
            post = (t >= c + guard) & (t <= c + half)
            if pre.sum() < 30 or post.sum() < 30:
                continue
            nulls.append(np.abs((np.median(I[post].real, 0) + 1j * np.median(I[post].imag, 0))
                                - (np.median(I[pre].real, 0) + 1j * np.median(I[pre].imag, 0))))
            k = int(c * 60)
            if TRANSIENT_LOOKBACK + 60 <= k < n:
                for cc in ch:
                    tr_null[cc].append(float(ch[cc][k]))
    return {"recovery": rec, "null": np.array(nulls),
            "tr_sig": tr_sig, "tr_null": tr_null}


def print_audit(au):
    nl = au["null"]
    print()
    print("=" * 96)
    print("[C] 특징 심사 기준 — 차분이 배경을 지우는가 / 신호가 널보다 큰가 (12.53.6)")
    print("=" * 96)
    print("  1) 차분 벡터가 격리 지문을 복원하는가")
    print("     " + "기기".ljust(10) + "n".rjust(4) + "코사인".rjust(12) + "크기배율".rjust(13))
    for app, v in au["recovery"].items():
        c = np.array([x["cos"] for x in v])
        g = np.array([x["gain"] for x in v])
        print("     " + KOR.get(app, app).ljust(10) + str(len(v)).rjust(4)
              + ("%.3f +-%.2f" % (np.median(c), c.std())).rjust(14)
              + ("%.2f +-%.2f" % (np.median(g), g.std())).rjust(13))
    print()
    print("  2) 널 분포 — 전이 없는 시각 %d개, 같은 창·같은 계산" % len(nl))
    print("     " + "차수".ljust(8) + "널 중앙(mA)".rjust(14) + "널 p95(mA)".rjust(14))
    for h, lab in ((0, "|dI1|"), (1, "|dI2|"), (2, "|dI3|"), (8, "|dI9|"), (12, "|dI13|")):
        print("     " + lab.ljust(8) + ("%.2f" % (1000 * np.median(nl[:, h]))).rjust(14)
              + ("%.2f" % (1000 * np.percentile(nl[:, h], 95))).rjust(14))
    print()
    print("  3) 과도 채널 48~50 심사 (12.54) — 전이 1.5초 뒤 vs 조용한 시각")
    print("     " + "채널".ljust(8) + "널중앙".rjust(10) + "널p95".rjust(10)
          + "|전이값|".rjust(11) + "전이vs널 AUC".rjust(15) + "최대 분리비".rjust(13))
    for cc in ("c48", "c49", "c50"):
        nn = np.abs(np.array(au["tr_null"][cc]))
        ss = np.abs(np.concatenate([np.array(au["tr_sig"][x][cc]) for x in au["tr_sig"]]))
        best = max(sepratio(np.array(au["tr_sig"][x][cc]), np.array(au["tr_sig"][y][cc]))
                   for x in au["tr_sig"] for y in au["tr_sig"] if x < y)
        print("     " + cc.ljust(8) + ("%.3f" % np.median(nn)).rjust(10)
              + ("%.3f" % np.percentile(nn, 95)).rjust(10)
              + ("%.3f" % np.median(ss)).rjust(11)
              + ("%.3f" % _auc(ss, nn)).rjust(15) + ("%.2f" % best).rjust(13))


def main() -> int:
    ap = argparse.ArgumentParser(description="차분 고조파 특징 검증 (12.53절)")
    ap.add_argument("--audit", action="store_true",
                    help="[C] 널 분포·지문 복원·과도 채널 심사 (12.53.6, 12.54)")
    ap.add_argument("--out", default="results/delta_feature_probe.json")
    a = ap.parse_args()

    print("=" * 96)
    print("[A] 격리 녹화 — 배경을 복소로 뺀 순수 기기 벡터")
    print("=" * 96)
    iso = isolated()
    print(f"  {'기기':<14s}{'구간':>5s}{'ΔP중앙':>9s}"
          f"{'|I2|/|I1|':>22s}{'|I13|/|I3|':>22s}{'|I3|/W (mA/W)':>22s}")
    print("  " + "-" * 94)
    for app in ("laptop_charger", "beam_projector", "minipc",
                "hotplate", "oven", "electiric_kettle"):
        v = iso.get(app)
        if not v:
            continue
        f = lambda k: np.array([x[k] for x in v])
        print(f"  {KOR.get(app, app):<14s}{len(v):>5d}{np.median(f('dP')):>8.0f}W"
              + "".join(f"{np.median(f(k)):>13.4f} ±{f(k).std():<8.4f}"
                        for k in ("I2_I1", "I13_I3", "I3_per_W")))

    print()
    print("  분리비 (기기간 차이 / 기기내 변동, 2 미만이면 겹침 — 12.35.1)")
    for k, pairs in (("I2_I1", [("laptop_charger", "beam_projector"),
                                ("laptop_charger", "minipc")]),
                     ("I13_I3", [("beam_projector", "minipc"),
                                 ("beam_projector", "laptop_charger")]),
                     ("I9_I3", [("beam_projector", "minipc")]),
                     ("I3_per_W", [("hotplate", "oven"),
                                   ("hotplate", "electiric_kettle")])):
        for x, y in pairs:
            if x in iso and y in iso:
                A = np.array([r[k] for r in iso[x]]); B = np.array([r[k] for r in iso[y]])
                s = sepratio(A, B)
                print(f"    {k:<10s}{KOR.get(x,x):>8s} vs {KOR.get(y,y):<8s}"
                      f"  {A.mean():8.4f} vs {B.mean():8.4f}   분리비 {s:5.2f}"
                      f"  {'갈린다' if s >= 2 else '겹침'}")

    print()
    print("=" * 96)
    print("[B] 복합 실측 전이 — test_5/6/7 라벨된 SMPS 전이에서 복소 차분")
    print("=" * 96)
    rows = composite()
    print(f"  전이 {len(rows)}개 (|ΔP| >= 5W 인 것만)")
    print(f"  {'스위칭 기기':<12s}{'n':>4s}{'|ΔP|':>8s}{'|ΔI2|(mA)':>12s}"
          f"{'|ΔI2|/|ΔI1|':>14s}{'|ΔI13|/|ΔI3|':>15s}{'|ΔI9|/|ΔI3|':>14s}")
    print("  " + "-" * 82)
    by = {}
    for r in rows:
        by.setdefault(r["app"], []).append(r)
    for app in SMPS:
        v = by.get(app)
        if not v:
            continue
        f = lambda k: np.array([x[k] for x in v])
        print(f"  {KOR.get(app, app):<12s}{len(v):>4d}{np.median(np.abs(f('dP'))):>7.0f}W"
              f"{1000*np.median(f('dI2')):>12.1f}{np.median(f('dI2_dI1')):>14.4f}"
              f"{np.median(f('dI13_dI3')):>15.4f}{np.median(f('dI9_dI3')):>14.4f}")
    print()
    print("  분리비 (복합 전이에서)")
    for k in ("dI2", "dI2_dI1", "dI13_dI3", "dI9_dI3"):
        for x, y in (("laptop_charger", "beam_projector"),
                     ("beam_projector", "minipc"),
                     ("laptop_charger", "minipc")):
            if x in by and y in by and len(by[x]) > 2 and len(by[y]) > 2:
                A = np.array([r[k] for r in by[x]]); B = np.array([r[k] for r in by[y]])
                s = sepratio(A, B)
                flag = "  <<< 갈린다" if s >= 2 else ""
                print(f"    {k:<10s}{KOR.get(x,x):>6s} vs {KOR.get(y,y):<6s}"
                      f"  {A.mean():9.4f} vs {B.mean():9.4f}  분리비 {s:5.2f}{flag}")

    if a.audit:
        print_audit(audit())

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"isolated": iso, "composite": rows},
                                      ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print()
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
