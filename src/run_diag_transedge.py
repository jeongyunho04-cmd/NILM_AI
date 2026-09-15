# -*- coding: utf-8 -*-
"""**대부하 전이 근처의 헛detect — 원인이 창 안의 전이인가** (14.154).

사용자: *"게이트에 의한 오염을 분리해도 이렇게 대부하기기가 켜고 꺼질때 오탐 축퇴가
발생하는데 이제는 무엇때문이지 진단해줘"*

14.152 가 게이트 가설을 죽였다. 남은 후보는 **창이 전이를 물고 있다는 것 자체**다.
창의 시간 범위부터 못 박는다 (`target_index(3600) = 3239`):

```
  세밀  타깃 **−4.0초 ~ +6.0초**    (600사이클 @60Hz 중 index 239 가 타깃)
  광역  타깃 **−54.0초 ~ +6.0초**   (3600사이클을 2Hz 로)
```
즉 **54초 전에 꺼진 대부하도 아직 창 안**이다.

거리 곡선을 또 그리지 않는다 ([[a-flat-plateau-with-a-cliff-is-not-a-distance-effect]]).
대신 **개입**한다 — 원시 45채널 타임라인에서 창 안의 전이 **반대쪽**을 타깃 쪽 내용으로
덮어서 *전이가 없는 창*으로 만든다. **타깃 시점의 원시값은 한 톨도 안 건드린다.**

```
  [1] 크기   확실 OFF 칸의 헛ON율을 '창 안 대부하 전이 유무' 로 가른다 (독립 구간도 센다)
  [2] 개입   전이 지우기 — 헛ON 이 사라지면 원인은 **창 안의 반대쪽**이다
  [3] 위약   같은 양을 **전이가 아닌 곳**에서 잘라 붙인다. 이게 같이 없애면 [2] 는 못 읽는다
  [4] 갈래   사라졌으면 어느 경로인지 — 물리 프라이어의 **창-최대** 철회 / Δ채널 / 그 외
```

전이는 라벨이 아니라 **관측 전력**에서 뽑는다 (라벨 타이밍 오차를 안 탄다).

    python -X utf8 -m src.run_diag_transedge --ckpt results/cnn_gfp_s0.pt results/cnn_pcap_s0.pt
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.model.inputs import build_inputs  # noqa: E402
from src.model.net import P_CH_FINE, P_CH_WIDE  # noqa: E402
from src.model.realdata import RealWindows, target_index  # noqa: E402
from src.preprocessing import load_nilm_npz  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
FS = 60
WIN = 3600
STEP_W = 300.0        #: 이 와트 이상 변하면 **대부하 전이**
NMS_S = 2.0           #: 전이 사이 최소 간격


def big_steps(P: np.ndarray) -> np.ndarray:
    """관측 전력에서 대부하 전이 사이클을 뽑는다 (라벨을 안 쓴다)."""
    k = FS                                   # ±1초
    c = np.convolve(P, np.ones(k) / k, mode="same")
    d = np.zeros_like(c)
    d[k:-k] = c[2 * k:] - c[:-2 * k]
    a = np.abs(d)
    cand = np.flatnonzero(a >= STEP_W)
    if not len(cand):
        return cand
    out, last = [], -10 ** 9
    for i in cand[np.argsort(-a[cand])]:
        if all(abs(i - j) > NMS_S * FS for j in out):
            out.append(int(i))
    return np.array(sorted(out))


def labels(stem, t_s, apps, ev):
    """(참ON, 채점제외). `uncertain` 은 빼고 센다 — 그림의 규약과 같다."""
    on = np.zeros((len(t_s), len(apps)), bool)
    unc = np.zeros_like(on)
    iv = ev[stem]["intervals"]
    pres = np.array([a in ev[stem]["appliances_present"] for a in apps])
    for k, a in enumerate(apps):
        for t0, t1 in iv.get(a, {}).get("on", []):
            on[(t_s >= t0) & (t_s <= t1), k] = True
        for t0, t1 in iv.get(a, {}).get("uncertain", []):
            unc[(t_s >= t0) & (t_s <= t1), k] = True
    return on, unc, pres


def fwd(m, fine, wide, dev, bs=256):
    G, PR, M0 = [], [], []
    with torch.no_grad():
        for i in range(0, len(fine), bs):
            o = m(torch.from_numpy(np.ascontiguousarray(fine[i:i + bs])).to(dev),
                  torch.from_numpy(np.ascontiguousarray(wide[i:i + bs])).to(dev))
            G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
            PR.append(o["power"].float().cpu().numpy())
            M0.append(o["power_mix"].float()[..., 0].cpu().numpy())
    return np.concatenate(G), np.concatenate(PR), np.concatenate(M0)


def prior_term(m, fine, wide, dev, bs=256):
    """`logsigmoid(κ·(창최대 − 문턱))` — 물리 프라이어가 **얼마나 막고 있나**."""
    if float(m.prior_kappa) <= 0:
        return np.zeros((len(fine), len(m.appliances)), np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(fine), bs):
            f = torch.from_numpy(np.ascontiguousarray(fine[i:i + bs])).to(dev)
            w = torch.from_numpy(np.ascontiguousarray(wide[i:i + bs])).to(dev)
            pm = torch.maximum(f[:, P_CH_FINE].amax(-1), w[:, P_CH_WIDE].amax(-1))
            gap = pm[:, None] - m.on_threshold_asinh[None]
            out.append(F.logsigmoid(float(m.prior_kappa) * gap).float().cpu().numpy())
    return np.concatenate(out)


def _mirror_fill(src, n, at_start):
    """`src` 를 **거울 반복**으로 늘려 길이 `n` 을 만든다. 이어붙이는 자리가 연속이 되게.

    `at_start=True` 면 결과의 **첫 값**이 `src[:, -1]` 이 되고(그 왼쪽이 src 의 끝이므로),
    아니면 결과의 **끝 값**이 `src[:, 0]` 이 된다. 잘라 붙인 자리에 **가짜 계단이 안 생긴다**.
    """
    a = src if not at_start else src[:, ::-1]
    b = src[:, ::-1] if not at_start else src
    unit = np.concatenate([a, b], 1)
    r = int(np.ceil(n / unit.shape[1])) + 1
    full = np.concatenate([unit] * r, 1)
    return full[:, :n] if at_start else full[:, -n:]


def splice(x, t, e, off=None):
    """창 `[t-off, t-off+WIN)` 안에서 **전이 `e` 의 반대쪽**을 타깃 쪽 내용으로 덮는다.

    `e < t` 면 과거를, `e > t` 면 미래를 지운다. **타깃 시점 `t` 의 원시값은 안 건드린다.**
    """
    off = target_index(WIN) if off is None else off
    lo, hi = t - off, t - off + WIN
    if lo < 0 or hi > x.shape[1]:
        return None
    y = x[:, lo:hi].copy()
    j = e - lo                                          # 창 안 좌표
    if j <= 1 or j >= WIN - 1:
        return None
    if e < t:
        src = y[:, j:min(j + max(t - e, 2), WIN)]        # 전이 이후 ~ 타깃
        if src.shape[1] < 2:
            return None
        y[:, :j] = _mirror_fill(src, j, at_start=False)
    else:
        src = y[:, max(0, j - max(e - t, 2)):j]          # 타깃 ~ 전이 이전
        if src.shape[1] < 2:
            return None
        y[:, j:] = _mirror_fill(src, WIN - j, at_start=True)
    return y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_gfp_s0.pt"])
    ap.add_argument("--stride", type=int, default=60, help="창 간격(사이클). 60=1초")
    ap.add_argument("--max-cells", type=int, default=400, help="개입을 걸 헛ON 칸 수 상한")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    off = target_index(WIN)
    print("창: 타깃 −%.1f초 ~ +%.1f초 (광역) · 세밀은 −4.0~+6.0초 · 전이 문턱 %.0fW"
          % (off / FS, (WIN - 1 - off) / FS, STEP_W))

    # ── 파일별 원시 타임라인·창·전이 ─────────────────────────────────────
    D = {}
    for stem in FILES:
        rw = RealWindows(stems=[stem], stride=a.stride, require_valid=False)
        raw = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        x = RealWindows._to_33ch(raw)
        P = np.asarray(raw["power_features"], np.float64)[:, 0]
        st = big_steps(P)
        t = np.asarray(rw.target_cycle)
        on, unc, pres = labels(stem, t / FS, apps, ev)
        # 창 안에 대부하 전이가 있나 (광역 창 기준)
        inwin = np.zeros(len(t), bool)
        near = np.full(len(t), np.nan)
        for i, tc in enumerate(t):
            m = (st >= tc - off) & (st < tc - off + WIN)
            inwin[i] = m.any()
            if m.any():
                near[i] = (st[m] - tc)[np.argmin(np.abs(st[m] - tc))] / FS
        D[stem] = dict(rw=rw, x=x, P=P, st=st, t=t, on=on, unc=unc,
                       pres=pres, inwin=inwin, near=near)
        print("  %-8s 창 %5d · 대부하 전이 %2d개 · 창 안 전이 있음 %5.1f%%"
              % (stem, len(t), len(st), 100 * inwin.mean()))

    for ck in a.ckpt:
        m = load_model(ck, dev)[0]
        m.eval()
        gf = bool(getattr(m, "gate_free_power", False))
        print("\n" + "=" * 96)
        print("■ %s%s" % (ck.split("/")[-1].replace(".pt", ""),
                          "  (완전분해)" if gf else ""))
        print("=" * 96)

        G, PW, PRI, OK, ONL, STEMS, TC, NEAR, INW = [], [], [], [], [], [], [], [], []
        for stem, d in D.items():
            g, pw, _ = fwd(m, d["rw"].fine, d["rw"].wide, dev)
            G.append(g); PW.append(pw)
            PRI.append(prior_term(m, d["rw"].fine, d["rw"].wide, dev))
            OK.append((~d["unc"]) & d["pres"][None])      # 채점에 쓰는 칸
            ONL.append(d["on"])
            STEMS += [stem] * len(d["t"])
            TC.append(d["t"]); NEAR.append(d["near"]); INW.append(d["inwin"])
        G, PW, PRI = np.concatenate(G), np.concatenate(PW), np.concatenate(PRI)
        OK, ONL = np.concatenate(OK), np.concatenate(ONL)
        TC, NEAR, INW = (np.concatenate(x) for x in (TC, NEAR, INW))
        STEMS = np.array(STEMS)

        offc = OK & (~ONL)                                # 확실 OFF 칸
        fa = offc & (G > 0.5)                             # 헛ON
        print("\n[1] **크기** — 확실 OFF 칸의 헛ON율 (창 안 대부하 전이 유무로 가른다)")
        print("  %-18s %20s %20s %10s"
              % ("기기", "전이 **있는** 창", "전이 **없는** 창", "배"))
        for k, ap_ in enumerate(apps):
            c1 = offc[:, k] & INW
            c0 = offc[:, k] & (~INW)
            if c1.sum() < 20 or c0.sum() < 20:
                continue
            r1 = fa[c1, k].mean() if c1.sum() else np.nan
            r0 = fa[c0, k].mean() if c0.sum() else np.nan
            print("  %-18s %8.1f%% (%5d칸) %8.1f%% (%5d칸) %10.2f"
                  % (ap_, 100 * r1, c1.sum(), 100 * r0, c0.sum(),
                     r1 / max(r0, 1e-9)))
        tot1, tot0 = offc & INW[:, None], offc & (~INW[:, None])
        print("  %-18s %8.1f%% (%5d칸) %8.1f%% (%5d칸) %10.2f"
              % ("**전체**", 100 * fa[tot1].mean(), tot1.sum(),
                 100 * fa[tot0].mean(), tot0.sum(),
                 fa[tot1].mean() / max(fa[tot0].mean(), 1e-9)))

        # ── [2][3] 개입 ──────────────────────────────────────────────────
        idx = np.flatnonzero(fa.any(1) & INW)
        if len(idx) > a.max_cells:
            idx = np.random.default_rng(0).choice(idx, a.max_cells, replace=False)
        rows = []
        for mode in ("전이 지우기", "위약(엉뚱한 곳)"):
            cut, keep_i, keep_k = [], [], []
            for i in idx:
                stem = STEMS[i]
                d = D[stem]
                tc = int(TC[i])
                inw = d["st"][(d["st"] >= tc - off) & (d["st"] < tc - off + WIN)]
                if not len(inw):
                    continue
                e = int(inw[np.argmin(np.abs(inw - tc))])
                if mode.startswith("위약"):
                    # 같은 쪽·같은 거리만큼 떨어진 **전이가 아닌** 자리
                    sgn = 1 if e > tc else -1
                    e2 = tc + sgn * max(int(0.5 * abs(e - tc)), 3 * FS)
                    if not (tc - off < e2 < tc - off + WIN):
                        continue
                    if np.min(np.abs(d["st"] - e2)) < 2 * FS:
                        continue
                    e = e2
                y = splice(d["x"], tc, e, off)
                if y is None:
                    continue
                cut.append(y); keep_i.append(i)
                keep_k.append(np.flatnonzero(fa[i]))
            if not cut:
                continue
            fine2, wide2 = build_inputs(np.stack(cut))
            g2, pw2, _ = fwd(m, fine2, wide2, dev)
            pri2 = prior_term(m, fine2, wide2, dev)
            b, af, n, dpri = [], [], 0, []
            for r, (i, ks) in enumerate(zip(keep_i, keep_k)):
                for k in ks:
                    b.append(G[i, k]); af.append(g2[r, k])
                    dpri.append(pri2[r, k] - PRI[i, k]); n += 1
            b, af = np.array(b), np.array(af)
            rows.append((mode, n, b.mean(), af.mean(), (af > 0.5).mean(),
                         float(np.mean(dpri))))
        print("\n[2]+[3] **개입** — 창 안의 전이를 지운다 (타깃 시점 원시값은 **비트 동일**)")
        print("  %-18s %6s %12s %12s %14s %12s"
              % ("판", "헛ON칸", "게이트 전", "게이트 후", "아직 >0.5", "Δ프라이어"))
        for nm, n, b, af, still, dp in rows:
            print("  %-18s %6d %12.3f %12.3f %13.1f%% %12.4f"
                  % (nm, n, b, af, 100 * still, dp))
        if len(rows) == 2:
            print("  (전이 지우기만 떨어지면 원인은 **창 안의 반대쪽**이다. "
                  "위약도 같이 떨어지면 잘라 붙인 것 자체가 원인이라 못 읽는다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
