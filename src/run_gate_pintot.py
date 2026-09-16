# -*- coding: utf-8 -*-
"""**총량 고정**의 관문 — 14.344 의 실측 수를 되찾나 (14.345).

사용자 지적에서 나온 칸이다: *"잘못된걸 거부하면 거기에 뭘 채워놔야지 그냥 비워놓아
버리면 어떡하니"*. 거부권만 걸면 test_5 총잔차가 20.7 -> 39.5W 로 터진다.

이 관문이 **기각선**이다 — 조합 머리(학습판)를 짓기 전에, 후처리가 내는 수를 모듈과
관문으로 고정해 둔다. 학습판이 이 수를 못 넘으면 구조로 할 이유가 없다.

⚠ `processed_data/composite_eval` 을 읽으므로 **HPC 에서 못 돈다** (HPC_RULES §0).

    python -X utf8 -m src.run_gate_pintot
"""
import argparse
import glob
import os
import sys
from math import comb

import numpy as np

from src import env_guard  # noqa: F401

from src.model import inputs as _I  # noqa: E402

_I.EVEN_MEDIAN = 5
from src.evaluation.real_events import build_on_off_truth, load_events  # noqa: E402
from src.model import gbudget as GB  # noqa: E402
from src.model.inputs import build_inputs, target_index  # noqa: E402
from src.model.losses import S_STATE  # noqa: E402
from src.model.realdata import RealWindows  # noqa: E402

APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
WC, STRIDE = 3600, 30
KK, KO = APPS.index("electiric_kettle"), APPS.index("oven")
RI = [APPS.index(a) for a in GB.RESISTIVE]
OK = []


def chk(no, name, cond, msg):
    OK.append(bool(cond))
    print("  [%d] %-40s %s\n      %s" % (no, name, "통과" if cond else "**실패**", msg))


def sgn(d):
    d = [x for x in d if x != 0]
    n = len(d)
    if not n:
        return 1.0
    k = sum(1 for x in d if x < 0)
    return min(sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n * 2, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_v49base_s*.pt")
    a = ap.parse_args()

    # ── [1][2] 자료 없이 되는 구조 관문 ─────────────────────────────────────
    rng = np.random.default_rng(0)
    P0 = rng.random((512, len(APPS))) * 1500.0
    g0 = rng.random(512) * 60.0
    v0 = 210.0 + rng.random(512) * 20.0
    q, _ = GB.apply(P0, g0, v0, APPS, veto=False, floor=False, pin=False, total=False)
    chk(1, "전부 끄면 **비트 동일**", np.array_equal(q, P0),
        "최대차 %.3e" % float(np.abs(q - P0).max()))
    #: ⚠ 14.345 — 첫 판에서 *"총량이 Ĝ·V² 와 소수점까지 같다"* 를 걸었다가 220.9W 로
    #  실패했다. **자가 틀렸다** — `fill` 경로는 저항이 하나도 안 남았을 때 24개 후보 중
    #  하나를 고르므로 총량이 정확히 안 맞는다. 그게 설계다 (이산 조합이라 그럴 수밖에).
    #  그래서 둘로 갈라 잰다: **비율이 정의된 창에서는 정확히**, 채운 창은 **후보 위에**.
    tot0 = g0 * 1e-3 * v0 ** 2
    q2, _ = GB.apply(P0, g0, v0, APPS, total=True)
    pre = GB.apply(P0, g0, v0, APPS, total=False)[0]
    alive = pre[:, RI].sum(1) > 1e-9
    qnf = GB.pin_total(pre, g0, v0, APPS, fill=False)
    #: ⚠ **`alive` 에서만** 성립한다. 비율이 0/0 인 창은 `fill=False` 면 전력이 0 인데
    #  `Ĝ·V²` 는 0 이 아니다 — 첫 두 판에서 그걸 안 갈라 220.9W·339.7W 로 실패했다.
    d2 = float(np.abs(qnf[alive][:, RI].sum(1) - tot0[alive]).max())
    chk(2, "★ 총량이 `Ĝ·V²` 와 **소수점까지** 같나 (`fill` 끔)", d2 < 1e-6,
        "|ΣP_저항 − Ĝ·V²| 최대 **%.3e W** (비율이 사는 창 %d/%d) — 구멍이 원리상 없다는 보증"
        % (d2, int(alive.sum()), len(alive)))
    #: 채운 창은 **후보 위에** 앉아야 한다 (총량은 못 맞춰도 조합은 물리적이어야 한다)
    filled = (~alive) & (tot0 > 100.0)
    names = [APPS[j] for j in RI]
    cg = np.array([sum(y[0] for y in c) * 1e-3 for c in __import__("itertools").product(
        *[[(0.0, None)] + [(gg, s_) for s_, gg in GB.app_states(x)] for x in names])])
    if filled.any():
        got = q2[filled][:, RI].sum(1) / np.maximum(v0[filled] ** 2, 1.0)
        off = np.abs(got[:, None] - cg[None, :]).min(1)
        chk(3, "채운 창이 **후보 위에** 앉나", float(off.max()) < 1e-9,
            "채운 창 %d개 · 후보에서 벗어난 양 최대 **%.3e mS** · "
            "총량 오차는 |Ĝ−가장가까운후보|·V² 라 최대 %.0fW (이산 조합이라 그렇다)"
            % (int(filled.sum()), 1e3 * float(off.max()),
               float(np.abs(q2[filled][:, RI].sum(1) - tot0[filled]).max())))
    else:
        chk(3, "채운 창이 **후보 위에** 앉나", True, "채운 창이 없다 (실측에서도 0이다)")
    non = [j for j, x in enumerate(APPS) if x not in GB.PIN_MS]
    chk(4, "저항 아닌 %d종이 **한 칸도 안 바뀐다**" % len(non),
        np.array_equal(q2[:, non], P0[:, non]),
        "최대차 %.3e" % float(np.abs(q2[:, non] - P0[:, non]).max()))

    # ── 실측 창 ────────────────────────────────────────────────────────────
    cks = sorted(glob.glob(a.ckpt))
    from src.run_gate_check import load_model
    good = []
    for c in cks:
        import torch
        d = torch.load(c, map_location="cpu", weights_only=False)
        if (int(d.get("fine_channels", -1)) == _I.FINE_CHANNELS
                and tuple(d.get("volt_orders") or ()) == tuple(_I.VOLT_ORDERS)):
            good.append(c)
    if not good:
        print("  [5~7] **건너뜀** — `%s` 중 지금 배치(FINE %d · 차수 %s)와 맞는 것이 없다.\n"
              "        배치를 바꾼 직후의 부트스트랩이다. 한 팔이라도 구워지면 다시 돈다."
              % (a.ckpt, _I.FINE_CHANNELS, tuple(_I.VOLT_ORDERS)))
        print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
        return 0 if all(OK) else 1

    import torch
    ev = load_events()
    lo = target_index(WC)
    F, W, L, G, V1, OB = [], [], [], [], [], []
    pool = None
    for stem in FILES:
        z = np.load("processed_data/composite_eval/%s.npz" % stem, allow_pickle=True)
        x = RealWindows._to_33ch(z).astype(np.float64)
        n = x.shape[1]
        iv = np.asarray(z["is_valid"], bool)
        tg = np.arange(lo, n - (WC - 1 - lo), STRIDE, dtype=np.int64)
        tg = np.array([t for t in tg if iv[t - lo:t - lo + WC].all()], np.int64)
        f, w = build_inputs(np.stack([x[:, t - lo:t - lo + WC] for t in tg]))
        F.append(f.astype(np.float16)); W.append(w.astype(np.float16))
        aps = sorted(ev[stem]["intervals"].keys())
        on, _ = build_on_off_truth(stem, aps, n, events=ev)
        on = np.asarray(on, bool)
        lb = np.zeros((len(tg), len(APPS)), bool)
        for x_ in aps:
            if x_ in APPS:
                lb[:, APPS.index(x_)] = on[tg, aps.index(x_)]
        L.append(lb)
        if pool is None:
            from src.synthesis.segment_pool import SegmentPool
            pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
        nv = len(_I.VOLT_ORDERS)
        v15 = np.zeros(15, complex)
        md = np.median(x[33:33 + nv], 1) + 1j * np.median(x[33 + nv:33 + 2 * nv], 1)
        for s_, h in enumerate(_I.VOLT_ORDERS):
            v15[h - 1] = md[s_]
        bud = GB.Budget(APPS, v15, pool=pool, volt_re0=33, volt_orders=_I.VOLT_ORDERS)
        G.append(bud.g_sum(x[None][:, :, tg])[0])
        V1.append(np.abs(x[33, tg] + 1j * x[33 + nv, tg]))
        OB.append(x[30, tg])
    F = np.concatenate(F); W = np.concatenate(W); L = np.concatenate(L)
    g = np.concatenate(G); v1 = np.concatenate(V1); obs = np.concatenate(OB)

    #: §31.1 괄호 자 (모델 무관)
    PT = (100.0 * np.sinh(F[:, 23, :].astype(np.float64)))[:, _I.fine_target_index()]
    los = np.array([min(S_STATE[x].values()) for x in APPS])
    oth = np.ones(len(APPS), bool); oth[KO] = False
    ener = L[:, KO] & ~((PT - (L & oth[None]) @ los) < 800.0)
    offo, kon, koff = ~L[:, KO], L[:, KK], ~L[:, KK]

    def score(P):
        return np.array([int((offo & (P[:, KO] >= 300)).sum()),
                         int((ener & (P[:, KO] < 300)).sum()),
                         int((koff & (P[:, KK] >= 300)).sum()),
                         int((kon & (P[:, KK] < 300)).sum())])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    PWS = []
    for c in good:
        m = load_model(c, dev)[0]
        m.eval()
        #: ★ 14.349 — **조합 머리는 Ĝ 를 같이 먹어야 돈다.** 여기 `g` 는
        #  `bud.g_sum` 이 낸 mS 로 `g_hat.npy` 와 **같은 양**이다. 안 넘기면
        #  `net.forward` 가 멈춘다 (`run_gate_comb` [7] 이 잠근 가드).
        #  ⚠ 다만 기준전압은 **이 녹화의 것**이고 학습은 **캐시의 것**이다 —
        #    그 차가 Ĝ 를 최대 0.275 mS 움직인다 (14.346). 결과를 읽을 때 센다.
        _cb = float(getattr(m, "comb_tau", 0.0) or 0.0) > 0
        o = []
        with torch.no_grad():
            for i in range(0, len(F), 256):
                o.append(m(torch.from_numpy(F[i:i + 256].astype(np.float32)).to(dev),
                           torch.from_numpy(W[i:i + 256].astype(np.float32)).to(dev),
                           torch.from_numpy(g[i:i + 256].astype(np.float32)).to(dev)
                           if _cb else None)["power"].cpu().numpy())
        PWS.append(np.concatenate(o).astype(np.float64))
        del m
    print("  실측 창 %d · 씨앗 %d (%s) · 오븐확실통전 %d · 포트ON %d"
          % (len(F), len(PWS), os.path.basename(good[0]).rsplit("_s", 1)[0],
             int(ener.sum()), int(kon.sum())))

    def arm(**kw):
        sc = np.array([score(P if not kw else GB.apply(P, g, v1, APPS, **kw)[0]) for P in PWS])
        rs = [float(np.abs(obs - (P if not kw else GB.apply(P, g, v1, APPS, **kw)[0]).sum(1)).mean())
              for P in PWS]
        return sc, float(np.median(rs))

    base, rb = arm()
    tot_only, _ = arm(veto=False, floor=False, pin=False, total=True)
    full, rf = arm(total=True)
    mb, mf = np.median(base, 0), np.median(full, 0)
    dd = [int(full[i].sum() - base[i].sum()) for i in range(len(PWS))]

    chk(5, "★ 사중+총량이 **14.344 의 수**를 되찾나",
        mf.sum() <= 33 and rf <= 11.5,
        "바닥 A%.0f B%.0f C%.0f D%.0f 합 **%.0f** · |r| %.1fW  ->  "
        "사중+총량 A%.0f B%.0f C%.0f D%.0f 합 **%.0f** · |r| **%.1fW**  (기준 합<=33 · |r|<=11.5)"
        % (*mb, mb.sum(), rb, *mf, mf.sum(), rf))
    chk(6, "씨앗 짝검정이 **6/6** 인가", sgn(dd) <= 0.05 and all(x < 0 for x in dd),
        "합차 %s · p=%.3f" % (dd, sgn(dd)))
    chk(7, "★ 거부권 **없이** 총량만 고정하면 C오탐이 는다",
        np.median(tot_only, 0)[2] >= mb[2],
        "C오탐 바닥 %.0f -> 총량만 **%.0f** — 오탐이 유일한 주장이면 재정규화가 그놈을 키운다. "
        "**거부권이 먼저 와야 한다**는 증거다" % (mb[2], np.median(tot_only, 0)[2]))

    print("\n%s  (%d/%d)" % ("전부 통과" if all(OK) else "**실패 있음**", sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == "__main__":
    raise SystemExit(main())
