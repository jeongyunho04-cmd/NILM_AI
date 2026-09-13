"""기울기 귀속 — **어느 항이 프로젝터를 끌어내리는가** (13.61)
=================================================================
13.59·13.60 이 같은 벽에 두 번 부딪혔다: **반사실 지형에서 최소를 참값으로 옮겼는데
학습이 안 따라온다.** 지형은 정적이고 학습은 기울기 과정이다. 그러면 기울기를 재야 한다.

    g_S = ∇θ S      S = 자리 D · SMPS 전용 · 셋 다 켜진 창의 **프로젝터 예측 전력**
    g_T = ∇θ T      T = 학습이 실제로 쓰는 각 손실 항 (가중 포함)

    ⟨−g_T, g_S⟩ > 0   ->  그 항을 줄이면 프로젝터가 **올라간다**
    ⟨−g_T, g_S⟩ < 0   ->  그 항이 프로젝터를 **끌어내린다**

배치는 `run_adapt` 와 같이 뽑는다 (실측은 `RealWindows` 에서 무작위, 합성은 캐시에서).
그래야 "그 항이 그 배치에서 실제로 내는 기울기" 를 재는 것이 된다 — 판정 창만으로
재면 학습이 안 보는 압력을 놓친다.

⚠ 정규화한 코사인과 **원 내적**을 같이 읽는다. 코사인이 커도 노름이 작으면 학습을
  못 움직인다. 표의 `|g_T|` 가 그 크기다.

    python -m src.run_grad_attrib --ckpt results/cnn_v24b.pt
    python -m src.run_grad_attrib --ckpt results/adapt_v24b_z_s0.pt --batch 512
"""
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.realdata import RealWindows, dense_targets
from src.model.traincache import CachedWindows
from src.run_adapt import real_sample_weights, real_targets
from src.run_gate_check import load_model
from src.run_loss_compare import SMPS, BIG, build_loss
from src.run_train_cnn import to_targets

TERMS = ("cons", "harm", "hedge", "pref", "sb", "synth")


def flat_grad(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if x is None else x).reshape(-1)
                      for p, x in zip(params, g)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="cache/train60_v24")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=4, help="배치를 이만큼 평균낸다")
    ap.add_argument("--harm-weight", default="inv_h2")
    ap.add_argument("--w-cons", type=float, default=0.1)
    ap.add_argument("--w-harm", type=float, default=4.0)
    ap.add_argument("--w-hedge", type=float, default=0.2)
    ap.add_argument("--w-pref", type=float, default=0.02)
    ap.add_argument("--w-standby", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--smps-boost", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-standby-sig", action="store_true",
                    help="손실의 **대기 지문**을 0 으로 두고 다시 잰다 (13.61.3). 게이트를 "
                         "내려 대기 지문을 얻는 것이 압력의 통로인지 가른다")
    ap.add_argument("--by-subset", action="store_true",
                    help="실측 창을 **자리 x 구성**으로 갈라 각 묶음의 압력을 따로 잰다 — "
                         "붕괴가 다른 창에서 수입된 것인지 본다")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, apps, _ = load_model(a.ckpt, dev)
    m.train()
    for p in m.parameters():
        p.requires_grad_(True)
    params = [p for p in m.parameters() if p.requires_grad]
    from src.model.net import standby_powers
    from src.synthesis.segment_pool import SegmentPool
    crit = build_loss(apps, dev, a.harm_weight, "",
                      pref_apps=([] if a.w_pref <= 0 else ["beam_projector"]))
    if a.w_standby > 0:
        pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
        crit.standby_w.copy_(torch.from_numpy(standby_powers(pool, apps)).to(dev))
        del pool
    if a.no_standby_sig:
        crit.standby_sig.zero_()
        print("  ** 대기 지문 0 으로 (13.61.3) **")
    jp = apps.index("beam_projector")
    js = [apps.index(x) for x in SMPS]
    jb = [apps.index(x) for x in BIG]

    # ── 판정 방향 g_S: 자리 D 창의 프로젝터 예측 전력 ────────────────────────
    ev = load_events()
    F, W = [], []
    for stem in ("test_3", "test_4", "test_5"):
        rw = dense_targets(stem, stride=30, site_transfer=getattr(m, "site_transfer", None))
        n_cyc = int(ev[stem]["cycles"])
        on, sc = build_on_off_truth(stem, apps, n_cyc, ev)
        t = np.asarray(rw.target_cycle, int).clip(0, n_cyc - 1)
        k = np.flatnonzero((~on[t][:, jb].any(1)) & on[t][:, js].all(1)
                           & sc[t][:, js].all(1))
        if len(k) == 0:
            continue
        k = k[:: max(1, len(k) // 192)][:192]
        f, w, *_ = rw.batch(k)
        F.append(f); W.append(w)
    fS = torch.from_numpy(np.ascontiguousarray(np.concatenate(F))).to(dev)
    wS = torch.from_numpy(np.ascontiguousarray(np.concatenate(W))).to(dev)
    S = m(fS, wS)["power"][:, jp].mean()
    gS = flat_grad(S, params)
    gSn = gS / gS.norm().clamp(min=1e-12)
    print(f"{a.ckpt}\n판정 방향 g_S: 자리 D 창 {len(fS)}개의 프로젝터 예측 전력 "
          f"(현재 {float(S):.2f}W, |g_S| {float(gS.norm()):.4g})\n")

    rwin = RealWindows(stride=30)
    cache = CachedWindows(a.cache)
    rng = np.random.default_rng(a.seed)
    acc = {k: [] for k in TERMS}
    nrm = {k: [] for k in TERMS}
    for _ in range(a.repeat):
        ridx = rng.choice(len(rwin), a.batch, replace=False)
        rf, rwd, rtg = real_targets(rwin.batch(ridx), dev)
        sw = real_sample_weights(rtg["p_observed"], "none", a.smps_boost)
        parts = crit.unlabeled(m(rf, rwd), rtg, w_cons=a.w_cons, w_harm=a.w_harm,
                               w_hedge=a.w_hedge, w_over=0.0, w_pref=a.w_pref,
                               w_sb=a.w_standby, sample_w=sw)
        sidx = np.sort(rng.choice(len(cache), a.batch, replace=False))
        sb_ = tuple(torch.from_numpy(x) for x in cache.batch(sidx))
        sf, swd, stg = to_targets(sb_, dev)
        synth = crit(m(sf, swd), stg)["total"]
        wmap = {"cons": a.w_cons, "harm": a.w_harm, "hedge": a.w_hedge,
                "pref": a.w_pref, "sb": a.w_standby, "synth": a.lam}
        for name in TERMS:
            v = synth if name == "synth" else parts.get(name)
            if v is None or wmap[name] <= 0 or not v.requires_grad:
                continue
            g = flat_grad(wmap[name] * v, params)
            acc[name].append(float(-(g @ gSn)))
            nrm[name].append(float(g.norm()))

    if a.by_subset:
        # ── 창 묶음별 압력 (13.61.2) ──────────────────────────────────────
        # 압력이 판정 창 자신에서 오는지, 다른 자리·구성에서 수입되는지 가른다.
        from src.preprocessing.file_registry import site_of
        ev2 = load_events()
        groups = {}
        for i, stem in enumerate(rwin.stem):
            groups.setdefault(str(stem), []).append(i)
        res_of = {}
        for stem, idxs in groups.items():
            n_cyc = int(ev2[stem]["cycles"])
            on, _sc = build_on_off_truth(stem, apps, n_cyc, ev2)
            t = np.asarray(rwin.target_cycle, int).clip(0, n_cyc - 1)
            res_of[stem] = on[t][:, jb].any(1)
        buckets = {}
        for stem, idxs in groups.items():
            st = site_of(stem)
            r = res_of[stem]
            for nm, sel in (("SMPS 만", ~r[idxs]), ("저항 있음", r[idxs])):
                k = np.asarray(idxs)[sel]
                if len(k) >= 64:
                    buckets.setdefault(f"자리 {st} · {nm}", []).extend(k.tolist())
        print(f"  {'창 묶음':22s}{'창':>7s}{'창당 평균':>14s}{'최소':>11s}"
              f"{'최대':>11s}{'|g|':>10s}   판정   (창당 x1000)")
        for nm in sorted(buckets):
            k = np.asarray(buckets[nm])
            # ⚠ 배치 하나로 재면 잡음이 2.6배까지 난다 (13.61.2). `--repeat` 번
            #   다른 표본으로 재서 평균과 폭을 같이 낸다.
            ds, gs = [], []
            for _ in range(max(1, a.repeat)):
                take = (k if len(k) <= a.batch
                        else rng.choice(k, a.batch, replace=False))
                rf, rwd, rtg = real_targets(rwin.batch(np.sort(take)), dev)
                sw = real_sample_weights(rtg["p_observed"], "none", a.smps_boost)
                pp = crit.unlabeled(m(rf, rwd), rtg, w_cons=a.w_cons, w_harm=a.w_harm,
                                    w_hedge=a.w_hedge, w_over=0.0, w_pref=a.w_pref,
                                    w_sb=a.w_standby, sample_w=sw)
                g = flat_grad(a.w_harm * pp["harm"], params)
                ds.append(float(-(g @ gSn)) / len(take) * 1000)
                gs.append(float(g.norm()))
            d, lo, hi = float(np.mean(ds)), min(ds), max(ds)
            print(f"  {nm:22s}{len(k):7d}{d:14.4g}{lo:11.4g}{hi:11.4g}"
                  f"{float(np.mean(gs)):10.4g}   "
                  + ("올린다" if lo > 0 else
                     "**끌어내린다**" if hi < 0 else "부호 섞임"))
        print("\n  '창당' 은 묶음 크기로 나눈 것 (x1000). 학습 배치는 묶음을 섞으므로")
        print("  실제 압력은 (창당) x (그 묶음이 배치에서 차지하는 창 수) 다.")
        return

    print(f"  {'항':10s}{'⟨−g_T, ĝ_S⟩':>14s}{'|g_T|':>12s}{'코사인':>10s}   판정")
    tot = 0.0
    for name in TERMS:
        if not acc[name]:
            continue
        d, n = float(np.mean(acc[name])), float(np.mean(nrm[name]))
        tot += d
        cos = d / max(n, 1e-12)
        verdict = "프로젝터를 올린다" if d > 0 else "**끌어내린다**"
        print(f"  {name:10s}{d:14.5g}{n:12.4g}{cos:10.3f}   {verdict}")
    print(f"  {'합계':10s}{tot:14.5g}")
    print("\n  합계가 음수면 그 배치에서 학습 한 걸음이 프로젝터를 내린다.")
    print("  ⚠ 코사인이 커도 |g_T| 가 작으면 못 움직인다 — 둘을 같이 읽을 것.")


if __name__ == "__main__":
    main()
