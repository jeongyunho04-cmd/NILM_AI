# -*- coding: utf-8 -*-
"""**연결점 여섯이 각각 `p_raw` 에 무엇을 하나** (14.151).

사용자: *"각 항의 영향과 부작용을 시뮬레이션해서 혹시 p_raw 의 예측정확도에
악영향을 끼칠 만한 요소가 있는지 검토해줘"*

14.150 이 센 여섯:

```
  (1) 값      power = σ(on)·p_raw
  (2) 경사    ∂power/∂p_raw = σ(on)        **흡수 상태**
  (3) 경사    ∂power/∂on    = σ'·p_raw
  (4) 프라이어 on_logit += logsigmoid(κ·gap)
  (5) 공유머리 on·state·p_states 가 같은 hd(z)
  (6) L_harm(0.1) 도 out["power"] 를 쓴다
```

**추론하지 않고 잰다.**

```
  [1] 자유도   `p_raw` 가 창마다 움직이나 — (1)의 보상 목표 `y/g` 를 따라갈 수 있나
  [2] 흡수     참ON 창의 `g` 분포 = `p_raw` 가 받는 **경사 이득**. 참OFF 는 누수
  [3] 물렁함   `∂L/∂on_logit` 대 `∂L/∂ln p_raw` — `L_power` 는 이 비가 **(1−g) 항등식**이다
  [4] 프라이어 κ 를 8 과 0 으로 두고 `g`·`power` 를 견준다
  [5] 공유머리 같은 손실을 p_raw 경유 / 게이트 경유 / 검출BCE 로 갈라 `∂/∂θ`
  [6] 항별     항마다 `∂/∂p_raw` 와 `∂/∂on_logit` 의 몫
```

기본은 실측 5파일이다 — 다만 실측에는 기기별 참전력이 없어 **혼자켜짐·전부꺼짐 창만**
유효하고 표본이 작다. `--holdout` 을 주면 **합성 홀드아웃**에서 잰다. 결합이 일을 하는
곳은 학습이고 학습은 합성에서 도니까 **그쪽이 본 자리**다.

    python -X utf8 src/run_diag_gatepath.py --ckpt results/cnn_pcap_s0.pt
    python -X utf8 -m src.run_diag_gatepath --holdout processed_data/holdout60_v32h \
        --ckpt results/cnn_pcap_s0.pt results/cnn_hv2_s0.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.model.inputs import V_CENTER  # noqa: E402
from src.model.net import P_CH_FINE, P_CH_WIDE  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

SOLO_MIN = 20          #: 이보다 적으면 그 기기는 건너뛴다
POWER_FLOOR_W = 5.0    #: `run_score_seq` 와 같은 하한
ONF = 5.0              #: 이 와트 밑은 "켜짐" 으로 안 센다


# ───────────────────────────────────────────────────────────── 모집단
def populations(cache, apps):
    """(y, y_power, 창유효, 파일). 실측은 **혼자켜짐/전부꺼짐 창만** 유효하다."""
    Y, TRUE, STEM = [], [], []
    for stem, d in cache.items():
        Y.append(d["y"].astype(np.int8))
        TRUE.append(d["p_obs"].astype(np.float64) - float(d["p_base"]))
        STEM += [stem] * len(d["t"])
    Y = np.concatenate(Y)
    TRUE = np.concatenate(TRUE)
    n = Y.sum(1)
    wvalid = ((n == 1) & (TRUE > POWER_FLOOR_W)) | (n == 0)
    YP = np.where(Y > 0, np.maximum(TRUE, 0.0)[:, None], 0.0)
    return Y, YP, wvalid, np.array(STEM)


def holdout_cache(path, apps, n_max):
    """합성 홀드아웃을 `cache` 모양으로 — **학습이 실제로 도는 분포**다."""
    import json
    from pathlib import Path

    from src.model.inputs import V_SPAN, build_inputs
    d = Path(path)
    ha = json.loads((d / "meta.json").read_text(encoding="utf-8"))["appliances"]
    X = np.load(d / "X.npy", mmap_mode="r")
    n = min(n_max, len(X))
    raw = np.asarray(X[:n])
    ti = raw.shape[-1] - 60
    obs = np.stack([raw[:, 0:15, ti], raw[:, 15:30, ti]], -1).astype(np.float32)
    fine, wide = build_inputs(raw)
    col = [ha.index(x) for x in apps]

    def L(k):
        return np.asarray(np.load(d / (k + ".npy"), mmap_mode="r"))[:n]

    Y = L("y_on")[:, col].astype(np.int8)
    YP = L("y_power")[:, col].astype(np.float64)
    ST = L("y_state")[:, col].astype(np.int64)
    v = fine[:, 25].mean(-1) * V_SPAN + V_CENTER
    cache = {"synth": dict(fine=fine, wide=wide, obs_harm=obs,
                           p_noise=L("p_noise").astype(np.float32),
                           v_obs=v.astype(np.float64), y=Y,
                           t=np.arange(n), p_obs=L("p_observed").astype(np.float32),
                           p_base=0.0, present=np.ones(len(apps), bool))}
    return cache, Y, YP, np.ones(n, bool), ST


def forward_all(m, cache, dev, kappa=None):
    """창마다 (p_raw, on_logit, 게이트, 프라이어항, power). `kappa` 를 주면 갈아 끼운다."""
    keep = float(m.prior_kappa)
    if kappa is not None:
        m.prior_kappa = float(kappa)
    acc = {k: [] for k in ("p_raw", "on_logit", "power", "prior", "mix0")}
    with torch.no_grad():
        for stem, d in cache.items():
            n = len(d["t"])
            for i in range(0, n, 256):
                f = torch.from_numpy(np.ascontiguousarray(d["fine"][i:i + 256])).to(dev)
                w = torch.from_numpy(np.ascontiguousarray(d["wide"][i:i + 256])).to(dev)
                o = m(f, w)
                acc["p_raw"].append(o["power_raw"].float().cpu().numpy())
                acc["mix0"].append(o["power_mix"].float()[..., 0].cpu().numpy())
                acc["on_logit"].append(o["on_logit"].float().cpu().numpy())
                acc["power"].append(o["power"].float().cpu().numpy())
                if float(m.prior_kappa) > 0:
                    pm = torch.maximum(f[:, P_CH_FINE].amax(-1), w[:, P_CH_WIDE].amax(-1))
                    gap = pm[:, None] - m.on_threshold_asinh[None]
                    acc["prior"].append(
                        F.logsigmoid(float(m.prior_kappa) * gap).float().cpu().numpy())
                else:
                    acc["prior"].append(np.zeros_like(acc["on_logit"][-1]))
    m.prior_kappa = keep
    R = {k: np.concatenate(v) for k, v in acc.items()}
    R["gate"] = 1.0 / (1.0 + np.exp(-R["on_logit"]))
    #: **문** = 출력을 낮추는 곱셈 인자. 기본판은 게이트, 완전분해판은 `1 − mix_0`
    #: (`power = Σ_{s≥1} mix_s·p_s = (1−mix_0)·(켜짐 상태들의 가중평균)` 이라 정확히 같은 자리다)
    R["door"] = ((1.0 - R["mix0"]) if bool(getattr(m, "gate_free_power", False))
                 else R["gate"])
    R["p_on"] = R["power"] / np.clip(R["door"], 1e-9, None)
    return R


def _cv(x):
    x = np.asarray(x, float)
    mu = x.mean()
    return float(x.std() / abs(mu)) if abs(mu) > 1e-9 else float("nan")


# ───────────────────────────────────────────────────── 손실 (학습과 같게)
def build_loss(apps, dev, w_harm=0.1, w_z=0.3, background=True):
    from src.model.companion import standby_operating_signatures
    from src.model.losses import (LossWeights, NILMLoss, PHASE_COHERENT_EVEN,
                                  build_state_scales)
    from src.model.net import (harmonic_scales, harmonic_signature_vref,
                               harmonic_signatures, harmonic_signatures_by_state,
                               noise_signature, standby_signatures)
    from src.run_baseline import S_I
    from src.run_train_cnn import _vnorm_exp
    from src.synthesis.genopts import build_synthesizer, resolve
    from src.synthesis.sp_curves import background_signature
    from src.synthesis.synthesizer import SESSION_PLUGGED_APPS

    pool = build_synthesizer(resolve("v32"), "processed_data/npz", "train").pool
    sig = harmonic_signatures(pool, apps)
    sb_sig = standby_signatures(pool, apps)
    sb_op, _pw, sb_used = standby_operating_signatures(
        pool, apps, only=SESSION_PLUGGED_APPS)
    for x in sb_used:
        sb_sig[apps.index(x)] = sb_op[apps.index(x)]
    nz = noise_signature(pool)
    if background:
        nz = nz + background_signature()          # 캐시가 상시 배경을 넣는다 (12.166)
    vref, vref_st = harmonic_signature_vref(pool, apps)
    sig_state = harmonic_signatures_by_state(pool, apps)[0]
    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz),
        harm_scale=torch.from_numpy(harmonic_scales(pool, apps)),
        harm_vnorm_frac=1.0,
        harm_vnorm_vref=torch.from_numpy(vref),
        harm_vnorm_vref_state=torch.from_numpy(vref_st),
        signatures_state=torch.from_numpy(sig_state),
        harm_even_magnitude=True,
        harm_sig_vnorm=True,
        harm_vnorm_exp=_vnorm_exp(apps, "RESISTIVE"),
        even_coherent=torch.tensor(
            [1.0 if x in PHASE_COHERENT_EVEN else 0.0 for x in apps],
            dtype=torch.float32),
        smps_group=[apps.index(x) for x in
                    ("beam_projector", "laptop_charger", "minipc") if x in apps],
        weights=LossWeights(harm=w_harm, cons=0.0, over=0.0, z=w_z),
        s_state=build_state_scales(apps, [S_I[x] for x in apps]),
    ).to(dev)
    del pool
    return crit


def state_labels(apps, Y, YP):
    """실측에 없는 `y_state` 를 만든다 — 켜진 기기는 `S_STATE` 에서 **가장 가까운 상태**."""
    from src.model.losses import S_STATE
    st = np.zeros(Y.shape, np.int64)
    for k, a in enumerate(apps):
        tab = S_STATE.get(a)
        on = Y[:, k] > 0
        if not on.any():
            continue
        if not tab:
            st[on, k] = 1
            continue
        sids = np.array(sorted(tab))
        pw = np.array([tab[s] for s in sids])
        st[on, k] = sids[np.argmin(np.abs(YP[on, k][:, None] - pw[None]), 1)]
    return st


# ───────────────────────────────────────────────────────────── 경사
def _gather(cache, order, sel):
    """전역 색인 `sel` 을 파일별로 갈라 (fine, wide, obs_harm, p_noise, v_obs)."""
    F_, W_, H_, N_, V_ = [], [], [], [], []
    for stem, lo, hi in order:
        s = sel[(sel >= lo) & (sel < hi)] - lo
        if not len(s):
            continue
        d = cache[stem]
        F_.append(d["fine"][s])
        W_.append(d["wide"][s])
        H_.append(d["obs_harm"][s])
        N_.append(d["p_noise"][s])
        V_.append(np.asarray(d["v_obs"], np.float64)[s])
    return (np.concatenate(F_), np.concatenate(W_), np.concatenate(H_),
            np.concatenate(N_), np.concatenate(V_))


def _tgt(Yc, YPc, STc, Hc, Nc, Vc, dev):
    t = {"y_power": torch.from_numpy(YPc.astype(np.float32)).to(dev),
         "y_on": torch.from_numpy(Yc.astype(np.float32)).to(dev),
         "y_plugged": torch.from_numpy((Yc > 0).astype(np.float32)).to(dev),
         "y_standby": torch.zeros(Yc.shape, dtype=torch.float32, device=dev),
         "obs_harm": torch.from_numpy(np.ascontiguousarray(Hc)).to(dev),
         "p_noise": torch.from_numpy(np.ascontiguousarray(Nc)).to(dev),
         "vrel": torch.from_numpy(Vc / V_CENTER).float().to(dev)}
    if STc is not None:
        t["y_state"] = torch.from_numpy(STc).to(dev)
    return t


def out_level(m, crit, cache, order, sel, Y, YP, ST, dev, w_harm, chunk=96):
    """**출력 수준** — 항마다 `∂L/∂p_raw`·`∂L/∂on_logit` 를 창별로."""
    terms = ("power", "harm", "on")
    gp = {k: [] for k in terms}
    ga = {k: [] for k in terms}
    keep = []
    for i in range(0, len(sel), chunk):
        s = np.sort(sel[i:i + chunk])
        f, w, h, nz, v = _gather(cache, order, s)
        tg = _tgt(Y[s], YP[s], ST[s], h, nz, v, dev)
        with torch.no_grad():
            o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                  torch.from_numpy(np.ascontiguousarray(w)).to(dev))
        pr = o["power_raw"].detach().float().requires_grad_(True)
        al = o["on_logit"].detach().float().requires_grad_(True)
        d2 = {k: (v2.detach().float() if torch.is_tensor(v2) and v2.is_floating_point()
                  else v2) for k, v2 in o.items()}
        d2["power_raw"], d2["on_logit"] = pr, al
        d2["power"] = torch.sigmoid(al) * pr
        parts = crit(d2, tg)
        for nm in terms:
            if nm not in parts or not parts[nm].requires_grad:
                gp[nm].append(np.zeros(tuple(pr.shape), np.float64))
                ga[nm].append(np.zeros(tuple(pr.shape), np.float64))
                continue
            a_, b_ = torch.autograd.grad(parts[nm], [pr, al], retain_graph=True,
                                         allow_unused=True)
            gp[nm].append((torch.zeros_like(pr) if a_ is None else a_)
                          .double().cpu().numpy())
            ga[nm].append((torch.zeros_like(al) if b_ is None else b_)
                          .double().cpu().numpy())
        keep.append(s)
    return ({k: np.concatenate(v) for k, v in gp.items()},
            {k: np.concatenate(v) for k, v in ga.items()},
            np.concatenate(keep))


def param_level(m, crit, cache, order, sel, Y, YP, ST, dev, w_harm, chunk=48):
    """**파라미터 수준** — 같은 손실을 `p_raw` 경유 / 게이트 경유 / 검출 BCE 로 갈라
    `∂/∂θ` 를 모은다. (5) 공유 머리의 간섭은 여기서만 보인다."""
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    params = [p for n, p in m.named_parameters() if p.requires_grad]
    TAGS = ("praw", "gate", "on", "praw_pw", "gate_pw", "praw_hm", "gate_hm")
    acc = {t: [torch.zeros_like(p) for p in params] for t in TAGS}
    N = len(sel)
    for i in range(0, N, chunk):
        s = np.sort(sel[i:i + chunk])
        f, w, h, nz, v = _gather(cache, order, s)
        tg = _tgt(Y[s], YP[s], ST[s], h, nz, v, dev)
        o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
              torch.from_numpy(np.ascontiguousarray(w)).to(dev))
        wt = len(s) / float(N)
        for tagx in ("praw", "gate", "on"):
            d2 = {k: (v2.detach() if torch.is_tensor(v2) else v2)
                  for k, v2 in o.items()}
            if tagx == "praw":
                d2["power_raw"] = o["power_raw"]
                d2["power"] = torch.sigmoid(o["on_logit"].detach()) * o["power_raw"]
            else:
                d2["on_logit"] = o["on_logit"]
                d2["power"] = torch.sigmoid(o["on_logit"]) * o["power_raw"].detach()
            parts = crit(d2, tg)
            if tagx == "on":
                Ls = [("on", 0.3 * parts["on"])]
            else:
                Ls = [(tagx, 1.0 * parts["power"] + w_harm * parts["harm"]),
                      (tagx + "_pw", 1.0 * parts["power"]),
                      (tagx + "_hm", w_harm * parts["harm"])]
            for nmx, L in Ls:
                g = torch.autograd.grad(L, params, retain_graph=True,
                                        allow_unused=True)
                for j, gj in enumerate(g):
                    if gj is not None:
                        acc[nmx][j] += wt * gj.detach()
        del o
    return names, acc


# ───────────────────────────────────────────────────────────── 본문
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_pcap_s0.pt"])
    ap.add_argument("--holdout", default="", help="주면 **합성 홀드아웃**에서 잰다")
    ap.add_argument("--n", type=int, default=4096, help="홀드아웃에서 쓸 창 수")
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--w-harm", type=float, default=0.1)
    ap.add_argument("--n-grad", type=int, default=768, help="경사를 잴 창 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--even-median", type=int, default=0,
                    help="짝수차 중앙값을 **강제**한다 (14.160) — 분포 밖 시험용. 안 주면 체크포인트를 따라간다")
    a = ap.parse_args()
    from src.run_gate_check import sync_even_median
    sync_even_median(a.ckpt, a.even_median)   #: 창을 짓기 **전에** 전역을 맞춘다

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu",
                           weights_only=False)["appliances"])
    if a.holdout:
        cache, Y, YP, wvalid, ST = holdout_cache(a.holdout, apps, a.n)
        src = "합성 홀드아웃 %s" % a.holdout
    else:
        from src.run_train_seq import real_windows
        cache = real_windows(apps, a.grid_s, dev)
        Y, YP, wvalid, _ = populations(cache, apps)
        ST = state_labels(apps, Y, YP)
        src = "실측 5파일 (격자 %.1fs, **혼자켜짐/전부꺼짐 창만**)" % a.grid_s
    order, lo = [], 0
    for stem, d in cache.items():
        order.append((stem, lo, lo + len(d["t"])))
        lo += len(d["t"])
    alloff = (Y.sum(1) == 0) & wvalid
    print("%s · 창 %d · 유효 %d · 전부꺼짐 %d · 장치 %s"
          % (src, len(Y), int(wvalid.sum()), int(alloff.sum()), dev))
    if not a.holdout:
        print("⚠ `y_state` 는 실측에 없다 — `S_STATE` 최근접으로 붙였다")

    crit = build_loss(apps, dev, w_harm=a.w_harm, background=not a.holdout)
    rng = np.random.default_rng(a.seed)
    pool_g = np.flatnonzero(wvalid)
    gsel = rng.choice(pool_g, size=min(a.n_grad, len(pool_g)), replace=False)

    for ck in a.ckpt:
        tag = ck.split("/")[-1].replace(".pt", "")
        m = load_model(ck, dev)[0]
        m.eval()
        R = forward_all(m, cache, dev)
        print(chr(10) + "=" * 100)
        print("■ %s   (prior_kappa %.3g · gate_free %s)"
              % (tag, float(m.prior_kappa), bool(getattr(m, "gate_free_power", False))))
        print("=" * 100)
        SEL = {}
        for k, x in enumerate(apps):
            sel = wvalid & (Y[:, k] > 0) & (YP[:, k] > ONF)
            if sel.sum() >= SOLO_MIN:
                SEL[k] = sel

        # [1] 자유도 — (1) 값 경로가 표현 가능한가
        print(chr(10) + "[1] **자유도** — `p_raw` 가 창마다 움직이나. (1)의 보상 목표는 `y/g` 다")
        print("    보상 기울기 = log(p_raw/y) 를 −log g 로 회귀. **1 이면 완전 보상 · 0 이면 없음**")
        print("  %-18s %6s %9s %8s %8s %9s %10s %10s %10s"
              % ("기기", "창", "CV(p_on)", "CV(문)", "CV(y)", "CV(y/문)",
                 "보상기울기", "log(p_on/y)", "log(out/y)"))
        for k, sel in SEL.items():
            g = np.clip(R["door"][sel, k], 1e-6, 1 - 1e-9)
            pr = np.maximum(R["p_on"][sel, k], 1e-6)
            ou = np.maximum(R["power"][sel, k], 1e-6)
            y = YP[sel, k]
            lx, ly = -np.log(g), np.log(pr / y)
            sl = float(np.polyfit(lx, ly, 1)[0]) if lx.std() > 1e-6 else float("nan")
            print("  %-18s %6d %9.3f %8.3f %8.3f %9.3f %10.3f %10.4f %10.4f"
                  % (apps[k], int(sel.sum()), _cv(pr), _cv(g), _cv(y), _cv(y / g),
                     sl, float(ly.mean()), float(np.log(ou / y).mean())))

        # [2] 흡수
        print(chr(10) + "[2] **문** — 기본판은 σ(on) · 완전분해판은 (1−mix_0). 지렛대가 같은 자리다")
        print("  %-18s %8s %8s %9s %8s %8s  | %10s %9s"
              % ("기기", "평균g", "중앙g", "조화평균", "g<0.5", "g<0.1",
                 "꺼짐평균g", "꺼짐p99"))
        for k, sel in SEL.items():
            g = R["door"][sel, k]
            go = R["door"][alloff, k] if alloff.sum() else np.zeros(1)
            print("  %-18s %8.3f %8.3f %9.3f %7.1f%% %7.1f%%  | %10.4f %9.4f"
                  % (apps[k], g.mean(), np.median(g),
                     float(1.0 / np.mean(1.0 / np.maximum(g, 1e-6))),
                     100 * (g < 0.5).mean(), 100 * (g < 0.1).mean(),
                     go.mean(), np.percentile(go, 99)))

        # [4] 프라이어
        print(chr(10) + "[4] **프라이어** — `on_logit += logsigmoid(κ·gap)`. κ=0 과 견준다")
        R0 = forward_all(m, cache, dev, kappa=0.0) if float(m.prior_kappa) > 0 else R
        print("  %-18s %11s %11s %10s %11s %11s"
              % ("기기", "켜짐항<-.01", "켜짐중앙항", "Δg(켜짐)", "Δpower/참",
                 "꺼짐항<-.01"))
        for k, sel in SEL.items():
            pv = R["prior"][sel, k]
            dg = R["gate"][sel, k] - R0["gate"][sel, k]
            dp = (R["power"][sel, k] - R0["power"][sel, k]) / np.maximum(YP[sel, k], 1e-6)
            off = R["prior"][alloff, k] if alloff.sum() else np.zeros(1)
            print("  %-18s %10.1f%% %11.4f %10.4f %11.4f %10.1f%%"
                  % (apps[k], 100 * (pv < -0.01).mean(), float(np.median(pv)),
                     float(np.mean(dg)), float(np.mean(dp)),
                     100 * (off < -0.01).mean()))

        # [3]+[6] 출력 수준
        gp, ga, ks = out_level(m, crit, cache, order, gsel, Y, YP, ST, dev, a.w_harm)
        onm = (Y[ks] > 0) & (YP[ks] > ONF)
        pr_s, g_s = R["p_raw"][ks], R["gate"][ks]
        W = {"power": 1.0, "harm": a.w_harm, "on": 0.3}
        print(chr(10) + "[3]+[6] **출력 수준 경사** (켜진 칸 %d개, 가중 반영)"
              % int(onm.sum()))
        print("  %-8s %15s %15s %11s %11s"
              % ("항", "‖∂L/∂ln p_raw‖", "‖∂L/∂on_logit‖", "비", "몫(p_raw)"))
        tot_p = sum(float(np.linalg.norm((W[n] * gp[n] * pr_s)[onm])) ** 2
                    for n in ("power", "harm"))
        for nm in ("power", "harm", "on"):
            a1 = (W[nm] * gp[nm] * pr_s)[onm]
            a2 = (W[nm] * ga[nm])[onm]
            n1, n2 = float(np.linalg.norm(a1)), float(np.linalg.norm(a2))
            sh = 100 * n1 ** 2 / max(tot_p, 1e-30) if nm != "on" else float("nan")
            print("  %-8s %15.6f %15.6f %11.4f %10.1f%%"
                  % (nm, n1, n2, n2 / max(n1, 1e-30), sh))
        print("  항등식 점검 — `L_power` 는 ∂/∂on = (1−g)·∂/∂ln p_raw 여야 한다")
        num = (W["power"] * ga["power"])[onm]
        den = (W["power"] * gp["power"] * pr_s)[onm]
        ok = np.abs(den) > 1e-12
        if ok.any():
            rr = num[ok] / den[ok]
            print("    창별 비의 중앙 %.4f  대  중앙 (1−g) %.4f   (최대 절대차 %.2e)"
                  % (float(np.median(rr)), float(np.median(1 - g_s[onm][ok])),
                     float(np.max(np.abs(rr - (1 - g_s[onm][ok]))))))

        # [5] 파라미터 수준
        names, acc = param_level(m, crit, cache, order, gsel, Y, YP, ST, dev, a.w_harm)

        def _flat(t, mask=None):
            v = [g.reshape(-1) for n, g in zip(names, acc[t])
                 if mask is None or mask(n)]
            return torch.cat(v) if v else torch.zeros(1, device=dev)

        def _cos(u, v):
            d = u.norm() * v.norm()
            return float((u @ v) / d) if float(d) > 0 else float("nan")

        print(chr(10) + "[5] **파라미터 수준** — 같은 손실을 세 갈래로 갈라 `∂/∂θ`")
        print("  %-14s %12s %12s %12s %11s %11s %11s"
              % ("무리", "‖p_raw경유‖", "‖게이트경유‖", "‖검출BCE‖",
                 "게/p 배", "cos(p,게)", "cos(게,BCE)"))
        for gname, msk in (("전체", None),
                           ("세밀 fine.*", lambda n: n.startswith("fine.")),
                           ("광역 wide.*", lambda n: n.startswith("wide.")),
                           ("몸통 trunk.*", lambda n: n.startswith("trunk.")),
                           ("머리 heads.*", lambda n: n.startswith("heads."))):
            a1, a2, a3 = _flat("praw", msk), _flat("gate", msk), _flat("on", msk)
            print("  %-14s %12.6f %12.6f %12.6f %11.2f %11.4f %11.4f"
                  % (gname, float(a1.norm()), float(a2.norm()), float(a3.norm()),
                     float(a2.norm()) / max(float(a1.norm()), 1e-30),
                     _cos(a1, a2), _cos(a2, a3)))
        print("  (게이트 경유가 p_raw 경유보다 크면 **전력 손실이 게이트를 진폭 손잡이로 쓴다**)")
        print("  항별 (전체 θ) — 누가 게이트를 미나")
        print("    %-10s %12s %12s %10s" % ("항", "p_raw경유", "게이트경유", "게/p 배"))
        for nmx, lab in (("pw", "power(1.0)"), ("hm", "harm(%.2g)" % a.w_harm)):
            u, v2 = _flat("praw_" + nmx), _flat("gate_" + nmx)
            print("    %-10s %12.6f %12.6f %10.2f"
                  % (lab, float(u.norm()), float(v2.norm()),
                     float(v2.norm()) / max(float(u.norm()), 1e-30)))

        # [7] 게이트를 떼면 꺼진 창이 p_raw 를 얼마나 세게 미나 (13.11 위험)
        offm = (Y[ks] == 0)
        gg = np.clip(g_s[offm], 1e-12, 1.0)
        print(chr(10) + "[7] **떼면 얼마나 세지나** — 꺼진 칸에서 `∂power/∂p_raw = g` 가 1 이 된다")
        print("    꺼진 칸 %d개 · g 평균 %.3e · 중앙 %.3e · p99 %.3e"
              % (int(offm.sum()), float(gg.mean()), float(np.median(gg)),
                 float(np.percentile(gg, 99))))
        print("    -> `p_raw` 가 받는 OFF 당김이 **평균 %.0f배**로 커진다 (13.11 의 그 병)"
              % float(1.0 / max(gg.mean(), 1e-12)))
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
