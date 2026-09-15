# -*- coding: utf-8 -*-
"""**물리 프라이어 해부** — 전이 근처 오탐의 구조적 통로를 찾는다 (14.138).

사용자: *"모델 구조 전체를 면밀히 분석해서 전이 근처에 오탐을 발생시키는 원인이
있는지 한번 더 검토해 줄 수 있어?"*

찾은 줄은 `forward` 의 이 세 줄이다.

```
  p_max    = max(fine[P].amax(전체 600), wide[P].amax(전체 120))
  gap      = p_max − asinh(MIN_ON_W x β / 100)
  on_logit = 머리 + logsigmoid(κ x gap)                  (κ=8, β=0.5)
```

여기서 **구조적으로** 걸리는 것이 셋이다.

```
  (1) `p_max` 가 **창 전체의 최대**다 — 수용영역이 사실상 ∞ 이고 미래 6.0초가
      그대로 들어간다. `--fine-time-split` 은 몸통만 쪼개고 **이 줄은 안 끊는다**
      (`fp` 는 쪼갠 `hp` 가 아니라 원시 `fine` 에서 뽑는다).
      ⇒ 전이 **2초 전** 창이 이미 전이 **후**의 전력을 본다.
  (2) 문턱이 기기의 **최소** ON 전력이다. 오븐은 `MIN_ON_W = 15.5W` 인데 그건
      **팬조명**이고 통전은 1357W 다. 문턱이 통전의 1/175 이라 통전 헛detect 에
      프라이어는 **아무 힘도 못 쓴다**.
  (3) 프라이어는 **놓아주기만** 한다 — `logsigmoid <= 0` 이라 gap>0 이면 정확히 0.
      즉 "총전력이 충분하면" 게이트를 그냥 머리에 맡긴다. 컨덕턴스 축퇴는
      정의상 총전력이 충분한 상황이라 **프라이어가 닿지 않는 병**이다.
```

무엇을 재나
-----------
```
  [A] 문턱이 무엇으로 잡혔나 (표)
  [B] 프라이어가 실제로 억누르는 창이 얼마나 되나 — 기기별
  [C] `p_max` 가 **어디서** 오나 (과거/미래) · 과거만으로 자르면 얼마나 바뀌나
  [E] 오븐 통전 헛창에서 프라이어 항 (0 일 것으로 본다 — 예측을 먼저 적는다)
      그리고 문턱을 통전 기준으로 올려도 / 과거만 봐도 안 고쳐진다는 것
```

    python -X utf8 src/run_diag_prior.py results/cnn_pcap_s0.pt
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

from src.evaluation.real_events import load_events  # noqa: E402
from src.model.inputs import wide_target_index  # noqa: E402
from src.model.net import MIN_ON_W, P_CH_FINE, P_CH_WIDE, POWER_SCALE  # noqa: E402
from src.model.realdata import dense_targets  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402
from src.model.losses import S_STATE  # noqa: E402

FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
RES4 = {"oven", "electiric_kettle", "hotplate", "hair_dryer"}
BIG3 = ("electiric_kettle", "hair_dryer", "hotplate")
NEAR_S = 20.0


def build():
    out = {}
    for s in FILES:
        rw = dense_targets(s, stride=30)
        t = np.asarray(rw.target_cycle, float) / 60.0
        iv = load_events()[s]["intervals"]
        ev = load_events()[s]["events"]

        def mask(app):
            m = np.zeros(len(t), bool)
            for t0, t1 in iv.get(app, {}).get("on", []):
                m |= (t >= t0) & (t <= t1)
            return m

        big = np.zeros(len(t), bool)
        for a in BIG3:
            big |= mask(a)
        xs = np.array(sorted(e["t_s"] for e in ev if e["appliance"] in RES4))
        nr = np.zeros(len(t), bool)
        for x in xs:
            nr |= np.abs(t - x) <= NEAR_S
        out[s] = dict(rw=rw, t=t, oven=mask("oven"), kettle=mask("electiric_kettle"),
                      big=big, near=nr)
    return out


def sweep(m, D, dev):
    """창마다 on_logit · 프라이어 항 · p_max 출처를 모은다."""
    tp = m.target_pos
    thr = m.on_threshold_asinh.detach().cpu().numpy()          # (K,)
    kap = m.prior_kappa
    acc = {k: [] for k in ("on", "st", "pri", "pri_past", "src", "pmax", "pmax_past")}
    with torch.no_grad():
        for s in FILES:
            rw = D[s]["rw"]
            for i in range(0, len(rw), 256):
                idx = np.arange(i, min(i + 256, len(rw)))
                f, w, *_ = rw.batch(idx)
                ft = torch.from_numpy(np.ascontiguousarray(f)).to(dev)
                wt = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o = m(ft, wt)
                acc["on"].append(o["on_logit"].float().cpu().numpy())
                acc["st"].append(torch.softmax(o["state"].float(), -1).cpu().numpy())
                fp, wp = f[:, P_CH_FINE], w[:, P_CH_WIDE]      # (B,600) (B,120)
                wti = wide_target_index(wp.shape[-1])
                a_f, a_w = fp.argmax(-1), wp.argmax(-1)
                mf, mw = fp.max(-1), wp.max(-1)
                pm = np.maximum(mf, mw)
                pmp = np.maximum(fp[:, :tp + 1].max(-1), wp[:, :wti + 1].max(-1))
                # p_max 가 **미래**에서 왔나 (이긴 쪽의 argmax 가 타깃보다 뒤인가)
                acc["src"].append(((mf >= mw) & (a_f > tp)) | ((mw > mf) & (a_w > wti)))
                acc["pmax"].append(pm)
                acc["pmax_past"].append(pmp)
                for key, v in (("pri", pm), ("pri_past", pmp)):
                    g = torch.from_numpy(v[:, None] - thr[None]).float()
                    acc[key].append(F.logsigmoid(kap * g).numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}



def intervene(m, D, dev, tp):
    """★ 14.116 의 **개입**을 그대로 하고 `Δon_logit` 을 머리와 프라이어로 가른다.

    개입: `fine[:, :, t+1:] = fine[:, :, t:t+1]` (미래를 타깃값으로 덮는다).
    이것은 `fp_max` 도 같이 바꾼다 — 미래에만 있던 큰 전력이 사라지면 `p_max` 가
    내려가고 프라이어가 **다시 억누르기 시작한다**. 그러면 게이트가 무너지는데,
    그건 몸통이 미래를 안 봐서가 아니라 **프라이어가 닫혀서**다.
    """
    thr = m.on_threshold_asinh.detach().cpu().numpy()
    kap = m.prior_kappa
    A = {k: [] for k in ("on0", "on1", "pri0", "pri1")}
    with torch.no_grad():
        for s in FILES:
            rw = D[s]["rw"]
            for i in range(0, len(rw), 256):
                idx = np.arange(i, min(i + 256, len(rw)))
                f, w, *_ = rw.batch(idx)
                f0 = np.ascontiguousarray(f)
                f1 = f0.copy()
                f1[:, :, tp + 1:] = f0[:, :, tp:tp + 1]
                wt = torch.from_numpy(np.ascontiguousarray(w)).to(dev)
                for tag, fx in (("0", f0), ("1", f1)):
                    ft = torch.from_numpy(fx).to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                        o = m(ft, wt)
                    A["on" + tag].append(o["on_logit"].float().cpu().numpy())
                    pm = np.maximum(fx[:, P_CH_FINE].max(-1), w[:, P_CH_WIDE].max(-1))
                    g = torch.from_numpy(pm[:, None] - thr[None]).float()
                    A["pri" + tag].append(F.logsigmoid(kap * g).numpy())
    return {k: np.concatenate(v) for k, v in A.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, apps, _ = load_model(a.ckpt, dev)
    print("물리 프라이어 해부 — `on_logit = 머리 + logsigmoid(κ(p_max − 문턱))`")
    print("  %s · κ=%.1f · β=%.2f · 장치 %s\n"
          % (a.ckpt, m.prior_kappa, float(m.prior_beta), dev))

    # ── [A] 문턱이 무엇으로 잡혔나 ────────────────────────────────────────
    print("[A] 문턱이 **최소 ON 전력**으로 잡혀 있다 — 상태가 여럿인 기기는 어긋난다")
    print("  %-17s %9s %8s %11s %11s %9s" % ("기기", "MIN_ON_W", "x β", "문턱 asinh",
                                             "S_STATE 최대", "문턱/최대"))
    for j, nm in enumerate(apps):
        smax = max(S_STATE.get(nm, {1: 0.0}).values())
        w = MIN_ON_W.get(nm, 0.0) * m.prior_beta
        bad = "  <- **어긋난다**" if smax > 0 and w < 0.2 * smax else ""
        print("  %-17s %9.1f %8.1f %11.4f %11.1f %8.3f%s"
              % (nm, MIN_ON_W.get(nm, 0.0), w, float(m.on_threshold_asinh[j]),
                 smax, (w / smax) if smax else 0.0, bad))

    D = build()
    R = sweep(m, D, dev)
    cat = lambda k: np.concatenate([D[s][k] for s in FILES])     # noqa: E731
    ov, big, near = cat("oven"), cat("big"), cat("near")
    print("\n  실측 %d창 (stride 0.5초 · 5파일)" % len(R["on"]))

    # ── [B] 프라이어가 실제로 억누르나 ───────────────────────────────────
    print("\n[B] 프라이어가 **실제로 억누르는** 창 (항 < -0.1)")
    print("  %-17s %11s %11s %11s" % ("기기", "억누름 창%", "평균 항", "최소 항"))
    for j, nm in enumerate(apps):
        p = R["pri"][:, j]
        print("  %-17s %10.2f%% %11.4f %11.3f"
              % (nm, 100 * (p < -0.1).mean(), p.mean(), p.min()))

    # ── [C] p_max 가 어디서 오나 ─────────────────────────────────────────
    fut = R["src"]
    print("\n[C] ★ `p_max` 의 출처 — **미래에서 오는 창** (수용영역 ∞ 인 줄)")
    print("  전체 %5.2f%% · 전이 ±20초 안 **%5.2f%%** · 밖 %5.2f%%"
          % (100 * fut.mean(), 100 * fut[near].mean(), 100 * fut[~near].mean()))
    d = R["pri_past"] - R["pri"]      # 과거만 쓰면 항이 얼마나 더 음수가 되나
    print("  과거만으로 자르면 프라이어 항이 **내려가는** 창 (Δ < -0.1)")
    print("  %-17s %10s %12s %16s" % ("기기", "전체%", "전이근처%", "전이근처 평균 Δ"))
    for j, nm in enumerate(apps):
        c = d[:, j] < -0.1
        print("  %-17s %9.2f%% %11.2f%% %+15.4f"
              % (nm, 100 * c.mean(), 100 * c[near].mean(), d[near, j].mean()))

    # ── [E] 오븐 통전 헛창에서 ───────────────────────────────────────────
    jo = apps.index("oven")
    M = (~ov) & big & near & (R["on"][:, jo] > 0) & (R["st"][:, jo, 2] > 0.5)
    print("\n[E] 오븐 **통전 헛창** %d개에서 프라이어가 하는 일" % M.sum())
    if M.sum():
        head = R["on"][M, jo] - R["pri"][M, jo]
        print("  프라이어 항 평균 **%+.6f** · 최소 %+.6f   (0 이면 아무 일도 안 한다)"
              % (R["pri"][M, jo].mean(), R["pri"][M, jo].min()))
        print("  머리 로짓 평균 %+.3f  ->  on_logit %+.3f" % (head.mean(), R["on"][M, jo].mean()))
        print("  `p_max` 가 미래에서 오는 창 %.1f%%" % (100 * fut[M].mean()))
        s2 = max(S_STATE["oven"].values())
        t2 = float(np.arcsinh(s2 * m.prior_beta / POWER_SCALE))
        g2 = F.logsigmoid(torch.tensor(m.prior_kappa * (R["pmax"][M] - t2))).numpy()
        print("  ⓐ 문턱을 **통전 %.0fW 기준**(asinh %.4f)으로 올려도 — 항 %+.4f · "
              "게이트 여전히 양수 **%.1f%%**"
              % (s2, t2, g2.mean(), 100 * ((head + g2) > 0).mean()))
        gp = R["pri_past"][M, jo]
        print("  ⓑ `p_max` 를 **과거만**으로 바꿔도 — 항 %+.6f · "
              "게이트 여전히 양수 **%.1f%%**" % (gp.mean(), 100 * ((head + gp) > 0).mean()))
        g3 = F.logsigmoid(torch.tensor(
            m.prior_kappa * (R["pmax_past"][M] - t2))).numpy()
        print("  ⓒ **둘 다** (통전 문턱 + 과거만) — 항 %+.4f · "
              "게이트 여전히 양수 **%.1f%%**" % (g3.mean(), 100 * ((head + g3) > 0).mean()))
    # ── [D] ★ 14.116 의 개입을 머리와 프라이어로 가른다 ────────────────────
    print("\n[D] ★ 14.116 의 개입(미래를 타깃값으로 덮기)을 **머리와 프라이어로** 가른다")
    print("  14.116 은 이 개입으로 핫플 헛게이트가 0.878 -> 0.004 인 것을 보고 **몸통이**")
    print("  미래를 본다고 읽었다. 그런데 개입은 `fp_max` 도 같이 바꾼다.")
    A = intervene(m, D, dev, m.target_pos)
    dall = A["on1"] - A["on0"]
    dpri = A["pri1"] - A["pri0"]
    dhead = dall - dpri
    print("  %-17s %12s %13s %13s %11s" % ("기기", "크게 변한 창", "Δ 평균",
                                           "그중 프라이어", "프라이어 몫"))
    for j, nm in enumerate(apps):
        c = np.abs(dall[:, j]) > 1.0
        if c.sum() == 0:
            print("  %-17s %11d  (없다)" % (nm, 0))
            continue
        share = dpri[c, j].sum() / dall[c, j].sum() if dall[c, j].sum() else 0.0
        star = "  <- **프라이어다**" if share > 0.7 else ""
        print("  %-17s %11d %13.3f %13.3f %10.1f%%%s"
              % (nm, int(c.sum()), dall[c, j].mean(), dpri[c, j].mean(),
                 100 * share, star))
    # test_5 260.0초 — 14.116 이 지목한 그 창
    off = 0
    for s_ in FILES:
        n = len(D[s_]["rw"])
        if s_ == "test_5":
            k = off + int(np.argmin(np.abs(D[s_]["t"] - 260.0)))
            jh = apps.index("hotplate")
            print("\n  14.116 이 지목한 창 — test_5 %.2f초 · 핫플" % D[s_]["t"][k - off])
            print("    게이트   %.4f  ->  **%.4f**"
                  % (1 / (1 + np.exp(-A["on0"][k, jh])), 1 / (1 + np.exp(-A["on1"][k, jh]))))
            print("    on_logit %+.3f -> %+.3f   (Δ %+.3f)"
                  % (A["on0"][k, jh], A["on1"][k, jh], dall[k, jh]))
            print("    ├ 머리     Δ %+.3f" % dhead[k, jh])
            print("    └ 프라이어 Δ **%+.3f**  (%.0f%%)"
                  % (dpri[k, jh], 100 * dpri[k, jh] / dall[k, jh] if dall[k, jh] else 0))
        off += n
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
