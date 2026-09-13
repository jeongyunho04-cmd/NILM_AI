# -*- coding: utf-8 -*-
"""13.84.41 의 처방 ②·③ 을 **재학습 없이** 잰다 (13.84.44).

② 방출·전환 균형
   지금은 `switch_bias` 하나가 방출 세기 25배 차이(저항성 7.5~9.0 대 미니PC 0.30)를
   동시에 맡는다. 방출에 배율 α 를 걸어 다시 복호하면 그 균형을 훑을 수 있다 —
   α->∞ 는 창별(독립) 판정, α=0 은 전이 전용. 중간에 둘 다보다 나은 자리가 있는가.
   ⚠ 실측 파일마다 승자가 다르다 (13.84.41: test_1·2 는 창별, test_3·4 는 사슬).
     **전역 α 가 없다면 그것도 답이다** — 균형이 파일에 따라 움직여야 한다는 뜻이다.

③ 설명 안 된 유효전력
   방출이 평평한 이유는 고조파 지문에 정보가 없어서다(13.84.31). 전력은 **독립인 근거**다.
   `resid(t) = P관측(t) − Σ_{k≠미니PC} P̂_k(t)` 가 미니PC 참 상태를 말하는가.
   ⚠ 파일마다 상시 배경 오프셋이 다르므로(13.84.40) **파일 안에서** 잰다.

    python -X utf8 src/run_diag_balance.py [results/seq_h38_base.pt]
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.chain import ChainHeads, viterbi
from src.preprocessing import load_nilm_npz
from src.run_gate_check import load_model
from src.run_train_seq import FILES, real_windows

FS = 60
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    w = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(w / (len(pos) * len(neg)))


def main():
    ck_path = sys.argv[1] if len(sys.argv) > 1 else "results/seq_h38_base.pt"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    apps = ck["appliances"]
    model = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)))[0]
    model.load_state_dict(ck["model"]); model.eval()
    heads = ChainHeads(ck["zdim"], ck["ddim"], len(apps), hidden=ck["hidden"],
                       score_norm=int(ck.get("score_norm", 0))).to(dev)
    heads.load_state_dict(ck["heads"]); heads.eval()
    cache = real_windows(apps, ck["meta"]["grid_s"], dev)
    km = apps.index("minipc")

    # 한 번만 돌려 두고 여러 α 로 복호만 다시 한다.
    S = {}
    with torch.no_grad():
        for stem, d in cache.items():
            Z, GL, PW = [], [], []
            for i in range(0, len(d["t"]), 512):
                o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                          torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                PW.append(o["power"].float())
            em, on, off, ini = heads(torch.cat(Z)[None],
                                     torch.from_numpy(d["dfeat"]).float()[None].to(dev),
                                     torch.cat(GL)[None])
            S[stem] = dict(em=em, on=on, off=off, ini=ini, gl=torch.cat(GL),
                           pw=torch.cat(PW).cpu().numpy(), y=d["y"].astype(bool),
                           t=d["t"], present=d["present"])

    # ── ② 방출 배율 훑기 ──────────────────────────────────────────────────
    print("② 방출 배율 α 훑기 — α->∞ 는 창별, α=0 은 전이 전용")
    print("   %-7s %9s | %s" % ("α", "전체평균", "".join("  %-8s" % s for s in FILES)))
    best = {}
    with torch.no_grad():
        for al in ALPHAS:
            accs, mp = [], {}
            for stem, s in S.items():
                p = viterbi(s["em"] * al, s["on"], s["off"],
                            s["ini"] * al)[0].cpu().numpy()
                for k in np.nonzero(s["present"])[0]:
                    accs.append(float((p[:, k] == s["y"][:, k]).mean()))
                if s["present"][km]:
                    mp[stem] = float((p[:, km] == s["y"][:, km]).mean())
            m = float(np.mean(accs))
            best[al] = (m, mp)
            print("   %-7.2f %9.4f | %s"
                  % (al, m, "".join("  %-8s" % ("%.3f" % mp[s] if s in mp else "-")
                                    for s in FILES)))
    # 창별 기준선
    wac, wmp = [], {}
    for stem, s in S.items():
        w = (s["gl"] > 0).cpu().numpy()
        for k in np.nonzero(s["present"])[0]:
            wac.append(float((w[:, k] == s["y"][:, k]).mean()))
        if s["present"][km]:
            wmp[stem] = float((w[:, km] == s["y"][:, km]).mean())
    print("   %-7s %9.4f | %s"
          % ("창별", float(np.mean(wac)),
             "".join("  %-8s" % ("%.3f" % wmp[s] if s in wmp else "-") for s in FILES)))
    bm = max(best, key=lambda x: best[x][0])
    print("   -> 전체 최고 α=%.2f (%.4f) · α=1 은 %.4f · 창별 %.4f"
          % (bm, best[bm][0], best[1.0][0], float(np.mean(wac))))
    print("   미니PC 파일별 최고 α:")
    for stem in FILES:
        row = [(al, best[al][1][stem]) for al in ALPHAS if stem in best[al][1]]
        if not row:
            continue
        b = max(row, key=lambda x: x[1])
        print("      %-8s α=%-6.2f %.3f   (α=1 %.3f · 창별 %.3f)"
              % (stem, b[0], b[1], best[1.0][1][stem], wmp[stem]))

    # ── ③ 설명 안 된 유효전력 ────────────────────────────────────────────
    print("\n③ 설명 안 된 유효전력 — resid = P관측 − Σ_{k≠미니PC} P̂_k")
    print("   미니PC 참 ON 때 resid 가 더 큰가. **파일 안에서** 잰다")
    print("   %-8s %8s %10s %10s %10s %8s"
          % ("파일", "표본", "ON 중앙", "OFF 중앙", "차이(W)", "AUC"))
    ref = None
    for stem in FILES:
        s = S.get(stem)
        if s is None or not s["present"][km]:
            continue
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        P = np.asarray(r["power_features"])[:, 0]
        ti = np.clip((s["t"] * FS).astype(int), 0, len(P) - 1)
        pobs = P[ti]
        other = s["pw"].sum(1) - s["pw"][:, km]
        resid = pobs - other
        y = s["y"][:, km]
        a_, b_ = resid[y], resid[~y]
        if len(a_) < 3 or len(b_) < 3:
            print("   %-8s   (한쪽 표본 부족: ON %d · OFF %d)" % (stem, len(a_), len(b_)))
            continue
        print("   %-8s %8d %10.1f %10.1f %10.1f %8.3f"
              % (stem, len(resid), np.median(a_), np.median(b_),
                 np.median(a_) - np.median(b_), auc(a_, b_)))
    print("   참값 미니PC 는 10~13W 다 (13.84.31 ⑤). 차이가 그 자릿수면 쓸 수 있다")

    print("\n   견줌 — 지금 방출(게이트 로짓)의 같은 자리 AUC")
    print("   %-8s %8s %8s" % ("파일", "resid AUC", "방출 AUC"))
    for stem in FILES:
        s = S.get(stem)
        if s is None or not s["present"][km]:
            continue
        y = s["y"][:, km]
        g = s["gl"].cpu().numpy()[:, km]
        if y.sum() < 3 or (~y).sum() < 3:
            continue
        print("   %-8s %8s %8.3f" % (stem, "", auc(g[y], g[~y])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
