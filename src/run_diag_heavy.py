# -*- coding: utf-8 -*-
"""오븐/핫플 잔차가 왜 터지나 — 후보 셋을 **갈라서** 잰다 (13.84.74 교락 점검).

13.84.74 는 "2단계가 **오븐/핫플에서만** 잔차를 터뜨린다" 고 단정했다 (59.3 -> 104.6W).
그런데 그 뒤 13.87 [4] 가 **겹칠수록 초선형으로 는다**는 것을 쟀다 (1대 12W -> 6대 125W).
오븐/핫플이 켜진 창은 **동시에 켜진 기기도 많다** — 둘이 안 갈려 있다.

후보 셋:
  (가) 겹침 교락      오븐/핫플이라서가 아니라 **기기가 많아서**다
  (나) 절대 W 눈금    1357W 오븐의 10% 는 136W, 25W 미니PC 의 10% 는 2.5W.
                     같은 상대오차여도 절대 잔차가 크다
  (다) 듀티 양봉      오븐·핫플만 `is_on=1` 구간의 전력이 **두 봉우리**다
                     (오븐 히터 1357W ↔ 팬/조명 16.8W · 핫플 릴레이 550W ↔ 0W).
                     매끄러운 예측은 그 안에서 늘 틀린다

가르는 법:
  (가) **같은 동시개수 안에서** 오븐/핫플 유무로 나눈다. 차이가 남으면 교락이 아니다
  (나) 절대 잔차와 **관측 대비 상대 잔차**를 같이 찍는다
  (다) 오븐/핫플 켜짐 구간을 **관측 총전력**으로 고/저로 갈라 본다. 듀티가 범인이면
       한쪽에 몰리거나 전환 근처에 몰린다

    python -X utf8 src/run_diag_heavy.py
    python -X utf8 src/run_diag_heavy.py --ckpt results/cnn_v37.pt results/seq_v36p_mask.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.run_train_seq import real_windows

HEAVY = ("oven", "hotplate")


def predict(path, apps, cache, dev):
    """체크포인트 하나를 실측 전 파일에 흘린다 -> {파일: (power (T,K),)}."""
    from src.run_gate_check import load_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "heads" in ck:                      # 시퀀스 판
        m = load_model(ck.get("ref", "results/cnn_v37.pt"), dev, weights=False,
                       mask=not bool(ck.get("no_mask", False)),
                       proj_from={k: ck[k] for k in
                                  ("proj", "proj_cap", "proj_floor", "proj_resp")
                                  if k in ck})[0]
        m.load_state_dict(ck["model"])
    else:
        m = load_model(path, dev)[0]
    m.eval()
    out = {}
    with torch.no_grad():
        for stem, d in cache.items():
            P = []
            for i in range(0, len(d["t"]), 512):
                P.append(m(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                           torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                         ["power"].float().cpu().numpy())
            out[stem] = np.concatenate(P)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=["results/cnn_v37.pt", "results/seq_v36p_mask.pt"])
    ap.add_argument("--grid-s", type=float, default=2.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    APPS = torch.load(a.ckpt[0], map_location="cpu", weights_only=False)["appliances"]
    apps = list(APPS)
    cache = real_windows(apps, a.grid_s, dev)
    hk = [apps.index(x) for x in HEAVY if x in apps]

    print("오븐/핫플 잔차의 원인 — 후보 셋을 갈라서 잰다", flush=True)
    print("=" * 78)

    all_r, all_n, all_h, all_o = {}, None, None, None
    for path in a.ckpt:
        pw = predict(path, apps, cache, dev)
        R, N, H, O = [], [], [], []
        for stem, d in cache.items():
            base = float(d["p_base"])
            if not np.isfinite(base):
                continue
            y = d["y"].astype(bool)
            r = pw[stem].sum(1) + base - d["p_obs"]
            heavy = np.zeros(len(r), bool)
            for k in hk:
                if d["present"][k]:
                    heavy |= y[:, k]
            R.append(r); N.append(y.sum(1)); H.append(heavy); O.append(d["p_obs"])
        all_r[path] = np.concatenate(R)
        if all_n is None:
            all_n = np.concatenate(N); all_h = np.concatenate(H); all_o = np.concatenate(O)
    n, h, obs = all_n, all_h, all_o
    tags = [p.split("/")[-1][:-3] for p in a.ckpt]

    # ── (가) 같은 동시개수 안에서 오븐/핫플 유무 ────────────────────────────
    print()
    print("(가) **같은 동시개수 안에서** 오븐/핫플 유무로 나눈다 — 겹침 교락인가")
    print("     교락이면 같은 열 안에서 두 값이 비슷해야 한다")
    for t, path in zip(tags, a.ckpt):
        r = np.abs(all_r[path])
        print("  [%s]  |잔차| 평균 W" % t)
        print("    동시개수   %s" % "".join("%8d" % k for k in range(1, 7)))
        for lab, m0 in (("오븐/핫플 있음", h), ("오븐/핫플 없음", ~h)):
            row = "    %-10s" % lab
            for k in range(1, 7):
                s = m0 & (n == k)
                row += "%8s" % ("%.1f" % r[s].mean() if s.sum() >= 15 else "-")
            print(row)
        row = "    %-10s" % "창 수(있음)"
        for k in range(1, 7):
            row += "%8d" % int((h & (n == k)).sum())
        print(row)
        row = "    %-10s" % "창 수(없음)"
        for k in range(1, 7):
            row += "%8d" % int(((~h) & (n == k)).sum())
        print(row)

    # ── (나) 절대 대 상대 ──────────────────────────────────────────────────
    print()
    print("(나) 절대 W 대 **관측 대비 상대** — 눈금 탓인가")
    print("  %-16s %10s %10s %10s %10s" % ("", "절대(있음)", "절대(없음)", "상대(있음)", "상대(없음)"))
    for t, path in zip(tags, a.ckpt):
        r = np.abs(all_r[path])
        rel = r / np.maximum(obs, 10.0)
        print("  %-16s %9.1fW %9.1fW %9.1f%% %9.1f%%"
              % (t, r[h].mean(), r[~h].mean(), 100 * np.median(rel[h]),
                 100 * np.median(rel[~h])))
    print("  ⇒ 절대는 벌어지는데 상대가 비슷하면 **눈금 탓**이다")

    # ── (다) 듀티 양봉 ────────────────────────────────────────────────────
    print()
    print("(다) 오븐/핫플 켜짐 구간을 **관측 총전력**으로 갈라 본다 — 듀티 양봉인가")
    if h.sum() >= 40:
        thr = np.median(obs[h])
        print("     그 구간 관측 중앙 %.0fW 로 가른다 (고 %d창 · 저 %d창)"
              % (thr, int((h & (obs >= thr)).sum()), int((h & (obs < thr)).sum())))
        print("  %-16s %12s %12s %12s %12s"
              % ("", "고|잔차|", "저|잔차|", "고 상대", "저 상대"))
        for t, path in zip(tags, a.ckpt):
            r = np.abs(all_r[path])
            rel = r / np.maximum(obs, 10.0)
            hi, lo = h & (obs >= thr), h & (obs < thr)
            print("  %-16s %11.1fW %11.1fW %11.1f%% %11.1f%%"
                  % (t, r[hi].mean(), r[lo].mean(),
                     100 * np.median(rel[hi]), 100 * np.median(rel[lo])))
        print("  ⇒ 듀티가 범인이면 **저전력 구간**(히터 쉼)에서 상대오차가 크게 튄다")

    # ── 부호 — 모자라나 넘치나 ────────────────────────────────────────────
    print()
    print("부호 있는 평균 잔차 (예측합 − 관측). 음수 = **덜 예측한다**")
    print("  %-16s %12s %12s" % ("", "오븐/핫플", "그 밖"))
    for t, path in zip(tags, a.ckpt):
        r = all_r[path]
        print("  %-16s %11.1fW %11.1fW" % (t, r[h].mean(), r[~h].mean()))


if __name__ == "__main__":
    main()
