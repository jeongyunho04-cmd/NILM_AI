# -*- coding: utf-8 -*-
"""오븐만 유독 불안정한 까닭 — 헛게이트를 **유형별로** 가른다 (14.105).

사용자: *"test_2 250초·690초·790초 부근 모두 오븐이 오탐된 곳인데 원인을 찾아줘.
타이밍 쪽은 고칠 만큼 고친 것 같은데 자꾸 오븐만 이렇게 불안정한 이유를."*

세 자리를 손으로 들여다봤더니 **성격이 서로 달랐다.** 그러면 하나의 원인을 찾는 것은
틀린 질문이다. 먼저 유형을 세고, 유형별로 세어야 한다.

```
유형 A **축퇴**   참으로 켜진 다른 저항 기기를 모델이 못 켰고 그 와트를 오븐이 받았다
                  -> 오븐을 빼면 잔차가 크게 +  (설명 안 된 와트가 있었다)
유형 B **팬조명**  게이트만 서고 전력은 ~0 (오븐 state 1 = 16.8W)
                  -> 오븐은 **ON 이 두 자리**(16.8W / 1357W)를 가리키는 유일한 기기다
유형 C **초과**    아무도 안 빼앗고 그냥 더 얹었다 — 예측합이 관측을 넘는다
                  -> 오븐을 빼야 잔차가 0 에 가까워진다
```
그리고 **오븐이 정말 유독한지**를 다른 기기와 같은 자로 견준다 (헛게이트 비율).

⚠ `라벨 != 예측` 을 곧바로 "모델이 틀렸다" 로 읽지 않는다 — 14.85 에서 그렇게 읽어
   없는 병을 지어냈다. 여기서는 **관측 총전력**이 심판이다: 오븐을 넣고 뺀 잔차를
   둘 다 찍으므로, 오븐이 채운 자리가 실재했는지 아닌지가 자료로 갈린다.

    python -X utf8 src/run_diag_ovenfp.py results/cnn_hzwt_s0.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.run_gate_check import load_model  # noqa: E402
from src.run_train_seq import real_windows  # noqa: E402

KO = {"oven": "오븐", "electiric_kettle": "포트", "hotplate": "핫플",
      "hair_dryer": "드라이", "air_conditioner": "에어컨", "fan": "선풍",
      "beam_projector": "빔프", "laptop_charger": "충전", "minipc": "미니PC"}
RES = ("oven", "electiric_kettle", "hotplate", "hair_dryer")
W_BIG = 200.0       #: 이보다 크면 "와트를 가진" 헛게이트다 (팬·조명 16.8W 와 가른다)
GAP_W = 250.0       #: 잔차가 이만큼 남아 있으면 "설명 안 된 와트가 있었다"


def _fwd(path, cache, dev):
    m = load_model(path, dev)[0]
    m.eval()
    out = {}
    with torch.no_grad():
        for stem, d in cache.items():
            pw, gl, st, sb = [], [], [], []
            for i in range(0, len(d["t"]), 512):
                o = m(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                      torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                pw.append(o["power"].float().cpu()); gl.append(o["on_logit"].float().cpu())
                st.append(o["state"].float().cpu()); sb.append(o["standby"].float().cpu())
            out[stem] = tuple(torch.cat(x).numpy() for x in (pw, gl, st, sb))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--grid-s", type=float, default=2.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu", weights_only=False)["appliances"])
    cache = real_windows(apps, a.grid_s, dev)
    i_ov = apps.index("oven")

    print("오븐 헛게이트 유형 나누기 (14.105) · 격자 %.1f초 · 체크포인트 %d개\n"
          % (a.grid_s, len(a.ckpt)))

    # ── ① 오븐이 정말 유독한가 — 기기별 헛게이트 비율 ────────────────────
    print("① **오븐만 그런가** — 참 OFF 인 창에서 게이트가 서는 비율 (기기별)")
    print("   %-8s %8s %8s %10s   %s" % ("기기", "참OFF창", "헛게이트", "비율",
                                         "그중 전력 200W 넘음"))
    tab = {}
    FW = [_fwd(p, cache, dev) for p in a.ckpt]
    for k, app in enumerate(apps):
        off = big = hit = 0
        for stem, d in cache.items():
            y = d["y"].astype(bool)[:, k]
            for F in FW:
                pw, gl = F[stem][0], F[stem][1]
                s = ~y
                off += int(s.sum())
                g = s & (gl[:, k] > 0)
                hit += int(g.sum())
                big += int((g & (pw[:, k] > W_BIG)).sum())
        tab[app] = (off, hit, big)
        mark = "  <-- " if app == "oven" else ""
        print("   %-8s %8d %8d %9.1f%%   %d (%.1f%%)%s"
              % (KO.get(app, app), off, hit, 100.0 * hit / max(off, 1), big,
                 100.0 * big / max(off, 1), mark))

    # ── ② 오븐 헛게이트를 유형으로 가른다 ────────────────────────────────
    print("\n② 오븐 헛게이트 **유형** (참 오븐 OFF · 게이트 섬)")
    cnt = dict(A=0, B=0, C=0, D=0)
    detail = []
    for stem, d in cache.items():
        y = d["y"].astype(bool)
        base = float(d["p_base"])
        for F in FW:
            pw, gl, st, sb = F[stem]
            s = (~y[:, i_ov]) & (gl[:, i_ov] > 0)
            for i in np.nonzero(s)[0]:
                w = float(pw[i, i_ov])
                tot = float(pw[i].sum() + sb[i].sum() + base)
                res_in = float(d["p_obs"][i]) - tot          # 오븐 포함 잔차
                res_out = res_in + w                         # 오븐 뺐을 때 잔차
                # 참으로 켜졌는데 모델이 못 켠 저항 기기
                miss = [KO[r] for r in RES if r != "oven"
                        and y[i, apps.index(r)] and gl[i, apps.index(r)] <= 0]
                if w <= W_BIG:
                    t = "B"                                  # 팬·조명 (전력 없음)
                elif miss and res_out > GAP_W:
                    t = "A"                                  # 축퇴 — 남의 자리를 받았다
                elif res_in < -GAP_W:
                    t = "C"                                  # 초과 배분
                else:
                    t = "D"                                  # 그 밖 (설명 안 된 와트를 받음)
                cnt[t] += 1
                detail.append((t, stem, float(d["t"][i]), w, res_in, res_out, miss))
    n = max(sum(cnt.values()), 1)
    nm = {"A": "**축퇴** — 참으로 켜진 저항 기기를 모델이 못 켰고 그 와트를 오븐이 받았다",
          "B": "**팬·조명** — 게이트만 서고 전력 %.0fW 이하 (오븐 state 1 = 16.8W)" % W_BIG,
          "C": "**초과** — 아무도 안 빼앗고 더 얹었다 (예측합 > 관측)",
          "D": "그 밖 — 설명 안 되던 와트를 받았는데 못 켠 저항 기기는 없다"}
    for t in "BACD":
        print("   %s  %5d창 (%4.1f%%)  %s" % (t, cnt[t], 100.0 * cnt[t] / n, nm[t]))

    # ── ③ 유형별 표본 ────────────────────────────────────────────────────
    print("\n③ 유형별 표본 (각 4개)")
    print("   %-3s %-8s %7s %8s %9s %9s  %s"
          % ("형", "파일", "t(초)", "오븐W", "잔차(포함)", "잔차(제외)", "못 켠 저항"))
    for t in "BACD":
        got = [x for x in detail if x[0] == t]
        got.sort(key=lambda x: -abs(x[3]))
        for x in got[:4]:
            print("   %-3s %-8s %7.1f %8.0f %9.0f %9.0f  %s"
                  % (x[0], x[1], x[2], x[3], x[4], x[5], "+".join(x[6]) or "-"))
    # ── ④ 축퇴 짝 · 전이 근처 ────────────────────────────────────────────
    from collections import Counter
    pair = Counter()
    for x in detail:
        if x[0] == "A":
            for q in x[6]:
                pair[q] += 1
    print("")
    print("④ **축퇴 짝** — A 유형에서 모델이 못 켠 저항 기기")
    for q, c in pair.most_common():
        print("   %-6s %4d창 (%.0f%%)" % (q, c, 100.0 * c / max(cnt["A"], 1)))

    print("")
    print("⑤ 전이 근처인가 — 참 전이(어느 기기든) %.0f초 안" % (2 * a.grid_s))
    #: ⚠ **바닥값을 먼저 잰다.** 창의 34%가 이미 전이 근처라, 바닥 없이 58% 를 보면
    #:   "전이가 원인" 이라고 읽게 된다 ([[check-conditioning-before-believing-a-fit]]).
    near = {t: [0, 0] for t in "BACD"}
    b_tot = b_near = 0
    for stem, d in cache.items():
        y = d["y"].astype(bool)
        ch = np.zeros(len(d["t"]), bool)
        ch[1:] = (y[1:] != y[:-1]).any(1)
        idxs = np.nonzero(ch)[0]
        b_tot += len(d["t"])
        if len(idxs):
            b_near += int(sum(np.abs(idxs - i).min() <= 2 for i in range(len(d["t"]))))
        for x in detail:
            if x[1] != stem:
                continue
            i = int(np.argmin(np.abs(d["t"] - x[2])))
            near[x[0]][0] += 1
            if len(idxs) and np.abs(idxs - i).min() <= 2:
                near[x[0]][1] += 1
    base_r = 100.0 * b_near / max(b_tot, 1)
    print("   바닥값 — 창 전체의 %.0f%% 가 이미 전이 근처다 (%d/%d)" % (base_r, b_near, b_tot))
    for t in "BACD":
        n_, k_ = near[t]
        if n_:
            r = 100.0 * k_ / n_
            print("   %s  %4d/%4d = %.0f%%   (바닥 대비 **%.1f배**)"
                  % (t, k_, n_, r, r / max(base_r, 1e-9)))

    # ── ⑥ ★2 의 61창을 유형으로 가른다 ──────────────────────────────────
    i_ke = apps.index("electiric_kettle")
    kb = ka = kc = kd = 0
    for x in detail:
        d = cache[x[1]]
        i = int(np.argmin(np.abs(d["t"] - x[2])))
        if not (d["y"].astype(bool)[i, i_ke] and not d["y"].astype(bool)[i, i_ov]):
            continue
        kb += x[0] == "B"; ka += x[0] == "A"; kc += x[0] == "C"; kd += x[0] == "D"
    print("")
    print("⑥ ★2 가 세는 창(포트 참ON·오븐 OFF)의 유형 — B %d · A %d · C %d · D %d"
          % (kb, ka, kc, kd))
    print("   ⇒ ★2 는 %s" % ("**대부분 팬·조명 게이트**를 세고 있다 (와트 병이 아니다)"
                             if kb > ka + kc else "실제 와트 축퇴를 세고 있다"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
