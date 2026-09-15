# -*- coding: utf-8 -*-
"""**오븐 통전 참값**을 관측에서 끌어내고, 그 위에서 헛detect 를 다시 센다 (14.139).

사용자: *"오븐 통전이랑 포트 전이 근처를 기준으로 해 봐. 거기가 제일 컸어."*

★★ 그 지적이 **모집단의 구멍**을 열었다.

```
  라벨의 오븐 이벤트는 `delta_p_w = +12.7W` — **팬조명(16.8W)** 이다.
  즉 `intervals["oven"]["on"]` 은 **팬조명 구간**이고, 그 안에서 서모스탯이
  통전(1357W)을 껐다 켰다 하는 것은 **라벨에 한 글자도 없다.**
  그런데 지금까지 모든 자가 모집단을 `~oven` 으로 잡았다 —
  **통전 헛detect 가 가장 많은 칸을 통째로 빼고 있었다.**
```

통전 참값 만드는 법 (라벨이 아니라 **관측**에서)
------------------------------------------------
```
  1. 팬조명 ON 구간 안에서 **지속 계단**을 찾는다
       before = mean(p[i-3s:i-1s])   after = mean(p[i+1s:i+3s])
       |after-before| in [1000, 1750]W      <- 통전 1357W 만. 드라이 1022 · 핫플 550 제외
     ⚠ 듀티 가장자리는 **되돌아오므로** 이 판정에서 저절로 빠진다
       (거친 `|p[i+6]-p[i-6]|>200W` 는 핫플 듀티를 1090개나 주웠다)
  2. 다른 기기가 ±3초 안에서 스위칭한 계단은 버린다
  3. 올림·내림을 **짝지어** 구간을 만들고 짝마다 R = V^2/ΔP 를 잰다
  4. ★ 가장자리를 **날카롭게** 다시 잡는다 — ±1.5초 안에서 `|p[i+6]-p[i-6]|` 최대
     ⚠⚠ 이걸 빼먹으면 가장자리가 중앙 **0.18초** 이르게 찍히고, 그것만으로
        "통전 직전 -1~0초에 36.8% 봉우리" 라는 **가짜 봉우리**가 생긴다.
        날카롭게 잡으면 그 봉우리가 **통째로 사라진다** (0.0%). 그 창들은 실제로
        통전 중이었고 모델이 **맞힌** 것이었다.
  5. 검증: 19쌍 전부 R 40.3~41.9Ω · 중앙 **41.0Ω** (참 `RESISTIVE_OHM` 40.6Ω)
```

    python -X utf8 src/run_diag_condtruth.py results/cnn_pcap_s{0,1,2}.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402

from src.evaluation.real_events import load_events  # noqa: E402
from src.model.realdata import DEFAULT_DIR, dense_targets  # noqa: E402
from src.preprocessing import load_nilm_npz  # noqa: E402
from src.run_gate_check import load_model  # noqa: E402

FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
BIG3 = ("electiric_kettle", "hair_dryer", "hotplate")
OTHER = ("electiric_kettle", "hair_dryer", "hotplate", "air_conditioner")
FS = 60
LO, HI = 1000.0, 1750.0
R_OK = (35.0, 48.0)


def _steps(p, W):
    i = np.arange(3 * FS, len(p) - 3 * FS)
    c = np.concatenate([[0.0], np.cumsum(p)])
    seg = lambda a, b: (c[b] - c[a]) / (b - a)          # noqa: E731
    j = seg(i + FS, i + 3 * FS) - seg(i - 3 * FS, i - FS)
    aj = np.abs(j)
    out = []
    for k in np.flatnonzero(aj > W):
        if aj[k] >= aj[max(0, k - FS):k + FS].max() and (not out or i[k] - out[-1][0] > FS):
            out.append((i[k] / FS, float(j[k])))
    return out


def _sharp(p, t):
    """★ ±1.5초 안에서 ±0.1초 차분이 최대인 자리로. 안 하면 가짜 봉우리가 선다."""
    k = int(round(t * FS))
    j = np.arange(max(6, k - 90), min(len(p) - 7, k + 90))
    return float(j[np.abs(p[j + 6] - p[j - 6]).argmax()]) / FS


def conduction_truth(verbose=True):
    """{stem: [(시작초, 끝초), ...]} — 짝마다 R 로 검증한 오븐 통전 구간."""
    out, rows = {s: [] for s in FILES}, []
    for s in FILES:
        raw = load_nilm_npz("%s/%s.npz" % (DEFAULT_DIR, s))
        p = np.asarray(raw["p_denoised_w"], float)
        v = np.asarray(raw["power_features"])[:, 4]
        E = load_events()[s]
        ovon = E["intervals"].get("oven", {}).get("on", [])
        if not ovon:
            continue
        oth = np.array(sorted(e["t_s"] for e in E["events"] if e["appliance"] in OTHER))
        st = [(x, d) for x, d in _steps(p, LO) if abs(d) <= HI
              and (not len(oth) or np.abs(oth - x).min() > 3.0)
              and any(t0 <= x <= t1 for t0, t1 in ovon)]
        cur = None
        for x, d in st:
            if d > 0:
                cur = (x, d)
            elif cur is not None:
                dP = 0.5 * (cur[1] - d)
                a, b = _sharp(p, cur[0]), _sharp(p, x)
                R = v[int(a * FS):int(b * FS)].mean() ** 2 / dP
                rows.append((s, a, b, R))
                if R_OK[0] <= R <= R_OK[1]:
                    out[s].append((a, b))
                cur = None
    if verbose:
        Rv = np.array([r[3] for r in rows])
        print("오븐 통전 참값 — 쌍 %d개 · R 중앙 **%.1fΩ** (참 40.6) · 범위 %.1f~%.1f"
              % (len(rows), np.median(Rv), Rv.min(), Rv.max()))
        for s in FILES:
            if out[s]:
                print("  %-8s 통전 %2d구간 · %6.1f초  (%s)"
                      % (s, len(out[s]), sum(b - a for a, b in out[s]),
                         " ".join("%.0f~%.0f" % ab for ab in out[s][:6])
                         + (" ..." if len(out[s]) > 6 else "")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    TRUTH = conduction_truth()
    print("")

    M, CP = {}, []
    for ci, ck in enumerate(a.ckpt):
        m, apps, _ = load_model(ck, dev)
        jo = apps.index("oven")
        acc = {k: [] for k in ("cd", "kd", "conT", "fan", "big", "ke", "hd", "hp",
                               "cp", "pw", "file")}
        for s in FILES:
            rw = dense_targets(s, stride=30)
            t = np.asarray(rw.target_cycle, float) / 60.0
            iv = load_events()[s]["intervals"]

            def mask(app):
                mm = np.zeros(len(t), bool)
                for t0, t1 in iv.get(app, {}).get("on", []):
                    mm |= (t >= t0) & (t <= t1)
                return mm
            cT = np.zeros(len(t), bool)
            for x0, x1 in TRUTH[s]:
                cT |= (t >= x0) & (t <= x1)
            ed = np.array(sorted([x for x, _ in TRUTH[s]] + [y for _, y in TRUTH[s]]))
            acc["cd"].append(t - ed[np.abs(t[:, None] - ed[None]).argmin(1)]
                             if len(ed) else np.full(len(t), 1e9))
            kev = np.array(sorted(e["t_s"] for e in load_events()[s]["events"]
                                  if e["appliance"] == "electiric_kettle"))
            acc["kd"].append(t - kev[np.abs(t[:, None] - kev[None]).argmin(1)]
                             if len(kev) else np.full(len(t), 1e9))
            b = np.zeros(len(t), bool)
            for ap_ in BIG3:
                b |= mask(ap_)
            acc["conT"].append(cT); acc["fan"].append(mask("oven")); acc["big"].append(b)
            acc["ke"].append(mask("electiric_kettle")); acc["hd"].append(mask("hair_dryer"))
            acc["hp"].append(mask("hotplate")); acc["file"].append(np.full(len(t), s))
            cp, pw = [], []
            with torch.no_grad():
                for i in range(0, len(rw), 256):
                    f, w, *_ = rw.batch(np.arange(i, min(i + 256, len(rw))))
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=dev == "cuda"):
                        o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                              torch.from_numpy(np.ascontiguousarray(w)).to(dev))
                    cp.append(((o["on_logit"][:, jo] > 0) &
                               (torch.softmax(o["state"].float(), -1)[:, jo, 2] > 0.5)
                               ).cpu().numpy())
                    pw.append(o["power"][:, jo].float().cpu().numpy())
            acc["cp"].append(np.concatenate(cp)); acc["pw"].append(np.concatenate(pw))
        R = {k: np.concatenate(v) for k, v in acc.items()}
        if ci == 0:
            M = {k: R[k] for k in ("cd", "kd", "conT", "fan", "big", "ke", "hd",
                                   "hp", "pw", "file")}
        CP.append(R["cp"])
    CP = np.stack(CP)
    cd, kd, conT, fan, big = M["cd"], M["kd"], M["conT"], M["fan"], M["big"]
    ke, hd, hp, fl, pw = M["ke"], M["hd"], M["hp"], M["file"], M["pw"]
    c0 = CP.mean(0) > 0.5

    print("[0] ★★ 통전 헛detect 가 **어디에 사는가** — 넷째 칸을 통째로 빼고 있었다")
    print("  %-42s %7s %12s %11s" % ("칸", "창", "통전판정", "평균 W"))
    for nm, mk in (("오븐 완전OFF · 큰부하 (지금까지 쓴 모집단)", ~fan & big),
                   ("오븐 완전OFF · 큰부하 없음", ~fan & ~big),
                   ("★ 팬조명ON · 통전 참OFF (**빠져 있었다**)", fan & ~conT),
                   ("통전 참ON (여기는 맞히면 정답 — 재현)", conT)):
        n = int(mk.sum())
        print("  %-42s %7d %6.1f%% (%4d) %10.1f"
              % (nm, n, 100 * c0[mk].mean() if n else 0, int(c0[mk].sum()),
                 pw[mk].mean() if n else 0))
    tot = int(c0[(~fan & big) | (fan & ~conT)].sum())
    print("  -> 헛창 %d개 중 **%d개(%.0f%%)** 가 빠져 있던 칸에 있다"
          % (tot, int(c0[fan & ~conT].sum()), 100 * c0[fan & ~conT].sum() / max(tot, 1)))

    BINS = [(-20, -10), (-10, -6), (-6, -4), (-4, -2), (-2, 0), (0, 2),
            (2, 4), (4, 6), (6, 10), (10, 20)]

    def tab(name, pop, d, tail=True):
        print("  %-26s 창 %d · 헛율 %.2f%%" % (name, int(pop.sum()),
                                              100 * c0[pop].mean() if pop.sum() else 0))
        print("      띠  " + " ".join("%7s" % ("%g~%g" % b) for b in BINS)
              + ("  |d|>20" if tail else ""))
        print("      창  " + " ".join("%7d" % (pop & (d >= x) & (d < y)).sum()
                                      for x, y in BINS)
              + ("  %7d" % (pop & (np.abs(d) > 20)).sum() if tail else ""))
        for si in range(len(CP)):
            c = CP[si]
            print("    씨%d 헛  " % si + " ".join(
                ("%6.1f%%" % (100 * c[pop & (d >= x) & (d < y)].mean()))
                if (pop & (d >= x) & (d < y)).sum() >= 6 else "      -"
                for x, y in BINS)
                + ("  %6.1f%%" % (100 * c[pop & (np.abs(d) > 20)].mean())
                   if tail and (pop & (np.abs(d) > 20)).sum() else ""))

    print("\n[1] ★ **오븐 통전 가장자리** 기준 (음수 = 통전이 곧 온다)")
    tab("팬조명ON · 통전 참OFF", fan & ~conT, cd)
    print("\n[2] ★ **포트 전이** 기준")
    tab("팬조명ON · 통전 참OFF", fan & ~conT, kd)
    tab("오븐 완전OFF · 큰부하", ~fan & big, kd)

    print("\n[3] 조합으로 가르면 (팬조명ON · 통전 참OFF)")
    print("  %-22s %8s %10s" % ("같이 켜진 큰 부하", "창", "통전 헛%"))
    for nm, mk in (("아무것도 없음", ~ke & ~hd & ~hp), ("포트만", ke & ~hd & ~hp),
                   ("드라이만", hd & ~ke & ~hp), ("핫플만", hp & ~ke & ~hd),
                   ("포트+드라이", ke & hd & ~hp), ("포트+핫플", ke & ~hd & hp),
                   ("드라이+핫플", ~ke & hd & hp), ("셋 다", ke & hd & hp)):
        q = (fan & ~conT) & mk
        print("  %-22s %8d %9.1f%%" % (nm, int(q.sum()),
                                       100 * c0[q].mean() if q.sum() else 0))
    print("\n[4] 파일별 (팬조명ON · 통전 참OFF)")
    for s in FILES:
        q = (fan & ~conT) & (fl == s)
        if q.sum():
            print("  %-8s 창 %4d · 헛 %4d (%5.1f%%) · 통전주기 %2d회"
                  % (s, int(q.sum()), int(c0[q].sum()), 100 * c0[q].mean(),
                     len(TRUTH[s])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
