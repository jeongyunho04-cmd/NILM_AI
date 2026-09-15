# -*- coding: utf-8 -*-
"""★1 면적 · ★2 축퇴 — 오븐↔포트 병을 재는 **미리 적은 두 자** (14.95).

왜 도구로 박나
--------------
이 둘은 14.89 · 14.94 · 14.95 세 판의 판정 기준으로 sbatch 머리에 **미리 적혀** 있는데
정작 재는 코드는 매번 즉석으로 썼다. 그러면 판마다 자가 미세하게 달라지고, 그 차이가
효과로 읽힌다 ([[match-the-scoring-convention-before-comparing]]). 여기에 못 박는다.

```
  ★1  test_5 의 128~142초에서 오븐 **통전(state 2) 확률의 면적** (참값 0, 작을수록 좋다)
        격자 2.0초 · `softmax(out["state"])[:, 오븐, 2]` 의 **합** (적분 아니다)
        그 구간의 참 오븐은 **팬·조명(~10W)** 뿐이라 통전 확률은 0 이어야 한다
  ★2  포트 참ON · 오븐 참OFF 창(5파일 통틀어 **61창**)에서 **오븐 게이트가 서는 비율**
        `on_logit > 0` · 격자 2.0초
```
⚠ **자를 먼저 검정한다.** `--control` 로 준 판이 미리 적은 값으로 되나온 뒤에야 처치를
   읽는다. `cnn_v32h_on_s{0,1,2}` 는 **4.454 ± 1.005 · 30.6 ± 1.5%** 여야 한다
   (sd 는 모표준편차 `ddof=0`). 안 나오면 내 자가 그 자가 아니다.

⚠ 격자가 **2.0초**인 것이 자의 일부다. 0.5초로 재면 같은 판이 23.5 가 나온다 — 창이
   4배라 그렇다. 문턱 3.0 도 2.0초 격자에서 정한 값이다.

    python -X utf8 src/run_diag_ovenstar.py results/cnn_hzwt_s0.pt ...
    python -X utf8 src/run_diag_ovenstar.py --control results/cnn_v32h_on_s{0,1,2}.pt \
                                            --arm wtapA results/cnn_hzwt_s{0,1,2}.pt
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

LO, HI = 128.0, 142.0     #: ★1 구간 (초)
GRID_S = 2.0              #: 두 자 모두 이 격자다 — 자의 일부이지 편의가 아니다
ON_STATE = 2              #: 오븐 통전 슬롯 (`losses.S_STATE["oven"] = {1: 16.8, 2: 1357.1}`)
#: 미리 적은 대조값 (14.95, `cnn_v32h_on_s{0,1,2}`). 자 검정에 쓴다.
PINNED = {"cnn_v32h_on": (4.454, 1.005, 30.6, 1.5)}


def _fwd(path, cache, dev):
    """(state 로짓, on 로짓) 파일별. `predict` 는 상태를 안 내놔서 직접 돈다."""
    m = load_model(path, dev)[0]
    m.eval()
    out = {}
    with torch.no_grad():
        for stem, d in cache.items():
            st, gl = [], []
            for i in range(0, len(d["t"]), 512):
                o = m(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                      torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                st.append(o["state"].float().cpu())
                gl.append(o["on_logit"].float().cpu())
            out[stem] = (torch.cat(st).numpy(), torch.cat(gl).numpy())
    return out


def f1_two_ways(path, cache, apps, dev, app="oven"):
    """오븐 검출을 **두 자로** 잰다 — 현행 라벨과 물리 마스크 (12.162).

    12.162.3 이 확정한 것: 라벨 ON 인데 **관측 총전력이 `V²/R` 의 절반 미만**인
    창이 오븐은 52.9% 고, 모델은 그 창에서 **한 번도 안 켠다**. 물리를 지키는데
    F1 은 미검출로 센다 (오븐 F1 0.530 -> 마스크 0.867).

    `L_gcond` 는 그 어긋남을 손실로 옮기는 손잡이라, **현행 F1 은 내려가는 것이
    정상**이고 마스크 F1 이 내려가면 진짜 악화다. 한 자로만 보면 12.162 의 결론을
    잊고 "손잡이가 오븐을 망쳤다" 로 읽는다 ([[check-the-label-and-the-shape-before-reading-a-confusion-table]]).
    """
    from src.model.postproc import RESISTIVE_OHM
    k = apps.index(app)
    ohm = RESISTIVE_OHM[app]
    pr = _fwd(path, cache, dev)
    acc = {}
    for nm in ("현행", "마스크"):
        tp = fp = fn = 0
        for stem, d in cache.items():
            y = d["y"].astype(bool)[:, k]
            g = pr[stem][1][:, k] > 0
            keep = np.ones(len(y), bool)
            if nm == "마스크":
                # 통전이 **물리적으로 불가능한** 창을 뺀다 (라벨 ON 쪽만)
                need = 0.5 * np.asarray(d["v_obs"], float) ** 2 / ohm
                keep = ~(y & (np.asarray(d["p_obs"], float) < need))
            tp += int((y & g & keep).sum())
            fp += int((~y & g & keep).sum())
            fn += int((y & ~g & keep).sum())
        rec = tp / max(tp + fn, 1)
        pre = tp / max(tp + fp, 1)
        acc[nm] = (2 * rec * pre / max(rec + pre, 1e-9), rec, pre)
    return acc


def stars(path, cache, apps, dev):
    """(★1 면적, ★2 축퇴 %, 축퇴 창 수)."""
    i_ov, i_ke = apps.index("oven"), apps.index("electiric_kettle")
    pr = _fwd(path, cache, dev)
    d = cache["test_5"]
    sel = (d["t"] >= LO) & (d["t"] <= HI)
    p2 = torch.from_numpy(pr["test_5"][0][:, i_ov]).softmax(-1).numpy()[:, ON_STATE]
    area = float(p2[sel].sum())
    hit = tot = 0
    for stem, dd in cache.items():
        y = dd["y"].astype(bool)
        s = y[:, i_ke] & ~y[:, i_ov]
        if not s.any():
            continue
        hit += int((pr[stem][1][s, i_ov] > 0).sum())
        tot += int(s.sum())
    return area, 100.0 * hit / max(tot, 1), tot


def _blk(name, vals):
    a = np.asarray(vals, float)
    return a, "%-11s %s | %7.3f ± %-6.3f" % (
        name, " ".join("%7.3f" % x for x in a), a.mean(), a.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="*", help="처치 체크포인트 (시드 순서대로)")
    ap.add_argument("--control", nargs="*", default=[], help="대조 (짝차의 기준)")
    ap.add_argument("--arm", default="처치", help="처치 이름")
    ap.add_argument("--f1", action="store_true",
                    help="★3/★4 — 오븐 검출을 현행 라벨과 **물리 마스크** 두 자로 "
                         "(12.162). `L_gcond` 판정에는 반드시 켠다.")
    a = ap.parse_args()
    paths = list(a.control) + list(a.ckpt)
    if not paths:
        raise SystemExit("체크포인트를 달라")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(paths[0], map_location="cpu", weights_only=False)["appliances"])
    cache = real_windows(apps, GRID_S, dev)
    print("★1 면적 · ★2 축퇴 (격자 %.1f초 · 구간 %.0f~%.0f초)" % (GRID_S, LO, HI))
    print("")

    got = {}
    for p in paths:
        got[p] = stars(p, cache, apps, dev)
    n_deg = got[paths[0]][2]

    def show(tag, ps):
        A, la = _blk(tag, [got[p][0] for p in ps])
        D, ld = _blk(tag, [got[p][1] for p in ps])
        return A, D, la, ld

    print("  %-11s %s | %-16s" % ("", "  씨0     씨1     씨2 ", "평균 ± sd(모)"))
    print("  ★1 면적 (참값 0 · 문턱 3.0)")
    if a.control:
        cA, cD, la, ld = show("대조", a.control)
        print("    " + la)
    if a.ckpt:
        tA, tD, la2, ld2 = show(a.arm, a.ckpt)
        print("    " + la2)
    print("  ★2 축퇴 %% (포트 참ON·오븐 OFF %d창)" % n_deg)
    if a.control:
        print("    " + ld)
    if a.ckpt:
        print("    " + ld2)

    if a.control and a.ckpt and len(a.control) == len(a.ckpt):
        for nm, c, t, lo_better in (("★1 면적", cA, tA, True), ("★2 축퇴", cD, tD, True)):
            d = t - c
            sgn = int((d < 0).sum()) if lo_better else int((d > 0).sum())
            print("    짝차 %-8s %s | %+7.3f ± %-6.3f  (%d/%d %s)"
                  % (nm, " ".join("%+7.3f" % x for x in d), d.mean(), d.std(),
                     sgn, len(d), "개선" if lo_better else "악화"))

    # ── ★3/★4 오븐 검출을 두 자로 (12.162) ──────────────────────────────
    if a.f1:
        print("")
        print("  ★3/★4 오븐 검출 — **두 자로** (12.162: 채점기가 두 정의를 섞어 잰다)")
        print("    %-11s %-22s %-22s" % ("", "★3 현행 라벨 F1(재현/정밀)",
                                         "★4 물리마스크 F1(재현/정밀)"))
        for tag, ps in (("대조", a.control), (a.arm, a.ckpt)):
            if not ps:
                continue
            r = [f1_two_ways(p, cache, apps, dev) for p in ps]
            c = np.asarray([x["현행"][0] for x in r])
            m = np.asarray([x["마스크"][0] for x in r])
            print("    %-11s %.3f ± %.3f (%.3f/%.3f)   %.3f ± %.3f (%.3f/%.3f)"
                  % (tag, c.mean(), c.std(),
                     np.mean([x["현행"][1] for x in r]), np.mean([x["현행"][2] for x in r]),
                     m.mean(), m.std(),
                     np.mean([x["마스크"][1] for x in r]),
                     np.mean([x["마스크"][2] for x in r])))
        print("    ⚠ ★3 은 **내려가는 것이 정상**이다 (라벨이 팬·조명도 ON 으로 적는다). "
              "★4 가 내려가면 진짜 악화다.")

    # ── 자 검정 — 미리 적은 값과 맞나 ────────────────────────────────────
    for key, (a1, s1, a2, s2) in PINNED.items():
        ps = [p for p in paths if key in p]
        if len(ps) < 2:
            continue
        A = np.asarray([got[p][0] for p in ps]); D = np.asarray([got[p][1] for p in ps])
        ok = abs(A.mean() - a1) < 0.05 and abs(D.mean() - a2) < 0.2
        print("")
        print("  자 검정 (%s): ★1 %.3f ± %.3f (적힌 값 %.3f ± %.3f) · "
              "★2 %.1f ± %.1f (%.1f ± %.1f)   %s"
              % (key, A.mean(), A.std(), a1, s1, D.mean(), D.std(), a2, s2,
                 "OK — 같은 자다" if ok else "** 다른 자다 — 판정하지 마라 **"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
