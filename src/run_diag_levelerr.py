# -*- coding: utf-8 -*-
"""저항 기기의 **수준 오차**를 컨덕턴스로 기기별로 가른다 (14.55).

14.55 가 잰 것: 앵커 뒤 남은 잔차는 **유령이 아니다** (유령 중앙 = 0.0W). 참으로 켜진
기기의 **수준** 오차다. 그럼 어느 기기인가 — 실측에는 기기별 참 전력이 없다.

**컨덕턴스가 준다.** 저항은 `P = V²/R` 이고 R 을 안다
(`postproc.RESISTIVE_OHM`: 포트 35.8 · 오븐 40.6 · 드라이 54.3 · 핫플 101.8Ω).
참 라벨로 **저항 하나만 켜진 창**을 고르고, 관측 총전력에서 기준선을 뺀 것이
`V²/R_k` 와 맞는지로 **그 기기가 통전 중인지** 확인한 뒤 (그래야 오븐 FAN_LIGHT 16W 나
핫플 릴레이 OFF 구간을 안 센다), `pred_k` 를 `V²/R_k` 와 견준다.

    ⚠ 이것이 실측에서 기기별 전력 참값을 얻는 **유일한** 길이다. 라벨은 on/off 만 준다.
    ⚠ 통전 확인을 안 하면 14.48 의 물리 프록시 오탐(~14%)을 그대로 먹는다.

    python -X utf8 src/run_diag_levelerr.py --ckpt results/a.pt results/b.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch  # noqa: E402

from src.model.postproc import RESISTIVE_OHM  # noqa: E402
from src.run_diag_rollback import predict  # noqa: E402
from src.run_train_seq import real_windows  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--grid-s", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=0.06,
                    help="통전 확인 허용오차 — |P관측−기준선 − V²/R| / (V²/R)")
    ap.add_argument("--postproc", default="off", choices=("off", "cap", "full"))
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt[0], map_location="cpu", weights_only=False)["appliances"])
    cache = real_windows(apps, a.grid_s, dev)
    R = {}
    for p in a.ckpt:
        _, pr = predict(p, cache, dev, postproc=a.postproc)
        R[p] = pr
        print("  %s 끝" % p.split("/")[-1], flush=True)
    print()

    res = [x for x in apps if x in RESISTIVE_OHM]
    ri = [apps.index(x) for x in res]
    # ── 창 고르기: 참 라벨로 저항이 **정확히 하나** 켜져 있고, 그 하나의 컨덕턴스가
    #    관측 총전력(−기준선)을 설명하는 창.
    sel = {x: {"m": [], "true": [], "stem": []} for x in res}
    for stem, d in cache.items():
        y = d["y"].astype(bool)
        v = np.asarray(d["v_rms"] if "v_rms" in d else d.get("v_obs"), dtype=np.float64)
        pobs = d["p_obs"].astype(np.float64)
        base = float(d.get("p_base", 0.0))
        n_res = y[:, ri].sum(1)
        for k, (x, j) in enumerate(zip(res, ri)):
            pk = v ** 2 / RESISTIVE_OHM[x]
            ok = (n_res == 1) & y[:, j] & (np.abs(pobs - base - pk) / np.maximum(pk, 1.0) < a.tol)
            if ok.any():
                sel[x]["m"].append(np.flatnonzero(ok))
                sel[x]["true"].append(pk[ok])
                sel[x]["stem"] += [stem] * int(ok.sum())
    names = [p.split("/")[-1].replace(".pt", "") for p in a.ckpt]
    print("저항 기기의 **수준 오차** — 컨덕턴스 참값 대비  (저항 하나만 켜지고 통전 확인된 창)")
    print("  %-20s%7s%9s" % ("기기", "창", "참 W") + "".join("%16s" % n[-15:] for n in names))
    for x in res:
        if not sel[x]["m"]:
            print("  %-20s%7d" % (x, 0) + "   (표본 없음)")
            continue
        tw = np.concatenate(sel[x]["true"])
        line = "  %-20s%7d%9.0f" % (x, len(tw), np.median(tw))
        for p in a.ckpt:
            got = []
            off = 0
            for stem, idx, t in zip(
                    [s for s in cache], [], []):
                pass
            # 파일별로 예측을 모은다
            vals = []
            i0 = 0
            for stem, d in cache.items():
                y = d["y"].astype(bool)
                v = np.asarray(d["v_rms"] if "v_rms" in d else d.get("v_obs"), dtype=np.float64)
                pobs = d["p_obs"].astype(np.float64); base = float(d.get("p_base", 0.0))
                n_res = y[:, ri].sum(1); j = apps.index(x)
                pk = v ** 2 / RESISTIVE_OHM[x]
                ok = (n_res == 1) & y[:, j] & (np.abs(pobs - base - pk) / np.maximum(pk, 1.0) < a.tol)
                if not ok.any():
                    continue
                on, pw = R[p][stem]
                vals.append(np.where(on[ok, j], pw[ok, j], 0.0) - pk[ok])
            e = np.concatenate(vals) if vals else np.zeros(0)
            line += "%+15.1fW" % (np.median(e) if len(e) else np.nan)
        print(line)
    print()
    print("  ⚠ 부호: **양수 = 과대예측** (참 컨덕턴스 전력보다 크게 냈다).")
    print("     `run_diag_hipow` 의 잔차와 **부호가 반대**다 (거기는 관측−예측).")
    print("  ⚠ 표본이 적으면 읽지 마라 — 씨앗 폭이 크다 (SOLO 계열의 옛 교훈).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
