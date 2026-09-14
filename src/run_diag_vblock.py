# -*- coding: utf-8 -*-
"""**전압 고조파 블록 민감도** — 모델이 V_h 를 물리로 쓰나 판별자로 쓰나 (14.54).

14.39 가 인라인으로 쟀던 것을 스크립트로 굳힌다. v32s(창-안 텍스처 교체)의 **겨냥 지표**다.

무엇을 하나
-----------
같은 파일의 **다른 창**에서 전압 고조파 블록(세밀 45~56 = Re/Im(V_h), h=1,3,5,7,9,11)만
떼어다 끼운다. **나머지는 한 채널도 안 건드린다** — 특히 `V_rms`(채널 25)는 그대로다.

    물리 기대치는 **~0%** 다.
      · SMPS 는 정전압 제어라 전압 파형이 바뀌어도 전력이 안 변한다 (14.6 ③)
      · 저항의 V² 의존은 **안 건드린 채널 25** 가 나른다
    그런데 14.39 가 잰 값: `cnn_sigc_vn_s0` 에서 SMPS p_raw |Δ| **9.2%** · 저항 **7.7%**.
    `--vswap-p 0.5` 가 그것을 1.7% / 0.9% 로 눌렀는데, 그것은 **증상 억제**였다 (14.50).

왜 v32s 가 이것을 고칠 것인가 (14.51)
------------------------------------
합성 창은 창-안 `V_h/|V_1|` 변동이 **정확히 0** 이었다 — 텍스처가 창 전체에 정적으로
곱해졌다. 모델은 "창 안에서 V_h 는 안 움직인다" 는 세계만 봤고, 그래서 V_h 블록을
**창의 지문(판별자)** 으로 읽었다. 실측은 그렇지 않으니 실측에서 흔들린다.
v32s 는 그 변동을 실측의 62~71% 까지 넣는다 ⇒ **민감도가 줄어야 한다.**

    python -X utf8 src/run_diag_vblock.py --ckpt results/a.pt results/b.pt

⚠ 판정은 **시드 셋**으로. `--pair N` 이면 앞 N 을 A팔, 뒤 N 을 B팔로 보고 짝차를 낸다.
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch  # noqa: E402

from src.model.inputs import FINE_VOLT0, VOLT_ORDERS, WIDE_VOLT0  # noqa: E402

SMPS = ("minipc", "laptop_charger", "beam_projector")
RES = ("oven", "hotplate", "electiric_kettle", "hair_dryer")
NV = len(VOLT_ORDERS)
#: ⚠⚠ **두 갈래를 따로 재야 한다** (14.58). 세밀(45~56)은 창의 **뒤 10초**뿐이고
#:   광역(35~46)은 60초 전부다. `--vtex-seg-s 10` 은 세밀 창 전체가 토막 하나라
#:   **세밀 블록을 한 톨도 안 바꾼다** — 광역만 x1.4~12.3 이 된다 (캐시 대조 실측).
#:   14.39 가 짚은 과민 채널은 **세밀** 쪽이라, 세밀만 재면 v32s 효과를 0 으로 읽고
#:   광역만 재면 14.39 의 병을 못 본다.
VBLK_FINE = slice(FINE_VOLT0, FINE_VOLT0 + 2 * NV)              # 45..56
VBLK_WIDE = slice(WIDE_VOLT0, WIDE_VOLT0 + 2 * NV)              # 35..46


def _run(m, fine, wide, dev, bs=256):
    P, G = [], []
    with torch.no_grad():
        for i in range(0, len(fine), bs):
            o = m(torch.from_numpy(fine[i:i + bs]).to(dev),
                  torch.from_numpy(wide[i:i + bs]).to(dev))
            P.append(o["power_raw"].float().cpu().numpy())
            G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
    return np.concatenate(P), np.concatenate(G)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--block", default="fine", choices=("fine", "wide", "both"),
                    help="어느 전압 고조파 블록을 바꿔 끼울 것인가 (14.58). "
                         "**fine**(세밀 45~56, 창의 뒤 10초)이 14.39 가 짚은 자리고, "
                         "**wide**(광역 35~46, 60초 전부)가 `--vtex-seg-s 10` 이 실제로 "
                         "바꾼 자리다. 둘은 **다른 것을 잰다** — 겨냥에 맞춰 골라라.")
    ap.add_argument("--stems", nargs="+", default=["test_1", "test_2", "test_3", "test_5"])
    ap.add_argument("--stride", type=int, default=120, help="실측 창 보폭 (사이클)")
    ap.add_argument("--seed", type=int, default=0, help="어느 창의 블록을 끼울지 섞는 씨앗")
    ap.add_argument("--pair", type=int, default=0, metavar="N",
                    help="앞 N 을 A팔, 뒤 N 을 B팔로 보고 짝차를 낸다")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from src.model.realdata import RealWindows
    from src.run_gate_check import load_model

    # 창을 파일별로 모으고, **같은 파일 안에서만** 블록을 섞는다
    F, W, FS = [], [], []
    for stem in a.stems:
        rw = RealWindows(stems=[stem], stride=a.stride, require_valid=False)
        idx = np.arange(len(rw))
        f, w, *_ = rw.batch(idx)
        F.append(np.asarray(f)); W.append(np.asarray(w)); FS += [stem] * len(idx)
    fine = np.concatenate(F); wide = np.concatenate(W)
    stems = np.asarray(FS)
    rng = np.random.default_rng(a.seed)
    swapped, swapw = fine.copy(), wide.copy()
    for stem in a.stems:                      # ⚠ **같은 파일 안에서만** — 자리·Z 를 안 바꾼다
        m = np.flatnonzero(stems == stem)
        if len(m) < 2:
            continue
        perm = m[rng.permutation(len(m))]
        bad = perm == m
        if bad.any():                         # 자기 자신이면 한 칸 민다 (Δ=0 을 안 만들려고)
            perm[bad] = m[(np.flatnonzero(bad) + 1) % len(m)]
        if a.block in ("fine", "both"):
            swapped[m, VBLK_FINE] = fine[perm, VBLK_FINE]
        if a.block in ("wide", "both"):
            swapw[m, VBLK_WIDE] = wide[perm, VBLK_WIDE]
    where = {"fine": "세밀 45~56 (뒤 10초)", "wide": "광역 35~46 (60초)",
             "both": "세밀 + 광역"}[a.block]
    print(f"창 {len(fine)}개 / 파일 {len(a.stems)}개 · 바꿔 끼운 곳 **{where}** "
          f"(같은 파일 안에서만)" + chr(10))

    names = [p.split("/")[-1].replace(".pt", "") for p in a.ckpt]
    rows = {}
    for p, nm in zip(a.ckpt, names):
        m, ck = load_model(p, dev)[:2] if isinstance(load_model(p, dev), tuple) else (None, None)
        m = load_model(p, dev)[0]
        m.eval()
        apps = list(torch.load(p, map_location="cpu", weights_only=False)["appliances"])
        p0, g0 = _run(m, fine, wide, dev)
        p1, g1 = _run(m, swapped, swapw, dev)
        rel = np.abs(p1 - p0) / np.maximum(p0, 1.0)
        flip = ((g0 > 0.5) != (g1 > 0.5)).mean(0)
        si = [apps.index(x) for x in SMPS if x in apps]
        ri = [apps.index(x) for x in RES if x in apps]
        rows[nm] = (float(np.median(rel[:, si])), float(np.median(rel[:, ri])),
                    float(flip.mean()), rel, apps, flip)
        print(f"  {nm} 끝", flush=True)
    print()

    print("  %-22s%14s%14s%14s" % ("체크포인트", "SMPS |Δp_raw|", "저항 |Δp_raw|", "게이트 뒤집힘"))
    for nm in names:
        s_, r_, f_, *_ = rows[nm]
        print("  %-22s%13.2f%%%13.2f%%%13.2f%%" % (nm[-22:], 100 * s_, 100 * r_, 100 * f_))
    if a.pair and len(a.ckpt) == 2 * a.pair:
        A, B = names[:a.pair], names[a.pair:]
        print()
        for k, lbl in ((0, "SMPS |Δp_raw|"), (1, "저항 |Δp_raw|"), (2, "게이트 뒤집힘")):
            d = np.array([rows[y][k] - rows[x][k] for x, y in zip(A, B)]) * 100
            sd = d.std(ddof=1) if len(d) > 1 else float("nan")
            mark = ("  ✅ **줄었다**" if (d.mean() < 0 and abs(d.mean()) > 2 * sd)
                    else ("  ~ 씨앗 폭 안" if abs(d.mean()) <= 2 * (sd if np.isfinite(sd) else 0)
                          else "  ⚠ 늘었다"))
            print("  짝차 (B − A)  %-16s%+7.2f%%p ± %.2f%s" % (lbl, d.mean(), sd, mark))
    print()
    print("  ⚠ 물리 기대치는 **~0%** 다 — SMPS 는 정전압 제어, 저항의 V² 는 채널 25 가 나른다.")
    print("  ⚠ 대조 (14.39 인라인): cnn_sigc_vn_s0 SMPS **9.2%** · 저항 7.7% · 뒤집힘 1.4%")
    print("                        cnn_v37(vswap 0.5)  1.7% ·      0.9% ·       0.4%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
