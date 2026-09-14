# -*- coding: utf-8 -*-
"""전압 지수 `--vexp` 의 관문 — 꺼지면 항등, 켜지면 물리 (14.7).

14.6 이 잰 것: 모델은 **상태 명목값**을 기울기 1.00 으로 잘 내는데 **같은 상태 안의
V² 의존**을 0.33~0.83 로만 읽어 **순 전압 지수가 0.85** 다 (물리는 2). 저전압에서
과예측한다 — 실측 test_5(210.6V · 드라이기 녹화 227.2V)가 v25 이후 모든 모델에서
+2.3~7.9% 다.

여기서 두 가지를 확인한다.
  ① **끄면 비트 동일** — `v_exp` 가 전부 0 이면 `vrel**0 = 1` 이다. 0 이 아니면 무언가 샌다.
  ② **켜면 물리** — 전압 α배 상황(전류 α · 전압 α · 전력 α²)에서 출력이 `α^k` 로 따라가는지.
     저항 **k≈2** · SMPS **k≈0** 이어야 한다. 안 나오면 안 걸린 것이다
     ([[verify-the-gate-runs-that-path]]).

⚠ 전류만 흔드는 훑기를 **쓰지 마라.** "같은 상태·같은 전압인데 전류만 α배" 는 저항이 변한
  것이라 물리적으로 안 생기고 학습 분포 밖이다. 14.6 에서 그 훑기로 `k_I≈0` 을 얻어
  "모델이 전류 크기를 못 읽는다" 고 잘못 결론냈다가 철회했다 — 분포 안에서 재면 오븐을
  13W~1326W 까지 기울기 1.00 으로 읽는다.

    python -X utf8 src/run_gate_vexp.py [--ckpt results/cnn_v31.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.inputs import build_inputs
from src.model.net import V_EXP, V_REL_CLAMP
from src.model.inputs import V_CENTER
from src.run_gate_check import load_model

AL = (0.92, 0.96, 1.00, 1.04, 1.08)
RES = ("electiric_kettle", "hair_dryer", "hotplate", "oven")
SMPS = ("beam_projector", "laptop_charger", "minipc")


def windows(cache, apps, want, n=400, seed=0):
    """`want` 무리 중 하나만 켜진 창을 모은다 (다른 무리는 전부 꺼짐)."""
    C = Path(cache)
    meta = json.loads((C / "meta.json").read_text(encoding="utf-8"))
    grid = np.arange(meta["target_offset"], int(meta["record_s"] * 60) - 13 * 60 - 1,
                     int(meta["grid_s"] * 60))
    raw = np.load(C / "raw.npy", mmap_mode="r"); yon = np.load(C / "y_on.npy", mmap_mode="r")
    ki = [apps.index(x) for x in want]
    other = [j for j in range(len(apps)) if j not in ki]
    rng = np.random.RandomState(seed)
    out = []
    for i in rng.permutation(len(raw))[:1500]:
        yo = np.asarray(yon[i]).astype(bool)
        ok = (yo[:, ki].sum(1) >= 1) & (yo[:, other].sum(1) == 0)
        ts = np.nonzero(ok)[0][:3]
        if not len(ts):
            continue
        r = np.asarray(raw[i])
        for c in grid[ts]:
            seg = r[:, c - meta["target_offset"]:c - meta["target_offset"] + meta["window_cycles"]]
            # ⚠ **상하한이 물리는 창은 뺀다.** `net.V_REL_CLAMP` 가 외삽을 막느라 vrel 을 자르는데,
            #   α 훑기 끝에서 잘리면 응답이 눌려 "지수가 덜 더해진 것"처럼 보인다(1.73 대 2.00).
            #   기본 전압이 가운데 대역인 창만 써야 **기제 자체**를 잰다.
            v0 = float(np.median(seg[32]))
            lo, hi = V_REL_CLAMP
            if not (V_CENTER * lo / min(AL) < v0 < V_CENTER * hi / max(AL)):
                continue
            out.append(seg)
        if len(out) >= n:
            break
    return (np.stack(out) if out else None), ki


def k_of(model, W, ki, dev):
    """전압 α배 상황에서 그 무리의 `p_raw` 합이 `α^k` 로 따라가는 k.

    ⚠ **게이트를 곱하지 않는다.** 저전력 무리(SMPS 50~100W)에서는 α 를 내리면 게이트가
    통째로 꺼져 비가 0.67 로 튄다 — 그것은 전력 응답이 아니라 판정 뒤집힘이다.
    지수는 `p_raw` 에 곱해지므로 `p_raw` 로 재는 것이 옳다.
    """
    vals = []
    for al in AL:
        W2 = W.copy()
        W2[:, 0:30] *= al          # 전류 고조파 (I = V/R 이므로 같이 α)
        W2[:, 30] *= al * al       # 전력 = V·I
        W2[:, 32:45] *= al         # 전압 실효 + 고조파
        f, w = build_inputs(W2)
        with torch.no_grad():
            P = []
            for i in range(0, len(f), 256):
                o = model(torch.from_numpy(f[i:i + 256]).to(dev),
                          torch.from_numpy(w[i:i + 256]).to(dev))
                P.append(o["power_raw"].float())
            p = torch.cat(P).cpu().numpy()
        vals.append(float(np.median(p[:, ki].sum(1))))
    rel = np.asarray(vals) / vals[2]
    return float(np.polyfit(np.log(AL), np.log(rel), 1)[0]), rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_v31.pt")
    ap.add_argument("--cache", default="cache/seqraw_v1")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    apps = list(torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"])
    off = load_model(a.ckpt, dev, proj_from={"vexp": False})[0]
    on = load_model(a.ckpt, dev, proj_from={"vexp": True})[0]
    print("기기별 지수 V_EXP: %s" % " · ".join("%s %.1f" % (k, v) for k, v in sorted(V_EXP.items())))
    print("  켠 판의 v_exp 버퍼: %s" % " ".join("%.1f" % x for x in on.v_exp.cpu().numpy()))
    print("  끈 판의 v_exp 버퍼: %s" % " ".join("%.1f" % x for x in off.v_exp.cpu().numpy()))

    W, _ = windows(a.cache, apps, RES, n=64)
    f, w = build_inputs(W)
    with torch.no_grad():
        fa = torch.from_numpy(f).to(dev); wa = torch.from_numpy(w).to(dev)
        d = float((on(fa, wa)["power"] - off(fa, wa)["power"]).abs().max())
    print("\n① 끄면 항등인가 — 켠/끈 판의 출력 최대차 %.3e  (0 이어야 한다)"
          % float((off(fa, wa)["power"] - load_model(a.ckpt, dev)[0](fa, wa)["power"]).abs().max()))
    print("   켠 판과의 차 %.3e  (0 이 아니어야 한다)" % d)

    print("\n② 켜면 지수가 **정확히 그만큼** 더해지는가 (p_raw 로, 게이트 없이)")
    print("  %-10s %-6s %s   %s" % ("무리", "판", "   ".join("α=%.2f" % x for x in AL), "k"))
    print("  ⚠ **이미 배운 체크포인트에서 `k_켬 = 2.0` 을 기대하면 안 된다** — 헤드가 이미")
    print("     0.8 쯤 하고 있어 지수를 얹으면 **이중 계산**이다. 재학습 전 관문은")
    print("     구조가 **더한 양**이 지수와 같은지를 본다.")
    for nm, want in (("저항 4종", RES), ("SMPS 3종", SMPS)):
        W, ki = windows(a.cache, apps, want, n=400)
        if W is None:
            print("  %-10s 창 없음" % nm); continue
        e = float(np.mean([V_EXP.get(x, 0.0) for x in want]))
        ks = {}
        for tag, mo in (("끔", off), ("켬", on)):
            ks[tag], rel = k_of(mo, W, ki, dev)
            print("  %-10s %-6s %s   k=%5.2f"
                  % (nm if tag == "끔" else "", tag, "   ".join("%6.3f" % x for x in rel), ks[tag]))
        d = ks["켬"] - ks["끔"]
        print("  %-10s %-6s 더해진 양 %+.2f · 기대 %+.2f -> %s   (재학습 뒤 목표 k_켬 = %.1f)"
              % ("", "", d, e, "통과" if abs(d - e) < 0.25 else "**실패**", e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
