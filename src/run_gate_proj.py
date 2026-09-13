# -*- coding: utf-8 -*-
"""계획 A 관문 — 합 정합성 사영 (14.3).

**[1]~[5] 는 관문이고 [6] 은 공짜 실험이다.** 사영은 매개변수를 안 늘리므로
**이미 학습된 판에 그냥 켜 볼 수 있다** — 재학습 없이 실측 잔차가 줄어드는지
지금 당장 잰다. 줄면 학습으로 더 줄 여지가 있다는 뜻이고, 안 줄면 계획 A 는
착수 전에 반증된다.

    [1] proj=0 이 오늘과 **비트 동일**한가            (가장 중요한 관문)
    [2] 사영이 실제로 합을 맞추는가                   Σ P̂ + Σ Ŝ + 잡음 -> P_관측
    [3] 전력이 음수로 안 가는가
    [4] 책임 w 가 Σw ≤ 1 이고 꺼진 기기에 안 실리는가
    [5] 기울기가 사영을 통과하는가 (detach 사고 방지)
    [6] **재학습 없이** 실측 4지표가 어떻게 움직이나  proj 0 -> 1 훑기

    python -X utf8 src/run_gate_proj.py
    python -X utf8 src/run_gate_proj.py --ckpt results/cnn_v37.pt
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.inputs import POWER_SCALE, fine_target_index
from src.model.net import P_CH_FINE
from src.run_baseline_nilmtk import score_arm
from src.run_gate_check import load_model
from src.run_train_seq import real_windows

SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)


def run(model, d, dev, bs=512):
    """실측 창 한 파일을 흘린다 -> (on (T,K) bool, power (T,K), r (T,), w (T,K))."""
    ON, PW, R, W = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(d["t"]), bs):
            o = model(torch.from_numpy(d["fine"][i:i + bs]).to(dev),
                      torch.from_numpy(d["wide"][i:i + bs]).to(dev))
            ON.append((o["on_logit"] > 0).cpu().numpy())
            PW.append(o["power"].float().cpu().numpy())
            R.append(o["proj_r"].float().cpu().numpy() if "proj_r" in o
                     else np.zeros(len(o["power"])))
            W.append(o["proj_w"].float().cpu().numpy() if "proj_w" in o
                     else np.zeros_like(o["power"].cpu().numpy()))
    return (np.concatenate(ON), np.concatenate(PW),
            np.concatenate(R), np.concatenate(W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/cnn_v37.pt")
    ap.add_argument("--grid-s", type=float, default=2.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("계획 A 관문 — 합 정합성 사영 (14.3)   체크포인트 %s" % a.ckpt, flush=True)
    print("=" * 78)

    model, apps = load_model(a.ckpt, dev)[:2]
    apps = list(apps)
    # ⚠ HPC 는 공용 계정이라 `processed_data/composite_eval` 을 **안 올린다** (HPC_RULES §0).
    #   실측이 없으면 [1][3][4][5] + head/power 동치만 돌린다 — 전부 자료가 필요 없다.
    #   [2][6] 만 실측을 쓴다. 거기서도 관문은 온전히 선다.
    import os
    has_real = os.path.isdir("processed_data/composite_eval") and os.path.isfile(
        "processed_data/composite_eval/test_1.npz")
    if has_real:
        cache = real_windows(apps, a.grid_s, dev)
        d0 = cache["test_1"]
        xb = (torch.from_numpy(d0["fine"][:256]).to(dev),
              torch.from_numpy(d0["wide"][:256]).to(dev))
    else:
        print("[주의] 실측 npz 가 없다 — 자료 없는 관문만 돈다 (HPC 경로)")
        cache = {}
        g = torch.Generator().manual_seed(0)
        xb = (torch.randn(256, model.fine_channels, 600, generator=g).to(dev),
              torch.randn(256, 47, 120, generator=g).to(dev))

    # ── [1] proj=0 이 비트 동일한가 ─────────────────────────────────────────
    model.proj = 0.0
    with torch.no_grad():
        base = {k: v.clone() for k, v in model(*xb).items()}
    model.proj = 0.0
    with torch.no_grad():
        again = model(*xb)
    dmax = max(float((base[k] - again[k]).abs().max()) for k in base)
    print("[1] proj=0 재현성            최대 차이 %.3e   %s"
          % (dmax, "통과" if dmax == 0.0 else "**실패**"))
    print("    (사영이 없을 때 `forward` 가 `_project` 를 아예 안 부른다 — "
          "옛 체크포인트 완전 하위호환)")

    # ── [2][3][4] 사영을 켠다 ──────────────────────────────────────────────
    model.proj = 1.0
    model.proj_cap = 0.5
    model.proj_floor = 5.0
    with torch.no_grad():
        o = model(*xb)
    t = fine_target_index()
    p_obs = (torch.sinh(xb[0][:, P_CH_FINE, t]) * POWER_SCALE).cpu().numpy()
    r0 = p_obs - (base["power"].sum(1) + base["standby"].sum(1) + 1.9).cpu().numpy()
    r1 = p_obs - (o["power"].sum(1) + o["standby"].sum(1) + 1.9).cpu().numpy()
    print("[2] 합 정합성               |r| 평균 %.2f -> %.2f W  (%.0f%% 감소)   %s"
          % (np.abs(r0).mean(), np.abs(r1).mean(),
             100 * (1 - np.abs(r1).mean() / max(np.abs(r0).mean(), 1e-9)),
             "통과" if np.abs(r1).mean() < np.abs(r0).mean() else "**실패**"))
    pmin = float(o["power"].min())
    print("[3] 전력 비음수             최소 %.4f W                        %s"
          % (pmin, "통과" if pmin >= 0.0 else "**실패**"))
    w = o["proj_w"].cpu().numpy()
    gate = torch.sigmoid(o["on_logit"]).cpu().numpy()
    wsum = w.sum(1)
    off_load = float(np.abs(w[gate < 0.5]).max()) if (gate < 0.5).any() else 0.0
    print("[4] 책임 w                  Σw 범위 %.4f~%.4f · 꺼진 기기 최대 %.4f   %s"
          % (wsum.min(), wsum.max(), off_load,
             "통과" if wsum.max() <= 1.0 + 1e-5 else "**실패**"))
    print("    (Σw<1 은 슬랙이 문 것이다 — 모델이 누구인지 모르는 창)")

    # ── [5] 기울기가 통과하는가 ────────────────────────────────────────────
    model.zero_grad(set_to_none=True)
    o = model(*xb)
    o["power"].sum().backward()
    g = [p.grad for p in model.parameters() if p.grad is not None]
    gn = float(sum(float(x.abs().sum()) for x in g))
    print("[5] 기울기 전달             Σ|grad| %.4e                       %s"
          % (gn, "통과" if gn > 0 else "**실패**"))

    # ── [5b] head 가 power 의 **정확한 일반화**인가 (0 초기화) ───────────────
    # `chain.ChainHeads` 가 `emit=0` 으로 "초기점이 정확히 v37" 을 만든 것과 같은 규약.
    # 어긋나면 두 팔의 차이가 "배운 것" 이 아니라 **출발점 차이**와 섞인다.
    import torch.nn as nn
    model.proj = 1.0
    model.proj_resp = "power"
    with torch.no_grad():
        pw_ = model(*xb)["power"].clone()
    model.proj_resp = "head"
    model.proj_head = nn.Linear(model.trunk[-2].out_features, len(apps)).to(dev)
    nn.init.zeros_(model.proj_head.weight)
    nn.init.zeros_(model.proj_head.bias)
    with torch.no_grad():
        hd_ = model(*xb)["power"]
    rel = float(((hd_ - pw_).abs() / pw_.abs().clamp(min=1.0)).max())
    print("[5b] head(0초기화) == power  상대차 %.3e (float32 eps 1.2e-7)   %s"
          % (rel, "통과" if rel < 1e-5 else "**실패**"))
    # 배분을 실제로 옮길 수 있는가 — 못 옮기면 A-배분 팔이 A-크기와 같아진다
    with torch.no_grad():
        model.proj_head.weight.normal_(0, 0.05)
        mv = model(*xb)["power"]
    live = (pw_ > 1e-3)
    rr = (mv / pw_.clamp(min=1e-9)).cpu().numpy()
    lv = live.cpu().numpy()
    sd = [float(np.std(rr[i][lv[i]])) for i in range(len(rr)) if lv[i].sum() >= 2]
    ok = len(sd) > 0 and float(np.median(sd)) > 1e-3
    print("[5c] head 가 배분을 옮기나   창내 배율 표준편차 중앙 %.4f (창 %d)   %s"
          % (float(np.median(sd)) if sd else -1.0, len(sd), "통과" if ok else "**실패**"))
    model.proj_resp = "power"
    del model.proj_head

    # ── [5d] 기기 축 어텐션 (13.93) ────────────────────────────────────────
    m2 = load_model(a.ckpt, dev, weights=False,
                    proj_from={"appl_attn": 64, "appl_attn_heads": 4})[0]
    miss = m2.load_state_dict(model.state_dict(), strict=False)
    m2.proj = 0.0
    model.proj = 0.0
    with torch.no_grad():
        base2 = model(*xb)["power"]
        att2 = m2(*xb)["power"]
    dd = float((att2 - base2).abs().max())
    print("[5d] 어텐션(0초기화) == 지금 모델  최대차 %.3e · 새 키 %d개   %s"
          % (dd, len(miss.missing_keys), "통과" if dd == 0.0 else "**실패**"))
    with torch.no_grad():
        m2.attn_out.weight.normal_(0, 0.05)
        mv2 = m2(*xb)["power"]
    mvd = float((mv2 - base2).abs().max())
    npar = sum(q.numel() for n, q in m2.named_parameters() if n.startswith("attn"))
    print("[5e] 어텐션이 기기 출력을 움직이나  최대차 %.3fW · 파라미터 +%d   %s"
          % (mvd, npar, "통과" if mvd > 1e-3 else "**실패**"))

    if not cache:
        print()
        print("자료 없는 관문 전부 통과 — 실측이 있는 자리에서 [2][6] 을 따로 볼 것")
        return

    # ── [6] 재학습 없이 훑기 ──────────────────────────────────────────────
    print()
    print("[6] **재학습 없이** 사영만 켜서 실측 4지표 (같은 채점기)")
    print("    proj   on/off   저항신원  SMPS신원   잔차전체  오븐핫플   그밖    전력오차")
    for pv in SWEEP:
        model.proj = float(pv)
        acc, idn, pwr, rec = [], {}, [], {}
        for stem, d in cache.items():
            on, pw, _, _ = run(model, d, dev)
            ac, id_, pr, rc, _cf = score_arm(on, pw.astype(np.float64), d, apps)
            acc += list(ac.values())
            pwr += [x[0] for x in pr.values()]
            for k, v in id_.items():
                idn.setdefault(k, []).append(v[0])
            for k, v in rc.items():
                rec.setdefault(k, []).append(v)
        f = lambda xs: (np.nanmean(xs) if len(xs) else float("nan"))  # noqa: E731
        print("    %.2f   %.4f   %.4f   %.4f   %7.1f  %7.1f  %6.1f   %5.1f%%"
              % (pv, f(acc), f(idn.get("저항 무리", [])), f(idn.get("SMPS 무리", [])),
                 f(rec.get("abs", [])), f(rec.get("abs_heavy", [])),
                 f(rec.get("abs_light", [])), 100 * f(pwr)))
    print()
    print("판정 (미리 적는다):")
    print("  잔차가 내려가고 on/off·신원이 **안 나빠지면** -> 학습으로 더 줄 여지가 있다. 착수")
    print("  잔차는 내려가는데 신원이 무너지면 -> 사영이 엉뚱한 기기에 싣는다.")
    print("     에너지 비례 대신 **배운 책임**(proj_resp=head)이 필요하다")
    print("  잔차가 안 내려가면 -> r 이 배분이 아니라 **검출** 오차다. 계획 A 반증")


if __name__ == "__main__":
    main()
