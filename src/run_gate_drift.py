# -*- coding: utf-8 -*-
"""고정 표류 사영 관문 — 꺼지면 항등, 켜지면 동작 (13.84.60).

13.84.52 ⓐ 를 구현했다: `run_build_drift.py` 가 굳힌 부분공간을 (2H,2H) 행렬로 만들어
`L_harm` 의 오차에서 깎는다. **파일별 추정도 신탁도 없다** — 학습 전에 한 번 굳힌 상수다.

왜: 13.84.56~57 이 표류가 **동작점의 함수**이고 실효 3차원임을 쟀다. 그 방향의 오차는
배분과 무관한 순방향 모형 오차인데 `L_harm` 이 그것도 줄이라고 밀면 기기 전력으로 흡수된다.

관문 넷:
  [1] 항등    `--drift-k 0` 이면 사영이 안 들어가고 손실이 **비트 단위로** 같다
  [2] 동작    k>=1 이면 표류 방향 오차가 벌점에서 **사라진다** (그 방향만 틀린 예측의 L_harm=0)
  [3] 비용    사영이 **각 기기 지문**을 얼마나 깎나 — 미니PC 만 보면 안 된다
  [4] 기울기  `∂L_harm/∂P̂` 가 얼마나 바뀌나 (배분을 정하는 것이 이 기울기다)

    python -X utf8 src/run_gate_drift.py [--k 1 2 3]
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch

from src.model.lossbuild import build_drift_proj, build_loss

#: 체크포인트에서 가져온다 — 저장소의 이름을 그대로 쓴다 (`electiric_kettle` 오타 포함)
APPS = ["air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven"]


def fake(crit, K, H, B=4, seed=0):
    """손실이 먹는 최소한의 (out, tgt). 라벨 없음 — 관문은 수치만 본다."""
    g = torch.Generator().manual_seed(seed)
    pw = torch.rand(B, K, generator=g) * 40.0
    out = {"power": pw.clone().requires_grad_(True),
           "power_raw": pw.clamp(min=1e-3),
           "on_logit": torch.randn(B, K, generator=g),
           "plugged_logit": torch.randn(B, K, generator=g),
           "power_mix": None, "power_states": None}
    obs = torch.einsum("bk,khc->bhc", pw, crit.sig) + crit.noise_sig[None]
    obs = obs + 0.02 * torch.randn(B, H, 2, generator=g)
    return out, {"obs_harm": obs}


def harm_of(crit, out, tgt):
    pred = crit._harm_pred_active(out, out["power"])
    idle = torch.sigmoid(out["plugged_logit"]) * (1.0 - torch.sigmoid(out["on_logit"]))
    pred = pred + torch.einsum("bk,khc->bhc", idle, crit.standby_sig) + crit.noise_sig[None]
    err = crit._harm_err(pred, tgt["obs_harm"], None)
    return ((err * crit.harm_mask[None, :, None]).mean() / crit.harm_mask.mean().clamp(min=1e-6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="results/drift_basis.npz")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()

    base = build_loss(APPS, "cpu", verbose=False)
    H = base.harm_scale.numel(); K = len(APPS)
    out, tgt = fake(base, K, H)
    L0 = harm_of(base, out, tgt)
    g0 = torch.autograd.grad(L0, out["power"], retain_graph=True)[0].clone()
    print("차수 %d · 기기 %d · 기준 L_harm %.8f" % (H, K, float(L0)))

    # ── [1] 항등 ────────────────────────────────────────────────────────────
    off = build_loss(APPS, "cpu", verbose=False, drift_basis=a.basis, drift_k=0)
    o1, t1 = fake(off, K, H)
    L1 = harm_of(off, o1, t1)
    same = (float(L1) == float(L0)) and off.drift_proj.numel() == 0
    print("\n[1] 항등  `--drift-k 0` -> 사영 버퍼 %d개 · L_harm %.8f · %s"
          % (off.drift_proj.numel(), float(L1), "**같다**" if same else "!! 다르다 !!"))

    B = np.load(a.basis, allow_pickle=True)
    ordr = [int(x) for x in B["orders"]]; idx = [o - 1 for o in ordr]; ho = len(ordr)
    Vb = np.asarray(B["V"], np.float64)

    print("\n[2] 동작 · [3] 비용 · [4] 기울기")
    print("   %-4s %11s %10s | %s" % ("k", "L_harm", "표류만틀린", "지문 남는 비율 (사영 뒤 / 원래)"))
    print("   %-4s %11s %10s | %s" % ("", "", "판의 L", "  ".join("%-7s" % x[:7] for x in APPS)))
    for k in a.k:
        crit = build_loss(APPS, "cpu", verbose=False, drift_basis=a.basis, drift_k=k)
        o, t = fake(crit, K, H)
        L = harm_of(crit, o, t)
        g = torch.autograd.grad(L, o["power"], retain_graph=True)[0]

        # [2] 표류 방향으로만 틀린 예측을 만들어 벌점이 0 인지 본다
        P = crit.drift_proj.numpy()
        d = np.zeros((1, H, 2))
        v = Vb[:k].sum(0)                          # 표류 부분공간 안의 한 벡터
        d[0, idx, 0] = v[:ho]; d[0, idx, 1] = v[ho:]
        o2, t2 = fake(crit, K, H)
        t2["obs_harm"] = t2["obs_harm"] + 0.05 * torch.from_numpy(d).float()
        t2b = {"obs_harm": t2["obs_harm"].clone()}
        Ld = float(harm_of(crit, o2, t2)) - float(harm_of(crit, o2, {"obs_harm": t2["obs_harm"] - 0.05 * torch.from_numpy(d).float()}))

        # [3] 각 기기 지문이 사영 뒤 얼마나 남나
        keep = []
        for j in range(K):
            sg = crit.sig[j].numpy()               # (H,2)
            w = np.concatenate([sg[:, 0], sg[:, 1]])
            keep.append(np.linalg.norm(P @ w) / max(np.linalg.norm(w), 1e-30))
        print("   %-4d %11.8f %10.2e | %s"
              % (k, float(L), abs(Ld), "  ".join("%6.1f%%" % (100 * v) for v in keep)))
        gd = float((g - g0).abs().max() / g0.abs().max())
        print("   %-4s %11s %10s |   기울기 최대 변화 %.1f%% · 미니PC 열 %.1f%% · L_harm %+.1f%%"
              % ("", "", "", 100 * gd,
                 100 * float((g[:, APPS.index("minipc")] - g0[:, APPS.index("minipc")]).abs().max()
                             / g0[:, APPS.index("minipc")].abs().max()),
                 100 * (float(L) / float(L0) - 1)))

    print("\n   읽는 법 — [2] 의 '표류만틀린 판의 L' 이 0 에 가까워야 사영이 **그 방향을**")
    print("   안 벌한다는 뜻이다. [3] 에서 미니PC 말고 다른 기기가 크게 깎이면 그 기기의")
    print("   전력이 L_harm 에서 식별을 잃는다 — k 를 줄여라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
