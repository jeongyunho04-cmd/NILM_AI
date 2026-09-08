"""자리·구성으로 갈라 본다 — 평균이 지우는 것을 되살린다 (13.57)
=================================================================
**AUC 로도 `run_power_check` 로도 안 보이는 실패가 있다.**

    AUC              순위 기반이라 그 기기의 ON 창이 **다 같이** 눌리면 안 움직인다
    run_power_check  `P̂ = σ(on)·p_raw` 라 게이트가 꺼진 창을 표본에서 빼버린다.
                     그래서 무너진 판과 멀쩡한 판이 똑같이 2~3W 로 나온다
                     (그 도구 문서의 "검출률이 낮으면 배분 오차를 읽지 말 것")

여기서는 **정답 ON 이고 채점 가능한 창 전부**에서 예측 전력을 참값과 직접 견준다.
모델 자신의 게이트로 창을 거르지 않는다. 그리고 창을 **자리 × 구성**으로 가른다 —
13.57 이 이걸로 "test_4 만 이상하다" 를 "자리 D × SMPS 전용 × 적응" 으로 고쳤다.

    표 ①  파일 × 구성별 Z 추정 · 프로젝터 관문 · 프로젝터 예측W
    표 ②  자리 D 의 SMPS 전용 창에서 3종 배분 (프로젝터 40W 가 어디로 가나)

    python -m src.run_site_split_check --ckpt results/cnn_v24b.pt results/adapt_v24b_z_s0.pt
"""
from typing import List
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.realdata import dense_targets
from src.preprocessing.file_registry import SITE_SESSIONS
from src.run_gate_check import load_model

STEMS = ["test_1", "test_2", "test_3", "test_4", "test_5"]
SMPS = ("beam_projector", "laptop_charger", "minipc")
BIG = ("electiric_kettle", "oven", "hotplate", "hair_dryer")
#: 자리 D 의 SMPS 전용 창에서 셋 다 켜졌을 때의 참 전력 (13.52 조합 차분)
TRUE_W = {"beam_projector": 43.0, "laptop_charger": 39.0, "minipc": 8.0}
#: stem -> 그 세션의 잰 Z. `SITE_SESSIONS` 에서 되짚는다 (하드코딩하지 않는다).
TRUE_Z = {s: v["z_ohm"] for v in SITE_SESSIONS.values() for s in v["stems"]
          if v["z_ohm"] is not None}
SITE_OF = {s: k for k, v in SITE_SESSIONS.items() for s in v["stems"]}


@torch.no_grad()
def scan(model, stem: str, apps: List[str], dev: str, stride: int = 30):
    """파일 하나를 훑어 관문·전력·Z추정과 정답 마스크를 돌려준다."""
    st = getattr(model, "site_transfer", None)
    rw = dense_targets(stem, stride=stride, site_transfer=st)
    G, LZ, P, POBS = [], [], [], []
    for i in range(0, len(rw), 512):
        f, w, pobs, _oh, _pn = rw.batch(np.arange(i, min(i + 512, len(rw))))
        o = model(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                  torch.from_numpy(np.ascontiguousarray(w)).to(dev))
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
        P.append(o["power"].float().cpu().numpy())
        POBS.append(pobs)
        if "log_z" in o:
            LZ.append(o["log_z"].float().cpu().numpy())
    ev = load_events()
    n_cyc = int(ev[stem]["cycles"])
    on, sc = build_on_off_truth(stem, apps, n_cyc, ev)
    t = np.asarray(rw.target_cycle, int).clip(0, n_cyc - 1)
    return (np.concatenate(G), np.concatenate(P), np.concatenate(POBS),
            np.concatenate(LZ) if LZ else None, on[t], sc[t])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--skip-split", action="store_true", help="표 ① 을 건너뛴다")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for p in a.ckpt:
        m, apps, _ = load_model(p, dev)
        jb = [apps.index(x) for x in BIG]
        jp = apps.index("beam_projector")
        js = [apps.index(x) for x in SMPS]
        D = {s: scan(m, s, apps, dev, a.stride) for s in STEMS}

        if not a.skip_split:
            print("=" * 104)
            print(f"① 파일 × 구성 — {p}")
            print("=" * 104)
            print(f"  {'파일 · 구성':30s}{'창':>6s}{'관측W':>9s}{'Z추정':>8s}{'참Z':>7s}"
                  f"{'프로관문':>10s}{'프로예측W':>11s}{'프로창':>8s}")
            for s in STEMS:
                G, P, POBS, LZ, ON, SC = D[s]
                res = ON[:, jb].any(1)
                for nm, mask in (("SMPS 만 (저항 0)", ~res), ("저항 있음", res)):
                    if mask.sum() < 20:
                        continue
                    pm = mask & ON[:, jp] & SC[:, jp]
                    z = float(np.exp(np.median(LZ[mask]))) if LZ is not None else float("nan")
                    tail = (f"{np.median(G[pm, jp]):10.3f}{np.median(P[pm, jp]):11.1f}"
                            if pm.sum() >= 20 else f"{'—':>10s}{'—':>11s}")
                    print(f"  {s + ' · ' + nm:30s}{int(mask.sum()):6d}"
                          f"{np.median(POBS[mask]):9.1f}{z:8.2f}"
                          f"{TRUE_Z.get(s, float('nan')):7.2f}{tail}{int(pm.sum()):8d}")
            print("  ⚠ 같은 파일 안에서 SMPS 창과 저항 창을 견줄 것. 파일끼리 견주면"
                  " 구성 차이가 파일 차이로 보인다 (13.57.1)")

        # ② 자리 D · SMPS 전용 · 셋 다 켜진 창
        acc = {"P": [], "O": []}
        for s in STEMS:
            if SITE_SESSIONS.get(SITE_OF.get(s, ""), {}).get("site") != "D":
                continue
            G, P, POBS, LZ, ON, SC = D[s]
            k = (~ON[:, jb].any(1)) & ON[:, js].all(1) & SC[:, js].all(1)
            if k.sum() < 20:
                continue
            acc["P"].append(P[k]); acc["O"].append(POBS[k])
        if acc["P"]:
            P = np.concatenate(acc["P"]); O = np.concatenate(acc["O"])
            med = [float(np.median(P[:, j])) for j in js]
            print("\n" + "=" * 104)
            print("② 자리 D · SMPS 전용 · 셋 다 켜진 창 — 배분 (참 43/39/8W)")
            print("=" * 104)
            print(f"  {'창':>6s}" + "".join(f"{n[:12]:>14s}" for n in SMPS)
                  + f"{'SMPS합':>10s}{'예측총합':>10s}{'관측W':>9s}")
            print(f"  {len(P):6d}" + "".join(f"{v:14.1f}" for v in med)
                  + f"{sum(med):10.1f}{np.median(P.sum(1)):10.1f}{np.median(O):9.1f}")
            print(f"  {'참값':>6s}" + "".join(f"{TRUE_W[n]:14.1f}" for n in SMPS)
                  + f"{sum(TRUE_W.values()):10.1f}")
        del m
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
