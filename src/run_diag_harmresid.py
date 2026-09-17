# -*- coding: utf-8 -*-
"""**`HARM_DEADZONE_PROFILE` 을 지금 계측기로 다시 잰다** (14.411).

⚠ 알려진 상계 요인: `harm_sig_vnorm`(창 전압 보정)은 안 걸었다. 걸면 바닥이
  조금 더 내려간다. 즉 여기서 내는 값은 **바닥의 상계**다.

왜 다시 재나
-----------
지금 표는 `losses.py:160` 에 박혀 있고 출처가 이렇다:

    "사람 라벨 5파일의 60초 창 55개에서 min_{P>=0} ‖y − A_정답·P‖ 의 잔차다.
     (2026-09-01 측정)"

**2026-09-01 은 새 계측기(2026-09-06~)보다 앞이다.** 그때의 '사람 라벨 5파일' 은
옛 계측기 녹화였고 지금은 삭제됐다. 즉 이 표는 **없는 자료에서 잰 값**이다.
그런데 §64 가 이 표로 "86% 가 못 줄이는 몫" 을 계산했고, 14.408 이 이 표를
1단계 불감대로 걸었다. 자를 안 재고 쓴 것이다 ([[check-conditioning-before-believing-a-fit]]).

무엇을 재나
----------
`min_{P>=0} ‖y − A_정답·P‖` 의 **차수별 잔차 중앙값**, 손실 단위로.
"손실 단위" 는 `NILMLoss._harm_err` 가 내는 값이다 — 짝수차 크기 공간 규약과
`harm_scale` 정규화가 거기 들어 있으므로 **손실 객체를 실제로 지어서 그 메서드로 잰다**
([[the-gate-must-build-the-real-object]]). 따로 구현하면 규약이 조용히 갈린다.

    python -X utf8 -m src.run_diag_harmresid
    python -X utf8 -m src.run_diag_harmresid --stems test_1 test_7 --stride 60

⚠ 이 도구는 `processed_data/composite_eval` 을 읽는다 — **채점 전용 자료**다.
  HPC 에 올리지 않는다 (HPC_RULES §0). 로컬에서만 돈다.
"""
from typing import Dict, List
import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401,E402

import torch  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from src.model.lossbuild import build_loss  # noqa: E402
from src.model.losses import HARM_DEADZONE_PROFILE as OLD  # noqa: E402
from src.model.realdata import dense_targets  # noqa: E402
from src.run_scorecard import load_events  # noqa: E402
from src.synthesis.segment_pool import SegmentPool  # noqa: E402

APPS = ("air_conditioner", "beam_projector", "electiric_kettle", "fan", "hair_dryer",
        "hotplate", "laptop_charger", "minipc", "oven")
#: 라벨 이름 -> 풀 이름 (실측 라벨은 `electric_`, 풀은 `electiric_` 오타가 정본이다)
ALIAS = {"electric_kettle": "electiric_kettle"}
DEFAULT_STEMS = ("test_1", "test_2", "test_3", "test_4", "test_5", "test_7")


def on_set(ev: dict, stem: str, t0: float, t1: float) -> List[str]:
    """창 [t0,t1] 에서 **내내 켜져 있던** 기기. 반쯤 걸친 것은 뺀다 —
    전이가 든 창은 정상상태 분해가 성립하지 않는다."""
    out = []
    f = ev[stem]
    for app, iv in f.get("intervals", {}).items():
        a = ALIAS.get(app, app)
        if a not in APPS:
            continue
        for lo, hi in iv.get("on", []):
            if lo <= t0 and hi >= t1:
                out.append(a)
                break
    return out


def partial(ev: dict, stem: str, t0: float, t1: float) -> bool:
    """창 안에 전이가 있나 (있으면 그 창은 버린다)."""
    for e in ev[stem].get("events", []):
        if t0 - 1.0 <= float(e["t_s"]) <= t1 + 1.0:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", nargs="*", default=list(DEFAULT_STEMS))
    ap.add_argument("--stride", type=int, default=120, help="창 간격 (사이클). 2초 기본")
    ap.add_argument("--min-windows", type=int, default=20)
    a = ap.parse_args()

    #: ⚠ 학습이 부르는 그대로 짓는다 — `cnn_comb_seeds.sbatch` 가 `--harm-even-magnitude
    #:   --state-signatures --standby-operating session` 을 건다. 규약이 갈리면
    #:   여기서 잰 잔차가 손실이 보는 잔차가 아니게 된다.
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    loss = build_loss(list(APPS), "cpu", harm_even_magnitude=True,
                      state_signatures=True, standby_operating="session")
    hs = loss.harm_scale.numpy().astype(np.float64)
    sig = loss.sig.numpy().astype(np.float64)              # (K,H,2)
    #: ★ 손실은 **상태별** 지문으로 편다 (`_harm_pred_active`: pw = gate·mix·p_states,
    #  pred = Σ_{k,s} pw·sig_state[k,s]). 기기별 `sig` 로 풀면 자유도가 모자라
    #  바닥을 **높게** 잡는다. 열을 (기기,상태) 로 놓아야 손실이 도달 가능한 값이다.
    sigs = loss.sig_state.numpy().astype(np.float64)       # (K,S,H,2)
    sb = loss.standby_sig.numpy().astype(np.float64)       # (K,H,2)
    nz = loss.noise_sig.numpy().astype(np.float64)         # (H,2)
    H = sig.shape[1]
    print("  손실 객체: K=%d · H=%d · harm_even_mag=%s · harm_mask 평균 %.3f"
          % (sig.shape[0], H, loss.harm_even_mag, float(loss.harm_mask.mean())))
    print("  harm_scale (mA): " + " ".join("h%d %.2f" % (h + 1, hs[h] * 1000)
                                           for h in range(0, H, 2)))

    ev = load_events()
    rows: List[np.ndarray] = []
    per_stem: Dict[str, int] = {}
    for stem in a.stems:
        if stem not in ev:
            print("  %s — 라벨이 없다, 건너뛴다" % stem)
            continue
        rw = dense_targets(stem, stride=a.stride, site_transfer=None)
        tgt = np.asarray(rw.target_cycle, float) / 60.0
        obs = np.asarray(rw.obs_harm, float)                # (n,H,2)
        n_ok = 0
        for i in range(len(rw)):
            t1 = float(tgt[i])
            t0 = t1 - 60.0
            if t0 < 0:
                continue
            if partial(ev, stem, t0, t1):
                continue
            on = on_set(ev, stem, t0, t1)
            if not on:
                continue
            k_on = [APPS.index(x) for x in on]
            #: 목표에서 **기기가 아닌 몫**을 먼저 뺀다 — 계측잡음 + 꺼진 기기의 대기전류.
            #  손실이 `pred` 에 더하는 것과 같은 항이다.
            off = [j for j in range(len(APPS)) if j not in k_on]
            base = nz + sb[off].sum(0)
            y = obs[i] - base                               # (H,2)
            #: 백색화한 실벡터로 펴서 NNLS. 표의 정의가 `min_{P>=0} ‖y − A·P‖` 다.
            cols = sigs[k_on].reshape(-1, sigs.shape[2], 2)          # (len(on)*S, H, 2)
            A = np.concatenate([cols[:, :, 0] / hs, cols[:, :, 1] / hs], axis=1).T
            b = np.concatenate([y[:, 0] / hs, y[:, 1] / hs])
            try:
                p, _ = nnls(A, b)
            except Exception:
                continue
            pred = base + np.einsum("k,khc->hc", p, cols)
            e = loss._harm_err(torch.from_numpy(pred[None]).float(),
                               torch.from_numpy(obs[i][None]).float()).numpy()[0]
            rows.append(e.mean(-1))                         # (H,) 성분 평균
            n_ok += 1
        per_stem[stem] = n_ok
        print("  %-8s 창 %d개 (stride %.1f초)" % (stem, n_ok, a.stride / 60.0))

    if len(rows) < a.min_windows:
        print("\n✖ 창이 %d개뿐이다 (>=%d 필요). stride 를 줄여라" % (len(rows), a.min_windows))
        return 1
    R = np.stack(rows)                                      # (n,H)
    med = np.median(R, axis=0)
    print("\n  창 **%d개** · 파일 %d개" % (len(R), len([k for k, v in per_stem.items() if v])))
    print("\n  차수 |  옛 표(12.122.16)  |  **새 측정**  |  비  | p25~p75")
    for h in range(H):
        o = OLD[h] if h < len(OLD) else float("nan")
        print("   h%-3d| %14.3f    | %11.3f  | %4.2f | %.3f~%.3f"
              % (h + 1, o, med[h], med[h] / max(o, 1e-9),
                 np.percentile(R[:, h], 25), np.percentile(R[:, h], 75)))
    odd = np.array([h % 2 == 0 for h in range(H)])          # h1,h3,... 는 색인 0,2,...
    print("\n  홀수 평균  옛 %.3f -> **%.3f**   ·  짝수 평균  옛 %.3f -> **%.3f**"
          % (np.mean([OLD[h] for h in range(H) if h % 2 == 0]), med[odd].mean(),
             np.mean([OLD[h] for h in range(H) if h % 2 == 1]), med[~odd].mean()))
    print("  전 차수 평균  옛 **%.3f** -> 새 **%.3f**"
          % (np.mean(OLD[:H]), med.mean()))
    print("\n  새 표 (losses.py 에 붙일 꼴):")
    print("  HARM_DEADZONE_PROFILE = [" + ", ".join("%.3f" % x for x in med) + "]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
