"""순방향 모델 오차는 **중첩이 깨져서** 생기는가 (12.125)

왜 재는가
--------
12.122.2 가 *"관측이 오답을 선호한다"* 를 확정했고 그 원인이 **순방향 모델
오차**다. `L_harm` 은 실측 창에서

    pred = Σ P_i · sig_i + Σ대기 + 계측        (sig_i 는 격리 녹화의 상수)

를 관측과 맞춘다. 정답 배분을 줘도 7~28% 가 안 맞고, 그 압력이 배분을 민다.

**그 오차가 어디서 오는가에 따라 고칠 방법이 완전히 갈린다:**

    (A) 기기 하나하나의 지문이 실측 조건에서 틀렸다
        -> in-situ 적합이 고친다 (12.122.11, 검증됨 −18%). 합성기와 무관하다

    (B) 여러 기기가 같이 돌 때 **중첩이 깨진다** (공통 전원 임피던스로 전압이
        일그러지고 각 기기의 전류 고조파가 서로 바뀐다)
        -> 지문을 아무리 다듬어도 못 고친다. **순방향 물리를 고쳐야 하고,
           그것은 합성기와 `L_harm` 이 함께 쓰는 자리다**

무엇을 재면 갈리는가
-----------------
정답 배분 잔차를 ‖y‖ 로 정규화해 **동시 통전 기기 수**와 **구성**으로 층을 낸다.

    (A) 라면 상대잔차가 기기 수에 대체로 평평하다
    (B) 라면 기기 수를 따라 **오른다**

그리고 **저항 전용 창**이 결정적이다 — 저항은 선형 부하라 지문이 가장 단순하다.
거기서도 상대잔차가 크면 (B) 든 (A) 든 SMPS 의 미묘함 얘기가 아니다.

    python -m src.run_forward_gap_probe
"""
from pathlib import Path
from typing import Dict, List, Sequence
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import numpy as np
import torch
from scipy.optimize import nnls

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.realdata import SMPS_APPLIANCES
from src.preprocessing import load_nilm_npz
from src.run_fit_insitu_sig import EVAL_DIR, WINDOW
from src.run_summarize_gate import human_stems

H = 15


def collect(apps: Sequence[str], nz: np.ndarray, stems: Sequence[str]) -> List[dict]:
    ev = load_events()
    NZ = nz[:H, 0] + 1j * nz[:H, 1]
    out: List[dict] = []
    for stem in stems:
        f = Path(EVAL_DIR) / f"{stem}.npz"
        if stem not in ev or not f.exists():
            continue
        z = load_nilm_npz(str(f))
        hc = np.asarray(z["harmonics_complex"])
        pf = np.asarray(z["power_features"])
        ok = np.asarray(z["is_valid"]).astype(bool)
        on, _ = build_on_off_truth(stem, apps, len(hc), ev)
        i = np.flatnonzero(ok)
        for k in range(0, len(i) - WINDOW, WINDOW):
            sl = i[k:k + WINDOW]
            sup = np.flatnonzero(on[sl].mean(0) > 0.5)
            if not len(sup):
                continue
            out.append({"stem": stem, "sup": sup, "y": hc[sl].mean(0)[:H] - NZ,
                        "pobs": float(pf[sl, 0].mean())})
    return out


def resid(sig: np.ndarray, idx: Sequence[int], y: np.ndarray) -> float:
    if not len(idx):
        return float(np.linalg.norm(np.concatenate([y.real, y.imag])))
    A = np.array([np.concatenate([sig[j, :H, 0], sig[j, :H, 1]]) for j in idx]).T
    _, r = nnls(A, np.concatenate([y.real, y.imag]))
    return float(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="results/cnn_ovh.pt")
    ap.add_argument("--sig-insitu", default="results/sig_insitu.npz")
    ap.add_argument("--holdout", default="processed_data/holdout60",
                    help="합성 홀드아웃 — **같은 자로** 합성을 재서 실측과 견준다")
    ap.add_argument("--n-synth", type=int, default=1500)
    ap.add_argument("--out", default="results/forward_gap.json")
    a = ap.parse_args()

    apps = list(torch.load(a.ckpt, map_location="cpu", weights_only=False)["appliances"])
    from src.model.net import harmonic_signatures, noise_signature
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    SIG = harmonic_signatures(pool, apps)
    NZ = noise_signature(pool)
    del pool
    SIG_INS = np.asarray(np.load(a.sig_insitu, allow_pickle=True)["sig"], np.float32)

    stems = human_stems(load_events())
    wins = collect(apps, NZ, stems)
    smps = {apps.index(x) for x in SMPS_APPLIANCES if x in apps}

    for w in wins:
        w["ny"] = float(np.linalg.norm(np.concatenate([w["y"].real, w["y"].imag])))
        w["r"] = resid(SIG, list(w["sup"]), w["y"])
        w["ri"] = resid(SIG_INS, list(w["sup"]), w["y"])
        w["rel"] = w["r"] / max(w["ny"], 1e-12)
        w["reli"] = w["ri"] / max(w["ny"], 1e-12)
        w["n_on"] = len(w["sup"])
        w["n_smps"] = len(set(w["sup"]) & smps)
        w["n_res"] = w["n_on"] - w["n_smps"]
        w["cls"] = ("저항 전용" if w["n_smps"] == 0 else
                    "SMPS 전용" if w["n_res"] == 0 else "혼합")

    print("=" * 84)
    print(f"순방향 모델 오차의 층 — 사람 기록 {len(stems)}파일 {len(wins)}창")
    print("정답 배분 NNLS 잔차 / ‖y‖. 모델도 손실도 없다")
    print("=" * 84)

    def block(title: str, key, order=None):
        print(f"\n[{title}]")
        print(f"  {'층':<14s}{'창':>5s}{'‖y‖중앙':>10s}"
              f"{'상대잔차 격리':>14s}{'상대잔차 in-situ':>17s}")
        print("  " + "-" * 62)
        groups: Dict[str, list] = {}
        for w in wins:
            groups.setdefault(str(key(w)), []).append(w)
        for g in (order or sorted(groups)):
            if g not in groups:
                continue
            v = groups[g]
            print(f"  {g:<14s}{len(v):>5d}{np.median([x['ny'] for x in v]):>10.3f}"
                  f"{np.mean([x['rel'] for x in v]):>14.1%}"
                  f"{np.mean([x['reli'] for x in v]):>17.1%}")

    block("구성별 — **저항 전용이 결정적이다**", lambda w: w["cls"],
          ["저항 전용", "SMPS 전용", "혼합"])
    block("동시 통전 기기 수 — 오르면 중첩이 깨지는 것이다",
          lambda w: f"{w['n_on']}종")
    block("동시 SMPS 수", lambda w: f"SMPS {w['n_smps']}종")

    # 중첩 붕괴의 직접 검정 — 기기 수에 대한 상대잔차 기울기
    n = np.array([w["n_on"] for w in wins], float)
    r = np.array([w["rel"] for w in wins], float)
    if len(set(n)) > 1:
        slope, icpt = np.polyfit(n, r, 1)
        rho = float(np.corrcoef(n, r)[0, 1])
        print(f"\n[중첩 검정] 상대잔차 = {icpt:.3f} + {slope:+.4f} x 기기수   "
              f"(상관 {rho:+.3f}, n={len(wins)})")
        print("  기울기가 0 이면 지문 오차(A), 양이면 상호작용(B) 이다.")

    # ── 합성은 같은 오차를 내는가 (sim2real) ──────────────────────────────
    # **이것이 "합성기를 고치면 배분이 낫는가" 의 판정자다.** 합성 창을 실측과
    # 똑같은 자로 잰다 — 지문은 train 분할, 창은 holdout 분할이라 일반화가 낀다.
    syn: List[dict] = []
    nz_c = NZ[:H, 0] + 1j * NZ[:H, 1]      # collect() 안의 변환과 같게
    hp = Path(a.holdout)
    if (hp / "X.npy").exists():
        X = np.load(hp / "X.npy", mmap_mode="r")
        yon = np.load(hp / "y_on.npy")
        pick = np.random.default_rng(0).choice(
            len(X), size=min(a.n_synth, len(X)), replace=False)
        for i in pick:
            sup = np.flatnonzero(yon[i] > 0)
            if not len(sup):
                continue
            w = np.asarray(X[i], np.float64)
            y = w[:H].mean(1) + 1j * w[15:15 + H].mean(1) - nz_c
            ny = float(np.linalg.norm(np.concatenate([y.real, y.imag])))
            if ny < 1e-9:
                continue
            n_s = len(set(sup.tolist()) & smps)
            syn.append({"rel": resid(SIG, list(sup), y) / ny, "ny": ny,
                        "n_on": len(sup), "n_smps": n_s,
                        "cls": ("저항 전용" if n_s == 0 else
                                "SMPS 전용" if n_s == len(sup) else "혼합")})
        print(f"\n[sim2real] **같은 자로 합성 홀드아웃을 잰다** ({len(syn)}창)")
        print(f"  {'층':<12s}{'실측창':>7s}{'실측 상대잔차':>14s}"
              f"{'합성창':>8s}{'합성 상대잔차':>14s}{'실측/합성':>10s}")
        print("  " + "-" * 68)
        for c in ("저항 전용", "SMPS 전용", "혼합", "전체"):
            rr = [w["rel"] for w in wins if c == "전체" or w["cls"] == c]
            ss = [w["rel"] for w in syn if c == "전체" or w["cls"] == c]
            if not rr or not ss:
                continue
            mr, ms = float(np.mean(rr)), float(np.mean(ss))
            print(f"  {c:<12s}{len(rr):>7d}{mr:>14.1%}{len(ss):>8d}{ms:>14.1%}"
                  f"{mr / max(ms, 1e-9):>9.1f}x")
        print("\n  **합성이 실측만큼 안 틀리면** 1단계 모델은 실측의 순방향 오차를")
        print("  한 번도 못 보고 배운다. 그 배수가 곧 합성기 상향의 여지다.")

    print("\n[규칙 14] 이 자는 **정답 on/off 를 아는 창**에서만 잰다. "
          "기기별 전력 참값은 안 쓴다 —")
    print("  전력은 NNLS 가 자유롭게 푸므로, 남는 잔차는 전적으로 순방향 모델의 것이다.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "n_windows": len(wins), "stems": list(stems),
        "by_class": {c: {"n": sum(1 for w in wins if w["cls"] == c),
                         "rel_iso": float(np.mean([w["rel"] for w in wins if w["cls"] == c])),
                         "rel_insitu": float(np.mean([w["reli"] for w in wins if w["cls"] == c]))}
                     for c in ("저항 전용", "SMPS 전용", "혼합")
                     if any(w["cls"] == c for w in wins)},
        "synth": ({"n": len(syn),
                   "rel_by_class": {c: float(np.mean([w["rel"] for w in syn if w["cls"] == c]))
                                    for c in ("저항 전용", "SMPS 전용", "혼합")
                                    if any(w["cls"] == c for w in syn)},
                   "rel_all": float(np.mean([w["rel"] for w in syn]))} if syn else None),
        "_config": {"argv": sys.argv, "args": vars(a)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
