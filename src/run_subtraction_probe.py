"""
뺄셈 가설 — 핫플 검출은 오븐 추정에 얹혀 있는가 (설계 문서 12.48절)
=====================================================================
12.20.4 가 후보를 하나만 남겼다.

> 파형 암기 · 듀티 사용 · 고조파 절대수준 · 분포 밖 입력 · 소프트게이트 · 광역 배관
> -> 전부 반증.  **뺄셈** (오븐을 확실히 빼면 잔여가 핫플일 수밖에 없다) -> 유일하게 남음

그리고 검증 실험을 적어 두고 실시하지 않았다:

> *합성 창에서 **오븐 신호만** 실측 수준으로 흐리고 핫플 검출이 무너지는지 본다.
> 무너지면 뺄셈이 확정이고, 처방은 "핫플을 더 잘 가르자" 가 아니라
> **"오븐 추정을 더 튼튼하게"** 로 바뀐다.*

여기서 그것을 한다. **학습하지 않는다.**

    python -m src.run_subtraction_probe --ckpt results/cnn_ov1.pt

[방법]
`resistive_overlap` 레시피로 **타깃 시점에 오븐·핫플이 동시 통전**하는 창을 만들고
(`compute_gt_harmonics=True` 로 기기별 성분을 받는다), 합계에서 **오븐 성분만**
비율 `s` 로 바꾼 뒤 추론한다.

    관측' = 관측 + (s-1) x 오븐성분        (고조파 15차 전부 + P)
    오븐은 순저항이라 Q 기여가 작다. Q·V 는 건드리지 않는다.

대조군으로 **같은 조작을 핫플에** 건다. 그리고 성분 합이 관측을 재현하는지
먼저 검산한다 (결합·전압강하·양자화가 있으므로 근사다).

[예측 — 돌리기 전에 적는다]
- **뺄셈이면**: 오븐을 키우면(s>1) 잔여가 먹히므로 핫플 검출이 **가파르게** 무너진다.
  오븐 1150W 기준 핫플 468W 를 다 먹는 지점이 s ~ 1.4 다. 그보다 훨씬 앞에서
  (s = 1.1~1.2) 이미 흔들려야 한다. 오븐을 줄이면(s<1) 잔여가 커져 핫플을 **과대**
  추정하거나 다른 큰 저항 부하(포트)를 켠다.
- **핫플 자체 지문(리플)으로 검출한다면**: 오븐을 바꿔도 핫플 게이트가 **평평**하다.
  핫플의 P 리플은 조작에 안 닿기 때문이다.
- 대조: 핫플을 직접 줄이면 어느 가설에서든 검출이 내려간다. 이것은 감도 기준선이다.

**판정: `s = 1.2` 에서 핫플 재현율이 0.1 이상 떨어지면 뺄셈 쪽이다.**

[⚠ 크기 조작만으로는 뺄셈을 못 시험한다 — 돌려 보고 알았다]
오븐을 s 배 해도 **모델이 그 커진 오븐을 여전히 정확히 추정하면 잔여는 그대로**다.
뺄셈이어도 정답이 나온다. 12.20.4 의 표현은 "흐린다" 였지 "키운다" 가 아니었다.

그래서 두 번째 조작을 넣는다 — **핫플의 리플 결만 지우고 전력은 남긴다.**
핫플은 릴레이가 약 2초 주기로 끊는데(0.4절: 0.9초 통전 / 1.1초 휴지),
그 시간 구조를 이동평균으로 뭉개고 **창 평균 전력은 보존**한다.

    잔여의 **크기**는 그대로   <- 뺄셈이 쓰는 것
    잔여의 **결**만 사라진다   <- 핫플 자체 지문이 쓰는 것 (ch36~37, 12.42)

    뺄셈이면      -> 검출이 유지된다
    자체 지문이면 -> 검출이 무너진다

**판정: 반폭 150사이클(±2.5초, 릴레이 주기를 넘는다) 평활에서 핫플 재현율이
0.1 이상 떨어지면 자체 지문 쪽이고, 유지되면 뺄셈 쪽이다.**
"""
from pathlib import Path
from typing import Dict, List
import argparse
import json
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.model.inputs import build_inputs, target_index
from src.run_gate_check import load_model
from src.run_live import KOR

N_HARM = 15
SCALES = (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4)
# 이동평균 반폭(사이클). 핫플 릴레이 주기가 약 2초(120사이클)이므로
# 60(±1초)이면 절반쯤 뭉개지고 150(±2.5초)이면 주기가 사라진다.
SMOOTH_HALF = (0, 15, 30, 60, 150, 300)


def make_windows(n: int, window_cycles: int, seed: int) -> List:
    """오븐·핫플이 타깃 시점에 동시 통전하는 창 n개 (기기별 고조파 포함)."""
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import LoadSynthesizer
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    syn = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=True)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        np.random.seed(int(rng.integers(1 << 30)))
        s = syn.synthesize_resistive_overlap_window(
            window_size_cycles=window_cycles, compute_gt_harmonics=True,
            pair=("oven", "hotplate"), exclude_active=("electiric_kettle",))
        ti = target_index(window_cycles)
        if (s.gt_is_on.get("oven", np.zeros(1))[ti]
                and s.gt_is_on.get("hotplate", np.zeros(1))[ti]
                and "oven" in s.gt_harmonics_ri and "hotplate" in s.gt_harmonics_ri):
            out.append(s)
    return out


def to_33ch(hr: np.ndarray, pf: np.ndarray) -> np.ndarray:
    """(N,15,2)+(N,6) -> (33,N) 합성기 배치와 같은 채널 배열."""
    n = hr.shape[0]
    x = np.empty((33, n), np.float32)
    x[0:15] = hr[:, :, 0].T
    x[15:30] = hr[:, :, 1].T
    x[30], x[31], x[32] = pf[:, 0], pf[:, 1], pf[:, 4]
    return x


def _movavg(a: np.ndarray, half: int) -> np.ndarray:
    """시간축(0번) 이동평균. 가장자리는 반사 없이 edge 로 채운다. 평균 보존."""
    if half <= 0:
        return a
    k = 2 * half + 1
    pad = np.pad(a, [(half, half)] + [(0, 0)] * (a.ndim - 1), mode="edge")
    c = np.cumsum(pad, axis=0)
    c = np.concatenate([np.zeros((1,) + a.shape[1:], np.float64), c], axis=0)
    return ((c[k:] - c[:-k]) / k).astype(np.float32)


@torch.no_grad()
def run(model, samples: List, app: str, dev: str, window_cycles: int,
        scale: float = 1.0, smooth_half: int = 0) -> np.ndarray:
    """`app` 성분을 scale 배 하거나 시간축으로 평활한 뒤 추론."""
    X = []
    for s in samples:
        hr = np.asarray(s.harmonics_ri, np.float32).copy()
        pf = np.asarray(s.power_features, np.float32).copy()
        comp_h = np.asarray(s.gt_harmonics_ri[app], np.float32)
        comp_p = np.asarray(s.gt_target_power_w[app], np.float32)
        if smooth_half > 0:
            # 성분을 평활한 것으로 **교체**한다. 창 평균은 보존된다.
            hr += _movavg(comp_h, smooth_half) - comp_h
            pf[:, 0] += _movavg(comp_p, smooth_half) - comp_p
        d = scale - 1.0
        if d != 0.0:
            hr += d * comp_h
            pf[:, 0] += d * comp_p
        X.append(to_33ch(hr, pf))
    fine, wide = build_inputs(np.stack(X))
    G = []
    for i in range(0, len(fine), 256):
        f = torch.from_numpy(np.ascontiguousarray(fine[i:i + 256])).to(dev)
        w = torch.from_numpy(np.ascontiguousarray(wide[i:i + 256])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(f, w)
        G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
    return np.concatenate(G)


def main() -> int:
    ap = argparse.ArgumentParser(description="뺄셈 가설 검증 (12.48절)")
    ap.add_argument("--ckpt", nargs="+", default=["results/cnn_ov1.pt"])
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--window-cycles", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/subtraction_probe.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 92)
    print("[뺄셈 가설] 오븐 성분만 s 배 하고 핫플 검출이 무너지는지 본다 (12.20.4)")
    print("=" * 92)
    print(f"  창 {a.windows}개 합성 중 (오븐·핫플 타깃 동시 통전, 포트 제외)...")
    samples = make_windows(a.windows, a.window_cycles, a.seed)
    ti = target_index(a.window_cycles)

    # ── 검산: 성분 합이 관측을 재현하는가 ────────────────────────────────
    err_p, err_h, p_ov, p_hp = [], [], [], []
    for s in samples:
        tot_p = sum(np.asarray(v, np.float32)[ti] for v in s.gt_target_power_w.values())
        tot_p += sum(np.asarray(v, np.float32)[ti] for v in s.gt_standby_power_w.values())
        tot_p += float(np.asarray(s.p_noise_w, np.float32)[ti])
        obs_p = float(np.asarray(s.power_features, np.float32)[ti, 0])
        err_p.append(obs_p - tot_p)
        h = np.zeros((N_HARM, 2), np.float32)
        for v in s.gt_harmonics_ri.values():
            h += np.asarray(v, np.float32)[ti]
        oh = np.asarray(s.harmonics_ri, np.float32)[ti]
        err_h.append(float(np.abs(np.hypot(*oh.T) - np.hypot(*h.T)).sum()))
        p_ov.append(float(np.asarray(s.gt_target_power_w["oven"], np.float32)[ti]))
        p_hp.append(float(np.asarray(s.gt_target_power_w["hotplate"], np.float32)[ti]))
    print(f"  검산: 관측 P − 성분합 P = 중앙 {np.median(err_p):+.1f}W "
          f"(p95 |{np.percentile(np.abs(err_p), 95):.1f}|W)  "
          f"| 고조파 크기 합 오차 중앙 {np.median(err_h) * 1000:.1f}mA")
    print(f"  타깃 시점 전력: 오븐 중앙 {np.median(p_ov):.0f}W  "
          f"핫플 중앙 {np.median(p_hp):.0f}W")
    print(f"  -> 오븐을 s 배 하면 잔여가 {np.median(p_ov):.0f}x(s-1) W 만큼 바뀐다. "
          f"핫플 {np.median(p_hp):.0f}W 를 다 먹는 지점은 "
          f"s = {1 + np.median(p_hp) / np.median(p_ov):.2f}")

    payload: Dict[str, dict] = {"n": len(samples),
                                "p_oven_med": float(np.median(p_ov)),
                                "p_hotplate_med": float(np.median(p_hp))}
    for ck in a.ckpt:
        model, apps, _ = load_model(ck, dev)
        jh, jo = apps.index("hotplate"), apps.index("oven")
        jk = apps.index("electiric_kettle")
        tag = Path(ck).stem
        print()
        print(f"  [{tag}]")
        rows = {}
        for pert in ("oven", "hotplate"):
            print(f"    -- {KOR[pert]} 성분을 s 배 --")
            print(f"       {'s':>6s}{'핫플 게이트':>12s}{'핫플 검출':>11s}"
                  f"{'오븐 게이트':>12s}{'포트 게이트':>12s}{'포트 켜짐':>10s}")
            sub = {}
            for s in SCALES:
                g = run(model, samples, pert, dev, a.window_cycles, scale=s)
                r = {"hp_gate": float(np.median(g[:, jh])),
                     "hp_recall": float((g[:, jh] > 0.5).mean()),
                     "ov_gate": float(np.median(g[:, jo])),
                     "kt_gate": float(np.median(g[:, jk])),
                     "kt_fire": float((g[:, jk] > 0.5).mean())}
                sub[f"{s:.1f}"] = r
                mark = "  <- 원본" if s == 1.0 else ""
                print(f"       {s:>6.1f}{r['hp_gate']:>12.3f}{r['hp_recall']:>11.3f}"
                      f"{r['ov_gate']:>12.3f}{r['kt_gate']:>12.3f}"
                      f"{r['kt_fire']:>10.3f}{mark}")
            rows[pert] = sub
        # ── 리플 결 제거 (진짜 뺄셈 시험) ────────────────────────────────
        for pert in ("hotplate", "oven"):
            print(f"    -- {KOR[pert]} 성분의 시간 결만 지운다 (전력 보존) --")
            print(f"       {'반폭':>7s}{'초':>7s}{'핫플 게이트':>12s}{'핫플 검출':>11s}"
                  f"{'오븐 게이트':>12s}{'포트 켜짐':>10s}")
            sub = {}
            for h in SMOOTH_HALF:
                g = run(model, samples, pert, dev, a.window_cycles, smooth_half=h)
                r = {"hp_gate": float(np.median(g[:, jh])),
                     "hp_recall": float((g[:, jh] > 0.5).mean()),
                     "ov_gate": float(np.median(g[:, jo])),
                     "kt_fire": float((g[:, jk] > 0.5).mean())}
                sub[str(h)] = r
                print(f"       {h:>7d}{2 * h / 60:>7.1f}{r['hp_gate']:>12.3f}"
                      f"{r['hp_recall']:>11.3f}{r['ov_gate']:>12.3f}"
                      f"{r['kt_fire']:>10.3f}" + ("  <- 원본" if h == 0 else ""))
            rows[f"smooth_{pert}"] = sub
        sm = rows["smooth_hotplate"]
        d150 = sm["0"]["hp_recall"] - sm["150"]["hp_recall"]
        print(f"    판정(리플): 반폭 150(±2.5초)에서 핫플 재현율 "
              f"{sm['0']['hp_recall']:.3f} -> {sm['150']['hp_recall']:.3f} "
              f"(Δ {d150:+.3f}) -> {'자체 지문 쪽' if d150 >= 0.1 else '뺄셈 쪽'}")

        base = rows["oven"]["1.0"]["hp_recall"]
        d12 = base - rows["oven"]["1.2"]["hp_recall"]
        print(f"    판정: s=1.2 에서 핫플 재현율 {base:.3f} -> "
              f"{rows['oven']['1.2']['hp_recall']:.3f}  (Δ {d12:+.3f})  "
              f"-> {'뺄셈 쪽' if d12 >= 0.1 else '뺄셈 아님'}")
        payload[tag] = rows

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                           encoding="utf-8")
    print()
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
