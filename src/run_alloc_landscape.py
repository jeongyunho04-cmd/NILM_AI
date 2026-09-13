"""배분 지형 — **진짜 `NILMLoss` 로** 항별 최소가 어디인지 훑는다 (13.60)
=================================================================
`run_loss_compare` 는 배분 **두 점**(모델 A, 참값 B)만 견준다. 그것으로는
"손실이 오답을 선호한다" 까지만 알고 **최소가 어디인지**는 모른다. 13.59.8 이
근사 격자로 최소를 찾았다가 짝수차·`harm_scale`·상태 지문을 빼먹어 진짜 손실과
어긋났다. 여기서는 `unlabeled()` 를 **그대로 불러** 격자를 훑는다.

    반사실 가족:  SMPS 셋을 **정답대로 검출했다고 두고**(게이트 1, `on_logit` 큼)
                  전력만 (W_프로, W_충전) 격자로 준다. 미니PC 는 참값에 고정.

    왜 게이트를 1 로 두나: `P = σ(on)·p_raw` 라 게이트를 안 만지면 "배분" 과
    "검출" 이 섞인다. 게이트를 정답으로 박아야 **배분 축만** 남는다. 그리고
    그 점에서 `hedge` 는 그 열에 대해 만족되므로 hedge 가 배분에 주는 압력과
    **게이트를 눌러 얻는 이득**이 갈린다 (전자는 이 표, 후자는 `--gate-scan`).

⚠ **전력 경로를 통째로 맞춰야 한다** (`_set` 주석). 상태 지문이 켜져 있으면
  `power`/`power_raw` 만 바꿔서는 고조파 예측이 배분에 **전혀 반응하지 않는다**.

⚠ **`모델 그대로` 줄이 격자 최소보다 낮으면** 반사실 가족이 도달 가능 집합을 못 덮는
  것이다. 13.60 이 그것으로 답을 찾았다 — 게이트를 정답으로 박는 것 자체가 `cons` 에서
  비쌌고, 그 비용이 **자유 대기 헤드** 때문이었다.

이 도구가 13.60 에서 확정한 것:

    1단계 cnn_v24b       참값 total 1.7354 < 모델 자신 2.0651   -> 손실이 정답을 원한다
    2단계 adapt_v24b_z   참값 total 2.0097 > 모델 자신 1.1928   -> 뒤집힌다
                         격차의 90%가 `cons`. 대기 합이 1.06 -> 6.93W (예산 0.6)
    `--anchor-standby`   참값 1.6242 < 격자 최소 1.6547         -> 다시 정답이 최소

    python -m src.run_alloc_landscape --ckpt results/cnn_v24b.pt
    python -m src.run_alloc_landscape --ckpt results/adapt_v24b_z_s0.pt --anchor-standby
    python -m src.run_alloc_landscape --ckpt results/adapt_v24b_z_s0.pt --gate-scan
"""
from typing import Sequence
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.evaluation.real_events import build_on_off_truth, load_events
from src.model.realdata import dense_targets
from src.run_gate_check import load_model
from src.run_loss_compare import SMPS, BIG, build_loss

#: 13.59.7 에서 조합 차분으로 다시 잰 참 ON 전력 (대기 되돌림 포함)
TRUE_W = {"beam_projector": 44.6, "laptop_charger": 38.9, "minipc": 8.9}


@torch.no_grad()
def collect(m, apps, stems: Sequence[str], dev: str):
    """자리 D · SMPS 전용 · 셋 다 켜진 창의 모델 출력과 타깃을 모은다."""
    js = [apps.index(x) for x in SMPS]
    jb = [apps.index(x) for x in BIG]
    ev = load_events()
    O, T = [], []
    for stem in stems:
        rw = dense_targets(stem, stride=30, site_transfer=getattr(m, "site_transfer", None))
        n_cyc = int(ev[stem]["cycles"])
        on, sc = build_on_off_truth(stem, apps, n_cyc, ev)
        t = np.asarray(rw.target_cycle, int).clip(0, n_cyc - 1)
        keep = np.flatnonzero((~on[t][:, jb].any(1)) & on[t][:, js].all(1)
                              & sc[t][:, js].all(1))
        for i in range(0, len(keep), 256):
            idx = keep[i:i + 256]
            f, w, pobs, oh, pn = rw.batch(idx)
            o = m(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                  torch.from_numpy(np.ascontiguousarray(w)).to(dev))
            O.append({k: v.float() for k, v in o.items() if torch.is_tensor(v)})
            T.append({"p_observed": torch.from_numpy(pobs).float().to(dev),
                      "obs_harm": torch.from_numpy(oh).float().to(dev),
                      "p_noise": torch.from_numpy(pn).float().to(dev),
                      "harm_offset": None})
    return O, T


def standby_watts(apps) -> np.ndarray:
    """기기별 **측정된** 대기 전력 (K,) W. `standby_sig` 와 같은 출처다."""
    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    out = np.zeros(len(apps), np.float32)
    for j, a in enumerate(apps):
        try:
            out[j] = float(np.mean(pool.get_standby_profile(a).power_w))
        except Exception:
            out[j] = 0.0
    del pool
    return out


def anchor_standby(o, sb_w):
    """자유 대기 헤드를 `idle x 측정 프로파일` 로 못박는다 (13.60.4).

    `y_standby_power` 의 규약이 "활성 중이면 0" 이라 이 헤드는 꺼져-꽂힘일 때만
    값을 내야 한다. 1단계는 라벨(`y_standby`)이 잡아 주지만 **2단계에는 라벨도
    구조적 게이트도 없어** `L_cons` 안의 자유 슬랙이 된다. 자리 D 창에서 모델은
    여기에 6.93W 를 넣는데 조합 차분이 준 예산은 0.6W 다.
    """
    d = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in o.items()}
    idle = torch.sigmoid(d["plugged_logit"]) * (1 - torch.sigmoid(d["on_logit"]))
    d["standby"] = idle * torch.as_tensor(sb_w, device=idle.device, dtype=idle.dtype)[None]
    return d


def zero_idle(o, keep):
    """`keep` 밖 기기의 `plugged` 를 0 으로 눌러 **대기 항을 끈다** (13.60.3).

    `idle = σ(plugged)·(1−σ(on))` 이므로 이렇게 하면 그 기기의 대기 전력·대기 지문이
    둘 다 빠진다. 자리 D 창에서 모델은 대기로 6.9W 를 넣는데 조합 차분이 준 예산은
    0.6W 다 — 그 초과가 `L_cons` 를 통해 SMPS 에서 빠져나간다.
    """
    d = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in o.items()}
    m = torch.ones_like(d["plugged_logit"][0], dtype=torch.bool)
    m[list(keep)] = False
    d["plugged_logit"][:, m] = -12.0
    return d


def _set(o, js, w_vec, force_gate: bool, force_state: bool, sb_w=None):
    """배분을 넣은 사본.

    ⚠ **전력 경로를 통째로 맞춰야 한다.** 상태 지문이 켜져 있으면
    `_harm_pred_active` 는 `power` 를 `gate = power/power_raw` 로만 보고 크기는
    `power_mix · power_states` 에서 가져간다. `power` 와 `power_raw` 만 같이 바꾸면
    게이트가 1 로 고정되어 **고조파 예측이 배분에 전혀 반응하지 않는다** — 처음에
    그렇게 짜서 `harm` 이 격자 전체에서 상수로 나왔다 (13.60.1).

        power_mix   -> 그 기기의 최빈 통전 상태에 원핫
        power_states-> 그 상태에 W
        power_raw   -> W,   power -> σ(on)·W
    """
    d = {k: v.clone() for k, v in o.items()}
    for c, j in enumerate(js):
        w = float(w_vec[c])
        if "power_mix" in d and "power_states" in d:
            k = int(d["power_mix"][:, j].mean(0).argmax())
            d["power_mix"][:, j] = 0.0
            d["power_mix"][:, j, k] = 1.0
            d["power_states"][:, j] = 0.0
            d["power_states"][:, j, k] = w
        d["power_raw"][:, j] = w
        if force_gate and sb_w is not None:
            d["standby"][:, j] = 0.0        # 정답대로 켜졌으면 대기는 0 이다
        if force_gate:
            d["on_logit"][:, j] = 12.0
            d["plugged_logit"][:, j] = 12.0
            d["power"][:, j] = w
        else:
            d["power"][:, j] = torch.sigmoid(d["on_logit"][:, j]) * w
    return d


@torch.no_grad()
def score(crit, O, T, js, w_vec, kw, force_gate, force_state, sb_w=None):
    acc, n = {}, 0
    for o, t in zip(O, T):
        p = crit.unlabeled(_set(o, js, w_vec, force_gate, force_state, sb_w), t, **kw)
        b = len(t["p_observed"])
        for k in ("cons", "harm", "hedge", "pref", "total"):
            if k in p:
                acc[k] = acc.get(k, 0.0) + float(p[k]) * b
        n += b
    return {k: v / n for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stems", nargs="+", default=["test_3", "test_4", "test_5"])
    ap.add_argument("--harm-weight", default="inv_h2")
    ap.add_argument("--sig-site", default="", choices=["", "D", "E"])
    ap.add_argument("--w-cons", type=float, default=0.1)
    ap.add_argument("--w-harm", type=float, default=4.0)
    ap.add_argument("--w-hedge", type=float, default=0.2)
    ap.add_argument("--w-pref", type=float, default=0.02)
    ap.add_argument("--step", type=float, default=4.0, help="격자 간격 W")
    ap.add_argument("--max-w", type=float, default=80.0)
    ap.add_argument("--free-gate", action="store_true",
                    help="게이트를 모델 것 그대로 둔다 (배분과 검출이 섞인다)")
    ap.add_argument("--zero-idle", action="store_true",
                    help="SMPS 아닌 기기의 대기 항을 끈다 (13.60.3) — 모델이 자리 D 창에서 "
                         "대기로 6.9W 를 넣는데 조합 차분 예산은 0.6W 다")
    ap.add_argument("--anchor-standby", action="store_true",
                    help="자유 대기 헤드를 idle x 측정 프로파일로 못박는다 (13.60.4)")
    ap.add_argument("--gate-scan", action="store_true",
                    help="배분을 참값에 두고 **프로젝터 게이트만** 훑는다")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, apps, _ = load_model(a.ckpt, dev)
    crit = build_loss(apps, dev, a.harm_weight, a.sig_site,
                      pref_apps=([] if a.w_pref <= 0 else ["beam_projector"]))
    js = [apps.index(x) for x in SMPS]
    kw = dict(w_cons=a.w_cons, w_harm=a.w_harm, w_hedge=a.w_hedge,
              w_over=0.0, w_pref=a.w_pref)
    O, T = collect(m, apps, a.stems, dev)
    if a.zero_idle:
        O = [zero_idle(o, js) for o in O]
    if a.anchor_standby:
        SBW = standby_watts(apps)
        O = [anchor_standby(o, SBW) for o in O]
        print("  ** 대기 못박음: "
              + ", ".join(f"{apps[j]} {SBW[j]:.2f}W" for j in range(len(apps))
                          if SBW[j] > 0.05) + " **")
    nwin = int(sum(len(t["p_observed"]) for t in T))
    fg, fs = (not a.free_gate), True
    print(f"{a.ckpt}  창 {nwin}개  (자리 D · SMPS 전용 · 셋 다 켜짐)")
    print(f"대기 {'못박음' if a.anchor_standby else ('SMPS 만' if a.zero_idle else '모델 것')} · "
          f"게이트 {'정답으로 고정' if fg else '모델 것'} · "
          f"상태 최빈 통전으로 고정 · 지문 "
          f"{'자리 ' + a.sig_site if a.sig_site else '뭉친'}")
    tw = np.array([TRUE_W[x] for x in SMPS])
    print(f"참 배분 {'/'.join(f'{v:.1f}' for v in tw)}W")
    # ── 모델 그대로 (반사실이 도달 가능 집합을 덮는지 확인한다) ──────────────
    # 이 줄이 격자 최소보다 **낮으면** 반사실 가족이 모델의 실제 상태를 못 담는
    # 것이라 표를 그대로 읽으면 안 된다 — 게이트·상태를 정답으로 박는 것 자체가
    # 다른 항에서 그만큼 비싸다는 뜻이다.
    acc, n, mw = {}, 0, np.zeros(3)
    with torch.no_grad():
        for o, t in zip(O, T):
            pp = crit.unlabeled(o, t, **kw)
            b = len(t["p_observed"])
            for k in ("cons", "harm", "hedge", "pref", "total"):
                acc[k] = acc.get(k, 0.0) + float(pp[k]) * b
            mw += o["power"][:, js].float().cpu().numpy().sum(0)
            n += b
    print(f"모델 그대로  {'/'.join(f'{v:.1f}' for v in mw / n)}W  ->  "
          + "  ".join(f"{k} {acc[k] / n:.4f}"
                      for k in ("cons", "harm", "hedge", "pref", "total")) + "\n")

    if a.gate_scan:
        # 배분은 참값. 프로젝터 게이트만 움직인다 — `P = σ(on)·p_raw` 의 다른 축이다.
        print(f"  {'σ(on) 프로':>10s}{'P̂ 프로':>9s}"
              + "".join(f"{k:>10s}" for k in ("cons", "harm", "hedge", "pref", "total")))
        jp = apps.index("beam_projector")
        for g in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 0.99):
            acc, n = {}, 0
            for o, t in zip(O, T):
                d = _set(o, js, tw, True, fs)
                d["on_logit"][:, jp] = float(np.log(g / (1 - g)))
                d["power"][:, jp] = g * tw[0]
                p = crit.unlabeled(d, t, **kw)
                b = len(t["p_observed"])
                for k in ("cons", "harm", "hedge", "pref", "total"):
                    acc[k] = acc.get(k, 0.0) + float(p[k]) * b
                n += b
            print(f"  {g:10.2f}{g*tw[0]:9.1f}"
                  + "".join(f"{acc[k]/n:10.4f}" for k in
                            ("cons", "harm", "hedge", "pref", "total")))
        print("\n  게이트를 낮춰 total 이 내려가면 **검출을 죽여 손실을 버는 것**이다.")
        return

    SBW0 = SBW if a.anchor_standby else None
    grid = np.arange(0.0, a.max_w + 0.5 * a.step, a.step)
    best = {}
    rows = []
    for wp in grid:
        for wc in grid:
            v = score(crit, O, T, js, np.array([wp, wc, tw[2]]), kw, fg, fs, SBW0)
            rows.append((wp, wc, v))
            for k, x in v.items():
                if k not in best or x < best[k][0]:
                    best[k] = (x, wp, wc)
    print(f"  {'항':10s}{'최소값':>10s}{'프로 W':>9s}{'충전 W':>9s}"
          f"{'참값에서':>10s}{'초과':>8s}")
    tv = score(crit, O, T, js, tw, kw, fg, fs, SBW0)
    for k in ("cons", "harm", "hedge", "pref", "total"):
        if k not in best:
            continue
        v, wp, wc = best[k]
        ex = 100.0 * (tv[k] - v) / max(abs(v), 1e-9)
        print(f"  {k:10s}{v:10.4f}{wp:9.1f}{wc:9.1f}{tv[k]:10.4f}{ex:7.0f}%")
    print("\n  `total` 의 최소가 참값에서 멀면 **그 배분이 손실의 답**이다.")
    print("  항마다 최소가 다르면 어느 항이 어디로 끄는지가 그 표다.")


if __name__ == "__main__":
    main()
