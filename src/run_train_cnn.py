"""
Phase 3 — 2갈래 CNN 학습 (1단계: 합성 사전학습)
================================================
설계 문서 2·3·4.1절. 12.8절에서 확정한 창 구성을 쓴다.

    세밀 10초 @ 60Hz (36,600)  +  광역 60초 @ 2Hz (12,120)
    타깃 = 60초 창의 끝-1초 (인덱스 3539)  ->  추론 지연 1초

**이 모델은 Phase 1 baseline 을 이겨야 의미가 있다** (4.1절, 10절).
기준선: 평균 MAE 1.43W / F1 0.952 / 저항3종 0.968 / 오븐→포트 9%

# 기본
python -m src.run_train_cnn

# 짧게 확인
python -m src.run_train_cnn --epochs 3 --epoch-windows 20000

# 시드 3개 (분산 확인 - 데이터가 얇아 단일 실행을 믿으면 안 된다)
for s in 0 1 2; do python -m src.run_train_cnn --seed $s --tag seed$s; done
"""
from pathlib import Path
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.evaluation import (
    format_table, load_holdout, resistive_confusion,
    score_appliances, summarize, total_power_residual,
)
from src.model.inputs import build_inputs
from src.model.traincache import CachedWindows
from src.model.losses import LossWeights, NILMLoss
from src.model.net import (
    NILMNet, appliance_state_counts, harmonic_scales, harmonic_signatures,
    noise_signature, standby_signatures,
)
from src.run_baseline import LOW_LOAD, S_I

WINDOW_CYCLES = 3600          # 60초. 광역 갈래가 이 전체를 2Hz 로 본다
HOLDOUT_DIR = "processed_data/holdout60"


class SynthBatchDataset(Dataset):
    """실시간 합성. 12.1절 측정대로 캐시보다 워커가 낫다 (창 재사용 0).

    **배치 단위로 돌려준다** (`DataLoader(batch_size=None)`). 창 1개씩 변환하면
    numpy 호출 오버헤드가 지배해 261 win/s 까지 떨어진다 — 생성기 자체(2,700 win/s)의
    1/10 이다. 배치로 묶으면 벡터 연산이 살아난다.
    또 원시 창(512 x 33 x 3600 = 243MB)이 아니라 변환 결과(47MB)만 워커 경계를
    넘으므로 IPC 도 5배 가볍다.
    """

    def __init__(self, n_batches: int, batch_size: int, seed: int):
        self.n_batches = n_batches
        self.bs = batch_size
        self.seed = seed
        self.gen = None

    def __len__(self) -> int:
        return self.n_batches

    def _ensure(self):
        if self.gen is not None:
            return
        import os
        from src.synthesis.dataset import NILMBatchGenerator
        from src.synthesis.segment_pool import SegmentPool
        from src.synthesis.synthesizer import LoadSynthesizer
        np.random.seed((self.seed * 7919 + os.getpid()) % (2 ** 31))
        pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
        self.gen = NILMBatchGenerator(
            segment_pool=pool, window_size_cycles=WINDOW_CYCLES,
            synthesizer=LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False),
            compute_gt_harmonics=False)

    def __getitem__(self, i: int):
        self._ensure()
        g, n = self.gen, self.bs
        k = len(g.appliance_list)
        ti = g.target_index
        xs = np.empty((n, 33, WINDOW_CYCLES), np.float32)
        yp = np.empty((n, k), np.float32); yo = np.empty((n, k), np.float32)
        ypl = np.empty((n, k), np.float32); ys = np.empty((n, k), np.float32)
        yst = np.empty((n, k), np.int64)
        oh = np.empty((n, 15, 2), np.float32)
        pn = np.empty(n, np.float32); pobs = np.empty(n, np.float32)
        for j in range(n):
            smp, _ = g._synthesize_window()
            t = g._format_targets(smp)
            xs[j] = g._format_inputs(smp)
            yp[j], yo[j] = t["y_power"], t["y_on"]
            ypl[j], ys[j] = t["y_plugged"], t["y_standby_power"]
            yst[j] = t["y_state"]
            oh[j] = smp.harmonics_ri[ti]
            pn[j] = smp.p_noise_w[ti]
            pobs[j] = smp.power_features[ti, 0]
        fine, wide = build_inputs(xs)
        return tuple(torch.from_numpy(a) for a in
                     (fine, wide, yp, yo, ypl, ys, yst, oh, pn, pobs))


def to_targets(batch, dev):
    (fine, wide, yp, yo, ypl, ys, yst, oh, pn, pobs) = [b.to(dev, non_blocking=True) for b in batch]
    return fine, wide, {
        "y_power": yp, "y_on": yo, "y_plugged": ypl, "y_standby": ys, "y_state": yst,
        "obs_harm": oh, "p_noise": pn, "p_observed": pobs, "harm_offset": None,
    }


def prepare_holdout_inputs(hs, batch: int = 512):
    """홀드아웃 입력을 한 번만 변환해 RAM 에 둔다.

    매 평가마다 3.8GB memmap 을 읽고 build_inputs 를 돌리면 24초가 걸려,
    학습 1 epoch(9초)보다 오래 걸린다. 변환 후는 8,000창 x 45KB = 360MB 라
    올려 둘 수 있다.
    """
    F, W = [], []
    for i in range(0, len(hs), batch):
        f, w = build_inputs(np.asarray(hs.X[i:i + batch]))
        F.append(f); W.append(w)
    return np.concatenate(F), np.concatenate(W)


@torch.no_grad()
def evaluate(model, prep, dev, batch: int = 512) -> tuple:
    fine_all, wide_all = prep
    model.eval()
    P, ON = [], []
    for i in range(0, len(fine_all), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(torch.from_numpy(fine_all[i:i + batch]).to(dev),
                      torch.from_numpy(wide_all[i:i + batch]).to(dev))
        P.append(o["power"].float().cpu().numpy())
        ON.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
    model.train()
    return np.concatenate(P), np.concatenate(ON)


def report(pred, on_prob, hs, tag=""):
    sc = score_appliances(hs.y_power, pred, hs.appliances, S_I,
                          on_true=hs.y_on.astype(bool), on_pred=on_prob > 0.5)
    summ = summarize(sc, low_load=LOW_LOAD)
    cm = resistive_confusion(hs.y_power, pred, hs.appliances)
    resid = total_power_residual(pred, hs.p_observed, p_noise=hs.p_noise)
    return sc, summ, cm, resid


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 2갈래 CNN 학습")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--epoch-windows", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--width", type=float, default=1.0, help="채널 폭 배수 (용량 부족 시 2)")
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-harm", type=float, default=0.1)
    ap.add_argument("--w-cons", type=float, default=0.0, help="1단계는 0 (3.3절)")
    ap.add_argument("--w-over", type=float, default=0.1,
                    help="물리 상한 힌지. 예측 합이 관측 총전력을 넘을 때만 벌한다")
    ap.add_argument("--eval-every", type=int, default=1, help="N epoch 마다 홀드아웃 평가")
    ap.add_argument("--block-windows", type=int, default=24_000,
                    help="캐시 블록 셔플 단위. 작을수록 메모리가 덜 든다 (24000 = 약 1.1GB)")
    ap.add_argument("--cache", default="cache/train60",
                    help="학습 캐시 경로. 'none' 이면 실시간 합성 (12.8.2절 참조)")
    ap.add_argument("--tag", default="cnn")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    env_guard.verify_numerics()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 84)
    print(f"[Phase 3] 2갈래 CNN — 세밀 10초@60Hz + 광역 60초@2Hz, 타깃 끝-1초")
    print("=" * 84)

    hs = load_holdout(HOLDOUT_DIR)
    apps = hs.appliances
    prep = prepare_holdout_inputs(hs)
    # 변환이 끝나면 3.8GB 원시 memmap 은 더 필요 없다. 놓아 주어야
    # 그 페이지가 작업집합에 남지 않는다.
    hs.X = np.zeros((len(prep[0]), 1, 1), np.float32)
    print(f"평가: 홀드아웃 {len(hs):,}창 (뒤 {hs.meta['holdout_frac']:.0%}) "
          f"| sha {hs.meta['content_sha256']} | 타깃 {hs.meta['target_index']}/{hs.meta['window_cycles']}")

    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, apps)
    sb_sig = standby_signatures(pool, apps)
    nz_sig = noise_signature(pool)
    h_scale = harmonic_scales(pool, apps)
    del pool

    model = NILMNet(apps, appliance_state_counts(apps), width=a.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz_sig),
        harm_scale=torch.from_numpy(h_scale),
        weights=LossWeights(harm=a.w_harm, cons=a.w_cons, over=a.w_over),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * max(1, a.epoch_windows // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    print(f"모델 {n_par/1e6:.2f}M 파라미터 | 배치 {a.batch} | {a.epochs} epoch x "
          f"{a.epoch_windows:,}창 = {steps:,} step | 장치 {dev}")
    print(f"손실 가중치: power 1.0 / state .3 / on .3 / plugged .1 / standby .1 "
          f"/ harm {a.w_harm} / cons {a.w_cons} / over {a.w_over}")

    n_batches = max(1, a.epoch_windows // a.batch)
    cache = None
    if a.cache and a.cache.lower() != "none":
        cache = CachedWindows(a.cache)
        print(f"학습 데이터: 캐시 {a.cache} — 독립 창 {len(cache):,}개 "
              f"({cache.meta['bytes']/1e9:.1f}GB) | epoch 당 {a.epoch_windows:,}창 "
              f"-> 전체 {a.epochs * a.epoch_windows / len(cache):.1f}회 재사용")
        dl = None
    else:
        print(f"학습 데이터: 실시간 합성 (워커 {a.workers}) — 창 재사용 0")
        dl = DataLoader(SynthBatchDataset(n_batches, a.batch, a.seed), batch_size=None,
                        num_workers=a.workers, persistent_workers=a.workers > 0,
                        prefetch_factor=2 if a.workers else None,
                        pin_memory=(dev == "cuda"))

    rng = np.random.default_rng(a.seed)

    def epoch_batches():
        """캐시면 블록 셔플로 뽑고, 아니면 DataLoader 를 돈다.

        전역 셔플을 쓰면 13GB 캐시 전체가 작업집합에 올라와 물리 메모리 여유가
        0 이 된다. 블록 셔플이면 동시에 손대는 구간이 block_windows 로 제한된다.
        """
        if cache is None:
            yield from dl
            return
        for arrays in cache.iter_batches(a.batch, n_batches, rng,
                                         block_windows=a.block_windows):
            yield tuple(torch.from_numpy(x) for x in arrays)

    hist, best = [], None
    t_all = time.time()
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); agg, nb = {}, 0
        for batch in epoch_batches():
            fine, wide, tgt = to_targets(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(fine, wide)
                parts = crit(out, tgt)
            opt.zero_grad(set_to_none=True)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            nb += 1
        agg = {k: v / max(nb, 1) for k, v in agg.items()}
        t_train = time.time() - t0

        if ep % a.eval_every and ep != a.epochs:
            print(f"  ep{ep:>3d}  loss {agg['total']:.4f} (pw {agg['power']:.4f} "
                  f"harm {agg['harm']:.4f})  [{t_train:.0f}s, "
                  f"{a.epoch_windows/max(t_train,1e-9):,.0f} win/s]", flush=True)
            continue

        pred, onp = evaluate(model, prep, dev)
        sc, summ, cm, resid = report(pred, onp, hs)
        row = {"epoch": ep, "loss": agg, "mae": summ["mae_w_mean"], "f1": summ["f1_mean"],
               "resistive_acc": cm["accuracy"] if cm else None,
               "oven_err": cm["matrix"][1][0] if cm else None,
               "resid_abs": resid["mean_abs_w"], "sec": round(time.time() - t0, 1),
               "train_sec": round(t_train, 1)}
        hist.append(row)
        print(f"  ep{ep:>3d}  loss {agg['total']:.4f} (pw {agg['power']:.4f} "
              f"harm {agg['harm']:.4f})  |  MAE {row['mae']:.3f}W  F1 {row['f1']:.4f}  "
              f"저항3종 {row['resistive_acc']:.3f}  잔차 {row['resid_abs']:.1f}W  "
              f"[{t_train:.0f}s 학습 / {row['sec']-t_train:.0f}s 평가, "
              f"{a.epoch_windows/max(t_train,1e-9):,.0f} win/s]", flush=True)
        if best is None or row["f1"] > best["f1"]:
            best = row
            torch.save({"model": model.state_dict(), "appliances": apps,
                        "width": a.width, "epoch": ep},
                       Path(a.out) / f"{a.tag}.pt")

    pred, onp = evaluate(model, prep, dev)
    sc, summ, cm, resid = report(pred, onp, hs)
    print("\n" + format_table(sc))
    print(f"\n  기기 평균 MAE {summ['mae_w_mean']:.2f}W | F1 평균 {summ['f1_mean']:.3f} "
          f"| 최악 F1 {summ['worst_f1'][0]:.3f} ({summ['worst_f1'][1]})")
    if cm:
        print(f"  저항3종 혼동 정확도 {cm['accuracy']:.3f} | "
              f"오븐→포트 {cm['matrix'][1][0]}/{sum(cm['matrix'][1])} "
              f"({100*cm['matrix'][1][0]/max(sum(cm['matrix'][1]),1):.1f}%)")
    print(f"  저부하 FA_rel(고부하 동시) < 0.15 : {summ['fa_target_pass']}")
    print(f"  총전력 잔차 절대 평균 {resid['mean_abs_w']:.2f}W")

    print(f"\n  {'':22s}{'Phase1 GBM':>14s}{'Phase3 CNN':>14s}")
    for lab, base, got in [("기기 평균 MAE (W)", 1.43, summ["mae_w_mean"]),
                           ("F1 평균", 0.952, summ["f1_mean"]),
                           ("저항3종 정확도", 0.968, cm["accuracy"] if cm else float("nan")),
                           ("총전력 잔차 (W)", 12.64, resid["mean_abs_w"])]:
        better = "승" if (got < base if "MAE" in lab or "잔차" in lab else got > base) else "패"
        print(f"  {lab:22s}{base:>14.3f}{got:>14.3f}   {better}")

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.tag}.json").write_text(json.dumps({
        "phase": "3-cnn", "params_m": n_par / 1e6,
        "config": vars(a), "window_cycles": WINDOW_CYCLES,
        "holdout": {k: hs.meta[k] for k in ("n_windows", "content_sha256", "target_index")},
        "history": hist, "per_appliance": [x.as_row() for x in sc],
        "summary": summ, "resistive_confusion": cm, "total_power_residual": resid,
        "elapsed_s": round(time.time() - t_all, 1),
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n완료 {time.time()-t_all:.0f}s | {Path(a.out)/f'{a.tag}.json'}")
    print("=" * 84 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
