"""
Phase 3 — 2갈래 CNN 학습 (1단계: 합성 사전학습)
================================================
설계 문서 2·3·4.1절. 12.8절에서 확정한 창 구성을 쓴다.

    세밀 10초 @ 60Hz (36,600)  +  광역 60초 @ 2Hz (12,120)
    타깃 = 60초 창의 끝-1초 (인덱스 3539)  ->  추론 지연 1초

**이 모델은 Phase 1 baseline 을 이겨야 의미가 있다** (4.1절, 10절).
기준선은 `results/baseline_gbm.json` 에서 읽는다 (`baseline_reference()`).
2026-08-21 기준 평균 MAE 1.45W / F1 0.951 / 저항3종 0.990 / 오븐→포트 8.1%

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
from src.model.inputs import FINE_CYCLES, TARGET_LOOKAHEAD, build_inputs
from src.synthesis.dataset import chunk_seed
from src.model.traincache import CachedWindows
from src.model.losses import LossWeights, NILMLoss, build_state_scales
from src.model.net import (
    NILMNet, appliance_state_counts, harmonic_scales, harmonic_signatures,
    noise_signature, standby_signatures,
)
from src.run_baseline import LOW_LOAD, S_I, baseline_reference

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
        from src.synthesis.dataset import NILMBatchGenerator
        from src.synthesis.segment_pool import SegmentPool
        from src.synthesis.synthesizer import LoadSynthesizer
        # 시드는 여기서 걸지 않는다 - `__getitem__` 이 배치 번호로 건다.
        pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
        self.gen = NILMBatchGenerator(
            segment_pool=pool, window_size_cycles=WINDOW_CYCLES,
            synthesizer=LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False),
            compute_gt_harmonics=False)

    def __getitem__(self, i: int):
        self._ensure()
        # 시드는 **배치 번호**로 건다. 워커 번호로 걸면 `--workers` 를 바꾸는 것만으로
        # 같은 시드가 다른 학습 데이터를 만든다 (`chunk_seed` 주석, 12.11절).
        np.random.seed(chunk_seed(self.seed, i))
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
    ap.add_argument("--w-state-power", type=float, default=0.0, metavar="W",
                    help="상태별 전력 출력을 그 상태의 실제 전력에 묶는 항 (12.35). "
                         "0 이면 끈다 - 그러면 전력 손실이 섞인 뒤에만 걸려 "
                         "충전기·미니PC 의 상태별 출력이 붕괴한다 (분화비 1.02 / 1.16).")
    ap.add_argument("--fine-dropout", type=float, default=0.0,
                    help="학습 중 세밀 갈래를 통째로 가릴 확률 (12.21절). 합성에서 학습한 "
                         "선형 probe 가 실측에서 세밀은 AUC 0.32 로 뒤집히고 광역은 0.69 를 "
                         "유지한다 - 광역을 쓰는 법을 배우게 강제한다")
    ap.add_argument("--wide-summary", action="store_true",
                    help="광역 갈래에도 amax + 창끝 슬라이스를 준다 (12.19.4 후보 1)")
    ap.add_argument("--periodicity", action="store_true",
                    help="자기상관·교차율을 헤드 직전에 직접 준다 (12.19.4 후보 2)")
    ap.add_argument("--w-over", type=float, default=0.1,
                    help="물리 상한 힌지. 예측 합이 관측 총전력을 넘을 때만 벌한다")
    ap.add_argument("--prior-kappa", type=float, default=8.0,
                    help="on 게이트 물리 프라이어 세기 (12.9.8절). 0 이면 끈다")
    ap.add_argument("--prior-beta", type=float, default=0.5,
                    help="최소 ON 전력에 곱하는 안전 여유. 작을수록 느슨하다")
    ap.add_argument("--eval-every", type=int, default=1, help="N epoch 마다 홀드아웃 평가")
    ap.add_argument("--select", choices=("final", "best-f1"), default="final",
                    help="체크포인트 선택. 기본 final - 홀드아웃으로 고르면 평가셋이 "
                         "모델 선택을 겸해 보고 숫자가 편향된다 (12.9.9절)")
    ap.add_argument("--snapshot-every", type=int, default=50,
                    help="N epoch 마다 results/snapshots/ 에 중간 체크포인트 저장 (0=끄기). "
                         "중단 대비 + 나중에 epoch 수가 적당했는지 사후 판정용")
    ap.add_argument("--per-state-scale", dest="per_state_scale",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="손실 척도를 (기기,상태)별로 (12.9.9절). --no-per-state-scale 로 끈다")
    ap.add_argument("--block-windows", type=int, default=24_000,
                    help="캐시 블록 셔플 단위. 작을수록 메모리가 덜 든다 (24000 = 약 1.1GB)")
    ap.add_argument("--harm-odd-only", action="store_true",
                    help="L_harm 에서 짝수차를 뺀다 (12.75절). 짝수차는 계측 인공물이라 "
                         "(12.72) 손실이 가장 큰 가중을 그것에 걸고 있었다 (12.70.3)")
    ap.add_argument("--holdout", default=HOLDOUT_DIR, metavar="DIR",
                    help="합성 홀드아웃 디렉터리. TARGET_LOOKAHEAD 를 바꾸면 라벨 시점이 "
                         "달라지므로 홀드아웃도 그 값으로 다시 만들어야 한다 (12.45)")
    ap.add_argument("--cache", default="cache/train60",
                    help="학습 캐시 경로. 'none' 이면 실시간 합성 (12.8.2절 참조)")
    ap.add_argument("--fine-channels", type=int, default=None, metavar="N",
                    help="세밀 갈래가 쓸 채널 수 (기본: inputs.FINE_CHANNELS). "
                         "캐시는 그대로 두고 앞에서부터 N 개만 쓴다. "
                         "12.34 의 고조파 위상 6채널을 빼고 대조군을 학습할 때 "
                         "--fine-channels 38 로 준다. 캐시가 같으므로 채널 수 "
                         "말고는 아무것도 안 달라진다.")
    ap.add_argument("--tag", default="cnn")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    env_guard.verify_numerics()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 84)
    print(f"[Phase 3] 2갈래 CNN — 세밀 10초@60Hz + 광역 60초@2Hz, 타깃 끝-1초")
    print("=" * 84)

    hs = load_holdout(a.holdout)
    apps = hs.appliances
    prep = prepare_holdout_inputs(hs)
    # 변환이 끝나면 3.8GB 원시 memmap 은 더 필요 없다. 놓아 주어야
    # 그 페이지가 작업집합에 남지 않는다.
    hs.X = np.zeros((len(prep[0]), 1, 1), np.float32)
    from src.model.inputs import target_index as _tgt_idx
    want_tgt = _tgt_idx(int(hs.meta["window_cycles"]))
    if int(hs.meta["target_index"]) != want_tgt:
        # TARGET_LOOKAHEAD 를 바꾸면 라벨 시점이 옮겨간다. 안 막으면 조용히 틀린다.
        raise SystemExit(
            "홀드아웃의 타깃 시점이 현재 코드와 다릅니다: "
            f"{hs.meta['target_index']} vs {want_tgt}  ({a.holdout})" + chr(10)
            + "  TARGET_LOOKAHEAD 를 바꿨다면 홀드아웃도 다시 만드십시오:" + chr(10)
            + f"  python -m src.run_build_holdout --out <새 디렉터리> "
            + f"--window-cycles {hs.meta['window_cycles']} "
            + f"--windows {hs.meta['n_windows']} --seed {hs.meta['seed']}")
    print(f"평가: 홀드아웃 {len(hs):,}창 (뒤 {hs.meta['holdout_frac']:.0%}) "
          f"| sha {hs.meta['content_sha256']} | 타깃 {hs.meta['target_index']}/{hs.meta['window_cycles']}")

    from src.synthesis.segment_pool import SegmentPool
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
    sig = harmonic_signatures(pool, apps)
    sb_sig = standby_signatures(pool, apps)
    nz_sig = noise_signature(pool)
    h_scale = harmonic_scales(pool, apps)
    del pool

    model = NILMNet(apps, appliance_state_counts(apps), width=a.width,
                    wide_summary=a.wide_summary, periodicity=a.periodicity,
                    fine_dropout=a.fine_dropout,
                    prior_kappa=a.prior_kappa, prior_beta=a.prior_beta,
                    fine_channels=a.fine_channels).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz_sig),
        harm_scale=torch.from_numpy(h_scale),
        harm_odd_only=a.harm_odd_only,
        weights=LossWeights(harm=a.w_harm, cons=a.w_cons, over=a.w_over,
                            state_power=a.w_state_power),
        s_state=(build_state_scales(apps, [S_I[x] for x in apps])
                 if a.per_state_scale else None),
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

    def save_ckpt(path: Path, ep_saved: int) -> None:
        """프라이어 설정도 함께 저장한다. 빠뜨리면 재평가가 kappa=0 으로 모델을
        되살려 **학습과 다른 모델을 채점한다** (12.9.8절)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "appliances": apps,
                    "width": a.width, "epoch": ep_saved,
                    "prior_kappa": a.prior_kappa, "prior_beta": a.prior_beta,
                    "wide_summary": a.wide_summary, "periodicity": a.periodicity,
                    "fine_dropout": a.fine_dropout,
                    # 세밀 채널 수를 반드시 남긴다. 12.34 에서 38 -> 44 로
                    # 늘었고, 이 키가 없는 체크포인트는 38 로 간주된다.
                    "fine_channels": model.fine_channels,
                    # 타깃 시점 구성 (12.45). 채널 수와 달리 슬라이스로 못 맞춘다 —
                    # 어긋나면 입력과 라벨이 다른 순간을 가리켜 조용히 틀린다.
                    "target_lookahead": TARGET_LOOKAHEAD,
                    "fine_cycles": FINE_CYCLES,
                    "select": a.select}, path)

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
            # 손실 항은 **GPU 텐서로** 누적한다. 여기서 `float(v)` 를 부르면 항마다
            # 스트림 동기화가 걸려(9개 항 x 매 스텝) CPU 가 다음 배치를 미리 읽지
            # 못하고 GPU 앞에서 멈춰 선다. 캐시 읽기 24.5ms 가 GPU 유휴 시간에
            # 통째로 노출되어, 실측에서 79.4 -> 59.3 ms/배치 (6,451 -> 8,639 win/s,
            # +34%) 의 차이가 났다. 12.9.7절 참조.
            for k, v in parts.items():
                d = v.detach()
                agg[k] = d if k not in agg else agg[k] + d
            nb += 1
        # epoch 이 끝난 뒤 한 번만 CPU 로 가져온다.
        agg = {k: float(v) / max(nb, 1) for k, v in agg.items()}
        t_train = time.time() - t0

        # 중간 스냅샷. 두 가지를 준다 — 중단되면 잃는 것이 최대 N epoch 이고,
        # 나중에 "epoch 수가 적당했는가" 를 재학습 없이 사후 판정할 수 있다.
        # 최종 체크포인트(`{tag}.pt`)와 섞이지 않게 하위 디렉터리에 둔다.
        if a.snapshot_every > 0 and ep % a.snapshot_every == 0 and ep != a.epochs:
            save_ckpt(Path(a.out) / "snapshots" / f"{a.tag}_ep{ep:04d}.pt", ep)

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
        if a.select == "best-f1" and best is row:
            save_ckpt(Path(a.out) / f"{a.tag}.pt", ep)

    if a.select == "final":
        # **홀드아웃으로 체크포인트를 고르지 않는다** (12.9.9절).
        # 고르면 홀드아웃이 모델 선택과 성능 보고를 겸하게 되어 보고 숫자가
        # 낙관 쪽으로 편향된다. 4.3절이 실측에는 봉인까지 두면서 합성 홀드아웃의
        # 이 오염은 방치돼 있었다. cosine 이 마지막 epoch 에서 0 으로 떨어지고
        # 12.9.6절에서 곡선이 ep210 부터 평평한 것을 확인했으므로 마지막을 쓴다.
        save_ckpt(Path(a.out) / f"{a.tag}.pt", a.epochs)
        if best is not None and best["epoch"] != a.epochs:
            print(f"  [참고] 최고 F1 은 ep{best['epoch']} ({best['f1']:.4f}, "
                  f"MAE {best['mae']:.2f}W) 였다. 저장한 것은 마지막 ep{a.epochs} 이다.")

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
    ref = baseline_reference()
    for lab, base, got in [("기기 평균 MAE (W)", ref["mae"], summ["mae_w_mean"]),
                           ("F1 평균", ref["f1"], summ["f1_mean"]),
                           ("저항3종 정확도", ref["resistive_acc"],
                            cm["accuracy"] if cm else float("nan")),
                           ("총전력 잔차 (W)", ref["resid_abs"], resid["mean_abs_w"])]:
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
