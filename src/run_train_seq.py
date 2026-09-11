# -*- coding: utf-8 -*-
"""끝까지 함께 — 몸통 + 방출 머리 + 전이 머리를 사슬 손실로 같이 학습한다 (13.84.26).

```
지금   입력 -> 2갈래 CNN -> 기기별 머리(켜짐·전력) -> 창마다 독립 판정
바꿈   입력 -> 같은 CNN  -> 머리 둘(방출·전이)     -> 사슬 디코딩 -> 파일 전체 상태 열
```

손실은 **켜짐 항만** 바뀐다. `NILMLoss` 의 `total` 에서 `w.on * parts["on"]` 을 빼고
`w_crf * CRF` 를 더한다. 전력·상태·플러그·대기·고조파·z 는 창마다 정답이 있으므로 그대로다.

13.84.24 의 얼린 몸통 판이 실측 0.852 -> 0.912~0.922 를 냈다(시드 6판). 이 판은 몸통도 배운다.

⚠ **실측 채점은 매번 지금 몸통으로 다시 돈다.** 얼린 판의 z 를 재사용하면 몸통이 바뀐 뒤로
   옛 표현을 채점하는 꼴이 된다.

    python -X utf8 src/run_train_seq.py --cache cache/seqraw_v1 --epochs 40
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401

import torch
from torch.utils.data import DataLoader, Dataset

from src.model.chain import BgHead, ChainHeads, crf_nll, viterbi
from src.model.inputs import build_inputs
from src.model.lossbuild import build_loss
from src.model.losses import LossWeights
from src.model.transition import N_FEAT, feat_at
from src.model.realdata import RealWindows
from src.run_gate_check import load_model
from src.run_train_cnn import vswap

FS = 60
FILES = ["test_1", "test_2", "test_3", "test_4", "test_5"]
PRE = 13
LABELS = ("y_on", "y_power", "y_plugged", "y_standby", "y_state",
          "obs_harm", "p_noise", "p_observed", "z_grid")


def dfeat_seq(H, P, cycles):
    df = np.zeros((len(cycles), N_FEAT + 2), np.float32)
    for q, c in enumerate(cycles):
        f_, dp_ = feat_at(H, P, int(c))
        df[q, :N_FEAT] = f_
        df[q, -2] = np.sign(dp_)
        df[q, -1] = np.log10(abs(dp_) + 1e-3)
    return df


class SeqChunks(Dataset):
    """기록 하나에서 연속 `chunk` 단계를 잘라 입력을 **그때 만든다**.

    ⚠ 입력을 미리 구우면 격자 2초·세밀 10초라 같은 사이클이 다섯 번 저장돼 5.9배다
      (단계당 155.6KB · 120만 단계면 187GB). 워커가 만드는 편이 싸다 (코어당 572단계/초).
    """

    def __init__(self, cache, idx, chunk, tgt_off, w_cyc, grid, fixed_start=None):
        #: 홀드아웃 채점은 **자리를 고정**한다 — 매번 다른 조각을 보면 epoch 사이 값이 흔들려
        #: 평평해졌는지 못 읽는다.
        self.fixed_start = fixed_start
        self.c = Path(cache)
        self.idx = np.asarray(idx)
        self.chunk, self.tgt_off, self.w_cyc = int(chunk), int(tgt_off), int(w_cyc)
        self.grid = np.asarray(grid)
        self._a = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        if self._a is None:
            self._a = {n: np.load(self.c / (n + ".npy"), mmap_mode="r")
                       for n in ("raw",) + LABELS}
        i = int(self.idx[j])
        T = len(self.grid)
        if self.chunk >= T:
            s = 0
        elif self.fixed_start is not None:
            s = int(self.fixed_start) % (T - self.chunk + 1)
        else:
            s = int(np.random.randint(0, T - self.chunk + 1))
        sl = slice(s, s + min(self.chunk, T))
        g = self.grid[sl]
        raw = np.asarray(self._a["raw"][i])
        win = np.stack([raw[:, c - self.tgt_off:c - self.tgt_off + self.w_cyc] for c in g])
        fine, wide = build_inputs(win)
        out = {"fine": fine, "wide": wide,
               "dfeat": dfeat_seq((raw[0:15] + 1j * raw[15:30]).T, raw[30], g)}
        for n in LABELS:
            out[n] = np.asarray(self._a[n][i][sl])
        return out


def collate(b):
    return {k: torch.from_numpy(np.stack([x[k] for x in b])) for k in b[0]}


def real_windows(apps, grid_s, dev):
    """실측 파일의 입력·Δ특징·참값을 **한 번만** 만든다. 몸통은 채점 때마다 새로 돈다."""
    from src.preprocessing import load_nilm_npz
    ev = json.load(open("processed_data/real_events.json", encoding="utf-8"))["files"]
    stride = int(grid_s * FS)
    out = {}
    for stem in FILES:
        rw = RealWindows(stems=[stem], stride=stride, require_valid=False)
        tgt = rw.target_cycle
        r = load_nilm_npz("processed_data/composite_eval/%s.npz" % stem)
        H = np.asarray(r["harmonics_complex"])
        P = np.asarray(r["power_features"])[:, 0]
        n = len(P)
        ok = (tgt >= PRE * FS + 1) & (tgt < n - PRE * FS - 1)
        sel = np.nonzero(ok)[0]
        F, W = [], []
        for i in range(0, len(sel), 512):
            f, w, _, _, _ = rw.batch(sel[i:i + 512])
            F.append(f); W.append(w)
        t = tgt[sel] / FS
        y = np.zeros((len(t), len(apps)), np.int8)
        for k, a in enumerate(apps):
            for t0, t1 in ev[stem]["intervals"].get(a, {}).get("on", []):
                y[(t >= t0) & (t < t1), k] = 1
        out[stem] = dict(fine=np.concatenate(F), wide=np.concatenate(W),
                         dfeat=dfeat_seq(H, P, tgt[sel]), y=y, t=t,
                         present=np.array([a in ev[stem]["appliances_present"] for a in apps]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/seqraw_v1")
    ap.add_argument("--init", default="results/cnn_v37.pt",
                    help="몸통 초기점. 'scratch' 면 **무작위 초기화**로 처음부터 배운다 (13.84.27)")
    ap.add_argument("--ref", default="results/cnn_v37.pt",
                    help="--init scratch 일 때 **구조·가림**을 가져올 체크포인트")
    ap.add_argument("--no-mask", action="store_true",
                    help="v37 이 가린 고차 위상 채널(세밀 4~7,12~15,37,38 · 광역 33,34)을 **살린다** "
                         "(13.84.27). ⚠ --init scratch 에서만 쓸 것 — 물려받은 가중치에는 본 적 없는 "
                         "입력이 된다. 가린 이유는 생성기가 h9~15 위상을 못 만들어서였는데(13.84.12), "
                         "지금 캐시는 sibling_rotate 로 그 단서를 원천에서 없앴고 전이 머리는 어차피 "
                         "원시를 가리지 않고 본다")
    ap.add_argument("--vswap-p", type=float, default=0.0, metavar="P",
                    help="학습 배치에서 전압 고조파 채널을 다른 창 것으로 바꿔 끼운다 (13.84.11). "
                         "⚠ v37 은 0.5 로 배웠다 — 0 으로 이어 학습하면 그 규제가 풀린다")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--records-per-batch", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-heads", type=float, default=3e-4)
    ap.add_argument("--w-crf", type=float, default=0.3)
    ap.add_argument("--pow-sig", action="store_true",
                    help="전력 의존 지문 (13.84.38). `L_harm` 이 전력에 선형인 것을 고친다 — "
                         "와트당 고차가 동작점에 따라 47~109%% 변한다 (13.84.32). 순방향 잔차 −13~34%%")
    ap.add_argument("--bg-head", action="store_true",
                    help="기록(=세션)당 배경 전류 하나 (13.84.38). `noise_sig` 는 전역 상수 하나인데 "
                         "실측 배경은 파일 간 3.6~22.6mA 로 다르고 h15 에서 미니PC 보다 크다 (13.84.35)")
    ap.add_argument("--w-bg", type=float, default=3.0, metavar="W",
                    help="배경 항의 크기 벌점. 0 이면 자유 항이 되어 **슬랙**이 된다")
    ap.add_argument("--keep-on", type=float, default=0.0, metavar="F",
                    help="켜짐 BCE 를 얼마나 남길지 (0=통째로 뺌, 1=그대로 두고 CRF 를 더함). "
                         "처음부터 배우는 판은 0 이면 게이트가 무감독이 된다 (13.84.27)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--score-norm", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--no-real", action="store_true",
                    help="실측 채점을 건너뛴다 — 실측 npz 가 없는 곳에서 (13.84.27). "
                         "대신 **합성 홀드아웃**으로 사슬 대 창별을 잰다. 실측은 체크포인트를 "
                         "받아 로컬에서 채점한다 (실측은 원래 학습에 안 들어간다)")
    ap.add_argument("--holdout-records", type=int, default=300,
                    help="--no-real 일 때 채점에 쓸 홀드아웃 기록 수")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="seq_v1")
    ap.add_argument("--freeze-trunk", action="store_true", help="몸통을 얼린다 (대조군)")
    ap.add_argument("--w-harm", type=float, default=0.1)
    ap.add_argument("--w-z", type=float, default=0.3)
    ap.add_argument("--w-over", type=float, default=0.0)
    ap.add_argument("--no-window-loss", action="store_true",
                    help="CRF 만 쓴다 (절제용). 기본은 창 단위 항을 다 쓰고 켜짐만 CRF 로 바꾼다")
    a = ap.parse_args()

    c = Path(a.cache)
    meta = json.loads((c / "meta.json").read_text(encoding="utf-8"))
    apps = meta["appliances"]
    K = len(apps)
    N = int(meta["record_s"] * FS)
    grid = np.arange(meta["target_offset"], N - PRE * FS - 1, int(meta["grid_s"] * FS))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # `--init scratch` 는 **v37 과 같은 구조·같은 가림에 무작위 가중치**다 (13.84.27).
    # 구조는 체크포인트에서 읽되 가중치만 새로 뽑아, 바뀐 것이 초기점 하나가 되게 한다.
    scratch = str(a.init).lower() in ("scratch", "random", "none")
    if scratch:
        torch.manual_seed(a.seed)
    if a.no_mask and not scratch:
        raise SystemExit("--no-mask 는 --init scratch 에서만 쓴다 (물려받은 가중치엔 본 적 없는 입력이다)")
    model, apps_m = load_model(a.ref if scratch else a.init, dev,
                               weights=not scratch, mask=not a.no_mask)[:2]
    assert list(apps_m) == list(apps), "기기 열 순서가 캐시와 다르다"
    with torch.no_grad():
        zdim = int(model(torch.zeros(1, model.fine_channels, 600, device=dev),
                         torch.zeros(1, 47, 120, device=dev))["z"].shape[1])
    torch.manual_seed(a.seed)
    heads = ChainHeads(zdim, N_FEAT + 2, K, hidden=a.hidden, score_norm=a.score_norm).to(dev)
    bghead = BgHead(zdim).to(dev) if a.bg_head else None

    n_rec = meta["records"]
    n_ho = max(1, int(n_rec * a.holdout_frac))
    idx = np.random.RandomState(0).permutation(n_rec)
    tr_i = idx[n_ho:]
    ho_i = idx[:n_ho]
    ds = SeqChunks(a.cache, tr_i, a.chunk, meta["target_offset"], meta["window_cycles"], grid)
    dl = DataLoader(ds, batch_size=a.records_per_batch, shuffle=True,
                    num_workers=a.workers, collate_fn=collate, drop_last=True,
                    persistent_workers=a.workers > 0)
    print("[seq] 기록 %d (학습 %d) · 격자 %d단계 · 조각 %d · z %d · 기기 %d · %s%s"
          % (n_rec, len(tr_i), len(grid), a.chunk, zdim, K, dev,
             (" · 몸통 얼림" if a.freeze_trunk else "")
             + (" · **처음부터**(무작위 몸통)" if scratch else " · 초기점 " + Path(a.init).stem)
             + (" · vswap %.2f" % a.vswap_p if a.vswap_p else "")
             + (" · 가림 해제" if a.no_mask else "")
             + (" · keep-on %.2f" % a.keep_on if a.keep_on else "")
             + (" · 전력의존지문" if a.pow_sig else "")
             + (" · 세션배경(벌점 %.1f)" % a.w_bg if a.bg_head else "")), flush=True)

    if a.freeze_trunk:
        for p in model.parameters():
            p.requires_grad_(False)
    groups = [{"params": heads.parameters(), "lr": a.lr_heads}]
    if bghead is not None:
        groups.append({"params": bghead.parameters(), "lr": a.lr_heads})
    if not a.freeze_trunk:
        groups.append({"params": model.parameters(), "lr": a.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.epochs * (len(ds) // a.records_per_batch)))

    crit = None
    if not a.no_window_loss:
        crit = build_loss(apps, dev, weights=LossWeights(
            harm=a.w_harm, cons=0.0, over=a.w_over, z=a.w_z),
            power_signatures=a.pow_sig)
        print("[seq] 창 단위 손실 조립 · 켜짐 가중 %.2f 를 CRF %.2f 로 갈아 끼운다"
              % (crit.w.on, a.w_crf), flush=True)

    real = None
    ho_dl = None
    if a.no_real:
        ho_ds = SeqChunks(a.cache, ho_i[:a.holdout_records], a.chunk, meta["target_offset"],
                          meta["window_cycles"], grid, fixed_start=0)
        ho_dl = DataLoader(ho_ds, batch_size=a.records_per_batch, shuffle=False,
                           num_workers=min(4, a.workers), collate_fn=collate)
        print("[seq] 실측 채점 **건너뜀** — 합성 홀드아웃 %d기록으로 잰다 (13.84.27)"
              % len(ho_ds), flush=True)
    else:
        real = real_windows(apps, meta["grid_s"], dev)
        print("[seq] 실측 준비 " + " · ".join("%s %d단계" % (s, len(d["t"]))
                                            for s, d in real.items()), flush=True)

    def score_holdout():
        """합성 홀드아웃에서 사슬 대 창별.

        ⚠ **실측 값과 직접 견주지 마라** — 여기는 전 기기·전 단계를 세고, 실측 채점은
        그 파일에 **있는 기기만** 세서 기기별로 평균한다. 규약이 다르다
        ([[match-the-scoring-convention-before-comparing]]). 두 판(A/B) 을 서로 견주는 데만 쓴다.
        """
        model.eval(); heads.eval()
        ca = wa = nn_ = 0.0
        with torch.no_grad():
            for bt in ho_dl:
                B, T = bt["y_on"].shape[0], bt["y_on"].shape[1]
                o = model(bt["fine"].to(dev).flatten(0, 1), bt["wide"].to(dev).flatten(0, 1))
                z = o["z"].reshape(B, T, -1)
                gl = o["on_logit"].reshape(B, T, K)
                em, on, off, ini = heads(z, bt["dfeat"].to(dev), gl)
                y = bt["y_on"].to(dev).bool()
                ca += float((viterbi(em, on, off, ini) == y).float().sum())
                wa += float(((gl > 0) == y).float().sum())
                nn_ += y.numel()
        model.train(); heads.train()
        return ca / nn_, wa / nn_, {}

    def score_real():
        model.eval(); heads.eval()
        accs, mp, base = [], {}, []
        with torch.no_grad():
            for stem, d in real.items():
                Z, GL = [], []
                for i in range(0, len(d["t"]), 512):
                    o = model(torch.from_numpy(d["fine"][i:i + 512]).to(dev),
                              torch.from_numpy(d["wide"][i:i + 512]).to(dev))
                    Z.append(o["z"].float()); GL.append(o["on_logit"].float())
                z = torch.cat(Z)[None]
                gl = torch.cat(GL)[None]
                em, on, off, ini = heads(z, torch.from_numpy(d["dfeat"]).float()[None].to(dev), gl)
                p = viterbi(em, on, off, ini)[0].cpu().numpy()
                w = (gl[0] > 0).cpu().numpy()
                y = d["y"].astype(bool)
                for k in np.nonzero(d["present"])[0]:
                    accs.append(float((p[:, k] == y[:, k]).mean()))
                    base.append(float((w[:, k] == y[:, k]).mean()))
                    mp.setdefault(stem, {})[k] = (base[-1], accs[-1])
        model.train(); heads.train()
        return float(np.mean(accs)), float(np.mean(base)), mp

    def save(ep, r, b):
        """⚠ **epoch 마다** 덮어쓴다 (13.84.27). 시간 한도에 걸려 죽어도 거기까지는 건진다 —
        SLURM 이 FIFO 라 짧은 시간을 걸어야 backfill 에 끼는데, 그러면 초과 위험이 생긴다."""
        torch.save({"model": model.state_dict(), "heads": heads.state_dict(),
                    "appliances": apps, "meta": meta, "hidden": a.hidden,
                    "score_norm": a.score_norm, "init": a.init, "ref": a.ref,
                    "vswap_p": a.vswap_p, "no_mask": a.no_mask, "keep_on": a.keep_on,
                    "pow_sig": a.pow_sig, "bg_head": a.bg_head, "w_bg": a.w_bg,
                    "bghead": (bghead.state_dict() if bghead is not None else None),
                "epoch": ep, "epochs": a.epochs,
                    "holdout_chain": r, "holdout_window": b,
                    "zdim": zdim, "ddim": N_FEAT + 2}, "results/%s.pt" % a.tag)
        print("     저장 results/%s.pt (epoch %d)" % (a.tag, ep), flush=True)

    score = score_holdout if a.no_real else score_real
    r, b, _ = score()
    print("  epoch  0 (학습 전)  %s 사슬 %.4f / 창별 %.4f"
          % ("홀드아웃" if a.no_real else "실측", r, b), flush=True)
    t0 = time.time()
    nwin = a.records_per_batch * a.chunk
    for ep in range(1, a.epochs + 1):
        tot, nb = 0.0, 0
        te = time.time()
        for bt in dl:
            B, T = bt["y_on"].shape[0], bt["y_on"].shape[1]
            fine = bt["fine"].to(dev).flatten(0, 1)
            wide = bt["wide"].to(dev).flatten(0, 1)
            vswap(fine, wide, a.vswap_p)          # 13.84.11 — 0 이면 아무것도 안 한다
            o = model(fine, wide)
            z = o["z"].reshape(B, T, -1)
            em, on, off, ini = heads(z, bt["dfeat"].to(dev), o["on_logit"].reshape(B, T, K))
            crf = crf_nll(em, on, off, bt["y_on"].to(dev).bool(), ini)
            # 세션 배경 — 조각 하나당 값 **하나**를 내고 그 안의 모든 창에 같이 건다 (13.84.38).
            bg_off = None
            if bghead is not None:
                bg = bghead(z)                                          # (B,H,2)
                bg_off = bg[:, None].expand(B, T, -1, -1).flatten(0, 1)
            if crit is None:
                loss = a.w_crf * crf
            else:
                # 창 단위 항은 그대로 쓰고 **켜짐 BCE 만** CRF 로 갈아 끼운다.
                tg = {"y_power": bt["y_power"].to(dev).flatten(0, 1),
                      "y_on": bt["y_on"].to(dev).float().flatten(0, 1),
                      "y_plugged": bt["y_plugged"].to(dev).float().flatten(0, 1),
                      "y_standby": bt["y_standby"].to(dev).flatten(0, 1),
                      "y_state": bt["y_state"].to(dev).long().flatten(0, 1),
                      "obs_harm": bt["obs_harm"].to(dev).flatten(0, 1),
                      "p_noise": bt["p_noise"].to(dev).flatten(0, 1),
                      "p_observed": bt["p_observed"].to(dev).flatten(0, 1),
                      "harm_offset": bg_off,
                      "log_z": torch.log(bt["z_grid"].to(dev)[..., 0].clamp(min=1e-3)).flatten(0, 1)}
                parts = crit(o, tg)
                # ⚠ `--keep-on` 이 0 이면 켜짐 BCE 를 **통째로** 뺀다 — v37 에서 이어 배울 때는
                #   그 감독을 이미 받은 몸통이라 괜찮지만, **처음부터 배우면 게이트가 한 번도
                #   감독을 못 받는다** (13.84.27). 그러면 방출이 base 와 emit 의 **큰 값 상쇄**로
                #   풀려 실측에서 부서진다 (항 규모 ±0.6 대 v37 판 ±0.05, 잔차는 둘 다 −0.05~−0.09).
                #   일부를 남기면 한 판으로 끝난다 — 2단계 학습을 안 해도 된다.
                loss = (parts["total"] - (1.0 - a.keep_on) * crit.w.on * parts["on"]
                        + a.w_crf * crf)
                if bghead is not None:
                    # ⚠ 크기 벌점이 없으면 자유 항이 되어 기기에서 전류를 빼앗는다.
                    loss = loss + a.w_bg * (bg / crit.harm_scale[None, :, None]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(heads.parameters()) + list(model.parameters())
                + (list(bghead.parameters()) if bghead is not None else []), 5.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        el = time.time() - te
        if ep % a.eval_every == 0 or ep == a.epochs:
            r, b, mp = score()
            print("  epoch %2d  손실 %8.4f · %.0f초/epoch (%.0f창/초) · %s 사슬 %.4f / 창별 %.4f · 남은 %.0f분"
                  % (ep, tot / max(nb, 1), el, nb * nwin / max(el, 1e-6),
                     "홀드아웃" if a.no_real else "실측", r, b,
                     (a.epochs - ep) * el / 60), flush=True)
            save(ep, r, b)
        else:
            print("  epoch %2d  손실 %8.4f · %.0f초/epoch (%.0f창/초) · 남은 %.0f분"
                  % (ep, tot / max(nb, 1), el, nb * nwin / max(el, 1e-6),
                     (a.epochs - ep) * el / 60), flush=True)
    r, b, mp = score()
    km = apps.index("minipc")
    if mp:                                    # --no-real 이면 실측 표가 없다
        print("\n미니PC 시간 정확도 (실측)")
        print("  파일      창별   사슬")
        for stem in FILES:
            if km in mp.get(stem, {}):
                print("  %-8s %6.3f %6.3f" % (stem, mp[stem][km][0], mp[stem][km][1]))
    save(a.epochs, r, b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
