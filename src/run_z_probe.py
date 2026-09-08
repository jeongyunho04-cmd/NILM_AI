"""선로 임피던스 Z 를 모델이 아는가 (13.55~13.56)
===================================================
세 가지를 잰다. `--probe` / `--real` / `--sweep` 로 골라 켠다 (기본 앞의 둘).

    ① 탐침   몸통 z 256차원에서 log(r_grid) 를 선형으로 얼마나 복원하나.
             `aux_z` 헤드가 있으면 그 출력의 R² 도 함께 낸다
    ② 실측   실측 녹화의 Z 를 맞히나. **모델이 본 적 없는 라벨**이다
             (계단법으로 따로 쟀다, 13.4 / `SITE_SESSIONS.z_ohm`)
    ③ 掃引   같은 창을 r 만 바꿔 만들면 예측이 얼마나 흔들리나

⚠ **掃引의 축이 학습 범위 안인지 확인할 것** (13.55.1). 13.54 의 掃引은 x=0.3r 이라
  Z=3.0 에서 x=0.9Ω 로 학습 범위(0.02~0.15)의 6배 밖이었고, 거기서 본 "86% 흔들림" 은
  표현의 결함이 아니라 외삽이었다. 여기 기본 掃引은 r 만 범위 안에서 움직이고
  x 를 중앙값 0.085Ω 에 고정한다.

⚠ Z 를 바꾸면 텍스처 난수도 같이 바뀐다 (13.2 가 텍스처를 V·R·X 에서 파생시킨다).
  그래서 이것은 "Z 단독" 이 아니라 **선로 조건 전체**에 대한 불변성이다.

⚠ 입력 57채널의 탐침 R² 는 **상한이 아니다** — 타깃 한 순간의 선형 읽기일 뿐이고,
  모델은 창 전체를 비선형으로 본다. 실제로 몸통이 그것을 크게 넘는다 (13.55.2).

    python -m src.run_z_probe --ckpt results/cnn_v24b.pt results/adapt_v24b_z_s0.pt
    python -m src.run_z_probe --ckpt results/cnn_v24b.pt --sweep --n-sweep 40
"""
from typing import Optional, Sequence
import argparse
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.model.inputs import build_inputs, fine_target_index
from src.model.realdata import dense_targets
from src.preprocessing.file_registry import SITE_SESSIONS
from src.run_gate_check import load_model

TI = fine_target_index()
STEMS = ["test_1", "test_2", "test_3", "test_4", "test_5"]
TRUE_Z = {s: (k, v["z_ohm"]) for k, v in SITE_SESSIONS.items()
          for s in v["stems"] if v["z_ohm"] is not None}
#: 학습 범위 안의 掃引 값 (`GridSimulator.r_grid_range` = (0.30, 2.00), 무리는 0.42~1.19)
SWEEP_R = (0.20, 0.50, 1.00, 1.90)
SWEEP_X = 0.085                      #: `x_grid_range` (0.02, 0.15) 의 중앙값
LAMS = (1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)


def _fit(Xtr, ytr, lam):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ ytr)
    return mu, sd, w


def _pred(model, X):
    mu, sd, w = model
    return np.c_[(X - mu) / sd, np.ones(len(X))] @ w


def ridge_r2(Xtr, ytr, Xte, yte):
    """정칙화를 **학습 절반 안에서** 고른다.

    몸통 z 는 256차원, 입력 슬라이스는 57차원이라 고정 lam 으로 견주면 차원이 큰 쪽이
    불리하다 (실제로 lam=1e-2 에서 R² 가 −27만까지 갔다). 안쪽 분할로 각자 최적 lam 을
    고른 뒤 같은 시험 절반에서 잰다.
    """
    n = len(Xtr)
    i = np.random.RandomState(1).permutation(n)
    a, b = i[:n // 2], i[n // 2:]
    best, blam = -np.inf, LAMS[0]
    for lam in LAMS:
        p = _pred(_fit(Xtr[a], ytr[a], lam), Xtr[b])
        r2 = 1.0 - np.var(ytr[b] - p) / max(np.var(ytr[b]), 1e-12)
        if r2 > best:
            best, blam = r2, lam
    p = _pred(_fit(Xtr, ytr, blam), Xte)
    return (1.0 - np.var(yte - p) / max(np.var(yte), 1e-12),
            float(np.std(yte - p)), blam)


def make_gen(recipe):
    """학습 캐시(v22/v24)와 같은 구성. meta.json 에서 되짚었다 (13.35.7 규율)."""
    from src.synthesis.augmentor import (DataAugmentor, POWER_SCALE_STD_PRESETS,
                                         STATE_MIX_PRESETS)
    from src.synthesis.dataset import NILMBatchGenerator
    from src.synthesis.segment_pool import SegmentPool
    from src.synthesis.synthesizer import LoadSynthesizer
    from src.run_recipe_mix_probe import PRESETS
    mix = PRESETS[recipe] if isinstance(recipe, str) else recipe
    pool = SegmentPool(npz_dir="processed_data/npz", time_split="train",
                       carrier_apps=("oven",))
    aug = DataAugmentor(power_scale_std_map=POWER_SCALE_STD_PRESETS["measured"],
                        state_mix={a: {int(s): float(v) for s, v in m.items()}
                                   for a, m in STATE_MIX_PRESETS["minipc_balanced"].items()},
                        sp_curves=True)
    syn = LoadSynthesizer(segment_pool=pool, compute_gt_harmonics=False,
                          augmentor=aug, background=False, couple_ext=True)
    return NILMBatchGenerator(segment_pool=pool, window_size_cycles=3600, synthesizer=syn,
                              recipe_mix=mix, compute_gt_harmonics=False)


@torch.no_grad()
def trunk_and_out(model, fine, wide, dev, batch: int = 256):
    """몸통 출력 z 와 예측을 같이 뽑는다 (`trunk` 에 forward hook)."""
    buf = []
    h = model.trunk.register_forward_hook(
        lambda _m, _i, o: buf.append(o.float().cpu().numpy()))
    P, G, LZ = [], [], []
    try:
        for i in range(0, len(fine), batch):
            # autocast 를 안 쓴다 — 탐침은 정밀도가 중요하다
            o = model(torch.from_numpy(fine[i:i + batch]).to(dev),
                      torch.from_numpy(wide[i:i + batch]).to(dev))
            P.append(o["power"].float().cpu().numpy())
            G.append(torch.sigmoid(o["on_logit"]).float().cpu().numpy())
            if "log_z" in o:
                LZ.append(o["log_z"].float().cpu().numpy())
    finally:
        h.remove()
    return (np.concatenate(buf), np.concatenate(P), np.concatenate(G),
            np.concatenate(LZ) if LZ else None)


@torch.no_grad()
def file_logz(model, stem: str, dev: str, stride: int = 30) -> Optional[np.ndarray]:
    rw = dense_targets(stem, stride=stride,
                       site_transfer=getattr(model, "site_transfer", None))
    out = []
    for i in range(0, len(rw), 512):
        f, w, *_ = rw.batch(np.arange(i, min(i + 512, len(rw))))
        o = model(torch.from_numpy(np.ascontiguousarray(f)).to(dev),
                  torch.from_numpy(np.ascontiguousarray(w)).to(dev))
        if "log_z" not in o:
            return None
        out.append(o["log_z"].float().cpu().numpy())
    return np.concatenate(out)


def gen_train(n: int, seed0: int = 9000):
    """학습 분포 그대로. 참 Z 는 `metadata` 에서 읽는다."""
    g = make_gen("smps_hi_res3")
    X, R = [], []
    for i in range(n):
        np.random.seed(seed0 + i)
        smp, _ = g._synthesize_window()
        X.append(g._format_inputs(smp))
        R.append(smp.metadata["r_grid_ohm"])
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{n}", flush=True)
    f, w = build_inputs(np.stack(X))
    return f, w, np.log(np.asarray(R))


def gen_sweep(rs: Sequence[float], n_seed: int, seed0: int = 4000):
    """같은 시드로 r 만 바꾼다. x 는 학습 중앙값에 고정 (13.55.1)."""
    g = make_gen({"smps_overlap": 1.0})
    out = {}
    for r in rs:
        g.synthesizer.grid_sim.r_grid_range = (r, r)
        g.synthesizer.grid_sim.x_grid_range = (SWEEP_X, SWEEP_X)
        X, TP = [], []
        for i in range(n_seed):
            np.random.seed(seed0 + i)
            smp, _ = g._synthesize_window()
            X.append(g._format_inputs(smp))
            TP.append([smp.gt_target_power_w[x][g.target_index] for x in g.appliance_list])
        out[r] = (*build_inputs(np.stack(X)), np.asarray(TP))
        print(f"    r={r:.2f} 완료", flush=True)
    return out, g.appliance_list


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-sweep", type=int, default=40)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="기본은 꺼져 있다 (합성이 오래 걸린다)")
    a = ap.parse_args()
    if not (a.probe or a.real or a.sweep):
        a.probe = a.real = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.probe:
        print("=" * 96)
        print("① 탐침 — 학습 분포에서 log(r_grid) 를 얼마나 아는가")
        print("=" * 96)
        print(f"  창 {a.n_train} 개 합성 중...", flush=True)
        fine, wide, lz = gen_train(a.n_train)
        idx = np.random.RandomState(0).permutation(len(lz))
        tr, te = idx[:len(lz) // 2], idx[len(lz) // 2:]
        print(f"\n  참 log r  평균 {lz.mean():+.3f}  분산 {lz.var():.4f}"
              f"  (= '평균만 맞히기' 의 MSE)")
        r2i, ei, li = ridge_r2(fine[:, :, TI][tr], lz[tr], fine[:, :, TI][te], lz[te])
        print(f"  입력 57채널 (타깃 순시, 선형)  R² {r2i:6.3f}  잔차 σ {ei:.3f}  (lam {li:g})")
        print(f"  ⚠ 이것은 **상한이 아니다** — 모델은 창 전체를 비선형으로 본다\n")
        print(f"  {'체크포인트':34s}{'몸통 z R²':>11s}{'잔차 σ':>9s}{'헤드 R²':>9s}{'헤드 σ':>8s}")
        for p in a.ckpt:
            m, _, _ = load_model(p, dev)
            zt, _, _, hz = trunk_and_out(m, fine, wide, dev)
            r2t, et, _ = ridge_r2(zt[tr], lz[tr], zt[te], lz[te])
            if hz is None:
                tail = f"{'—':>9s}{'—':>8s}"
            else:
                r2h = 1.0 - np.var(lz - hz) / max(np.var(lz), 1e-12)
                tail = f"{r2h:9.3f}{float(np.std(lz - hz)):8.3f}"
            print(f"  {p.replace(chr(92), '/').split('/')[-1]:34s}{r2t:11.3f}{et:9.3f}{tail}")
            del m
            torch.cuda.empty_cache()

    if a.real:
        print("\n" + "=" * 96)
        print("② 실측 녹화의 Z 를 맞히나 — 모델은 이 라벨을 본 적이 없다")
        print("=" * 96)
        print(f"  {'체크포인트':24s}" + "".join(f"{s:>12s}" for s in STEMS)
              + f"{'평균|오차|':>12s}")
        print(f"  {'참 Z [Ω]':24s}" + "".join(f"{TRUE_Z[s][1]:12.2f}" for s in STEMS))
        print(f"  {'세션':24s}" + "".join(f"{TRUE_Z[s][0]:>12s}" for s in STEMS))
        for p in a.ckpt:
            m, _, ck = load_model(p, dev)
            short = p.replace(chr(92), "/").split("/")[-1]
            if not ck.get("aux_z", False):
                print(f"  {short:24s}  (보조 헤드 없음)")
            else:
                row, err = [], []
                for s in STEMS:
                    z = float(np.exp(np.median(file_logz(m, s, dev))))
                    row.append(z)
                    err.append(abs(np.log(z / TRUE_Z[s][1])))
                print(f"  {short:24s}" + "".join(f"{v:12.2f}" for v in row)
                      + f"{np.mean(err) * 100:11.1f}%")
            del m
            torch.cuda.empty_cache()
        print("\n  값은 파일 전체 창의 **중앙값**. 오차는 |log(예측/참)| 을 % 로 읽은 것이다.")
        print("  ⚠ 같은 파일 안에서도 SMPS 전용 창은 낮게 읽는다 (13.57.1) —"
              " 구성별로 보려면 `run_site_split_check`.")

    if a.sweep:
        print("\n" + "=" * 96)
        print(f"③ 掃引 — r 만 학습 범위 안에서, x 는 {SWEEP_X}Ω 고정")
        print("=" * 96)
        print(f"  창 {a.n_sweep} 개 x r {len(SWEEP_R)} 값 합성 중...", flush=True)
        sw, apps = gen_sweep(SWEEP_R, a.n_sweep)
        watch = [x for x in ("beam_projector", "laptop_charger", "minipc") if x in apps]
        ji = [apps.index(x) for x in watch]
        tp = sw[SWEEP_R[0]][2][:, ji].mean(0)
        print("\n  참 전력 (r 에 불변): "
              + "  ".join(f"{n} {tp[k]:.2f}W" for k, n in enumerate(watch)))
        for p in a.ckpt:
            m, mapps, _ = load_model(p, dev)
            jj = [mapps.index(x) for x in watch]
            print(f"\n  {p}")
            print(f"    {'r[Ω]':>6s}" + "".join(f"{n[:12]:>14s}" for n in watch)
                  + f"{'관문(첫째)':>12s}{'예측 총합':>11s}")
            rows = []
            for r in SWEEP_R:
                f, w, _ = sw[r]
                _, P, G, _ = trunk_and_out(m, f, w, dev)
                rows.append((P[:, jj].mean(0), G[:, jj[0]].mean(), P.sum(1).mean()))
                print(f"    {r:6.2f}" + "".join(f"{v:14.2f}" for v in rows[-1][0])
                      + f"{rows[-1][1]:12.3f}{rows[-1][2]:11.2f}")
            arr = np.stack([q[0] for q in rows])
            print(f"    {'폭':>6s}" + "".join(f"{v:14.2f}" for v in np.ptp(arr, 0))
                  + f"{max(q[1] for q in rows) - min(q[1] for q in rows):12.3f}"
                  + f"{max(q[2] for q in rows) - min(q[2] for q in rows):11.2f}")
            print(f"    {'참값비':>6s}" + "".join(
                f"{np.ptp(arr, 0)[k] / max(tp[k], 1e-9) * 100:13.0f}%"
                for k in range(len(watch))))
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
