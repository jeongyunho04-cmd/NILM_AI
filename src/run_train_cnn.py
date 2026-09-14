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
from src.model.inputs import (ZERO_EVEN_HARMONICS, FINE_CYCLES, FINE_LAYOUT,
                             FINE_VOLT0, TARGET_LOOKAHEAD, V_CENTER, V_SPAN,
                             VOLT_ORDERS, WIDE_CHANNELS,
                             WIDE_VOLT0, build_inputs, RAW_CHANNELS)
from src.synthesis.dataset import chunk_seed
from src.model.traincache import CachedWindows
from src.model.losses import (LossWeights, NILMLoss, PHASE_COHERENT_EVEN,
                             build_state_scales)
from src.model.net import (
    NILMNet, appliance_state_counts, harmonic_scales, harmonic_signatures,
    noise_signature, standby_signatures,
)
from src.run_baseline import LOW_LOAD, S_I, baseline_reference

WINDOW_CYCLES = 3600          # 60초. 광역 갈래가 이 전체를 2Hz 로 본다
HOLDOUT_DIR = "processed_data/holdout60"


class SynthBatchDataset(Dataset):
    """실시간 합성 (창 재사용 0).

    ⚠⚠ **본선 학습에 쓰지 마라 — 캐시보다 50배 느리다** (14.9, 2026-09-13).
    여기 원래 "12.1절 측정대로 캐시보다 워커가 낫다" 고 적혀 있었는데 **13.10.5 가 그것을
    이미 뒤집었다.** 13.2 의 회로모델 델타(`SmpsCircuit`, `randomize_r=True` 라 창마다
    캐시 미스)가 생성기를 430 -> 56창/초로 **10배** 늦췄다. 12.1 은 그 전 세대 숫자다.
    ```
    캐시    30만창 한 번 굽기 + 에폭당 5만창 x 300에폭 = 창 재사용 50.0회
            HPC 64코어 굽기 19분 + 학습 19분          = ~40분
    실시간  5만창 x 300에폭 = 1,500만창을 매번 새로    = 캐시의 50배 = 16시간+
    ```
    979793 이 이 경로로 나갔다가 **14분에 1에폭도 못 끝내고** 취소됐다.
    쿼터(79.5G/100G) 때문에 GPFS 에 캐시를 못 두면 `/dev/shm` 에 굽는다
    (`patches/cnn_vexp2.sbatch`). 이 경로는 **작은 진단·항등검정 전용**이다.
    [[derived-limits-outlive-their-reason]] [[legacy-artifacts-shadow-new-files]]

    **배치 단위로 돌려준다** (`DataLoader(batch_size=None)`). 창 1개씩 변환하면
    numpy 호출 오버헤드가 지배해 261 win/s 까지 떨어진다 — 생성기 자체(2,700 win/s)의
    1/10 이다. 배치로 묶으면 벡터 연산이 살아난다.
    또 원시 창(512 x 33 x 3600 = 243MB)이 아니라 변환 결과(47MB)만 워커 경계를
    넘으므로 IPC 도 5배 가볍다.
    """

    def __init__(self, n_batches: int, batch_size: int, seed: int,
                 gen_spec: str = "v32", recipe_mix: str = "steady2",
                 smps_focus_off_p: float = 0.4):
        self.n_batches = n_batches
        self.bs = batch_size
        self.seed = seed
        #: 생성기 설정 (14.8). ⚠ **이 경로는 오래 맨몸이었다** — `LoadSynthesizer(segment_pool=pool)`
        #: 만 만들어 `cache_v32.sbatch` 가 캐시에 건 설정 **열하나가 빠져 있었다**
        #: (couple_ext · sp_curves · sp_per_texture · vtail · float_fill · steady_crop ·
        #:  standby_jitter_cap · state_mix · power_scale_std, 그리고 창 층의 recipe_mix ·
        #:  smps_focus_off_p). 13.84.27 이 `run_build_seqraw` 의 **같은 결함**을 찾아
        #: `genopts.py` 를 만들었는데 여기는 안 고쳐져 있었다.
        #: [[derive-commands-from-config-not-prose]] [[verify-the-gate-runs-that-path]]
        self.gen_spec = gen_spec
        self.recipe_mix = recipe_mix
        self.smps_focus_off_p = float(smps_focus_off_p)
        self.gen = None

    def __len__(self) -> int:
        return self.n_batches

    def _ensure(self):
        if self.gen is not None:
            return
        import json as _json
        from src.synthesis.dataset import NILMBatchGenerator
        from src.synthesis.genopts import build_synthesizer, check, describe, resolve
        # 시드는 여기서 걸지 않는다 - `__getitem__` 이 배치 번호로 건다.
        opts = resolve(self.gen_spec)
        syn = build_synthesizer(opts, "processed_data/npz", "train")
        bad = check(opts, syn)
        if bad:
            raise SystemExit("[train_cnn] 생성기 설정이 안 걸렸다: " + " / ".join(bad))
        mix = None
        if self.recipe_mix:
            from src.run_recipe_mix_probe import PRESETS
            mix = (PRESETS[self.recipe_mix] if self.recipe_mix in PRESETS
                   else _json.loads(self.recipe_mix))
        self.gen = NILMBatchGenerator(
            segment_pool=syn.pool, window_size_cycles=WINDOW_CYCLES,
            synthesizer=syn, compute_gt_harmonics=False,
            recipe_mix=mix, smps_focus_off_p=self.smps_focus_off_p)
        print("[train_cnn] 실시간 합성 생성기 '%s': %s · recipe_mix=%s · smps_focus_off_p=%.2f"
              % (self.gen_spec, describe(opts), self.recipe_mix, self.smps_focus_off_p),
              flush=True)

    def __getitem__(self, i: int):
        self._ensure()
        # 시드는 **배치 번호**로 건다. 워커 번호로 걸면 `--workers` 를 바꾸는 것만으로
        # 같은 시드가 다른 학습 데이터를 만든다 (`chunk_seed` 주석, 12.11절).
        np.random.seed(chunk_seed(self.seed, i))
        g, n = self.gen, self.bs
        k = len(g.appliance_list)
        ti = g.target_index
        # ⚠ **33 이 박혀 있었다** (14.8). 13.26 이 전압 고조파 12채널을 더해 `RAW_CHANNELS`
        #   가 45 가 됐는데 이 경로만 안 따라왔다 — 그 뒤로 한 번도 안 돌았다는 뜻이다.
        #   상수를 박지 말고 `inputs.RAW_CHANNELS` 를 쓴다.
        xs = np.empty((n, RAW_CHANNELS, WINDOW_CYCLES), np.float32)
        yp = np.empty((n, k), np.float32); yo = np.empty((n, k), np.float32)
        ypl = np.empty((n, k), np.float32); ys = np.empty((n, k), np.float32)
        yst = np.empty((n, k), np.int64)
        oh = np.empty((n, 15, 2), np.float32)
        pn = np.empty(n, np.float32); pobs = np.empty(n, np.float32)
        zg = np.empty((n, 2), np.float32)          # 13.55 선로 임피던스 [r, x]
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
            zg[j] = (smp.metadata.get("r_grid_ohm", np.nan),
                     smp.metadata.get("x_grid_ohm", np.nan))
        fine, wide = build_inputs(xs)
        return tuple(torch.from_numpy(a) for a in
                     (fine, wide, yp, yo, ypl, ys, yst, oh, pn, pobs, zg))


def cache_index_plan(n: int, batch_size: int, n_batches: int,
                     rng, block_windows: int = 24_000):
    """`CachedWindows.iter_batches` 와 **똑같은 색인 열**을 만든다 (I/O 없이).

    같은 `rng` 를 같은 차례로 소모하므로, 이 계획으로 뽑은 배치는 옛 경로가 낸
    배치와 **정확히 같다.** 동치는 `tests` 의 `test_cache_index_plan_matches_iter_batches`
    가 못박는다. 저쪽을 고치면 이쪽도 같이 고쳐야 한다.
    """
    n_blocks = max(1, (n + block_windows - 1) // block_windows)
    plan, made = [], 0
    while made < n_batches:
        for b in rng.permutation(n_blocks):
            lo = int(b) * block_windows
            hi = min(lo + block_windows, n)
            if hi - lo < batch_size:
                continue
            order = lo + rng.permutation(hi - lo)
            for k in range(0, len(order) - batch_size + 1, batch_size):
                plan.append(order[k:k + batch_size])
                made += 1
                if made >= n_batches:
                    return plan
    return plan


class CacheBatchDataset(Dataset):
    """미리 정한 색인으로 캐시에서 배치를 뽑는다 (13.47).

    `batch_size=None` · `shuffle=False` 로 쓴다 — DataLoader 가 순서를 지키므로
    워커를 붙여도 배치 열이 그대로다. 윈도우는 spawn 이라 memmap 손잡이가 pickle 이
    안 되므로 **워커마다 처음 쓸 때** 연다.
    """

    def __init__(self, cache_dir: str, plan):
        self.cache_dir = str(cache_dir)
        self.plan = plan
        self._c = None

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int):
        if self._c is None:
            from src.model.traincache import CachedWindows
            self._c = CachedWindows(self.cache_dir)
        return tuple(torch.from_numpy(x) for x in self._c.batch(self.plan[i]))


def vswap(fine: torch.Tensor, wide: torch.Tensor, p: float) -> None:
    """전압 고조파 채널(V_h h1~11 Re/Im, 세밀 45~56 · 광역 35~46)을 **같은 자리의 다른 창** 것으로
    바꿔 끼운다 — 창 단위 확률 `p`, 제자리 수정 (13.84.11).

    왜: 모델이 생성기의 **정확한 V→I 법칙**(회로 델타 + 텍스처별 s(p))을 판별자로 배운다. 합성 충전기+
    프로젝터 창의 전압 채널을 같은 자리의 다른 합성 창 것으로 바꾸기만 해도(전압도 전류도 각각은 학습 분포
    안) 미니PC 유령이 8~12% -> 24~32% 로 뛰고, 실측 전압을 붙이면 0.12 -> 0.50 이다. 실측의 (V, I) 짝은
    생성기 법칙에서 벗어나므로 그 잔차가 "설명 안 되는 SMPS 전류 = 미니PC" 로 읽힌다 (실패 ②의 운반자).
    같은 자리 안에서만 바꾸므로 자리·Z 수준의 정보는 남고 텍스처 단위의 정확한 짝만 깨진다. V 실효값 채널
    (세밀 25 · 광역 2)은 안 건드린다 — 자리 대리이자 Z 정보다. 0 이면 아무것도 안 한다.
    """
    if p <= 0:
        return
    B = fine.shape[0]
    nv = 2 * len(VOLT_ORDERS)
    site_d = wide[:, 2].mean(1) < 0                 # 광역 V 채널 부호: 자리 D(216V) 음수 / E(229V) 양수
    perm = torch.arange(B, device=fine.device)
    for s in (True, False):
        idx = torch.nonzero(site_d == s).squeeze(1)
        if len(idx) > 1:
            perm[idx] = idx[torch.randperm(len(idx), device=fine.device)]
    sel = torch.where(torch.rand(B, device=fine.device) < p, perm, torch.arange(B, device=fine.device))
    fine[:, FINE_VOLT0:FINE_VOLT0 + nv] = fine[sel][:, FINE_VOLT0:FINE_VOLT0 + nv]
    wide[:, WIDE_VOLT0:WIDE_VOLT0 + nv] = wide[sel][:, WIDE_VOLT0:WIDE_VOLT0 + nv]


def _vnorm_exp(apps, classes: str = ""):
    """기기별 `I/P` 의 전압 지수 `(i_exp − p_exp)` (K,) — 14.28.

    저항 `(1.0, 2.0)` · SMPS `(−1.0, 0.0)` 은 둘 다 **−1**, 모터 `(0.7, 0.7)` 과
    수동 `(1.0, 1.0)` 은 **0** 이다. 균일 −1 을 걸면 모터에 틀린 물리를 가르친다.

    `classes` 를 주면 **그 부하 분류에만** 보정을 건다 (14.33). 나머지는 0 이다.

    왜 그런 손잡이가 필요한가 — 두 절반의 근거가 다르다:
      · **저항**은 유도다. `I_h = V_h/R`, `P = V²/R` 이라 전압 *모양*이 고정된 채
        크기만 변하면 **모든 차수에서** `I_h/P ∝ 1/V` 다. 적합이 필요 없다.
      · **SMPS** 는 `h1` 에서만 선다. `P ≈ V·I₁·cosφ` 라 `I₁/P ≈ 1/(V·cosφ)` 는
        거의 **항등식**이고(기기와 무관하게 −1 이 나온다), `h>=3` 은 도통각이
        전압에 따라 변해 `V^e` 꼴이 아니다. 실측으로 차수별 지수를 재 봤지만
        **녹화 사이에 재현이 안 된다** (충전기 h11 폭 79.5 · 미니PC h1 폭 9.2) —
        확인도 반증도 못 하는 자리다.
      14.29 ④ 가 적어 둔 대가가 정확히 그 절반에 있다: F1 −0.0029/−0.0003/−0.0015 ·
      충전기 MAE +0.06/+0.31/+0.18W (셋 다 같은 부호).
    """
    from src.preprocessing.file_registry import get_load_class
    from src.synthesis.grid_simulator import GridSimulator
    t = GridSimulator()._LOAD_EXPONENTS
    want = {x.strip().upper() for x in classes.split(",") if x.strip()}
    # ⚠ **모르는 이름은 죽는다.** 안 그러면 오타 하나가 `want` 를 아무것도 안 맞게 만들어
    #   **전 기기 지수가 조용히 0** 이 된다 — 보정이 통째로 꺼진 판을 A/B 로 착각하게 된다.
    #   `BASE` 변수에 `--harm-vnorm-classes ""` 를 넣었다가 확장 뒤 따옴표 두 글자가
    #   값으로 넘어간 적이 있다 (14.37). 14.8 의 `resolve()` 와 같은 규약이다.
    known = {c.name.upper() for c in t}
    bad = want - known
    if bad:
        raise SystemExit("--harm-vnorm-classes: 모르는 분류 %s — 있는 것: %s"
                         % (sorted(bad), sorted(known)))
    out = []
    for a in apps:
        c = get_load_class(a)
        e = float(t[c][0] - t[c][1])
        if want and c.name.upper() not in want:
            e = 0.0
        out.append(e)
    return torch.tensor(out, dtype=torch.float32)


def _res_ohm(apps, res_apps: str, half: bool) -> torch.Tensor:
    """(K,) 등가저항 Ω. 목록에 없는 기기는 0 = 안 건다 (14.32)."""
    from src.model.postproc import HALFWAVE_OHM, RESISTIVE_OHM
    want = {x.strip() for x in res_apps.split(",") if x.strip()}
    tbl = HALFWAVE_OHM if half else RESISTIVE_OHM
    return torch.tensor([tbl[x] if (x in tbl and x in want) else 0.0 for x in apps],
                        dtype=torch.float32)


def _res_cond(apps, spec: str) -> torch.Tensor:
    """(K,) long 통전 상태 번호. `"oven:2,hotplate:2"` 꼴. 0 = 켜짐이 곧 통전 (14.32)."""
    d = {}
    for it in spec.split(","):
        if not it.strip():
            continue
        k, _, v = it.partition(":")
        k = k.strip()
        if k not in apps:
            raise SystemExit(f"--res-cond-state: 모르는 기기 {k!r} — 있는 것: {apps}")
        d[k] = int(v)
    return torch.tensor([d.get(x, 0) for x in apps], dtype=torch.long)


def to_targets(batch, dev):
    (fine, wide, yp, yo, ypl, ys, yst, oh, pn, pobs, zg) = [
        b.to(dev, non_blocking=True) for b in batch]
    return fine, wide, {
        "y_power": yp, "y_on": yo, "y_plugged": ypl, "y_standby": ys, "y_state": yst,
        "obs_harm": oh, "p_noise": pn, "p_observed": pobs, "harm_offset": None,
        # 14.26 — 창 전압비 V/V_CENTER. `L_harm` 의 지문을 이것으로 나눈다 (`--harm-sig-vnorm`).
        #   세밀 채널 25 가 `(v − V_CENTER)/V_SPAN` 이다 (`net.V_CH_FINE`).
        "vrel": (fine[:, 25].mean(-1) * V_SPAN + V_CENTER) / V_CENTER,
        # 14.32 — `L_swap` 이 쓰는 창 전압 (V). **같은 채널에서 같은 식으로** 낸다.
        #   ⚠ 이것이 없으면 `_swap_term` 의 가드가 **조용히 0** 을 낸다.
        "v_rms": fine[:, 25].mean(-1) * V_SPAN + V_CENTER,
        # 13.55 — 이 창의 선로 저항 [Ω]. 옛 캐시면 NaN 이고 손실이 알아서 건너뛴다.
        "log_z": torch.log(zg[:, 0].clamp(min=1e-3)),
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


def _cache_says_background(cache: str) -> bool:
    """캐시 meta 의 `background` 플래그. 없거나 못 읽으면 False."""
    if not cache or str(cache).lower() == "none":
        return False
    try:
        return bool(json.load(open(Path(cache) / "meta.json",
                                   encoding="utf-8")).get("background", False))
    except Exception:
        return False


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
    ap.add_argument("--vexp", action="store_true",
                    help="전압 지수를 구조에 박는다 (14.7). `p_raw *= (V/V_CENTER)^e_k` 로 "
                         "저항 e=2 · SMPS 0 · 유도기 0.6. 모델은 상태 명목값은 기울기 1.00 으로 "
                         "잘 내는데 **같은 상태 안의 V² 의존**을 0.33~0.83 로만 읽어 순 지수가 "
                         "0.85 다(물리는 2). 저전압에서 과예측한다 (14.6). 끄면 비트 동일")
    ap.add_argument("--w-harm", type=float, default=0.1)
    ap.add_argument("--harm-sig-vnorm", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="**기본 켜짐 (14.29 에서 채택).** L_harm 의 지문에 창 전압비를 "
                         "기기별 지수로 건다 (14.26·14.28). sig=median(I/P) 인데 I=P/V 라 "
                         "I/P ∝ V^(i_exp−p_exp) 다 — 상수 sig 는 저항·SMPS 에 power ∝ V¹ 을 "
                         "밀어 모델 지수를 1 에 앉힌다 (물리 2). 켜면 3/3 시드에서 지수가 "
                         "2 로 가고 실측 절대잔차가 41->24W 로 준다. 모터는 지수 0 이라 "
                         "보정이 안 걸린다. `--no-harm-sig-vnorm` 이면 옛 판과 비트 동일")
    ap.add_argument("--harm-vnorm-classes", default="RESISTIVE", metavar="LIST",
                    help="`--harm-sig-vnorm` 보정을 **이 부하 분류에만** 건다 (14.33). "
                         "**2026-09-14 기본이 `RESISTIVE` 로 바뀌었다** (14.37) — 시드 셋에서 "
                         "판정 줄 **+0.0067 ± 0.0007** 이고 저항 4종 신원은 여섯 판 전부 "
                         "0.9815 로 한 자리도 안 움직인다. `\"\"`(빈 문자열)이면 전 분류 = "
                         "14.28 판이고 옛 결과가 재현된다. "
                         "`RESISTIVE` 를 주면 **유도로 정확한 곳에만** 남는다 — SMPS 의 "
                         "−1 은 h1 에서 거의 항등식이고 h>=3 은 도통각 때문에 V^e 꼴이 "
                         "아니며, 실측 차수별 지수가 녹화 사이에 재현되지 않는다 "
                         "(충전기 h11 폭 79.5). 분류: RESISTIVE·SMPS·MOTOR·PASSIVE.")
    ap.add_argument("--w-cons", type=float, default=0.0, help="1단계는 0 (3.3절)")
    # ── 저항 조합 맞바꿈 `L_swap` 을 1단계에 (14.32) ──────────────────────────
    ap.add_argument("--w-swap", type=float, default=0.0, metavar="W",
                    help="**저항 조합 맞바꿈** `L_swap` (12.158 을 1단계로, 14.32). "
                         "니크롬선은 P=V²/R 이고 R 이 기기 고유값이라 컨덕턴스로 옮기면 "
                         "조합을 셀 수 있다 — 오븐 40.15Ω 대 포트 35.65Ω 은 겹치지 "
                         "않는데(각 폭 1.5~2.4%%, 녹화 사이 0.08%%) 고조파로는 각도가 "
                         "1.91°%% 라 못 가른다. 겹친 구간에서 몫을 통째로 뒤집어도 고조파 "
                         "모양이 0.1~0.2%%밖에 안 변한다. **기본 0 = 완전히 꺼짐.** "
                         "⚠ `L_res` 가 아니라 이 항을 쓴다 — `L_res` 는 비저항 일곱의 "
                         "게이트로 기울기가 샌다 (`run_gate_swap1.py` [1]).")
    ap.add_argument("--swap-tol", type=float, default=0.02, metavar="TOL",
                    help="상대오차 문턱. 12.112.3 이 0.02 를 최적으로 쟀다 "
                         "(0.01 은 아무것도 안 고치고 0.05 이상은 엉뚱한 조합을 문다).")
    ap.add_argument("--swap-slack", type=int, default=0, metavar="N",
                    help="켜진 기기 **개수**의 허용 변화. 0 이면 맞바꿈만 (12.112). "
                         "⚠ 후처리에서 이 제한을 풀었을 때 없는 기기를 발명했다.")
    ap.add_argument("--swap-tiebreak", default="mag", choices=("off", "h3", "mag"),
                    help="tol 안에 여러 조합이 들면 무엇으로 고르나 (12.165.6). "
                         "컨덕턴스가 같으면 **전력도 같아**(포트 1377W 대 드라이기강+핫플 "
                         "1392W) 정보가 0 이다. `mag` 는 차수별 크기만 봐 공통 위상 회전에 "
                         "면역이고, `h3`(복소)은 `harm_offset` 이 안 빠져 **반증됐다**.")
    ap.add_argument("--swap-tb-orders", default="3", metavar="LIST",
                    help="동점깨기에 쓸 차수. h1 은 안 쓴다(거기가 축퇴인 축이다). "
                         "12.165.6 이 h3 하나가 맞다고 쟀다.")
    ap.add_argument("--res-apps", default="electiric_kettle,oven,hotplate,hair_dryer",
                    metavar="LIST",
                    help="`--w-swap` 이 저항을 못 박을 기기. **넷 다 기본**이다 — 새 계측기 "
                         "격리 녹화에서 R 폭이 0.4~2.4%%이고 녹화 사이가 오븐 0.08%% · "
                         "핫플 0.26%% 다. 2단계 기본(포트·오븐)이 좁은 것은 옛 계측기 "
                         "결론이라 규칙 1 대상이다. 드라이기는 `HALFWAVE_OHM` 이 "
                         "약풍(108.6Ω)을 따로 가른다.")
    ap.add_argument("--res-cond-state", default="oven:2,hotplate:2", metavar="LIST",
                    help="`on=1` 인데 통전이 아닌 상태가 있는 기기의 **통전 상태 번호** "
                         "(14.32). 오븐 {1: 팬·조명 16.8W, 2: 히터 1357W} · 핫플 "
                         "{1: 표시등 10W, 2: 통전 549.6W} 가 그렇다. 안 주면 조합 탐색이 "
                         "오븐 팬·조명 창을 '통전' 으로 세어 `L_on` 과 싸운다 — 실측에서 "
                         "오븐 게이트가 켜진 창의 전력 중앙이 16.0W 인데 σ·V²/R 은 "
                         "1094W 를 요구한다. 빈 문자열이면 옛 동작(켜짐=통전).")
    ap.add_argument("--w-state-power", type=float, default=0.0, metavar="W",
                    help="상태별 전력 출력을 그 상태의 실제 전력에 묶는 항 (12.35). "
                         "0 이면 끈다 - 그러면 전력 손실이 섞인 뒤에만 걸려 "
                         "충전기·미니PC 의 상태별 출력이 붕괴한다 (분화비 1.02 / 1.16).")
    ap.add_argument("--w-z", type=float, default=0.0, metavar="W",
                    help="몸통 z 에서 log(r_grid) 를 맞히는 보조 감독 (13.55). "
                         "13.54 측정: 입력 57채널에서 log Z 를 R² 0.935 로 뽑는데 "
                         "몸통에서는 0.661 로 흐려진다. 참 전력은 Z 에 불변인데 "
                         "예측은 86%% 폭으로 흔들렸다. 라벨은 캐시의 z_grid 다")
    ap.add_argument("--harm-grad-balance", default="off",
                    choices=("off", "smps", "all"),
                    help="L_harm 의 기울기를 기기별로 균등화한다 (12.120). 값은 안 바뀌고 "
                         "기울기만 바뀐다. smps 는 SMPS 3종 안에서만, all 은 9종 전부. "
                         "**2단계 전용이었다** (`run_adapt`) — 13.84.17 에서 1단계에도 넣었다: "
                         "프로젝터가 와트당 지문 노름이 가장 작아(0.1219 대 미니PC 0.1697) "
                         "잔여의 값싼 흡수처가 되는데, 실측에는 `y_power` 가 없어 "
                         "'1단계는 L_power 가 붙잡아 준다' 가 실측 채점에는 안 걸린다. 기본 off")
    ap.add_argument("--fine-dropout", type=float, default=0.0,
                    help="학습 중 세밀 갈래를 통째로 가릴 확률 (12.21절). 합성에서 학습한 "
                         "선형 probe 가 실측에서 세밀은 AUC 0.32 로 뒤집히고 광역은 0.69 를 "
                         "유지한다 - 광역을 쓰는 법을 배우게 강제한다")
    ap.add_argument("--wide-target", action="store_true",
                    help="광역에 **타깃 블록 슬라이스**를 준다 (13.44). seq2point 인데 "
                         "광역은 hw.mean(-1) 하나뿐이라 순서에 불변이었다. "
                         "--wide-summary 의 창끝 슬라이스는 타깃에서 6초 어긋난다. "
                         "**평균을 대체하지 않고 더한다**")
    ap.add_argument("--wide-summary", action="store_true",
                    help="광역 갈래에도 amax + 창끝 슬라이스를 준다 (12.19.4 후보 1)")
    ap.add_argument("--periodicity", action="store_true",
                    help="자기상관·교차율을 헤드 직전에 직접 준다 (12.19.4 후보 2)")
    ap.add_argument("--seg-pool", type=int, default=0, metavar="N",
                    help="**구간별 풀링** (14.46). 전역 `mean`/`amax` 를 **타깃을 경계로 한 "
                         "N구간**으로 쪼갠다 (세밀·광역·원시 전력 통계 셋 다). "
                         "14.41~14.42 가 잰 것 — 세밀 창 600 중 360(6초)이 타깃보다 뒤인데 "
                         "깊은 탭의 수용영역은 ±93 뿐이라, 그 바깥 증거가 머리에 닿는 길이 "
                         "**위치를 모르는 전역 요약 하나**다. 수용영역 밖만 지워도 오븐 혼합이 "
                         "0.137 -> 0.941 로 살아난다. **0/1 이면 비트 동일**이다. "
                         "⚠ 없애지 않고 쪼갠다 — `amax` 는 conv 로 표현 안 되고(12.9.8) "
                         "`fp_max` 는 물리 프라이어가 쓴다.")
    ap.add_argument("--w-over", type=float, default=0.1,
                    help="물리 상한 힌지. 예측 합이 관측 총전력을 넘을 때만 벌한다")
    ap.add_argument("--gate-smooth", type=float, default=0.0, metavar="EPS",
                    help="**게이트 BCE 라벨 완화** (13.80). y -> y(1-2e)+e 라 최적 "
                         "로짓이 +-log((1-e)/e) 로 묶인다. e=0.05 면 sigma in "
                         "[0.05,0.95] 이라 dsigma/dlogit 이 0.0475 아래로 안 간다. "
                         "합성에서 d' 4~8 로 이미 풀린 과제가 포화해 **실측에 통하는 "
                         "방향을 고를 이유가 사라지는 것**을 막는다. 0 이면 옛 경로.")
    ap.add_argument("--gate-focal", type=float, default=0.0, metavar="GAMMA",
                    help="**쉬운 창 가중 낮추기** (13.80). BCE 에 (1-p_t)^gamma 를 "
                         "곱해 이미 맞은 창이 방향을 정하지 못하게 한다. 로짓은 안 "
                         "묶으므로 --gate-smooth 와 겨냥이 다르다 — 따로 켜서 갈라라.")
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
    ap.add_argument("--cache-workers", type=int, default=0, metavar="N",
                    help="캐시 배치를 뽑는 별도 프로세스 수 (13.47). 0 이면 메인 스레드가 "
                         "직접 훑는다(옛 동작). 배치 **색인 열은 그대로**라 결과가 바뀌지 "
                         "않는다 — 자료 공급을 계산과 겹치게 할 뿐이다. 스레드는 numpy "
                         "고급 색인이 GIL 을 안 놓아 소용이 없었다(0.95배)")
    ap.add_argument("--block-windows", type=int, default=24_000,
                    help="캐시 블록 셔플 단위. 작을수록 메모리가 덜 든다 (24000 = 약 1.1GB)")
    ap.add_argument("--off-detach-praw", action="store_true",
                    help="**꺼진 창에서 `p_raw` 에 경사를 주지 않는다** (13.11). 상태 전력 머리가 "
                         "죽는 것을 막는다 — 드라이기 HIGH 가 정확히 그렇게 0W 가 됐다. "
                         "게이트가 꺼진 창을 0 으로 만드는 일을 맡는다.")
    ap.add_argument("--harm-even-by-class", action="store_true",
                    help="짝수차 위상을 기기 부류별로 살린다 (13.45). "
                         "--harm-even-magnitude 와 같이 써야 뜻이 있다 — 위상이 뭉치는 "
                         "기기(오븐·포트·핫플·드라이기)가 그 창의 짝수차 예측 크기에서 "
                         "차지하는 몫만큼 복소 오차를 되살린다")
    ap.add_argument("--harm-even-magnitude", action="store_true",
                    help="L_harm 에서 **짝수차만 크기 공간**으로 잰다 (13.11). 플러그를 "
                         "반대로 꽂으면 짝수차가 180° 도므로(홀수차는 안 돈다) 짝수차 위상은 "
                         "기기 속성이 아니다 — 드라이기 약풍에서 격리 대 복합이 179° 어긋났다.")
    ap.add_argument("--state-signatures", action="store_true",
                    help="**상태별 고조파 지문** (13.11). 기기당 페이저 하나로는 "
                         "한 기기의 상태들이 고조파 모양이 다를 때 못 담는다 — 드라이기 "
                         "약풍(반파 |I2|/|I1| 0.431)과 강풍(0.0004)이 그 극단이고, "
                         "지문이 약풍에 앉는 바람에 강풍 창에서 정답 배분이 오답보다 "
                         "L_harm 27배 비쌌다. 상태 표본이 200사이클 미만이면 기기 지문으로 되돌린다.")
    ap.add_argument("--harm-odd-only", action="store_true",
                    help="L_harm 에서 짝수차를 뺀다 (12.75절). 짝수차는 계측 인공물이라 "
                         "(12.72) 손실이 가장 큰 가중을 그것에 걸고 있었다 (12.70.3)")
    ap.add_argument("--standby-operating", nargs="?", const="all", default="off",
                    choices=("off", "session", "all"),
                    help="`standby_sig` 를 **동작 중 휴지**의 지문으로 바꾼다 (12.164). "
                         "`SESSION_PLUGGED_APPS` 의 `gt_plugged` 가 '동작 중' 을 뜻하게 "
                         "바뀌면 `idle=σ(plugged)(1−σ(on))` 자리가 FAN_LIGHT 이므로 "
                         "고조파 자도 FAN_LIGHT(67.4mA) 여야 한다. `OFF_STANDBY`(6.44mA) "
                         "을 그대로 두면 전력과 고조파가 10배 어긋난 채 학습된다. "
                         "라벨을 바꿨으면 `session` 을 같이 켜야 짝이 맞는다.")
    ap.add_argument("--holdout", default=HOLDOUT_DIR, metavar="DIR",
                    help="합성 홀드아웃 디렉터리. TARGET_LOOKAHEAD 를 바꾸면 라벨 시점이 "
                         "달라지므로 홀드아웃도 그 값으로 다시 만들어야 한다 (12.45)")
    ap.add_argument("--gen", default="v32", metavar="NAME|JSON",
                    help="**실시간 합성**(`--cache none`)일 때의 생성기 설정 (14.8). "
                         "`genopts.PRESETS` 의 이름이나 JSON. 기본 'v32' = 사슬 이전 판 "
                         "(= v36 − sibling_rotate). ⚠ 캐시를 쓰면 이 값은 무시된다 — "
                         "그때는 캐시를 구울 때의 설정이 정본이다")
    ap.add_argument("--recipe-mix", default="steady2", metavar="NAME|JSON",
                    help="실시간 합성의 **창 층** 레시피 믹스. `cache_v32.sbatch` 와 같은 기본값")
    ap.add_argument("--smps-focus-off-p", type=float, default=0.4, metavar="P",
                    help="실시간 합성의 SMPS 집중 창 비율. `cache_v32.sbatch` 와 같은 기본값")
    ap.add_argument("--cache", default="cache/train60",
                    help="학습 캐시 경로. 'none' 이면 실시간 합성 (12.8.2절 참조)")
    ap.add_argument("--zero-wide-channels", default="", metavar="LIST",
                    help="**광역** 입력의 이 채널들을 0 으로 만든다 (13.12 절제용). "
                         "크기 블록 12~26, 위상 블록 27~34 "
                         "(`inputs.WIDE_MAG_CHANNELS` / `WIDE_PHASE_CHANNELS`). "
                         "캐시를 다시 굽지 않고 광역 확장의 효과를 가른다.")
    ap.add_argument("--zero-channels", default="", metavar="LIST",
                    help="세밀 입력의 이 채널들을 **0 으로 만든다** (쉼표). "
                         "채널 하나의 효과를 재는 **가장 조인 대조**다 — 채널 수를 "
                         "바꾸면 첫 층 모양이 달라져 초기화 난수까지 바뀌고, "
                         "12.114 가 바로 그 재학습 잡음에 묻혀 판정을 못 했다. "
                         "여기서는 구조·초기화·자료 순서가 전부 같고 그 채널의 "
                         "**값만** 없어진다")
    ap.add_argument("--vswap-p", type=float, default=0.0, metavar="P",
                    help="학습 배치에서 전압 고조파 채널을 **같은 자리의 다른 창** 것으로 바꿔 끼울 확률 "
                         "(13.84.11). 생성기의 정확한 V->I 법칙을 판별자로 배우는 것을 막는다 — 합성끼리 "
                         "전압만 바꿔도 미니PC 유령 8%% -> 32%%, 실측 전압을 붙이면 0.12 -> 0.50. 0 이면 옛 경로.")
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

    ZERO_CH = [int(x) for x in a.zero_channels.split(",") if x.strip()]
    if ZERO_CH:
        print(f"  ** 세밀 채널 {ZERO_CH} 를 0 으로 (조인 대조, 12.114 재시험) **")
    ZERO_W = [int(x) for x in a.zero_wide_channels.split(",") if x.strip()]
    if ZERO_W:
        print(f"  ** 광역 채널 {ZERO_W} 를 0 으로 (13.12 절제) **")
    hs = load_holdout(a.holdout)
    apps = hs.appliances
    prep = prepare_holdout_inputs(hs)
    if ZERO_CH:
        prep[0][:, ZERO_CH] = 0.0        # 학습과 평가가 같은 입력을 봐야 한다
    if ZERO_W:
        prep[1][:, ZERO_W] = 0.0
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
    # 동작 중 휴지의 지문 (12.164). `gt_plugged` 가 '동작 중' 으로 바뀐 기기는
    # `idle` 항이 가리키는 상태가 OFF_STANDBY 이 아니라 FAN_LIGHT 이다.
    if a.standby_operating != "off":
        from src.model.companion import standby_operating_signatures
        from src.synthesis.synthesizer import SESSION_PLUGGED_APPS
        _only = SESSION_PLUGGED_APPS if a.standby_operating == "session" else None
        sb_op, sb_pw, sb_used = standby_operating_signatures(pool, apps, only=_only)
        for x in sb_used:
            _j = apps.index(x)
            _o = float(np.hypot(sb_sig[_j, 0, 0], sb_sig[_j, 0, 1])) * 1000
            _n = float(np.hypot(sb_op[_j, 0, 0], sb_op[_j, 0, 1])) * 1000
            print(f"  ** 동작 중 휴지 지문 ({a.standby_operating}) {x}: "
                  f"|I1| {_o:.2f} -> {_n:.2f} mA, 전력 {sb_pw[_j]:.2f}W **")
            sb_sig[_j] = sb_op[_j]
    nz_sig = noise_signature(pool)
    # 상시 배경 (12.166.3). 캐시가 배경을 넣었으면 순방향 모형도 넣어야 짝이 맞는다.
    # **플래그가 아니라 캐시 meta 를 읽는다** — 따로 주면 어긋날 수 있다.
    if _cache_says_background(a.cache):
        from src.synthesis.sp_curves import background_signature, background_power
        _bg = background_signature()
        nz_sig = nz_sig + _bg
        print(f"  ** 상시 배경 (12.166): +{background_power():.2f}W, "
              f"|I1| +{np.hypot(_bg[0,0], _bg[0,1])*1000:.1f} mA -> noise_sig **")
    h_scale = harmonic_scales(pool, apps)
    # 상태별 지문 (13.11). `del pool` 앞에서 만들어야 한다.
    sig_state = None
    if a.state_signatures:
        from src.model.net import harmonic_signatures_by_state
        sig_state, _used = harmonic_signatures_by_state(pool, apps)
        print(f"  ** 상태별 지문 (13.11): {int(_used.sum())}개 상태를 따로 맞췄다 **")
        for _j, _a in enumerate(apps):
            for _s in range(_used.shape[1]):
                if not _used[_j, _s]:
                    continue
                _c = sig_state[_j, _s, :, 0] + 1j * sig_state[_j, _s, :, 1]
                _i1 = abs(_c[0])
                if _i1 > 1e-9:
                    print(f"       {_a:18s} s{_s}  와트당|I1| {1e3*_i1:.3f} mA/W"
                          f"   h2/h1 {abs(_c[1])/_i1:.4f}   h3/h1 {abs(_c[2])/_i1:.4f}")
    del pool

    model = NILMNet(apps, appliance_state_counts(apps), width=a.width,
                    wide_summary=a.wide_summary, wide_target=a.wide_target,
                    periodicity=a.periodicity,
                    fine_dropout=a.fine_dropout,
                    prior_kappa=a.prior_kappa, prior_beta=a.prior_beta,
                    fine_channels=a.fine_channels,
                    aux_z=(a.w_z > 0),
                    vexp=a.vexp, seg_pool=a.seg_pool).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    crit = NILMLoss(
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz_sig),
        harm_scale=torch.from_numpy(h_scale),
        harm_odd_only=a.harm_odd_only,
        off_detach_praw=a.off_detach_praw,
        signatures_state=(torch.from_numpy(sig_state) if a.state_signatures else None),
        harm_even_magnitude=a.harm_even_magnitude,
        harm_sig_vnorm=a.harm_sig_vnorm,
        # 14.28 — 기기별 `I/P` 전압 지수. 모터는 0 이라 보정이 안 걸린다.
        harm_vnorm_exp=(_vnorm_exp(apps, a.harm_vnorm_classes)
                        if a.harm_sig_vnorm else None),
        even_coherent=(torch.tensor(
            [1.0 if x in PHASE_COHERENT_EVEN else 0.0 for x in apps],
            dtype=torch.float32) if a.harm_even_by_class else None),
        gate_smooth=a.gate_smooth, gate_focal=a.gate_focal,
        harm_grad_balance=a.harm_grad_balance,
        smps_group=[apps.index(x) for x in
                    ("beam_projector", "laptop_charger", "minipc") if x in apps],
        weights=LossWeights(harm=a.w_harm, cons=a.w_cons, over=a.w_over,
                            state_power=a.w_state_power, z=a.w_z, swap=a.w_swap),
        s_state=(build_state_scales(apps, [S_I[x] for x in apps])
                 if a.per_state_scale else None),
        # ── 저항 조합 맞바꿈 (14.32). `--w-swap 0` 이면 버퍼만 생기고 안 걸린다 ──
        res_ohm=_res_ohm(apps, a.res_apps, half=False),
        res_ohm_half=_res_ohm(apps, a.res_apps, half=True),
        res_cond_state=_res_cond(apps, a.res_cond_state),
        swap_tol=a.swap_tol, swap_slack=a.swap_slack,
        swap_tiebreak=a.swap_tiebreak,
        swap_tb_orders=[int(x) for x in a.swap_tb_orders.split(",") if x.strip()],
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * max(1, a.epoch_windows // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    print(f"모델 {n_par/1e6:.2f}M 파라미터 | 배치 {a.batch} | {a.epochs} epoch x "
          f"{a.epoch_windows:,}창 = {steps:,} step | 장치 {dev}")
    if a.harm_sig_vnorm:
        _ex = _vnorm_exp(apps, a.harm_vnorm_classes)
        print("harm_vnorm 지수: " + " ".join("%s=%+.0f" % (x[:4], e)
                                             for x, e in zip(apps, _ex))
              + f"   (--harm-vnorm-classes {a.harm_vnorm_classes!r})")
    print(f"손실 가중치: power 1.0 / state .3 / on .3 / plugged .1 / standby .1 "
          f"/ harm {a.w_harm} / cons {a.w_cons} / over {a.w_over}")

    n_batches = max(1, a.epoch_windows // a.batch)
    cache = None
    if a.cache and a.cache.lower() != "none":
        cache = CachedWindows(a.cache)
        # 짝수차 규약은 캐시에 **구워져** 있다 (`build_fine` 이 0 으로 만든 것은 못 되돌린다).
        # 체크포인트에는 현재 코드 값이 적히므로, 둘이 다르면 체크포인트가 거짓을 주장하고
        # `run_gate_check` 의 검사가 그것을 통과시킨다. 여기서 막는다 (13.10).
        # 배치 대조 (13.12). 채널 수가 같아도 뜻이 다를 수 있으므로 **먼저** 본다.
        _cfl = str(cache.meta.get("fine_layout", "v1"))
        if _cfl != str(FINE_LAYOUT):
            raise SystemExit(
                f"캐시의 세밀 채널 배치가 현재 코드와 다릅니다: {a.cache}" + chr(10)
                + f"  캐시       fine_layout={_cfl!r}" + chr(10)
                + f"  현재 코드  FINE_LAYOUT={FINE_LAYOUT!r}" + chr(10)
                + "  앞부분의 뜻이 다르므로 슬라이스로 못 맞춥니다 — 캐시를 다시 구우십시오 (13.12).")
        _cze = cache.meta.get("zero_even_harmonics")
        if _cze is not None and bool(_cze) != bool(ZERO_EVEN_HARMONICS):
            raise SystemExit(
                f"캐시의 짝수차 구성이 현재 코드와 다릅니다: {a.cache}" + chr(10)
                + f"  캐시       zero_even_harmonics={bool(_cze)}" + chr(10)
                + f"  현재 코드  ZERO_EVEN_HARMONICS={bool(ZERO_EVEN_HARMONICS)}" + chr(10)
                + "  캐시를 다시 굽거나 src/model/inputs.py 를 캐시 값으로 되돌리십시오.")
        print(f"학습 데이터: 캐시 {a.cache} — 독립 창 {len(cache):,}개 "
              f"({cache.meta['bytes']/1e9:.1f}GB) | epoch 당 {a.epoch_windows:,}창 "
              f"-> 전체 {a.epochs * a.epoch_windows / len(cache):.1f}회 재사용")
        dl = None
    else:
        print(f"학습 데이터: 실시간 합성 (워커 {a.workers}) — 창 재사용 0")
        dl = DataLoader(SynthBatchDataset(n_batches, a.batch, a.seed,
                                          gen_spec=a.gen, recipe_mix=a.recipe_mix,
                                          smps_focus_off_p=a.smps_focus_off_p), batch_size=None,
                        num_workers=a.workers, persistent_workers=a.workers > 0,
                        prefetch_factor=2 if a.workers else None,
                        pin_memory=(dev == "cuda"))

    rng = np.random.default_rng(a.seed)

    # 13.47: 캐시 경로의 자료 공급을 계산과 겹친다. 계획을 **한 번에** 만들고
    # DataLoader 하나로 전 epoch 을 돈다 — epoch 마다 워커를 다시 띄우면 윈도우
    # spawn 이 1~2초라 300 epoch 에서 5~10분을 잃는다.
    cache_iter = None
    if cache is not None and a.cache_workers > 0:
        _plan = cache_index_plan(len(cache), a.batch, n_batches * a.epochs, rng,
                                 block_windows=a.block_windows)
        cache_iter = iter(DataLoader(
            CacheBatchDataset(a.cache, _plan), batch_size=None, shuffle=False,
            num_workers=a.cache_workers, persistent_workers=True, prefetch_factor=4,
            pin_memory=(dev == "cuda")))
        print(f"  ** 캐시 배치를 워커 {a.cache_workers}개로 미리 뽑는다 (13.47) — "
              f"색인 열은 옛 경로와 동일 **")

    def epoch_batches():
        """캐시면 블록 셔플로 뽑고, 아니면 DataLoader 를 돈다.

        전역 셔플을 쓰면 13GB 캐시 전체가 작업집합에 올라와 물리 메모리 여유가
        0 이 된다. 블록 셔플이면 동시에 손대는 구간이 block_windows 로 제한된다.

        `--cache-workers` 가 0 보다 크면 같은 색인 열을 별도 프로세스가 미리 뽑는다
        (13.47). 배치 내용은 바뀌지 않는다.
        """
        if cache is None:
            yield from dl
            return
        if cache_iter is not None:
            for _ in range(n_batches):
                yield next(cache_iter)
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
                    "wide_summary": a.wide_summary, "wide_target": a.wide_target,
                    "periodicity": a.periodicity,
                    "fine_dropout": a.fine_dropout,
                    # L_harm 기울기 균등화 (12.120). 13.84.17 에서 1단계에도 열었다.
                    "harm_grad_balance": a.harm_grad_balance,
                    # 세밀 채널 수를 반드시 남긴다. 12.34 에서 38 -> 44 로
                    # 늘었고, 이 키가 없는 체크포인트는 38 로 간주된다.
                    "fine_channels": model.fine_channels,
                    # 타깃 시점 구성 (12.45). 채널 수와 달리 슬라이스로 못 맞춘다 —
                    # 어긋나면 입력과 라벨이 다른 순간을 가리켜 조용히 틀린다.
                    "target_lookahead": TARGET_LOOKAHEAD,
                    "fine_cycles": FINE_CYCLES,
                    # 짝수차 배제 (12.77). 학습과 추론이 짝을 이뤄야 한다.
                    "zero_even_harmonics": ZERO_EVEN_HARMONICS,
                    # 세밀 채널 배치 (13.12). 채널 수와 달리 슬라이스로 못 맞춘다.
                    "fine_layout": FINE_LAYOUT,
                    # 광역 채널 수 (13.12). 세밀과 달리 슬라이스가 아예 없다 —
                    # `net.py` 가 conv 입력 채널로 직결한다.
                    "wide_channels": WIDE_CHANNELS,
                    "aux_z": bool(model.aux_z),
                    # ⚠ **추론에도 써야 한다** — 지수를 박고 배운 모델이다 (14.7).
                    "vexp": bool(model.vexp),
                    "seg_pool": int(model.seg_pool),
                    # 손실 설정이라 추론엔 안 쓴다. 계보 추적용이다 (13.80).
                    "gate_smooth": a.gate_smooth, "gate_focal": a.gate_focal,
                    "vswap_p": a.vswap_p,                 # 13.84.11 학습 시 전압 채널 바꿔 끼우기 (추론엔 무관)
                    # ⚠ 이것은 **추론에도 써야 한다** — 0 으로 배운 채널에 값을
                    # 주면 본 적 없는 입력이 된다. 채점 쪽이 읽어 같이 0 으로
                    # 만들 수 있게 남긴다 (13.80.10).
                    "zero_channels": a.zero_channels,
                    "zero_wide_channels": a.zero_wide_channels,
                    "select": a.select}, path)

    hist, best = [], None
    t_all = time.time()
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); agg, nb = {}, 0
        for batch in epoch_batches():
            fine, wide, tgt = to_targets(batch, dev)
            if ZERO_CH:
                fine[:, ZERO_CH] = 0.0        # 12.114 재시험의 조인 대조
            if ZERO_W:
                wide[:, ZERO_W] = 0.0
            vswap(fine, wide, a.vswap_p)          # 13.84.11 — 0 이면 아무것도 안 한다
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

        # 13.55: 보조 Z 항은 **표현이 잡히는지**를 바로 보여준다. 출발점은
        # 평균 예측 = Var(log r) 0.296 이다. 이 값이 안 내려가면 헤드가 논 것이다.
        _zs = f" z {agg['z']:.4f}" if a.w_z > 0 else ""
        # 14.32 — `L_swap` 이 **실제로 물고 있는지** 매 에포크 보인다.
        #   `swap_frac` 이 0 이면 항이 안 걸린 것이고, 그것이 이 계열의 기본 실패다
        #   (관문은 학습 전 한 번만 본다 — 중간에 0 으로 주저앉는 것은 못 잡는다).
        _sw = (f" swap {agg['swap']:.4f}/{agg['swap_frac']:.3f}"
               if a.w_swap > 0 else "")
        if ep % a.eval_every and ep != a.epochs:
            print(f"  ep{ep:>3d}  loss {agg['total']:.4f} (pw {agg['power']:.4f} "
                  f"harm {agg['harm']:.4f}{_zs}{_sw})  [{t_train:.0f}s, "
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
              f"harm {agg['harm']:.4f}{_zs}{_sw})  |  MAE {row['mae']:.3f}W  F1 {row['f1']:.4f}  "
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
