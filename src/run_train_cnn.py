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
import math
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
from src.model.inputs import fine_target_index as _fti
from src.model import net as _NET
from src.model.net import V_CH_FINE
FINE_TPOS = _fti()          #: 세밀 창 안의 타깃 위치 (239 = 600−1−360). 14.53 이 쓴다.
from src.model.inputs import (ZERO_EVEN_HARMONICS, EVEN_MAG0, EVEN2_CH,
                             FINE_CYCLES, FINE_LAYOUT,
                             FINE_VOLT0, HALFWAVE_CH, TARGET_LOOKAHEAD,
                             V_CENTER, V_SPAN,
                             VOLT_ORDERS, WIDE_CHANNELS, WIDE_MAG0,
                             WIDE_VOLT0, WIDE_PHI0, ODD_ORDERS, PHI_ORDERS, PHI0,
                             build_inputs, RAW_CHANNELS)
from src.synthesis.dataset import chunk_seed
from src.model.traincache import CachedWindows
from src.model.losses import (LossWeights, NILMLoss, PHASE_COHERENT_EVEN,
                             build_state_scales)
from src.model.net import (
    NILMNet, appliance_state_counts, harmonic_scales, harmonic_signature_vhrel,
    harmonic_signature_vref,
    harmonic_signatures,
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


#: 짝수차 크기를 나르는 채널. `run_gate_evenjit.py` 가 **원시를 흔들어** 이 표를 검증한다.
EVEN_FINE_MAG = list(range(EVEN_MAG0, EVEN_MAG0 + 7))   # |I2|,|I4|,...,|I14|
EVEN_FINE_R2 = 28                                        # |I2|/|I1|
EVEN_WIDE_H2 = (6, 9)                                    # |I2| · |I2|/|I1| (i_b 는 차수 키다)
EVEN_WIDE_MAG = [WIDE_MAG0 + 2 * s + 1 for s in range(7)]  # 13,15,...,25


def even_jitter(fine: torch.Tensor, wide: torch.Tensor, sigma: float) -> None:
    """짝수차 고조파 **크기의 모양**을 창마다 무작위로 흔든다 — 제자리 수정 (14.245).

    왜: 합성 캐시 안에서 짝수차가 **켜짐 조합의 결정함수**다. 300,000창 캐시에서 채널을
    9기기 전력·켜짐으로 회귀한 R² 가 `|I2|` **0.906** · 평활`|I2|` **0.929** ·
    `|I2|−|I4|` **0.934** 이고, 같은 조합 안 잔차 퍼짐이 전체의 **0.12~0.16** 뿐이다
    (14.243). 그러면 그 채널을 믿고 결정해도 합성 안에선 손실이 한 번도 안 오르니
    가중치를 깎을 `−∇L` 자체가 **생기지 않는다**.
    그런데 그 블록은 물리적으로 **5~7mA** 이고 `arcsinh(x·20)` 이 포화 아래라 기본파의
    **50.8배** 이득으로 들어온다. 판 147개 전부가 진폭 줄기보다 모양 방향에 **2.8배**
    민감하고(14.233), 짝수차 7채널을 **12.1° 돌리기만** 해도 건강한 판의 포트 게이트가
    0.979 -> 0.003 으로 죽는다(14.230). 숏컷이 부러지기 쉬운 자리에 박혀 있다.

    왜 이 폭인가: 분절 풀은 반파 드라이를 **녹화 하나**(`hair_dryer_1`, 149.7초)에서만
    뽑아 모양이 구간 사이 **0.06°** · 구간 안 0.15° 로 얼어 있는데, 실측 반파는 에피소드
    사이 **2.2~9.2°** · 안 0.7~4.9° 로 돈다 (14.234·14.237). 그 폭을 학습에 넣어
    "이 채널을 믿지 말라" 는 기울기를 **만들어 준다**.

    어떻게: 차수별 배수 `f_h = exp(σ·ε_h)` 를 창마다 뽑되 `log f` 의 평균을 빼서
    **기하평균을 1 로 고정**한다 — 크기(진폭 줄기)는 두고 **모양만** 돈다. 크기는
    라벨이 정해도 되는 진짜 정보라서 건드리면 안 된다 (14.220 의 DC 전량 제거가
    미탐을 17.5% -> 36.8% 로 올린 것과 같은 실수를 피한다).

    ⚠ **경로를 다 끊는다.** `|I2|` 는 세밀 16 말고도 28(`/|I1|`) · 44(평활) · 43(차) 과
    광역 6 · 9 · 13 에 실려 있다. 하나만 돌리면 망이 **안 돌린 쪽에서 원값을 도로 읽어**
    증강이 무효가 된다. 43 은 `|I2|−|I4|` 차라서 44 에서 `|I4|` 를 되살려 둘을 함께
    다시 짓는다. `asinh`/`sinh` 가 짝이라 `CURRENT_SCALE` 은 저절로 약분된다.
    ⚠ 세밀 39(역률)의 `i_rms` 에도 짝수차가 들어가지만 `|I1|` 2.55A 대 짝수차 6mA 라
    기여가 1e-5 미만이다 — 일부러 안 건드린다.

    `sigma <= 0` 이면 **난수도 안 뽑고** 곧장 돌아온다 (비트 동일).
    """
    if sigma <= 0:
        return
    e = torch.randn(fine.shape[0], 7, device=fine.device, dtype=torch.float32)
    f = torch.exp(sigma * (e - e.mean(1, keepdim=True)))[:, :, None]      # (B,7,1)

    fine[:, EVEN_FINE_MAG] = torch.asinh(torch.sinh(fine[:, EVEN_FINE_MAG]) * f)
    fine[:, EVEN_FINE_R2] = torch.asinh(torch.sinh(fine[:, EVEN_FINE_R2]) * f[:, 0])
    a = torch.sinh(fine[:, EVEN2_CH])                    # S·평활|I2|
    d = torch.sinh(fine[:, HALFWAVE_CH])                 # S·평활(|I2|−|I4|)
    a2 = a * f[:, 0]
    m4 = (a - d) * f[:, 1]                               # S·평활|I4| -> 돌린 것
    fine[:, EVEN2_CH] = torch.asinh(a2)
    fine[:, HALFWAVE_CH] = torch.asinh(a2 - m4)
    for c in EVEN_WIDE_H2:
        wide[:, c] = torch.asinh(torch.sinh(wide[:, c]) * f[:, 0])
    wide[:, EVEN_WIDE_MAG] = torch.asinh(torch.sinh(wide[:, EVEN_WIDE_MAG]) * f)


#: 홀수차 위상이 닿는 채널. `run_gate_phasejit.py` 가 **원시를 돌려** 이 표를 검증한다.
#: ⚠ 배치는 **블록**이다 — ch0~7 = Re(I1..I15), ch8~15 = Im(I1..I15).
#:   `run_diag_domaingap.fine_names` 가 교차로 적고 있었다 (14.267 에서 고침).
N_ODD = len(ODD_ORDERS)
PHASE_RE = [i for i, h in enumerate(ODD_ORDERS) if h >= 3]          # 1..7 (h=3..15)
PHASE_H = [h for h in ODD_ORDERS if h >= 3]


def odd_phase_jitter(fine: torch.Tensor, wide: torch.Tensor, sigma_deg: float) -> None:
    """홀수차 **h>=3 의 위상**을 창마다 무작위로 돌린다 — 제자리 수정 (14.268).

    왜: 분절 풀이 담는 세션 간 변이를 자유도별로 재 보니 (14.266, 여러 파일 있는 기기로),
    **차수별 위상이 파일 사이 18.73° rms** 로 여섯 자유도 중 절대 크기가 압도적이다
    (다음이 짝수차 모양 5.99°). 그런데 **포트와 반파/전파 드라이는 녹화가 각각 하나뿐**이라
    그 변이가 풀에 **구조적으로 0** 이다 — 실패하는 조합(포트+드라이)의 두 기기만 그렇다.
    `even_jitter` 가 통한 것과 **같은 구멍이 같은 기기에** 하나 더 뚫려 있다.

    ★ 라벨 안전: `h=1` 은 **안 건드린다.** P·Q 는 원시 `seg[:,30]·[:,31]` 에서 오고
    회전은 크기를 안 바꾸므로 `|I_h|`·역률·`I_rms`·모든 크기비(26·27·28·40)가 **불변**이다.
    움직이는 것은 `Re/Im(I3..I15)` 와 불변위상 `φ_h` 뿐이다.

    φ_h = arg(I_h) − h·arg(I_1) 이고 `I_1` 을 안 건드리므로 `φ_h -> φ_h + δ_h` 다.
    채널은 `w·cosφ`·`w·sinφ` 라 그 쌍을 δ_h 만큼 돌리면 된다 (게이트 `w` 는 크기라 불변).
    ⚠ 광역 φ 는 블록 **중앙값**이라 회전과 정확히 교환되지 않는다 — 관문이 그 오차를 잰다.

    `sigma_deg <= 0` 이면 **난수도 안 뽑고** 곧장 돌아온다 (비트 동일).
    """
    if sigma_deg <= 0:
        return
    B = fine.shape[0]
    d = torch.randn(B, len(PHASE_H), device=fine.device, dtype=torch.float32)         * (float(sigma_deg) * np.pi / 180.0)
    c, s = torch.cos(d), torch.sin(d)
    for j, i in enumerate(PHASE_RE):
        re = torch.sinh(fine[:, i])
        im = torch.sinh(fine[:, N_ODD + i])
        cj, sj = c[:, j, None], s[:, j, None]
        fine[:, i] = torch.asinh(re * cj - im * sj)
        fine[:, N_ODD + i] = torch.asinh(re * sj + im * cj)
    for j, h in enumerate(PHASE_H):
        if h not in PHI_ORDERS:
            continue
        k = PHI_ORDERS.index(h)
        cj, sj = c[:, j, None], s[:, j, None]
        for base, t in ((PHI0, fine), (WIDE_PHI0, wide)):
            a, b = t[:, base + 2 * k].clone(), t[:, base + 2 * k + 1].clone()
            t[:, base + 2 * k] = a * cj - b * sj
            t[:, base + 2 * k + 1] = a * sj + b * cj


class WeightAverager:
    """궤적 평균 (SWA, Izmailov 외 UAI 2018) — 14.295.

    **왜 이게 앙상블이 아닌가.** 앙상블은 따로 학습한 판 K 개의 *출력* 을 평균한다.
    그 판들은 서로 다른 분지에 있고, 신경망은 뉴런 순서 바꾸기에 대해 대칭이라
    **가중치를 평균하면 아무것도 아닌 것**이 나온다. 그래서 출력을 평균할 수밖에 없고,
    그러면 2대1 다수결이 되어 확신에 차서 틀린 다수가 살아남는다 (§19.1 이 죽인 것).
    여기서는 **한 궤적 안의 점들**을 평균한다. 그 점들은 손실이 낮은 경로로 이어져
    있으므로(같은 분지) 평균이 여전히 쓸 수 있는 점이고, 결과는 **판 하나**다.

    **왜 중심이 나은가.** 학습률이 0 이 아닌 동안 SGD 는 한 점이 아니라 분지 바닥
    주변의 정상분포에 든다. 마지막 가중치는 거기서 뽑은 표본 하나다. 골짜기는 한쪽
    벽이 가파르고 반대가 평평한데(He·Huang·Yuan, NeurIPS 2019), **학습** 손실의
    최소점은 가파른 쪽으로 치우쳐 있다. 평균은 평평한 쪽으로 더 들어간다. 분포
    이동은 곧 손실 곡면이 밀리는 것이므로, 평평한 안쪽 점이 덜 무너진다.

    ⚠ **부동소수 텐서만 평균한다.** `state_dict()` 에는 정수·불리언 버퍼가 섞여 있고
    (`_swap_combos` 같은 상수표), 그걸 누적 평균에 넣으면 dtype 이 깨지거나 값이
    바뀐다. 그런 항목은 **마지막 값을 그대로** 쓴다.

    ⚠ **BN 재추정이 필요 없다.** SWA 의 그 단계는 running mean/var 를 가진 정규화가
    있을 때만 필요한데, 이 망은 `_TGroupNorm`(= `nn.GroupNorm`) 뿐이라 통계를 순전파
    때마다 표본에서 낸다. 관문 [3] 이 실제 모델을 걸어 확인한다.
    """

    def __init__(self, model) -> None:
        self.n = 0
        self.avg = {k: (v.detach().clone().float() if v.is_floating_point()
                        else v.detach().clone())
                    for k, v in model.state_dict().items()}
        self.float_keys = [k for k, v in self.avg.items() if v.is_floating_point()]

    def add(self, model) -> None:
        """누적 평균 — `w̄ <- w̄ + (w − w̄)/(n+1)`. n 점의 산술평균과 정확히 같다."""
        sd = model.state_dict()
        if self.n == 0:
            for k, v in sd.items():
                self.avg[k].copy_(v.detach().float() if k in self.float_keys
                                  else v.detach())
        else:
            for k in self.float_keys:
                self.avg[k].add_(sd[k].detach().float().sub(self.avg[k]),
                                 alpha=1.0 / (self.n + 1))
            for k, v in sd.items():          # 정수·불리언은 마지막 값
                if k not in self.float_keys:
                    self.avg[k].copy_(v.detach())
        self.n += 1

    def state_dict(self, like) -> dict:
        """`like` 의 dtype 으로 되돌린 state_dict. 모델에 바로 실을 수 있다."""
        ref = like.state_dict()
        return {k: (v.to(ref[k].dtype) if k in ref else v)
                for k, v in self.avg.items()}

    def rel_shift(self, model) -> float:
        """`||w̄ − w|| / ||w||` (부동소수 파라미터만). 관문 [2] 의 자다 —
        이 값이 0 에 가까우면 **평균할 퍼짐이 없었다** = 무동작이다."""
        sd = model.state_dict()
        num = sum(float((self.avg[k] - sd[k].detach().float()).pow(2).sum())
                  for k in self.float_keys)
        den = sum(float(sd[k].detach().float().pow(2).sum()) for k in self.float_keys)
        return (num / max(den, 1e-30)) ** 0.5


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


def _vhrel_from_fine(fine, at_target: bool):
    """세밀 45~56 -> 그 창의 `V_h/V_1` (B, 15, 2) [Re, Im] (14.56).

    `build_inputs` 의 눈금을 **정확히 되돌린다**:
        h == 1 : `vr = x·V_SPAN + V_CENTER`,  `vi = x·V_SPAN`
        h  > 1 : `vr = sinh(x)/VOLT_HARM_SCALE`,  `vi = sinh(x)/VOLT_HARM_SCALE`
    관측 안 되는 차수(h13·h15·짝수)는 0 으로 둔다 — 손실의 가면이 거기를 막는다.

    ⚠ `at_target` 이면 타깃 사이클 하나, 아니면 창 평균이다 — `vrel` 과 **같은 자리**를
      봐야 한다 (한쪽만 타깃이면 그 자체가 새 불일치다, 14.53).
    """
    from src.model.inputs import VOLT_HARM_SCALE
    n = fine.shape[0]
    out = fine.new_zeros((n, 15, 2))
    nv = len(VOLT_ORDERS)
    for s_, h_ in enumerate(VOLT_ORDERS):
        xr = fine[:, FINE_VOLT0 + s_]
        xi = fine[:, FINE_VOLT0 + nv + s_]
        xr = xr[:, FINE_TPOS] if at_target else xr.mean(-1)
        xi = xi[:, FINE_TPOS] if at_target else xi.mean(-1)
        if h_ == 1:
            vr, vi = xr * V_SPAN + V_CENTER, xi * V_SPAN
        else:
            vr, vi = torch.sinh(xr) / VOLT_HARM_SCALE, torch.sinh(xi) / VOLT_HARM_SCALE
        out[:, h_ - 1, 0] = vr
        out[:, h_ - 1, 1] = vi
    v1 = torch.hypot(out[:, 0, 0], out[:, 0, 1]).clamp(min=1.0)[:, None]
    out[:, :, 0] = out[:, :, 0] / v1
    out[:, :, 1] = out[:, :, 1] / v1
    out[:, 0, 0] = 1.0                       # 정의상 rel[0] = 1+0j
    out[:, 0, 1] = 0.0
    return out


def to_targets(batch, dev, vrel_target: bool = False):
    """배치 -> `(fine, wide, tgt)`.

    `vrel_target` (14.53): 창 전압을 **타깃 사이클 하나**에서 읽는다. `False` 면 세밀 창
    600사이클(10초) 평균 — 옛 경로이고 **비트 동일**이다.

    ⚠⚠ **두 입구가 갈려 있었다.** 2단계(`run_adapt`)는 `RealWindows.v_observed` 를 쓰는데
      그것은 `power_features[targets, 4]` — **타깃 사이클** 값이다 (`realdata.py:199`).
      1단계만 10초 평균이었다 ([[pin-the-two-entry-points-against-each-other]]).
    ⚠ `L_harm` 이 맞추는 것은 **타깃 사이클 하나**의 `obs_harm` 이다. 그 사이클의 전압은
      그때 켜진 부하가 같이 만든 강하를 쓰고 있는데, 10초 평균은 통전 안 하는 구간의 높은
      전압을 섞는다 (핫플은 2.00초 주기로 0.47초만 통전한다, 14.48). 실측 오차
      (`run_diag_vrel.py`): 전체 창 **−0.01%** 인데 핫플통전·P>1500 **+0.28%** ·
      오븐통전·P>1000 **+0.30%** — **고전력 창에서만** 뜬다. `sig ∝ V^e`, 저항 `e=−1` 이라
      V 를 높게 읽으면 지문이 작아지고 `L_harm` 이 전력을 그만큼 **더** 요구한다
      (= 과대예측 쪽, 1500W 창에서 **+4.5W**).
    ⚠ `vrel` 과 `v_rms` **둘 다** 옮긴다 — 같은 순간의 같은 물리량이라 한쪽만 옮기면
      그 자체가 새 불일치다. `v_rms` 는 `L_swap`(14.32)과 `L_res` 의 `P = V²/R` 이 쓴다.
    """
    (fine, wide, yp, yo, ypl, ys, yst, oh, pn, pobs, zg, gh) = [
        b.to(dev, non_blocking=True) for b in batch]
    # 14.53 — 타깃 사이클 하나 대 10초 평균. 끄면 옛 식 그대로다.
    _v = (fine[:, V_CH_FINE, FINE_TPOS] if vrel_target
          else fine[:, V_CH_FINE].mean(-1)) * V_SPAN + V_CENTER
    return fine, wide, {
        # 14.56 — 그 창의 관측 **상대 전압 파형** `V_h/V_1` (B,H,2). `sig` 의 파형 몫을
        #   앵커하는 데 쓴다. 세밀 45~56 의 **눈금을 되돌려** 낸다 — 캐시를 다시 굽지
        #   않으려고 입력 채널에서 복원한다 (`inputs.build_inputs` 의 역).
        #   ⚠ 손실이 `use_vhrel` 이 아니면 **안 읽는다** (계산만 버린다).
        "vhrel": _vhrel_from_fine(fine, vrel_target),
        "y_power": yp, "y_on": yo, "y_plugged": ypl, "y_standby": ys, "y_state": yst,
        "obs_harm": oh, "p_noise": pn, "p_observed": pobs, "harm_offset": None,
        # 14.26 — 창 전압비 V/V_CENTER. `L_harm` 의 지문을 이것으로 나눈다 (`--harm-sig-vnorm`).
        #   세밀 채널 25 가 `(v − V_CENTER)/V_SPAN` 이다 (`net.V_CH_FINE`).
        "vrel": _v / V_CENTER,
        #: 14.348 — 조합 머리가 쓰는 창별 Ĝ_sum (mS). 캐시에 없으면 NaN 이다.
        "g_hat": gh,
        # 14.32 — `L_swap` 이 쓰는 창 전압 (V). **같은 채널에서 같은 식으로** 낸다.
        #   ⚠ 이것이 없으면 `_swap_term` 의 가드가 **조용히 0** 을 낸다.
        "v_rms": _v,
        # 13.55 — 이 창의 선로 저항 [Ω]. 옛 캐시면 NaN 이고 손실이 알아서 건너뛴다.
        "log_z": torch.log(zg[:, 0].clamp(min=1e-3)),
        #: ★ 14.354 — 모델 **입력**용 원값 r_grid [Ω]. 정규화는 모델이 한다
        #  (`net.Z_LOG_MEAN/STD`) — 호출부가 눈금을 틀릴 자리를 없앤다.
        "r_grid": zg[:, 0],
    }


def _z_drop(r, p: float):
    """★ 14.354 — 창마다 확률 `p` 로 Z 를 **"모름"(NaN)** 으로 가린다.

    왜 — 잰 자리에서는 Z 를 쓰고, **못 잰 자리에서도 돌아야** 한다. 이 가림이
    요구를 업그레이드로 바꾼다. ⚠ 이 손잡이는 **오직 그 일만** 한다 —
    `log_z` 보조감독은 `z_pre` 에서 읽으므로 세금이 안 줄어든다 (14.354).
    """
    if p <= 0:
        return r
    m = torch.rand(r.shape[0], device=r.device) < p
    return torch.where(m, torch.full_like(r, float("nan")), r)


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
    #: ★ 14.348 — 조합 머리가 홀드아웃에서도 `Ĝ` 를 받아야 한다. 홀드아웃에는
    #  `obs_harm.npy` 가 없지만 `X` 가 **원시 49채널**이라 타깃 색인에서 바로 푼다.
    #  ⚠ 없으면 평가만 반쪽으로 돌아 **학습과 평가가 다른 모델**이 된다.
    from src.model import gbudget as _GB
    from src.model.inputs import VOLT_ORDERS as _VO, target_index as _ti
    _t = _ti(hs.X.shape[-1])
    _x = np.asarray(hs.X[:, :, _t], np.float64)               # (N,49)
    _nv = len(_VO)
    _v15 = np.zeros(15, complex)
    for _s, _h in enumerate(_VO):
        _v15[_h - 1] = np.median(_x[:, 33 + _s]) + 1j * np.median(_x[:, 33 + _nv + _s])
    from src.model.losses import S_STATE as _SS
    _apps = list(getattr(hs, "appliances", None) or sorted(_SS))
    _bud = _GB.Budget(_apps, _v15, volt_re0=33, volt_orders=_VO)
    _g = _bud.g_sum(_x.T[None])[0].astype(np.float32)          # (N,)
    #: ★ 14.354 — 홀드아웃의 선로 저항. 14.353 이 빌더에 `z_grid` 를 넣었다.
    #  ⚠ 없는 옛 홀드아웃이면 **전부 NaN = "모름"** 으로 준다 — 그 경우 평가는
    #    대비책 갈래만 재는 것이고, 조용히 딴 값을 지어내지 않는다.
    _zg = getattr(hs, "z_grid", None)
    _r = (np.full(len(hs), np.nan, np.float32) if _zg is None
          else np.asarray(_zg[:, 0], np.float32))
    return np.concatenate(F), np.concatenate(W), _g, _r


@torch.no_grad()
def evaluate(model, prep, dev, batch: int = 512) -> tuple:
    fine_all, wide_all, g_all, r_all = prep
    model.eval()
    P, ON = [], []
    for i in range(0, len(fine_all), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            o = model(torch.from_numpy(fine_all[i:i + batch]).to(dev),
                      torch.from_numpy(wide_all[i:i + batch]).to(dev),
                      torch.from_numpy(g_all[i:i + batch]).to(dev)
                      if model.comb_tau > 0 else None,
                      torch.from_numpy(r_all[i:i + batch]).to(dev)
                      if getattr(model, "z_input", False) else None)
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
    # ── 14.183 **미래가 새는 구멍 넷** (14.182). 넷 다 기본이 옛 동작 = 비트 동일 ──
    #   관문 `run_gate_leaks` 가 막힌 것과 **일부러 남긴 것**을 둘 다 못 박는다.
    ap.add_argument("--state-power-src", default="table", choices=("table", "label"),
                    help="머리 바이어스 **초기값** 슬롯표 (14.186). `table` 이면 "
                         "`S_STATE` 그대로라 **비트 동일**. `label` 은 큰 슬롯(>=300W)만 "
                         "세그먼트 풀 **라벨 중앙값**으로 — 오븐 1357.1 -> **1100.6** · "
                         "핫플 549.6 -> 454.8 · 포트 1534.5 -> 1456.8. 표는 `L_power` 의 "
                         "**척도**(p90)였는데 13.84.68 이 초기값으로 재사용했고, 듀티 기기는 "
                         "초기값이 Huber δ **밖**이라 300에포크로 못 도착한다 (오븐 23%%만 이동). "
                         "⚠ 척도와 `p_state_cap` 은 **안 바꾼다** — 한 번에 하나만.")
    ap.add_argument("--fine-norm", default="window", choices=("window", "causal"),
                    help="ⓑ GroupNorm 통계를 **0..타깃**에서만 (14.183). 지금은 "
                         "(C_g, **T 전체**) 라 수용영역과 무관하게 미래가 기준선에 "
                         "들어간다. 경계에서 범인은 σ 가 아니라 **μ** 다 — μ만 "
                         "갈아끼우면 오븐 729 -> 45W, σ만은 597W. **window 면 비트 동일.**")
    ap.add_argument("--fine-conv", default="sym", choices=("sym", "causal"),
                    help="ⓓ 세밀 conv 를 **왼쪽 패딩**으로 (14.183). 블록5(반RF 189 "
                         "= 3.15초)부터 타깃이 계단(+148자리)을 문다. **sym 이 비트 동일.** "
                         "⚠ 파라미터 이름이 `conv.weight` 라 대칭판과 안 섞인다.")
    ap.add_argument("--fine-tpool", default="whole", choices=("whole", "split"),
                    help="ⓒ `h.mean/amax` 를 **과거/미래 따로** (14.183). `amax` 에는 "
                         "위치가 없어서 오탐 창의 argmax 가 128칸 중 **104칸이 미래**였다. "
                         "버리는 게 아니라 **시간 부호를 붙인다**. 머리 765 -> 1021. "
                         "**whole 이 비트 동일.**")
    ap.add_argument("--fine-derive", default="window", choices=("window", "both"),
                    help="ⓐ 파생 미래채널(41·42·29·30)의 **인과 짝 4개**를 ch23 에서 "
                         "되살려 몸통 입력에 더한다 (14.183). 41·42 는 수용영역이 1인데 "
                         "p(t+3.0s)·p(t+5.5s) 를 담는다 — dzn_s1 에서 이 넷이 **93%%**였다. "
                         "**캐시를 다시 안 구워도 된다.** window 면 비트 동일.")
    ap.add_argument("--wide-dg", action="store_true",
                    help="③ 광역 60초에서 **전이 컨덕턴스 계단** ΔĜ 를 뽑아 몸통 입력에 "
                         "채널 넷으로 더한다 (14.315). `G=P/V^2` 라 전압 계단이 이미 "
                         "나눠져 있고, 창 안 전이의 앞뒤 차이가 **기기 하나의 G** 다. "
                         "오븐 가열 창의 **89.1%%** 가 자기 듀티 전이를 60초 안에 담는다 "
                         "(10초 창은 47.9%%). **캐시를 다시 안 구워도 된다.** 끄면 비트 동일.")
    #: 14.224 — 시간축 DC 를 conv 에서 떼어 머리로 보낸다 (`net.fine_dc` 독스트링).
    ap.add_argument("--fine-dc", default="keep", choices=("keep", "split", "mag"),
                    help="split 이면 `_conv_in` 출력에서 채널별 시간평균을 빼서 conv 에 "
                         "넣고 그 평균을 **머리 특징**에 붙인다. 첫 conv 의 DC/AC 이득비가 "
                         "15판 중앙 **776배** 였다 (14.220). keep 이면 **비트 동일**. "
                         "mag 는 DC 를 몸통에 안 넣고 **크기 슬롯(p_states)에만** 되돌린다 "
                         "(14.251) — split 이 실측 게이트를 6/6 으로 고쳤지만 홀드아웃 MAE 를 "
                         "4.65 -> 7.75 로 올린 까닭이 DC 가 신원(on_logit)과 크기 양쪽에 "
                         "닿았기 때문이다. 신원 쪽 전이는 AUC 0.943 -> 0.521 로 깨지고 "
                         "크기 쪽은 진짜 정보다. RevIN 의 denorm 을 우리 머리에 맞춘 꼴이다.")
    ap.add_argument("--fine-pad", default="zeros", choices=("zeros", "replicate"),
                    help="세밀 몸통 conv 의 패딩 (14.131). **zeros 가 기본이라 비트 동일.** "
                         "zeros 는 창 밖을 0 으로 채우는데 `asinh` 눈금에서 0 은 "
                         "'고조파가 0' 이라 실측에 없는 값이다. block 6(d=64)은 600칸에서 "
                         "탭의 18.3%%, 240칸 조각에서 **45.7%%** 가 패딩이다. "
                         "★ 그리고 14.116 의 **진단 개입**이 한 것이 정확히 replicate 다 "
                         "(미래를 타깃값으로 덮었다) — 학습 처치(0 패딩)와 **같은 처치가 "
                         "아니었고**, 타깃 특징이 block 0(RF 7)에서 이미 cos **0.825** 로 "
                         "갈린다. 개입은 고쳤고(포트 0.003 -> 0.938) 처치는 악화시켰다.")
    ap.add_argument("--head-layout", default="v1", choices=("v1", "v2"),
                    help="머리 배치 (14.130). **v1 이 기본이고 비트 동일.** "
                         "v2 = *모든 요약을 타깃 기준 2.0초 격자로 낸다* — 세밀 600 = "
                         "**5x120** 이고 타깃 239 가 두 번째 토막의 끝이라 경계가 맞는다 "
                         "(K=4·6 은 안 맞는다). ★ 핵심은 풀링이 아니라 **정규화**다 — "
                         "`Conv1d -> GroupNorm(C,T 전체)` 라 수용영역과 무관하게 통계가 "
                         "창 전체에서 나와서, 풀링만 토막내면 대각/비대각이 **1.73배**밖에 "
                         "안 된다. v2 는 얕은 스택(RF 19)을 **토막마다 따로 태워** "
                         "비대각을 **0.0000** 으로 만든다. 광역은 14.94 가 닫은 축이라 "
                         "전역 평균 그대로 둔다. 머리 입력 765 -> 1475.")
    ap.add_argument("--head-drop", default="",
                    help="머리 입력에서 **빼는 덩이** (14.128). 쉼표로 여러 개. "
                         "**빈 값이면 비트 동일.** 이름: rawtgt · tap0/tap1/tap4 · "
                         "pastmean/pastmax · wide · rawstat. "
                         "왜 — 머리 입력 765칸의 유효 차원이 참여비 **20.2** 뿐이고 "
                         "(분산 90%%까지 79칸), 어느 덩이든 **나머지 전부로** R^2 "
                         "0.905~0.982 로 복원된다. 다만 3시드 판정 일치가 0.94~0.95 라 "
                         "**불안정의 원인이라는 증거는 없다** — 그래서 통과 조건은 "
                         "'좋아진다' 가 아니라 **'안 나빠진다'** 다.")
    ap.add_argument("--fine-future-segs", type=int, default=1,
                    help="미래 조각을 **몇 토막으로 나눠** 요약할지 (14.122). "
                         "`--fine-time-split` 전용. **1 이면 비트 동일.** "
                         "14.116 이 머리에 준 미래는 `hf.mean`/`hf.amax` 둘뿐인데 둘 다 "
                         "**순서 불변**이라 '앞으로 6초 안에 큰 게 있다' 는 말해도 "
                         "'**언제**' 는 못 말한다 — +0.2초 계단과 +5.9초 계단이 같은 값이다. "
                         "잰 것: `hf.amax` 가 머리 단일 덩이 기여 **1위 19.2%%** 이고, 미래 "
                         "덩이를 죽이면 오븐 헛게이트가 10.8%% -> **23.0%%** 로 두 배가 된다 "
                         "(= 미래는 순이득이다. 없애지 말고 시간 해상도를 줘라). "
                         "K=3 이면 2.0초 · K=6 이면 1.0초 해상도.")
    #: 14.160 — 짝수차 **크기**에 k탭 이동중앙값. 0/1 이면 **비트 동일**이다.
    #:   릴레이가 반주기에서 끊기면 그 **한 사이클이 물리적으로 반파**라 |I2| 가 100배
    #:   튀고(h2/h1 0.0018 -> 0.27, 최대 0.96), 사전에서 반파는 드라이기 s1(0.4326)
    #:   하나뿐이라 **드라이기가 없는 파일에 208~402W 유령**이 선다 (14.159).
    #:   추론에만 k=5 를 걸어 재 본 짝비교 (`cnn_pcap` 3시드, 학습은 안 한 상태):
    #:       드라이 유령 스파이크 **7개 -> 0개** · 절대 잔차 21.3 -> **18.5W** (3/3)
    #:       판정줄 .8820 -> .8774 · SMPS .9807 -> .9693   <- 분포 밖이라 생긴 손해로 본다
    #:   ⚠ ch43/44 는 이미 61주기 **평균**인데 그쪽을 되돌려도 유령이 안 죽는다
    #:     (402 -> 398.6W). 모델이 읽는 것은 평활 없는 **ch16~22** 다 (402 -> 67.5W).
    #: 14.162 — 보존 손실의 **죽은구역**(W). 음수면 옛 절대 와트 식이라 **비트 동일**.
    #:   `--w-cons` 가 0 이면 항 자체가 안 걸린다. δ 는 14.161 에서 재서 정했다.
    ap.add_argument("--cons-deadzone", type=float, default=-1.0, metavar="W",
                    help="보존 손실 죽은구역 (14.162). 음수면 옛 식과 비트 동일")
    ap.add_argument("--even-median", type=int, default=0, metavar="K",
                    help="짝수차 크기에 K탭 이동중앙값 (14.160). 0/1 이면 비트 동일")
    ap.add_argument("--p-state-cap", type=float, default=0.0,
                    help="상태 전력 슬롯의 **상한 배수** R — `p_states <= R x S_STATE` "
                         "(14.121). **0 이면 상한 없음 = 비트 동일.** 13.84.68 은 슬롯이 "
                         "아래로 죽는 것을 막았는데 위로 가는 쪽이 안 막혀 있었다. "
                         "9개 체크포인트에서 잰 max(p_states)/초기값: 빔 s1 **83.6배**"
                         "(자리채움 슬롯 10W -> 550W) · 충전기 s1 4.31배 · 미니PC s1 "
                         "2.17배 · 나머지 17개는 1.52배 이하. test_4 310~371초에서 "
                         "빔 p_raw 가 **99.2W** 로 튄 자리다. R=3 이면 병적인 둘만 걸린다.")
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
    ap.add_argument("--fine-time-split", action="store_true",
                    help="세밀 몸통을 **타깃에서 둘로** 쪼갠다 (14.116). 까닭: "
                         "`--fine-extra-dilations 32,64` 를 켜면 깊은 탭의 수용영역이 "
                         "763사이클(타깃 좌우 +-6.36초)이라 창 전체를 덮어, "
                         "`h[:,:,t]` 안에 **미래 6.02초가 섞여** 들어간다. 모델은 그것을 "
                         "미래라고 부를 길이 없어 무시할 수도 없다. 개입으로 확인: 미래를 "
                         "타깃값으로 덮으면 핫플 헛게이트가 0.878 -> 0.004 로 사라진다. "
                         "이 깃발은 정보를 **버리지 않고 시간 부호만** 준다 — 과거 조각과 "
                         "미래 조각을 따로 통과시켜 머리에 따로 준다. 가중치는 공유라 "
                         "**파라미터가 안 는다**. 끄면 **비트 동일**. "
                         "⚠ `--seg-pool` 과 다르다 — 저쪽은 전역 풀링을 쪼개는데 지금 "
                         "새는 길은 **탭 안**이라 풀링으로는 못 막는다.")
    ap.add_argument("--w-gate-cond", type=float, default=0.0, metavar="W",
                    help="게이트를 **통전**에 묶는 항 `L_gcond` (14.106). "
                         "`--res-cond-state` 가 지정한 기기에만, `on_logit` 을 "
                         "`1{y_state == 통전상태}` 에 맞추는 BCE 를 더한다. "
                         "까닭: 학습 자료에서 **오븐만** `on=1` 인데 전력 <50W 인 창이 "
                         "35.3%% 다 (포트·드라이·핫플 0.0%%, 그 창 p10 전력 14W). "
                         "`power = σ(on_logit)·p_raw` 라 게이트가 서면 전력 통로가 "
                         "열리고, 실측에서 오븐 헛게이트가 10.1%% 로 저항 4종 중 "
                         "압도적이다 (14.105). 팬·조명 창의 최적 게이트는 "
                         "`w_on/(w_on+W)` — `w_on=0.3` 이므로 0.3->0.50 · 0.9->0.25 · "
                         "2.7->0.10. **0 이면 항이 아예 안 생긴다** (비트 동일). "
                         "⚠ 실측 라벨은 팬·조명도 오븐 ON 이라 오븐 재현율이 떨어진다 "
                         "— 판정 줄에서 짝으로 본다.")
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
    ap.add_argument("--harm-vhrel-anchor", action="store_true",
                    help="**`sig` 의 파형 몫도 앵커한다** (14.56). 저항은 "
                         "`sig_h = v_h_rel,h / V_1` 인데 14.49 앵커는 `1/V_1` 쪽만 고쳤고 "
                         "`v_h_rel = V_h/V_1` 은 **녹화 세션 값 그대로 박제**돼 있었다. "
                         "실측 대 녹화 비 (14.55): 오븐 h7 0.81 · h11 0.61 / "
                         "포트·드라이 h3 **0.22~0.24** (E 녹화인데 고전력 창은 D). "
                         "13.69 가 생성기에서 없앤 가짜 판별자인데 손실에는 그 고침이 없었다. "
                         "생성기와 **같은 덧셈 꼴**로 건다: `sig_h += sig_1*(rel_창 − rel_녹화)`. "
                         "⚠ `--harm-vhrel-frac` 이 0 이면 **아무것도 안 한다**.")
    ap.add_argument("--harm-vhrel-src", default="conducting",
                    choices=("conducting", "file"),
                    help="`rel_녹화` 를 어디서 내나 (14.61). **conducting**(기본) 은 `sig` 와 "
                         "**같은 통전 사이클**의 `V_h/V_1` 이다. `file` 은 14.56 이 쓴 "
                         "`vtexture.file_rel`(파일 전체 중앙값)인데 **14.60 에서 기각됐다** — "
                         "둘이 1~15%% 다르고(오븐 h3 0.734 대 0.649), 실측 창 h3 0.694 에 대해 "
                         "보정 부호가 **−0.040 대 +0.045 로 뒤집힌다**.")
    ap.add_argument("--harm-vhrel-frac", type=float, default=0.0, metavar="F",
                    help="위 파형 앵커를 **몇 할만** 건다 (14.56). 0 = 끔(**비트 동일**). "
                         "⚠⚠ 기본이 0 인 까닭: 14.49 가 전량을 걸었다가 부호를 넘겼다 "
                         "(14.51, 오븐 −2.9%% -> +2.3%%). **처음부터 분수로 짠다.** "
                         "0.25 / 0.5 를 먼저 재고 직선인지부터 봐라.")
    ap.add_argument("--vrel-target", action="store_true",
                    help="창 전압(`vrel`·`v_rms`)을 **타깃 사이클 하나**에서 읽는다 (14.53). "
                         "끄면 세밀 창 600사이클(10초) 평균 = 옛 경로, **비트 동일**. "
                         "⚠⚠ **두 입구가 갈려 있었다**: 2단계 `run_adapt` 는 "
                         "`RealWindows.v_observed` = `power_features[targets, 4]` 로 "
                         "**이미 타깃 사이클**을 쓴다 (`realdata.py:199`). 1단계만 평균이었다. "
                         "⚠ `L_harm` 이 맞추는 것은 타깃 사이클 하나의 `obs_harm` 인데, 10초 "
                         "평균은 통전 안 하는 구간의 높은 V 를 섞는다 (핫플은 2.00초 주기로 "
                         "0.47초만 통전). 실측 오차: 전체 창 **−0.01%%** 인데 "
                         "핫플통전·P>1500 **+0.28%%** · 오븐통전·P>1000 **+0.30%%** — "
                         "**고전력 창에서만** 뜬다. V 를 높게 읽으면 `sig ∝ 1/V` 라 지문이 "
                         "작아지고 `L_harm` 이 전력을 더 요구한다 (과대예측 쪽, 1500W 창에서 "
                         "+4.5W = 14.51 이 남긴 과대 −16.3W 의 28%%).")
    ap.add_argument("--harm-vnorm-frac", type=float, default=1.0, metavar="F",
                    help="`--harm-vnorm-anchor` 의 보정을 **몇 할만** 건다 (14.51). "
                         "1.0 이 온전한 보정(기본), 0 이면 안 건 것과 같다. "
                         "⚠ **왜 1 보다 작게 거나**: 14.51 이 3시드로 쟀다 — f=1 은 실측 "
                         "고전력 과소를 `핫플통전·P>1500` 잔차 중앙 **+47.5 ± 4.1W -> "
                         "−16.3 ± 7.5W** 로 고치는데 **부호를 넘긴다**. 선형이라 보면 영점이 "
                         "실측 고전력 f=**0.745** · 합성 오븐 0.77 · 핫플 0.60 · 드라이 0.71 · "
                         "포트 0.32 다. ⚠ 선형은 **가정**이다 — 재라.")
    ap.add_argument("--harm-vnorm-anchor", action="store_true",
                    help="**`harm_sig_vnorm` 의 기준전압을 기기별 sig 적합값으로 옮긴다** (14.49). "
                         "지금은 `sig × (V/V_CENTER=222)^e` 인데 `sig` 는 각 기기의 격리 녹화 "
                         "전압에서 잰 값이다 (오븐 210.4V · 핫플 214.4V · 드라이 227.3V · 포트 227.7V). "
                         "어긋남 `(222/V_적합)^e` 가 기기별 상수 편향이 된다 — 오븐은 5.5%% 다. "
                         "합성 홀드아웃의 `p_states/참` 이 오븐 0.970 · 핫플 0.985 · 드라이 1.030 · "
                         "포트 1.014 로 **부호 4/4 · 순서 4/4** 맞는다. 끄면 **비트 동일**.")
    ap.add_argument("--wide-extra-dilations", default="", metavar="LIST",
                    help="광역 몸통 **뒤에 블록을 더한다** (14.91). 예 `8,16`. "
                         "기본 (1,2,4) 는 전폭 29블록 = 타깃에서 **+-7초** 뿐인데 창은 "
                         "+-30초다 (24%%). 머리는 그것을 60초에 걸쳐 평균 내므로 "
                         "'지난 20초가 평평했다' 를 만들 길이 없다. `8,16` 이면 전폭 "
                         "125블록 > 창 120 이라 창 전체를 덮는다. 세밀의 "
                         "`--fine-extra-dilations` 와 **같은 처방**이고, 비우면 비트 동일. "
                         "⚠ `--wide-target` 과 같이 써야 그 넓은 유닛을 타깃 자리에서 뽑는다.")
    ap.add_argument("--wide-seg-pool", type=int, default=0, metavar="N",
                    help="**광역 갈래에만** 구간 풀링 (14.88). 0 이면 `--seg-pool` 을 따라가 "
                         "**비트 동일**이다. 14.87 이 개입으로 잰 것 — 계단 위치를 담는 몫이 "
                         "`mean` 22%% · seg4 67%% · **seg8 86%%** 로 N 에 단조 증가한다. "
                         "그 정보가 필요한 곳은 광역뿐이라(세밀은 과거 3.98초 안에 계단이 "
                         "있으면 이미 맞힌다) 여기만 키운다. 파라미터는 `w2 x N` 만 는다.")
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
    #: ★ 14.347 — **조합 머리**. 저항 4종을 24개 조합 위의 softmax 로 낸다.
    #  `0` 이면 끔 = 비트 동일. 값은 물리 잔차의 눈금 (mS) — 14.343 의 λ 꼭지가 0.6~1.0 이다.
    #  ⚠ 캐시에 `g_hat.npy` 가 있어야 한다 (`run_build_ghat`).
    #: ★ 14.354 — 선로 임피던스 주입 + 가림
    ap.add_argument("--z-input", action="store_true",
                    help="r_grid 를 몸통 표현에 주입한다 (보조머리는 주입 전을 읽는다)")
    ap.add_argument("--z-drop", type=float, default=0.3,
                    help="창마다 이 확률로 Z 를 '모름'(NaN)으로 가린다")
    ap.add_argument("--comb-tau", type=float, default=0.0)
    #: ★ 14.352 — 물리 벌점을 **초과 주장 쪽만** 꺾는다. 0 이면 끔(비트 동일).
    ap.add_argument("--comb-over", type=float, default=0.0,
                    help="초과 주장 벌점의 눈금 [mS]. comb_tau 보다 훨씬 작게 (예 0.05)")
    ap.add_argument("--comb-over-margin", type=float, default=2.0,
                    help="초과로 치기 전 여유 [mS]. gbudget.VETO_MARGIN_MS 와 같은 자다")
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
    ap.add_argument("--on-detach-gate", action="store_true",
                    help="**켜진 창에서 게이트에 경사를 주지 않는다** (14.147). "
                         "`--off-detach-praw` 의 짝이다. `power = s(on)*p_raw` 라 전력 손실이 "
                         "게이트를 진폭 손잡이로 쓴다 — 참ON 창에서 게이트와 (참전력/슬롯 비)의 "
                         "상관이 에어컨 0.777 오븐 0.618 이다. 끊으면 게이트는 검출만, "
                         "p_raw 가 진폭을 맡는다. **값은 비트 동일, 경사만 바뀐다.**")
    ap.add_argument("--gate-free-power", action="store_true",
                    help="★ **게이트와 전력을 완전히 분해한다** (14.150). state 0 을 전력 혼합에 "
                         "넣고 그 전력을 0 으로 둬서 `p_raw` 가 OFF 를 표현할 수 있게 하고, "
                         "`power = p_raw` 로 곱을 뗀다. OFF 일은 상태 머리가 맡고 게이트는 "
                         "순수 검출기가 된다. `--on-power-praw`/`--on-detach-gate`/"
                         "`--off-detach-praw` 와 같이 못 쓴다 (저쪽은 반쪽 처치다).")
    ap.add_argument("--on-power-praw", action="store_true",
                    help="**마스크 독립 손실** (14.147B). 참ON 창에서 게이트를 식에서 빼고 "
                         "`L = huber(p_raw, y)` 로 건다. `--on-detach-gate` 는 경사만 끊어 "
                         "p_raw 가 y/gate 로 밀리는 **보상 왜곡**이 남는데 이쪽은 그것까지 막는다. "
                         "둘은 같이 못 쓴다.")
    ap.add_argument("--fine-dilations", default="",
                    help="세밀 conv 스택의 dilation 다섯 개 (쉼표). 비우면 1,2,4,8,16 = "
                         "**지금과 비트 동일**. 창은 '타깃 앞 3.98초 | 타깃 | 뒤 6.00초' 인데 "
                         "깊은 탭의 수용영역이 1+6*(1+2+4+8+16)=187사이클 = ±1.56초뿐이라 "
                         "창의 44%%(타깃 뒤 1.56~6.00초)가 **위치를 모르는** h.mean/h.amax "
                         "로만 머리에 닿는다. 14.47 의 반사실: 미래 6초를 다 지우면 "
                         "mix 0.137->0.939 인데 **수용영역 밖만** 지워도 0.941 이다 — "
                         "해로운 것은 '미래'가 아니라 '수용영역 밖'이다. "
                         "1,3,9,27,81 이면 RF 727사이클 = ±6.06초로 창을 다 덮고 "
                         "**블록 수가 같아 파라미터가 안 는다**(624,917 로 동일).")
    ap.add_argument("--fine-pool", default="both", choices=("both", "amax", "mean"),
                    help="세밀 갈래 전역 풀링 (14.79). both 가 기본 = **비트 동일**. "
                         "amax 면 **위치를 모르는 h.mean 을 뺀다** — 수용영역이 창을 덮으면 "
                         "conv 가 어떤 가중평균이든 만들 수 있어 중복이고, 남기면 합성에서 "
                         "잘 듣는 위치 불변 지름길만 준다 (14.47: 수용영역 밖만 지워도 "
                         "mix 0.941). h.amax 는 max 라 conv 가 표현 못 하고 듀티 기기에 "
                         "필요해서 남긴다. 원시 fp.amax/amin 과 물리 프라이어도 그대로.")
    ap.add_argument("--fine-extra-dilations", default="",
                    help="세밀 스택 **뒤에 덧붙일** dilation (쉼표, 예 32,64). 비우면 안 붙는다 "
                         "= **비트 동일**. 14.78 은 dilation 을 **교체**해 마지막 탭 RF 를 "
                         "187->727 로 늘렸는데 그 탭이 유일한 국소 탭이라 국소성을 잃었다 "
                         "(타깃 반응 2.393->1.494, +90 어깨 0.448->1.959 로 어깨가 더 높아짐). "
                         "저항 신원이 한 시드 0.9321, SMPS 0.9219, 판정 줄 0.8871 로 무너졌다. "
                         "⇒ **바꾸지 말고 더한다**: 앞 다섯을 그대로 두고 뒤에 붙인 뒤 "
                         "--tap-layers 에 4 를 넣어 옛 마지막 블록(RF 187)도 같이 뽑는다.")
    ap.add_argument("--tap-layers", default="",
                    help="타깃 슬라이스를 뽑을 블록 번호 (쉼표, 0-based). 비우면 0,1 = 비트 동일. "
                         "⚠ --fine-extra-dilations 를 쓸 때 **4 를 꼭 넣어라** — 안 넣으면 "
                         "14.78 과 똑같이 국소 탭이 사라진다.")
    ap.add_argument("--harm-even-by-class", action="store_true",
                    help="짝수차 위상을 기기 부류별로 살린다 (13.45). "
                         "--harm-even-magnitude 와 같이 써야 뜻이 있다 — 위상이 뭉치는 "
                         "기기(오븐·포트·핫플·드라이기)가 그 창의 짝수차 예측 크기에서 "
                         "차지하는 몫만큼 복소 오차를 되살린다")
    ap.add_argument("--harm-even-magnitude", action="store_true",
                    help="L_harm 에서 **짝수차만 크기 공간**으로 잰다 (13.11). 플러그를 "
                         "반대로 꽂으면 짝수차가 180° 도므로(홀수차는 안 돈다) 짝수차 위상은 "
                         "기기 속성이 아니다 — 드라이기 약풍에서 격리 대 복합이 179° 어긋났다.")
    ap.add_argument("--pow-sig", action="store_true",
                    help="`L_harm` 의 사전을 기기당 하나에서 **기기x전력대**로 바꾼다 "
                         "(13.84.38). `L_harm` 은 `Σ sig_k·power_k` 라 **전력에 선형**인데 "
                         "와트당 고차 함량이 동작점에 따라 47~109%% 변한다 (충전기 h11 이 "
                         "15~28W 에서 2.31, 63~65W 에서 1.10 mA/W). **지문을 갈아 끼우지 "
                         "않고 비 `g[k,b,h] = sig_대역/sig` 를 낸다** — 상태별 지문과 곱해서 "
                         "같이 쓸 수 있고, 표본이 얇은 칸은 g=1 이라 **끄면 비트 동일**이다. "
                         "⚠ 2단계(`run_train_seq --pow-sig`)에만 달려 있어 1단계에서는 "
                         "한 번도 안 켜졌다 (14.172). 겨냥은 **② SMPS 표류** 다 — "
                         "오늘 잰 보정 크기(대표전력·mA): 에어컨 1,714 · 충전기 517 · "
                         "미니PC 94 인데 저항은 모양 차수가 2~15 뿐이다.")
    ap.add_argument("--pow-bands", type=int, default=3, metavar="N",
                    help="전력대 개수. 경계는 그 기기 통전 전력의 분위수다")
    ap.add_argument("--pow-tau", type=float, default=0.15, metavar="T",
                    help="대역 경계의 **부드러움**. `u=σ((p−e)/(τ·e))` — 계단이 아니다")
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
    ap.add_argument("--even-jitter", type=float, default=0.0, metavar="SIGMA",
                    help="학습 배치에서 **짝수차 고조파 크기의 모양**을 창마다 무작위로 흔든다 (14.245). "
                         "차수별 배수 exp(sigma*eps) 를 기하평균 1 로 고정해 걸므로 크기는 두고 모양만 돈다. "
                         "까닭: 합성 캐시에서 짝수차가 켜짐 조합의 결정함수다 — R²(채널|9기기전력) 가 "
                         "|I2| 0.906 / 평활|I2| 0.929 / |I2|-|I4| 0.934 라 그 채널을 믿어도 손실이 안 올라 "
                         "가중치를 깎을 기울기가 안 생긴다. 그런데 물리적으로 5~7mA 인 그 블록이 "
                         "arcsinh(x*20) 포화 아래라 기본파의 50.8배 이득으로 들어오고, 12.1도만 돌려도 "
                         "포트 게이트가 0.979 -> 0.003 이다. 풀은 모양이 0.06도로 얼어 있는데 실측 반파는 "
                         "2.2~9.2도 돈다. 세밀 16~22,28,43,44 와 광역 6,9,13~25 를 **함께** 돌린다. 0 이면 옛 경로.")
    ap.add_argument("--odd-phase-jitter", type=float, default=0.0, metavar="DEG",
                    help="학습 배치에서 **홀수차 h>=3 의 위상**을 창마다 N(0, DEG) 로 돌린다 (14.268). "
                         "h=1 은 안 건드리므로 P·Q·역률·모든 크기비가 **불변**이다. 까닭: 풀의 "
                         "세션 간 변이를 자유도별로 재니 차수별 위상이 파일 사이 **18.73도 rms** 로 "
                         "가장 크고(다음이 짝수차 모양 5.99도), 포트·드라이는 녹화가 하나뿐이라 그 "
                         "변이가 풀에 0 이다. 0 이면 옛 경로.")
    ap.add_argument("--head-conductance", action="store_true",
                    help="전력 머리를 **전도도 영역**으로 바꾼다 (14.284). 순전파는 "
                         "`p_raw *= (V/222)^e_k` 로 vexp 와 같지만, **손실의 목표가** "
                         "`log(P/(V/222)^e)` 로 V 불변이 된다. 14.16 이 vexp 를 죽인 까닭은 "
                         "구조가 아니라 목표가 와트였던 것이다 — 와트가 V 를 따라가니 헤드가 "
                         "잔여 지수 0.66 을 또 배워 합이 2.68(물리 2.0)이 됐다. 목표를 V 불변으로 "
                         "두면 그 유인이 사라진다. e_k 는 net.V_EXP 표 고정 (저항 2 · SMPS 0 · "
                         "모터 0.6). 참값이 5W 위인 자리만 log 로 재고 꺼진 자리는 옛 척도 "
                         "Huber 를 그대로 써서 슬롯 사망(13.84.68)을 막는다. 끄면 **비트 동일**.")
    ap.add_argument("--hcond-scale", default="log", choices=("log", "watt"),
                    help="`--head-conductance` 의 **척도** (14.299). log 가 하던 일이 둘인데 "
                         "뗄 수 있다 — ① **V 불변**(목표를 vrel^e 로 나눈다. 14.16 이 요구한 "
                         "진짜 진단) ② **척도 불변**(log. '11,600배' 명분이자 피해의 원인). "
                         "`watt` 는 ①만 산다: `huber(P̂/(vp·s), y/(vp·s), power_delta)`. "
                         "저항 기기만 남기면 ②는 살 이유가 없다 — 11,600배는 1.4kW 대 11W "
                         "이야기인데 그 11W 짜리가 갈래에서 빠지고, 남는 넷은 529~1534W 라 "
                         "s_i 정규화가 이미 공평하다. 측정: 오븐↔포트 196W 를 가르는 와트당 "
                         "기울기가 바닥 7.37e-05 · **log 3.89e-05(0.53배)** · "
                         "**watt 7.93e-05(1.08배)** 다 (14.301 정정 — 손실의 분모는 "
                         "`s_i` 가 아니라 **상태별 척도** `s_state` 다. 오븐 통전은 "
                         "s_state[oven][2]=1357.1).")
    ap.add_argument("--hcond-on-w", type=float, default=5.0, metavar="W",
                    help="참값이 이 W 위인 자리만 전도도 목표로 잰다 (14.299 에서 손잡이로 뺐다. "
                         "옛 하드코딩 값이 5.0). ⚠⚠ **14.301 정정** — 처음엔 `S_STATE`(척도표)를 "
                         "학습 목표로 착각해 '오븐 s1 17W 가 4,626배' 라고 적었다. 실제 목표 "
                         "`target_power_w` 로 재면 오븐 팬·조명은 **0.0W** 라 문턱에 애초에 "
                         "안 걸리고, 5~100W 띠의 **75.6%%가 비저항**이다 (질량으로는 저항이 "
                         "1/44). 그래서 폭주를 뗄 주역은 `--hcond-classes` 이고 이 문턱은 "
                         "저항 쪽에 남는 **96창**(오븐 전이 35 · 핫플 61)을 떼는 마무리다.")
    ap.add_argument("--hcond-classes", default="", metavar="LIST",
                    help="전도도 목표를 **이 부하분류에만** 건다 (쉼표, 예: RESISTIVE). "
                         "빈 값이면 전부 = 옛 경로. ⚠ SMPS 는 `V_EXP=0` 이라 "
                         "`P/vrel^0 = P` 로 **그냥 log P** 다 — 전도도가 아니고 물리가 "
                         "하나도 없다. 모터 0.6 도 전도도가 아니다. `--harm-vnorm-classes` 와 "
                         "같은 규약이라 **모르는 이름은 죽는다**.")
    ap.add_argument("--swa-start", type=int, default=0, metavar="EPOCH",
                    help="**가중치 평균** (SWA, Izmailov 외 UAI 2018) 을 이 epoch 부터 켠다 "
                         "(14.295). 0 이면 **비트 동일**. 원리: LR 이 0 이 아닌 동안 SGD 는 "
                         "한 점이 아니라 분지 바닥 주변의 **정상분포**에 들고, 마지막 가중치는 "
                         "그 분포에서 뽑은 표본 하나다. 궤적을 평균하면 중심을 재는 것이고, "
                         "비대칭 골짜기(He 외 NeurIPS 2019)에서 중심은 **평평한 쪽**으로 더 "
                         "들어가 있어 곡면이 밀릴 때(= 분포 이동) 덜 무너진다. "
                         "⚠ **앙상블과 다른 물건이다** — 앙상블은 다른 분지의 판들의 *출력*을 "
                         "평균해 2대1 다수결이 되고 §19.1 에서 죽었다. 이건 **한 궤적 안의 "
                         "가중치**를 평균해 판 **하나**를 만든다. 씨앗이 다르면 초기값이 달라 "
                         "분지가 다르므로 **체크포인트끼리는 평균하면 안 된다**.")
    ap.add_argument("--swa-lr", type=float, default=0.0, metavar="LR",
                    help="평균 구간 동안 **고정할** 학습률. 0 이면 코사인이 `--swa-start` 에서 "
                         "내는 값을 그대로 쓴다. ⚠ 이 고정이 처치의 **일부다** — 지금 스케줄은 "
                         "`CosineAnnealingLR(T_max=steps)` 라 LR 이 0 으로 떨어져서, 그대로 "
                         "평균하면 마지막 구간에 퍼짐이 없어 w̄ ≈ w_T 로 **무동작**이 된다. "
                         "관문 [2] 가 그것을 잡는다.")
    ap.add_argument("--swa-every", type=int, default=1, metavar="N",
                    help="평균 구간에서 N epoch 마다 한 점씩 누적한다 (기본 1).")
    ap.add_argument("--fine-channels", type=int, default=None, metavar="N",
                    help="세밀 갈래가 쓸 채널 수 (기본: inputs.FINE_CHANNELS). "
                         "캐시는 그대로 두고 앞에서부터 N 개만 쓴다. "
                         "12.34 의 고조파 위상 6채널을 빼고 대조군을 학습할 때 "
                         "--fine-channels 38 로 준다. 캐시가 같으므로 채널 수 "
                         "말고는 아무것도 안 달라진다.")
    ap.add_argument("--tag", default="cnn")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.even_median > 1:
        from src.model import inputs as _I
        _I.EVEN_MEDIAN = int(a.even_median)
        print("** 14.160 짝수차 이동중앙값 k=%d — ch16~22·28·39·43·44 가 robust 해진다 **"
              % a.even_median)


    #: 14.150 — 반쪽 처치와 같이 못 쓴다. `gate_free_power` 가 그것들을 포함한다.
    if a.gate_free_power and (a.on_power_praw or a.on_detach_gate
                              or a.off_detach_praw):
        raise SystemExit("--gate-free-power 는 --on-power-praw/--on-detach-gate/"
                         "--off-detach-praw 와 같이 못 씁니다 (전자가 후자를 포함합니다)")

    #: 14.299 — 셋 다 `--head-conductance` 없이는 뜻이 없다. 조용히 무시되면
    #  "걸었는데 안 걸린 판"을 A/B 로 착각한다 ([[count-every-path-before-claiming-you-cut-one]]).
    if not a.head_conductance:
        for _n, _v, _d in (("--hcond-scale", a.hcond_scale, "log"),
                           ("--hcond-on-w", a.hcond_on_w, 5.0),
                           ("--hcond-classes", a.hcond_classes, "")):
            if _v != _d:
                raise SystemExit("%s 는 --head-conductance 와 같이 써야 합니다 "
                                 "(지금 그 플래그가 없어 조용히 무시됩니다)" % _n)

    #: 14.295 — SWA 는 `select=final` 이라야 뜻이 있다. `best-f1` 은 epoch 마다
    #  **돌고 있는** 판을 저장하므로 평균낸 가중치가 덮어써지거나 무시된다.
    if a.swa_start:
        if not (1 <= a.swa_start <= a.epochs):
            raise SystemExit("--swa-start 는 1..%d 여야 합니다 (받은 값 %d)"
                             % (a.epochs, a.swa_start))
        if a.select != "final":
            raise SystemExit("--swa-start 는 --select final 과만 씁니다 "
                             "(--select %s 는 epoch 마다 도는 판을 저장합니다)" % a.select)
        if a.swa_every < 1:
            raise SystemExit("--swa-every 는 1 이상이어야 합니다")

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
    # 14.49 — `harm_sig_vnorm` 의 기준전압. `--harm-vnorm-anchor` 가 아니면 안 넘긴다.
    _vref, _vref_st = harmonic_signature_vref(pool, apps)
    # 14.56 — 그 녹화의 상대 전압 파형. `--harm-vhrel-anchor` 가 아니면 안 넘긴다.
    _vhrel = harmonic_signature_vhrel(pool, apps, source=a.harm_vhrel_src)
    #: 파형 앵커를 걸 기기 — `--harm-vnorm-classes` 와 **같은 무리**다 (순저항).
    #: `I_h = V_h/R` 이 성립하는 곳만. SMPS 는 비선형이라 이 법칙이 없다 (13.69 와 같다).
    _vhrel_on = (np.asarray(_vnorm_exp(apps, a.harm_vnorm_classes), dtype=np.float32) != 0
                 ).astype(np.float32)
    if a.harm_vhrel_anchor:
        print("  ** 14.56/14.61 sig 파형 앵커 %.2f할 · 기준 %s · 기기 %s **"
              % (a.harm_vhrel_frac, a.harm_vhrel_src,
                 " ".join(x[:4] for x, o in zip(apps, _vhrel_on) if o)))
    if a.harm_vnorm_anchor:
        print("  ** 14.49 sig 기준전압을 기기별 적합값으로 (%.2f할): " % a.harm_vnorm_frac
              + " ".join("%s=%.1fV" % (x[:4], v) for x, v in zip(apps, _vref)) + " **")
    # 상태별 지문 (13.11). `del pool` 앞에서 만들어야 한다.
    sig_state = None
    pow_gain = pow_edges = None
    #: ⚠⚠ 14.172 — **`--state-signatures` 와 같이 쓰면 안 된다.** 둘이 같은 축을 잰다.
    #  대역 경계는 **통전 전력의 분위수**이고 상태 전력은 **그 통전 전력 자체**다:
    #      드라이 대역 487·963W  대  상태 {1:529W, 2:1022W}
    #      충전   대역  48· 63W  대  상태 {1: 36W, 2:  68W}
    #  손실은 `sig_state x pow_gain` 으로 **곱하므로** 같은 보정이 두 번 걸린다.
    #  실측 (`run_gate_powsig`): 드라이 s2 의 h2/h1 이 0.00018 -> **0.00000 (x0.000)**.
    #  반파 억제가 소멸한다 — ③ 스위칭 과도 유령이 기대는 바로 그 신호다.
    #  충전기도 x1.339 / x0.684 로 어긋난다.
    #  ⇒ 고치려면 대역을 **상태 안에서** 잡아야 한다 (`sig_state` 를 분모로).
    #    상태로 가른 뒤 남는 전력 의존은 SMPS 무리에서 7~23% 다 (저항은 <1.6%).
    if a.pow_sig and a.state_signatures:
        _nl = chr(10)
        raise SystemExit(
            "\u2716 --pow-sig 와 --state-signatures 는 **같은 축**을 잰다 (14.172)." + _nl
            + "  대역 경계 = 통전 전력의 분위수 = 상태 전력. 곱하면 제곱이 된다." + _nl
            + "  재현: python -X utf8 -m src.run_gate_powsig   (드라이 s2 h2/h1 x0.000)" + _nl
            + "  둘 중 하나만 쓰거나, 대역을 **상태 안에서** 잡는 판을 지어라.")
    if a.pow_sig:
        #: 14.172 — 13.84.38 의 전력대 사전. 2단계에만 달려 있었다.
        from src.model.net import harmonic_signatures_by_power
        pow_gain, pow_edges, _pu = harmonic_signatures_by_power(
            pool, apps, n_bands=a.pow_bands)
        print('  ** 전력 의존 지문 (13.84.38): %d/%d 칸을 따로 맞췄다 (나머지는 보정비 1) **' % (int(_pu.sum()), _pu.size))
    if a.state_signatures:
        from src.model.net import harmonic_signatures_by_state
        sig_state, _used = harmonic_signatures_by_state(pool, apps)
        print(f"  ** 상태별 지문 (13.11): {int(_used.sum())}개 상태를 따로 맞췄다 **")
        _src = getattr(harmonic_signatures_by_state, "last_source", None)
        if _src is not None and (_src == 2).any():
            _who = ["%s s%d" % (apps[k], s) for k, s in zip(*np.nonzero(_src == 2))]
            print("     ** 그중 %d칸은 **실측 전력**을 분모로 맞췄다 (14.167): %s **"
                  % (len(_who), " · ".join(_who)))
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
                    z_input=bool(a.z_input),
                    comb_tau=a.comb_tau, comb_over=a.comb_over,
                    comb_over_margin=a.comb_over_margin,
                    wide_summary=a.wide_summary, wide_target=a.wide_target,
                    periodicity=a.periodicity,
                    fine_dropout=a.fine_dropout,
                    gate_free_power=a.gate_free_power,
                    prior_kappa=a.prior_kappa, prior_beta=a.prior_beta,
                    fine_channels=a.fine_channels,
                    fine_dilations=(tuple(int(x) for x in a.fine_dilations.split(","))
                                    if a.fine_dilations else None),
                    fine_pool=a.fine_pool,
                    fine_extra_dilations=(tuple(int(x) for x in a.fine_extra_dilations.split(","))
                                          if a.fine_extra_dilations else None),
                    p_state_cap=a.p_state_cap,
                    head_drop=a.head_drop,
                    head_layout=a.head_layout,
                    state_power_src=a.state_power_src,
                    fine_norm=a.fine_norm, fine_conv=a.fine_conv,
                    fine_tpool=a.fine_tpool, fine_derive=a.fine_derive,
                    wide_dg=bool(a.wide_dg),
                    fine_dc=a.fine_dc,
                    fine_pad=a.fine_pad,
                    fine_future_segs=a.fine_future_segs,
                    tap_layers=(tuple(int(x) for x in a.tap_layers.split(","))
                                if a.tap_layers else None),
                    fine_time_split=a.fine_time_split,
                    aux_z=(a.w_z > 0),
                    vexp=a.vexp, head_conductance=a.head_conductance,
                    seg_pool=a.seg_pool,
                    wide_seg_pool=a.wide_seg_pool,
                    wide_extra_dilations=[int(x) for x in a.wide_extra_dilations.split(",")
                                          if x.strip()]).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    from src.model.lossbuild import hcond_cols_of     # 14.315
    crit = NILMLoss(
        head_conductance=a.head_conductance,          # 14.284
        hcond_scale=a.hcond_scale,                    # 14.299
        hcond_on_w=a.hcond_on_w,                      # 14.299
        #: 14.315 — **분류 이름을 그대로 넘기면 안 된다.** `NILMLoss` 는 열 번호를 받는다.
        #  여기서 직접 변환하던 시절은 없었고 `build_loss` 안에만 있어서 1단계가 죽었다.
        hcond_cols=hcond_cols_of(a.head_conductance, a.hcond_classes, apps),
        s_i=torch.tensor([S_I[x] for x in apps], dtype=torch.float32),
        signatures=torch.from_numpy(sig),
        standby_sig=torch.from_numpy(sb_sig),
        noise_sig=torch.from_numpy(nz_sig),
        harm_scale=torch.from_numpy(h_scale),
        # 14.49 — `harm_sig_vnorm` 의 기준전압을 **기기별 sig 적합값**으로 옮긴다.
        #   None 이면 222V 고정 = 옛 경로와 **비트 동일**.
        harm_vnorm_frac=float(a.harm_vnorm_frac),
        # 14.56 — `sig` 의 파형 몫. 끄면 `None`/0 이라 **비트 동일**이다.
        harm_vhrel_rec=(torch.from_numpy(_vhrel) if a.harm_vhrel_anchor else None),
        harm_vhrel_frac=float(a.harm_vhrel_frac),
        harm_vhrel_on=(torch.from_numpy(_vhrel_on) if a.harm_vhrel_anchor else None),
        harm_vnorm_vref=(torch.from_numpy(_vref)
                         if (a.harm_vnorm_anchor and a.harm_sig_vnorm) else None),
        harm_vnorm_vref_state=(torch.from_numpy(_vref_st)
                               if (a.harm_vnorm_anchor and a.harm_sig_vnorm) else None),
        harm_odd_only=a.harm_odd_only,
        cons_deadzone=a.cons_deadzone,
        off_detach_praw=a.off_detach_praw,
        on_detach_gate=a.on_detach_gate,
        on_power_praw=a.on_power_praw,
        signatures_state=(torch.from_numpy(sig_state) if a.state_signatures else None),
        harm_even_magnitude=a.harm_even_magnitude,
        power_gain=(torch.from_numpy(pow_gain) if pow_gain is not None else None),
        power_edges=(torch.from_numpy(pow_edges) if pow_edges is not None else None),
        power_tau=a.pow_tau,
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
                            state_power=a.w_state_power, z=a.w_z, swap=a.w_swap,
                            gate_cond=a.w_gate_cond),
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

    #: 14.295 — 궤적 평균. `swa_start` 부터 LR 을 **상수로 고정**하고 epoch 끝마다
    #  한 점씩 누적한다. 고정이 처치의 일부다 — 코사인이 0 으로 떨어지면 평균할
    #  퍼짐이 없어 w̄ ≈ w_T 가 된다 (관문 [2]).
    swa = WeightAverager(model) if a.swa_start else None
    swa_lr = 0.0
    if swa is not None:
        _spe = max(1, a.epoch_windows // a.batch)          # epoch 당 step
        swa_lr = float(a.swa_lr) if a.swa_lr > 0 else float(
            a.lr * (1.0 + math.cos(math.pi * ((a.swa_start - 1) * _spe) / steps)) / 2.0)
        print("** 14.295 가중치 평균 — ep%d 부터 LR 을 %.3e 로 고정하고 %d epoch 마다 "
              "누적한다 (예상 %d 점) **"
              % (a.swa_start, swa_lr, a.swa_every,
                 1 + (a.epochs - a.swa_start) // a.swa_every))

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
        #: ★ 14.348 — `--comb-tau` 를 켰는데 캐시에 `g_hat.npy` 가 없으면 배치가 **NaN**
        #  이고, 조합 softmax 가 그걸 퍼뜨려 **손실이 조용히 죽는다**. 여기서 멈춘다.
        if a.comb_over > 0 and a.comb_tau <= 0:
            raise SystemExit("--comb-over 는 --comb-tau 없이는 아무 일도 안 한다 "
                             "(조합 머리가 꺼져 있으면 후보 점수가 없다)")
        if a.comb_tau > 0 and not getattr(cache, "has_ghat", False):
            raise SystemExit(
                "--comb-tau 를 켰는데 캐시에 `g_hat.npy` 가 없다: %s@N@"
                "    python -X utf8 -m src.run_build_ghat --cache %s"
                .replace("@N@", chr(10)) % (a.cache, a.cache))
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
                    "on_detach_gate": a.on_detach_gate,
                    "on_power_praw": a.on_power_praw,
                    "gate_free_power": a.gate_free_power,
                    "even_median": int(a.even_median),
                    #: 14.331 — 입력 배치의 규약. 없으면 옛 6차수 판이다.
                    "volt_orders": list(VOLT_ORDERS),
                    #: 14.347 — 조합 머리의 눈금. 0 이면 안 썼다.
                    "z_input": bool(a.z_input),
                    "z_drop": float(a.z_drop),
                    #: 눈금을 체크포인트에 같이 적는다 — 바뀌면 옛 판이 조용히 어긋난다
                    "z_log_mean": float(_NET.Z_LOG_MEAN),
                    "z_log_std": float(_NET.Z_LOG_STD),
                    "comb_tau": float(a.comb_tau),
                    "comb_over": float(a.comb_over),
                    "comb_over_margin": float(a.comb_over_margin),
                    "cons_deadzone": float(a.cons_deadzone),
                    # 14.295 궤적 평균. 0 이면 안 쓴 것 = 옛 경로와 비트 동일.
                    "swa_start": int(a.swa_start),
                    "swa_lr": float(swa_lr),
                    "swa_every": int(a.swa_every),
                    "swa_n": int(swa.n) if swa is not None else 0,
                    "off_detach_praw": a.off_detach_praw,
                    "wide_summary": a.wide_summary, "wide_target": a.wide_target,
                    "periodicity": a.periodicity,
                    "fine_dropout": a.fine_dropout,
                    # L_harm 기울기 균등화 (12.120). 13.84.17 에서 1단계에도 열었다.
                    "harm_grad_balance": a.harm_grad_balance,
                    # 세밀 채널 수를 반드시 남긴다. 12.34 에서 38 -> 44 로
                    # 늘었고, 이 키가 없는 체크포인트는 38 로 간주된다.
                    "fine_channels": model.fine_channels,
                    "fine_dilations": list(model.fine_dilations),
                    "fine_pool": model.fine_pool,
                    "fine_extra_dilations": list(model.fine_extra_dilations),
                    "tap_layers": list(model.tap_layers),
                    # 세밀 몸통 시간 분할 (14.116). **이 키를 안 적어서** 학습은 됐는데
                    # 채점 경로가 모델을 못 지었다 — `trunk.0` 이 1021 대 765 로 어긋난다.
                    # 구조를 바꾸는 손잡이는 전부 여기 적혀야 한다
                    # ([[verify-the-input-path-not-just-the-model]]).
                    "fine_time_split": bool(model.fine_time_split),
                    # 상태 전력 상한 (14.121). 0 이면 상한 없음 -> 비트 동일.
                    "p_state_cap": float(model.p_state_cap),
                    # 미래 토막 수 (14.122). 1 이면 비트 동일.
                    "fine_future_segs": int(model.fine_future_segs),
                    # 머리에서 뺀 덩이 (14.128). 빈 값이면 비트 동일.
                    "head_drop": ",".join(model.head_drop),
                    # 머리 배치 (14.130). v1 이면 비트 동일.
                    "head_layout": str(model.head_layout),
                    # 세밀 패딩 (14.131). zeros 면 비트 동일.
                    "state_power_src": str(model.state_power_src),
                    "fine_norm": str(model.fine_norm),
                    "fine_conv": str(model.fine_conv),
                    "fine_tpool": str(model.fine_tpool),
                    "fine_derive": str(model.fine_derive),
                    "wide_dg": bool(model.wide_dg),
                    "fine_dc": str(model.fine_dc),
                    "fine_pad": str(model.fine_pad),
                    # 어느 캐시·홀드아웃으로 배웠나 (14.126). 판정할 때 "이 팔이 어느
                    # 캐시였지" 를 체크포인트에서 못 읽어 sbatch 를 뒤져야 했다.
                    # 처치가 **깃발이 아니라 캐시**인 판이 있으므로 반드시 남긴다.
                    "cache": str(a.cache), "holdout": str(a.holdout),
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
                    "wide_seg_pool": int(model.wide_seg_pool),
                    "wide_extra_dilations": list(model.wide_extra_dilations),
                    "harm_vnorm_anchor": bool(a.harm_vnorm_anchor),
                    #: 14.171 — 2단계가 **같은 순방향 모형**을 지으려면 이 둘이 필요하다.
                    #  없어서 `run_train_seq` 가 전압 앵커를 못 켜고 있었다.
                    "pow_sig": bool(a.pow_sig),
                    "pow_bands": int(a.pow_bands),
                    "pow_tau": float(a.pow_tau),
                    "harm_sig_vnorm": bool(a.harm_sig_vnorm),
                    "harm_vnorm_classes": str(a.harm_vnorm_classes),
                    "res_apps": str(a.res_apps),
                    "res_cond_state": str(a.res_cond_state),
                    "swap_tol": float(a.swap_tol),
                    "swap_tiebreak": str(a.swap_tiebreak),
                    "swap_slack": float(a.swap_slack),
                    "swap_tb_orders": str(a.swap_tb_orders),

                    "harm_vnorm_frac": float(a.harm_vnorm_frac),
                    "vrel_target": bool(a.vrel_target),
                    "harm_vhrel_anchor": bool(a.harm_vhrel_anchor),
                    "harm_vhrel_frac": float(a.harm_vhrel_frac),
                    "harm_vhrel_src": str(a.harm_vhrel_src),
                    # 손실 설정이라 추론엔 안 쓴다. 계보 추적용이다 (13.80).
                    "gate_smooth": a.gate_smooth, "gate_focal": a.gate_focal,
                    "vswap_p": a.vswap_p,                 # 13.84.11 학습 시 전압 채널 바꿔 끼우기 (추론엔 무관)
                    "head_conductance": a.head_conductance,  # 14.284 전도도 머리
                    # 14.299 전도도 목표의 척도·문턱·분류. 기본값이면 14.284 와 같다.
                    "hcond_scale": str(a.hcond_scale),
                    "hcond_on_w": float(a.hcond_on_w),
                    "hcond_classes": str(a.hcond_classes),
                    "even_jitter": a.even_jitter,         # 14.245 학습 시 짝수차 모양 흔들기 (추론엔 무관)
                    "odd_phase_jitter": a.odd_phase_jitter,  # 14.268 학습 시 홀수차 위상 돌리기
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
            fine, wide, tgt = to_targets(batch, dev, vrel_target=a.vrel_target)
            if ZERO_CH:
                fine[:, ZERO_CH] = 0.0        # 12.114 재시험의 조인 대조
            if ZERO_W:
                wide[:, ZERO_W] = 0.0
            vswap(fine, wide, a.vswap_p)          # 13.84.11 — 0 이면 아무것도 안 한다
            even_jitter(fine, wide, a.even_jitter)  # 14.245 — 0 이면 난수도 안 뽑는다
            odd_phase_jitter(fine, wide, a.odd_phase_jitter)   # 14.268 — 0 이면 무동작
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(fine, wide,
                            tgt["g_hat"] if model.comb_tau > 0 else None,
                            _z_drop(tgt["r_grid"], a.z_drop)
                            if getattr(model, "z_input", False) else None)
                parts = crit(out, tgt)
            opt.zero_grad(set_to_none=True)
            parts["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            #: 14.295 — 평균 구간에서는 코사인을 **멈추고** LR 을 상수로 둔다.
            #  SWA 원 논문의 처방이다. 퍼짐이 있어야 중심을 잴 수 있다.
            if swa is not None and ep >= a.swa_start:
                for _g in opt.param_groups:
                    _g["lr"] = swa_lr
            else:
                sched.step()
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
        #: 14.295 — 이 epoch 의 점을 평균에 넣는다. 마지막 epoch 은 항상 넣는다.
        if swa is not None and ep >= a.swa_start and (
                (ep - a.swa_start) % a.swa_every == 0 or ep == a.epochs):
            swa.add(model)

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

    #: 14.295 — 평균낸 가중치를 **모델에 싣고** 나서 저장·최종 평가한다. 순서를
    #  거꾸로 하면 보고한 숫자와 저장한 판이 다른 물건이 된다
    #  ([[verify-the-input-path-not-just-the-model]]).
    if swa is not None:
        _shift = swa.rel_shift(model)
        print("** 14.295 가중치 평균 %d 점 · ||w̄ − w_T||/||w_T|| = **%.3e** %s **"
              % (swa.n, _shift,
                 "" if _shift > 1e-4 else "<- ⚠ 퍼짐이 없다. 무동작에 가깝다"))
        #: ★ **짝을 공짜로 만든다** (14.295). `swa_start` 부터 LR 을 고정하면 짝과
        #  다른 것이 **둘**(스케줄 꼬리·평균)이 되어 이겨도 어느 쪽인지 모른다
        #  ([[count-how-many-things-differ-before-attributing]]). 같은 궤적의
        #  **끝점**을 따로 저장하면, 씨앗·자료순서·스케줄이 전부 같고 **평균만** 다른
        #  대조가 공짜로 생긴다.
        save_ckpt(Path(a.out) / f"{a.tag}_last.pt", a.epochs)
        print("   같은 궤적의 끝점을 %s_last.pt 로 따로 저장했다 (평균만 다른 대조)"
              % a.tag)
        model.load_state_dict(swa.state_dict(model))

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
