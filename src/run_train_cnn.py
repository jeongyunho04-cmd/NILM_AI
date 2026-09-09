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
                             TARGET_LOOKAHEAD, WIDE_CHANNELS, build_inputs)
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


def to_targets(batch, dev):
    (fine, wide, yp, yo, ypl, ys, yst, oh, pn, pobs, zg) = [
        b.to(dev, non_blocking=True) for b in batch]
    return fine, wide, {
        "y_power": yp, "y_on": yo, "y_plugged": ypl, "y_standby": ys, "y_state": yst,
        "obs_harm": oh, "p_noise": pn, "p_observed": pobs, "harm_offset": None,
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
    ap.add_argument("--w-harm", type=float, default=0.1)
    ap.add_argument("--w-cons", type=float, default=0.0, help="1단계는 0 (3.3절)")
    ap.add_argument("--w-state-power", type=float, default=0.0, metavar="W",
                    help="상태별 전력 출력을 그 상태의 실제 전력에 묶는 항 (12.35). "
                         "0 이면 끈다 - 그러면 전력 손실이 섞인 뒤에만 걸려 "
                         "충전기·미니PC 의 상태별 출력이 붕괴한다 (분화비 1.02 / 1.16).")
    ap.add_argument("--w-z", type=float, default=0.0, metavar="W",
                    help="몸통 z 에서 log(r_grid) 를 맞히는 보조 감독 (13.55). "
                         "13.54 측정: 입력 57채널에서 log Z 를 R² 0.935 로 뽑는데 "
                         "몸통에서는 0.661 로 흐려진다. 참 전력은 Z 에 불변인데 "
                         "예측은 86%% 폭으로 흔들렸다. 라벨은 캐시의 z_grid 다")
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
                    aux_z=(a.w_z > 0)).to(dev)
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
        even_coherent=(torch.tensor(
            [1.0 if x in PHASE_COHERENT_EVEN else 0.0 for x in apps],
            dtype=torch.float32) if a.harm_even_by_class else None),
        gate_smooth=a.gate_smooth, gate_focal=a.gate_focal,
        weights=LossWeights(harm=a.w_harm, cons=a.w_cons, over=a.w_over,
                            state_power=a.w_state_power, z=a.w_z),
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
        dl = DataLoader(SynthBatchDataset(n_batches, a.batch, a.seed), batch_size=None,
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
                    # 손실 설정이라 추론엔 안 쓴다. 계보 추적용이다 (13.80).
                    "gate_smooth": a.gate_smooth, "gate_focal": a.gate_focal,
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
        if ep % a.eval_every and ep != a.epochs:
            print(f"  ep{ep:>3d}  loss {agg['total']:.4f} (pw {agg['power']:.4f} "
                  f"harm {agg['harm']:.4f}{_zs})  [{t_train:.0f}s, "
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
              f"harm {agg['harm']:.4f}{_zs})  |  MAE {row['mae']:.3f}W  F1 {row['f1']:.4f}  "
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
