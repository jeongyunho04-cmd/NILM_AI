"""
실시간 추론 — 수신기 CSV 를 따라가며 분해하고, 예측과 실제를 함께 남긴다
==========================================================================
5.3절의 시연 동선이자, **다음 실측 라벨을 만드는 도구**다.

    python -m src.run_live --csv data/live.csv
    python -m src.run_live --replay data/test3.csv --speed 20     # 보드 없이 검증

[왜 수신기에 넣지 않고 CSV 를 따라가는가]
`nilm_receiver.py` 의 수신 루프는 ACK 타이밍에 묶여 있다. 펌웨어는 ACK 가 2.5초
안에 안 오면 같은 프레임을 다시 보내고, 5번 실패하면 큐가 밀린다(`nilm_link.h`).
그 루프 안에서 GPU 추론을 돌리면 ACK 가 늦어져 **프레임 유실을 만든다** —
측정 자체를 망가뜨리는 위험이다. 별도 프로세스로 CSV 를 따라가면 추론이 아무리
느려도 수신에 영향이 없다.

[예측과 실제를 함께 남기는 이유 — 이 파일의 진짜 목적]
지금 실측 라벨은 신호를 사람이 읽어 쓴 것이고, 12.25 에서 미니PC 구간이 틀린 것이
드러났다. 켜고 끈 시각을 손으로 적는 것도 방법이지만, **모델이 먼저 답하고 사람이
고치는 쪽이 배우는 게 많다**:

  - 사람은 "지금 뭐가 켜져 있나" 를 처음부터 쓰지 않고 **틀린 것만 고친다**
  - 고칠 거리가 생기는 지점이 곧 모델이 헷갈리는 지점이다 (능동 학습)
  - 예측 전체가 자동으로 기록되므로, 사람이 못 본 오답도 나중에 찾을 수 있다

**다만 사람이 알아챈 것만 기록하면 표본이 편향된다.** 모델이 확신하고 틀린
구간은 눈에 안 띈다 (12.15.1 이 그런 실패였다). 그래서 예측 스트림은 **전부**
남기고, 사람의 정정은 그 위에 얹는다.

    실행 중 키:  1~9 해당 기기 토글   0 전부 끔   space 현재 상태 확정
                 u 마지막 정정 취소   q 종료
"""
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src import env_guard  # noqa: F401  torch 보다 먼저

import numpy as np
import torch

from src.preprocessing.file_registry import NOISE_FLOOR_EXTERNAL_W
from src.model.inputs import build_inputs, target_index
from src.run_gate_check import load_model

WINDOW_CYCLES = 3600           # 60초 (12.8절에서 확정)
CYCLE_HZ = 60.0                # 계통 1주기 = 1행
HARMONICS = 15

# `--ckpt-smps` 로 따로 예측할 기기. 0.1절의 SMPS 무리다.
#
# [왜 이 옵션이 있는가 — 측정 근거]
# 2단계 적응(`adapt_v17`)은 총전력 잔차를 145.8W -> 28.9W 로 줄이고 유령 포트를
# 166.6W -> 3.7W 로 없앤다. 그 대가로 **SMPS 3종을 부순다**: test_5(사람이 기록한
# 라벨)에서 충전기 참값 68.4W 를 11.6W 로, 미니PC F1 을 0.574 -> 0.137 로 떨어뜨리고,
# 잃은 와트를 프로젝터에 얹는다(참 45.6W -> 72.6W). `L_cons` 가 총합만 보기 때문이다
# (12.27.3 의 4번이 예고한 것인데 크기가 훨씬 크다).
#
# 저항 무리는 2단계에서, SMPS 무리는 1단계에서 가져오면 양쪽 이득만 남는다.
# 재학습 없이 확인했다 — 잔차 30.0W / 유령 포트 3.7W / 충전기 F1 0.954 / 미니PC 0.574.
# 순전파가 2회로 늘지만 RTX 2050 에서 9.3ms 라 60Hz 요건의 1.8배가 남는다.
#
# **이것은 임시 조치다.** 근본 해결은 적응에서 SMPS 게이트를 얼리거나,
# 충전기의 충전 상태 궤적을 합성기 세그먼트 풀에 넣는 것이다 (12.30.6).
#
# [기본값 근거 — 2026-08-24, 12.52 절의 재채점]
# 위 수치는 `adapt_v17`/`cnn_v17` 시절의 것이다. 운영점이 그 뒤 바뀌었고,
# 최근접 보간 + stride 6 으로 네 조합을 나란히 재채점한 결과가 기본값을 정했다:
#
#   조합                        유령W  잔차W  미니PC 프로젝터 충전기  핫플   오븐
#   adapt_v17 단독 (옛 기본값)   9.56  14.50  0.389  0.792  0.861  0.980  0.931
#   adapt_ph1 단독              6.50   5.86  0.359  0.830  0.917  0.980  0.925
#   adapt_ph1 + cnn_ov1 (기본값) 7.90  11.49  0.805  0.841  0.937  0.980  0.925
#
# 하이브리드가 잔차(5.86 -> 11.49)를 내주고 미니PC(0.359 -> 0.805)를 산다.
# 충전기만은 `cnn_v17` 이 0.955 로 아직 최고다 (기본값 0.937).
# `--ckpt-smps ""` 로 단독 동작을 되돌릴 수 있다.
SMPS_GROUP = ("beam_projector", "laptop_charger", "minipc")

# 프로젝터 스냅 목표 (12.129). `postproc` 과 **같은 상수**를 쓴다 — 두 군데에
# 숫자를 적으면 언젠가 갈린다. 여기서 import 하는 것 자체가 운영 기본값이다.
from src.model.postproc import SNAP_TARGET_W  # noqa: E402

KOR = {"oven": "오븐", "hotplate": "핫플", "electiric_kettle": "포트",
       "hair_dryer": "드라이기", "minipc": "미니PC", "beam_projector": "프로젝터",
       "laptop_charger": "충전기", "fan": "선풍기", "air_conditioner": "에어컨"}


def csv_columns(header: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(header)}


def row_to_channels(row: List[str], col: Dict[str, int]) -> Optional[np.ndarray]:
    """CSV 한 행 -> 33채널 한 사이클.

    전처리(`feature_extractor` + `numpy_exporter`)와 **같은 식이어야 한다.**
        Re/Im  = ih_rms * (cos, sin)(radians(ih_deg))
        S      = vrms * irms
        Q      = sign(phase) * sqrt(max(0, S^2 - P^2))
    """
    try:
        irms = float(row[col["irms"]])
        p = float(row[col["p_w"]])
        v = float(row[col["vrms"]])
        phase = float(row[col["phase_deg"]])
        mag = np.array([float(row[col[f"ih{k}"]]) for k in range(1, HARMONICS + 1)], np.float32)
        deg = np.array([float(row[col[f"ihdeg{k}"]]) for k in range(1, HARMONICS + 1)], np.float32)
    except (ValueError, IndexError, KeyError):
        return None
    rad = np.radians(deg)
    s = v * irms
    q = np.sign(phase) * np.sqrt(max(0.0, s * s - p * p))
    x = np.empty(33, np.float32)
    x[0:15] = mag * np.cos(rad)
    x[15:30] = mag * np.sin(rad)
    x[30], x[31], x[32] = p, q, v
    return x


class CycleRing:
    """`t_s` 로 자리를 정하는 원형 버퍼. **순서가 뒤바뀐 행을 제자리에 꽂는다.**

    [왜 append 로는 안 되는가]
    수신기는 프레임을 **도착 순서 그대로** CSV 에 쓴다(`write_frame` 은 프레임이
    도착할 때 호출된다). 그런데 펌웨어는 송신 윈도우를 쓰면서 확인 못 받은 옛 장을
    골라 재전송하므로 `37 -> 28 -> 38 -> 29` 처럼 온다
    (`nilm_receiver.py` 의 `REORDER_MAX` 주석).

    프레임 하나가 30사이클(0.5초)이므로 9프레임 역전이면 **4.5초 전 사이클이
    '가장 최근' 자리에 앉는다.** 세밀 갈래는 뒤 10초만 보고 타깃은 그 안의
    539번째라(`inputs.py`), 오염되는 자리가 정확히 타깃 근방이다.

    설계 문서 12.29.4 는 이것을 "재생 검증에서만 걸린다" 고 적었는데 **틀렸다.**
    실시간에서도 이 도구는 수신기가 쓴 그 도착 순서를 그대로 읽는다.

    [고치는 방법 — 지연 0]
    `t_s` 는 보드 seq 로 계산된 값이라 도착 순서와 무관하게 정확하다
    (`t_s = (seq - seq0) * 0.5 + cycle / 60`). 그래서 `k = round(t_s * 60)` 을
    절대 사이클 번호로 삼아 제자리에 꽂는다. **버퍼링해서 정렬하는 것이 아니라
    자리를 지정할 뿐이라 지연이 늘지 않는다.**

    유실로 빈 자리는 직전 유효값으로 채운다 — 전처리의 60Hz 결측 보간에 해당한다.

    [세션 이어붙임 — 창 밖의 행을 그냥 버리면 안 되는 이유]
    보드가 리셋되면 `t_s` 가 0 으로 되돌아가고, 수신기는 같은 CSV 에 이어쓴다.
    실측 데이터에서 실제로 일어난다 (`test.2.csv` 2회, `test3.csv` 1회,
    `oven_1.csv`·`hotplate_1.csv`·`noise_selfpower.csv` 각 1회).

    되감긴 행은 창보다 훨씬 뒤라 '늦은 행' 판정에 걸리는데, 그냥 버리면
    `k_head` 가 영영 안 내려가 **리셋 이후 전부를 잃는다.** 그래서 창 밖 행이
    연달아 오면 세션 리셋으로 보고 버퍼를 비운다. 한 줄짜리 이상값에 속지
    않으려고 `reset_after` 개가 쌓일 때까지 들고 있다가 되살린다.

    펌웨어 재전송 폭은 최대 32프레임(16초)이라(`REORDER_MAX`) 60초 창보다
    뒤인 행은 원리적으로 늦은 프레임일 수 없다. 실측 최대 역전도 7.98초다.

    설계 문서 12.29.4 가 적어 둔 두 번째 한계(`is_segment_seam` 처리 없음)가
    여기서 해소된다.
    """

    def __init__(self, size: int, channels: int = 33, use_time: bool = True,
                 min_fill: float = 0.98, reset_after: int = 5):
        self.n = size
        self.buf = np.zeros((channels, size), np.float32)
        self.filled = np.zeros(size, bool)
        self.k_head: Optional[int] = None
        self.t_head = 0.0
        self.use_time = use_time
        self.min_fill = min_fill
        self.reset_after = reset_after
        self.pending: List[tuple] = []      # 창 밖 행. 리셋 판정 전까지 들고 있는다
        self.n_new = self.n_back = self.n_stale = self.n_seam = 0
        self.max_back_cycles = 0

    def push(self, x: np.ndarray, t_s: Optional[float]) -> str:
        """한 사이클을 넣는다. 'new' / 'backfill' / 'stale' / 'seam'."""
        if not self.use_time or t_s is None:
            k = 0 if self.k_head is None else self.k_head + 1
        else:
            k = int(round(t_s * CYCLE_HZ))

        if self.k_head is not None and k <= self.k_head - self.n:
            self.pending.append((x.copy(), k, t_s))
            self.n_stale += 1
            if len(self.pending) < self.reset_after:
                return "stale"              # 아직 리셋인지 이상값인지 모른다
            # 연달아 왔다 = 보드 리셋. 버퍼를 비우고 들고 있던 것부터 되살린다
            self.filled[:] = False
            self.k_head = None
            self.n_stale -= len(self.pending)
            self.n_seam += 1
            queued, self.pending = self.pending, []
            for xx, _, tt in queued:
                self.push(xx, tt)
            return "seam"

        self.pending.clear()                # 정상 행이 왔으니 이상값이었다

        if self.k_head is None:
            self.filled[:] = False
            self.k_head = k
        elif k > self.k_head:
            # 머리를 전진시킨다. 새로 열리는 자리는 옛 데이터가 남아 있으므로 비운다.
            for kk in range(self.k_head + 1, min(k, self.k_head + self.n) + 1):
                self.filled[kk % self.n] = False
            self.k_head = k
        elif k > self.k_head - self.n:
            self.buf[:, k % self.n] = x
            self.filled[k % self.n] = True
            self.n_back += 1
            self.max_back_cycles = max(self.max_back_cycles, self.k_head - k)
            return "backfill"

        self.buf[:, k % self.n] = x
        self.filled[k % self.n] = True
        if t_s is not None:
            self.t_head = t_s
        self.n_new += 1
        return "new"

    def ready(self) -> bool:
        return self.k_head is not None and float(self.filled.mean()) >= self.min_fill

    def window(self) -> np.ndarray:
        """(C, n). 시간 순으로 정렬된 창. 빠진 사이클은 직전 유효값으로 채운다."""
        idx = np.arange(self.k_head - self.n + 1, self.k_head + 1) % self.n
        w = self.buf[:, idx]
        f = self.filled[idx]
        if not f.all():
            pos = np.arange(self.n)
            src = np.maximum.accumulate(np.where(f, pos, 0))
            first = int(np.argmax(f))       # 창 머리가 비었으면 첫 유효값으로 채운다
            src[:first] = first
            w = w[:, src]
        return w

    def stats(self) -> dict:
        tot = self.n_new + self.n_back + self.n_stale
        return {"n_new": self.n_new, "n_backfill": self.n_back, "n_stale": self.n_stale,
                "n_seam": self.n_seam,
                "reorder_rate": (self.n_back + self.n_stale) / tot if tot else 0.0,
                "max_backfill_cycles": self.max_back_cycles,
                "max_backfill_s": self.max_back_cycles / CYCLE_HZ,
                "fill_ratio": float(self.filled.mean()) if self.k_head is not None else 0.0}


def tail_rows(path: Path, replay: bool, speed: float, poll: float = 0.2):
    """CSV 를 행 단위로 흘려보낸다. `replay` 면 파일 끝에서 멈춘다."""
    import csv as _csv
    while not path.exists():
        if replay:
            raise FileNotFoundError(path)
        print(f"  {path} 를 기다리는 중…", flush=True)
        time.sleep(1.0)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = _csv.reader(f)
        header = next(reader, None)
        if header is None:
            return
        col = csv_columns(header)
        yield ("header", col)
        while True:
            row = next(reader, None)
            if row is None:
                if replay:
                    return
                time.sleep(poll)          # 수신기가 더 쓸 때까지 기다린다
                continue
            yield ("row", row)
            if replay and speed > 0:
                time.sleep(1.0 / (60.0 * speed))


def render(apps: List[str], gate: np.ndarray, power: np.ndarray, p_obs: float,
           actual: Dict[str, bool], t_s: float) -> str:
    order = np.argsort(-power)
    parts = []
    for j in order:
        on = gate[j] > 0.5
        a = actual.get(apps[j])
        # 사람이 확정한 상태와 어긋나면 표시한다
        mark = "" if a is None else ("=" if a == on else "≠")
        if on or (a is True):
            parts.append(f"{KOR.get(apps[j], apps[j])}{mark} {power[j]:5.0f}W")
    body = "  ".join(parts) if parts else "(전부 꺼짐)"
    return f"t={t_s:7.1f}s  관측 {p_obs:7.1f}W | 예측 {power.sum():7.1f}W | {body}"


def main() -> int:
    ap = argparse.ArgumentParser(description="실시간 추론 + 예측/실제 기록")
    ap.add_argument("--csv", default="data/live.csv", help="수신기가 쓰는 CSV")
    ap.add_argument("--replay", default=None, help="기존 CSV 를 재생해 검증한다")
    ap.add_argument("--speed", type=float, default=20.0, help="재생 배속 (0=최대)")
    # ── 운영점 (2026-08-25, 12.102.5) ───────────────────────────────────
    # **단일 모델 + 물리 상한 후처리로 바꿨다.** 12.31.5 이래 하이브리드
    # (SMPS 는 1단계, 나머지는 2단계)였는데, 12.102 에서 단독+후처리가 네 지표를
    # 앞섰다:
    #
    #   adapt_ze1+cnn_ze1        전이 41/59  유령 8.67W  잔차 11.13W  미니PC 0.763
    #   adapt_smpsf --postproc   전이 44/59  유령 2.13W  잔차  8.88W  미니PC 0.716
    #
    # **2026-08-27 재교체: adapt_ovh + 저항 정합** (12.113). 오븐 라벨을 히터
    # 통전으로 바꾸고(12.111) 등가저항 정합을 넣자(12.112) 저항 부하가 살아났다:
    #
    #   adapt_smpsf + pp         잔차 58.4W  F1 0.779  저항전용파일 F1 0.268/0.516
    #   adapt_ovh + pp + rm      잔차 10.0W  F1 0.838  저항전용파일 F1 0.782/0.836
    #
    # 전이 귀속은 44 -> 38/59 로 진다. **저항이 700~1,500W 라 사용자에게 보이는
    # 값이고 SMPS 전이 차이는 15~50W 대에서 일어난다** — 사용자 판단으로 교체했다.
    #
    # 뒤지는 것은 프로젝터(−0.046)·충전기(−0.034) F1 이다.
    # **실행 간 폭은 안 쟀다** (12.102.5 의 유보). 사용자 결정으로 교체했다.
    # 되돌리려면 `--ckpt results/adapt_ze1.pt --ckpt-smps results/cnn_ze1.pt
    # --postproc off` 로 준다.
    ap.add_argument("--ckpt", default="results/adapt_ovh.pt")
    ap.add_argument("--ckpt-smps", default="", metavar="PT",
                    help="SMPS 3종(프로젝터/충전기/미니PC)만 이 체크포인트로 예측한다. "
                         "**기본은 빈 문자열 = 단독 동작**이다 (12.102.5). 하이브리드로 "
                         "되돌리려면 results/cnn_ze1.pt 를 준다")
    ap.add_argument("--no-rm-snap", action="store_true",
                    help="저항 정합의 전력 스냅을 끈다 (12.118 이전 동작). "
                         "기본은 켜짐 — 조합이 이미 맞을 때도 V^2/R 로 맞춘다")
    ap.add_argument("--resmatch", type=float, default=0.02, metavar="TOL",
                    help="저항 부하 정합 후처리 (12.112절). 관측 전력·전압으로 등가저항을 "
                         "역산해 저항 조합을 **맞바꾼다**(개수는 안 바꾼다). 운영 기본 0.02, 0=끔")
    ap.add_argument("--snap", type=float, default=SNAP_TARGET_W["beam_projector"],
                    metavar="W",
                    help="프로젝터를 격리 참값으로 스냅하고 차액을 다른 SMPS 로 넘긴다 "
                         "(12.129). **운영 기본 켜짐** — 프로젝터 중앙|오차| "
                         "8.09 -> 0.00W, 격리 폭 안에 드는 비율 2.5%% -> 99.9%%. "
                         "대가는 유령 +0.4W, 잔차 +0.04W. 0 이면 끔")
    ap.add_argument("--snap-no-redist", action="store_true",
                    help="스냅이 깎기만 하고 남에게 안 준다. 총합 보존이 깨지는 대신 "
                         "유령이 안 는다 (12.129 에서 잔차 4.24 -> 7.89W 로 터진다)")
    ap.add_argument("--absorb", type=float, default=0.0, metavar="FRAC",
                    help="총전력 잔차를 고조파가 닮은 SMPS 로 흡수한다 (12.104절). "
                         "0.5 면 실측 8파일에서 잔차 8.88 -> 7.35W. **기본은 꺼 둔다** — "
                         "미등록 부하의 잔차까지 SMPS 로 갈 수 있고, 그 위험은 우리 "
                         "실측 파일로는 못 잰다")
    ap.add_argument("--postproc", default="on", choices=("off", "on", "sync"),
                    help="물리 전력 상한 후처리 (12.102절). 프로젝터가 상한(55W)을 넘는 "
                         "만큼을 다른 SMPS 로 넘긴다. sync 는 게이트도 맞춘다. "
                         "**2단계 단독(--ckpt-smps \"\")에서만 이득이다** — 전이 귀속 "
                         "27 -> 44/59. 하이브리드에서는 41 -> 36/59 로 나빠진다")
    ap.add_argument("--every", type=int, default=30, help="추론 간격 (사이클). 30=0.5초")
    ap.add_argument("--log", default="results/live_log.jsonl")
    ap.add_argument("--no-reorder", action="store_true",
                    help="t_s 로 자리를 잡지 않고 파일 순서대로 쌓는다 (옛 동작. "
                         "순서 뒤바뀜의 영향을 비교할 때만)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    path = Path(a.replay or a.csv)
    replay = a.replay is not None
    if replay:
        # 재생으로 `test.csv` 를 들여다보는 것도 봉인을 깨는 것이다 (4.3절).
        # 실측이 35분뿐이라 한 번 보면 최종 평가가 오염된다. 코드가 막는다.
        from src.evaluation.sealing import assert_not_sealed
        assert_not_sealed(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, apps, ck = load_model(a.ckpt, dev)
    model_s = smps_ix = None
    if a.ckpt_smps:
        model_s, apps_s, ck_s = load_model(a.ckpt_smps, dev)
        if list(apps_s) != list(apps):
            raise SystemExit("두 체크포인트의 기기 목록이 다릅니다. 섞으면 안 됩니다.")
        smps_ix = [apps.index(x) for x in SMPS_GROUP if x in apps]
    ti = target_index(WINDOW_CYCLES)
    print("=" * 92)
    print(f"[실시간 추론] {a.ckpt} (stage {ck.get('stage', 1)}) | {path}"
          + ("  [재생]" if replay else "  [수신 대기]"))
    if model_s is not None:
        print(f"  SMPS 3종은 {a.ckpt_smps} (stage {ck_s.get('stage', 1)}) 에서 가져옵니다"
              f"  -> {', '.join(KOR.get(x, x) for x in SMPS_GROUP)}")
    print("  키: " + "  ".join(f"{i+1}={KOR.get(x, x)}" for i, x in enumerate(apps[:9]))
          + "   0=전부끔  space=확정  u=취소  q=종료")
    print("=" * 92)

    ring = CycleRing(WINDOW_CYCLES, use_time=not a.no_reorder)
    actual: Dict[str, bool] = {}
    log = Path(a.log); log.parent.mkdir(parents=True, exist_ok=True)
    logf = log.open("a", encoding="utf-8")
    sigs = None
    if a.absorb > 0:
        from src.model.net import (harmonic_signatures, noise_signature,
                                   standby_signatures)
        from src.synthesis.segment_pool import SegmentPool
        _pool = SegmentPool(npz_dir="processed_data/npz", time_split="train")
        sigs = (harmonic_signatures(_pool, apps), standby_signatures(_pool, apps),
                noise_signature(_pool))
        del _pool
        print(f"  ** 잔차 흡수 {a.absorb:g} (12.104절) **")

    n_seen = n_infer = 0
    t_s = 0.0
    col: Dict[str, int] = {}
    t_start = time.time()

    getch = None
    if sys.platform == "win32" and not replay:
        import msvcrt

        def getch():
            return msvcrt.getwch() if msvcrt.kbhit() else None

    try:
        for kind, item in tail_rows(path, replay, a.speed):
            if kind == "header":
                col = item
                continue
            x = row_to_channels(item, col)
            if x is None:
                continue
            try:
                t_row: Optional[float] = float(item[col["t_s"]])
            except (ValueError, IndexError, KeyError):
                t_row = None
            placed = ring.push(x, t_row)
            n_seen += 1
            if placed == "stale":
                continue                  # 창 밖으로 밀려난 행. 창은 안 바뀐다
            t_s = ring.t_head

            if getch is not None:
                c = getch()
                if c == "q":
                    break
                if c is not None:
                    if c.isdigit():
                        d = int(c)
                        if d == 0:
                            actual = {k: False for k in apps}
                        elif 1 <= d <= len(apps):
                            k = apps[d - 1]
                            actual[k] = not actual.get(k, False)
                    elif c == "u":
                        actual = {}
                    elif c == " ":
                        rec = {"t_s": t_s, "type": "actual", "state": dict(actual)}
                        logf.write(json.dumps(rec, ensure_ascii=False) + "\n"); logf.flush()
                        print(f"  [확정] t={t_s:.1f}s  "
                              + ", ".join(KOR.get(k, k) for k, v in actual.items() if v))

            if not ring.ready() or n_seen % a.every:
                continue

            win = ring.window()[None]                     # (1, 33, 3600)
            fine, wide = build_inputs(win)
            with torch.no_grad():
                o = model(torch.from_numpy(fine).to(dev), torch.from_numpy(wide).to(dev))
            gate = torch.sigmoid(o["on_logit"])[0].float().cpu().numpy()
            power = o["power"][0].float().cpu().numpy()
            if model_s is not None:
                with torch.no_grad():
                    os_ = model_s(torch.from_numpy(fine).to(dev),
                                  torch.from_numpy(wide).to(dev))
                gate[smps_ix] = torch.sigmoid(os_["on_logit"])[0].float().cpu().numpy()[smps_ix]
                power[smps_ix] = os_["power"][0].float().cpu().numpy()[smps_ix]
            standby_k = o["standby"][0].float().cpu().numpy()
            standby = float(standby_k.sum())
            p_obs = float(win[0, 30, ti])
            if a.postproc != "off":
                # 물리 전력 상한 후처리 (12.102). 프로젝터가 상한을 넘는 만큼을
                # 다른 SMPS 로 넘긴다. **오프라인 채점과 같은 함수**를 쓴다.
                from src.model.postproc import apply_postproc
                pp, gg = apply_postproc(power[None, :], gate[None, :], list(apps),
                                        gate_sync=(a.postproc == "sync"))
                power, gate = pp[0], gg[0]
            if a.snap > 0:
                # 프로젝터 스냅 (12.129). **상한 뒤, 저항 정합 앞** — `run_gate_check`
                # 와 같은 순서다. 순서가 다르면 오프라인 채점과 값이 갈린다.
                #
                # 왜 이것만 배분을 고치는가: 12.128 이 프로젝터 과대예측의 정체를
                # **충전기와의 제로섬 맞바꿈**(+17.00/−17.01W)으로 확정했다.
                # 프로젝터를 참값에 못 박으면 그 17W 는 갈 곳이 충전기뿐이고,
                # 12.129 가 실제로 되돌아오는 것을 독립 기준 셋으로 확인했다.
                from src.model.postproc import snap_power
                ps, gs = snap_power(power[None, :], gate[None, :], list(apps),
                                    targets={"beam_projector": float(a.snap)},
                                    redistribute=not a.snap_no_redist)
                power, gate = ps[0], gs[0]
            if a.resmatch > 0:
                # 저항 정합 (12.112). 등가저항이 기기 고유값이라 조합을 역산할 수 있다.
                from src.model.postproc import resistive_match
                obs_h = np.stack([win[0, 0:15, ti], win[0, 15:30, ti]], axis=-1)
                power, gate = resistive_match(
                    power[None, :], gate[None, :], list(apps),
                    np.array([p_obs]), np.array([float(win[0, 32, ti])]),
                    standby_k[None, :], np.array([NOISE_FLOOR_EXTERNAL_W]),
                    obs_harm=obs_h[None], tol=a.resmatch, snap=not a.no_rm_snap)
                power, gate = power[0], gate[0]
            if a.absorb > 0:
                # 총전력 잔차를 고조파가 닮은 SMPS 로 흡수 (12.104).
                from src.model.postproc import absorb_residual
                obs_h = np.stack([win[0, 0:15, ti], win[0, 15:30, ti]], axis=-1)
                power = absorb_residual(
                    power[None, :], gate[None, :], list(apps), standby_k[None, :],
                    np.array([NOISE_FLOOR_EXTERNAL_W]), np.array([p_obs]),
                    obs_h[None], sigs[0], sigs[1], sigs[2], frac=a.absorb)[0]
            n_infer += 1

            rec = {"t_s": round(t_s, 3), "type": "pred", "p_observed": round(p_obs, 2),
                   "pred_total": round(float(power.sum()) + standby, 2),
                   "gate": {k: round(float(g), 4) for k, g in zip(apps, gate)},
                   "power_w": {k: round(float(p), 2) for k, p in zip(apps, power)}}
            if actual:
                rec["actual"] = dict(actual)
                dis = [k for k in apps if k in actual and actual[k] != (gate[apps.index(k)] > 0.5)]
                if dis:
                    rec["disagree"] = dis
            logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not a.quiet:
                print("  " + render(apps, gate, power, p_obs, actual, t_s), flush=True)
    except KeyboardInterrupt:
        print("\n  중단")
    finally:
        st = ring.stats()
        # 순서 통계를 로그에도 남긴다. 채점할 때 "이 세션의 입력이 얼마나
        # 온전했는가" 를 성능과 나란히 봐야 한다.
        logf.write(json.dumps({"t_s": round(t_s, 3), "type": "ring_stats", **st},
                              ensure_ascii=False) + "\n")
        logf.close()

    el = time.time() - t_start
    print("=" * 92)
    print(f"  사이클 {n_seen:,}개 ({n_seen/60:.1f}초 분량)  추론 {n_infer:,}회"
          f"  경과 {el:.1f}초  ->  {n_infer/max(el, 1e-9):.1f} 추론/초")
    if a.no_reorder:
        print("  순서 보정: 꺼짐 (--no-reorder)")
    else:
        print(f"  순서 뒤바뀜: 되꽂음 {st['n_backfill']:,}행"
              f"  버림(창 밖) {st['n_stale']:,}행"
              f"  = 전체의 {st['reorder_rate']*100:.2f}%"
              f"  최대 역전 {st['max_backfill_s']:.2f}초")
        if st["n_seam"]:
            print(f"  세션 이어붙임 {st['n_seam']}회 감지 — 그때마다 창을 비웠다"
                  f" (보드 리셋. 창이 두 세션에 걸치면 안 된다)")
        if st["n_backfill"] or st["n_stale"]:
            print("    (--no-reorder 로 다시 돌리면 이 보정이 성능에 얼마나"
                  " 기여했는지 비교할 수 있다)")
    print(f"  기록: {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
