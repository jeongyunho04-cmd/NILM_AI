"""실시간 예측기 — UI 가 쓰는 API (배포 묶음)

    from nilm_runtime import NILMPredictor

    pred = NILMPredictor("models/adapt_smpsf.pt")
    pred.set_header(header_cols)             # 수신기 CSV 헤더 (한 번)
    for row in csv_rows:                     # 사이클 1개 = 행 1개
        pred.push_row(row)
        out = pred.predict()                 # 창(60초)이 차면 결과, 아니면 None
        if out:
            print(out.total_w, out.power_w["minipc"])

**이 파일은 학습 저장소의 `src/run_live.py` 에서 런타임 부분만 떼어 온 것이다.**
CSV 파싱과 링버퍼(순서 뒤바뀜 보정, 세션 이어붙임)는 원본 그대로다 — 그 주석에
왜 그렇게 해야 하는지가 적혀 있다.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import numpy as np
import torch

from .inputs import build_inputs, target_index, FINE_CYCLES, TARGET_LOOKAHEAD
from .net import NILMNet, appliance_state_counts
from .postproc import SMPS_GROUP, apply_postproc, resistive_match

CYCLE_HZ = 60
WINDOW_CYCLES = 3600                 # 60초
HARMONICS = 15

#: 화면 표시용 한글 이름.
APPLIANCE_KO = {
    "electiric_kettle": "전기포트", "oven": "오븐", "hotplate": "핫플레이트",
    "hair_dryer": "드라이기", "minipc": "미니PC", "beam_projector": "프로젝터",
    "laptop_charger": "충전기", "fan": "선풍기", "air_conditioner": "에어컨",
}


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


@dataclass
class PredictionResult:
    """추론 1회의 결과. UI 는 이것만 보면 된다."""
    t_s: float                          # 창 머리의 시각 (초, 보드 기준)
    observed_w: float                   # 관측 총전력
    total_w: float                      # 예측 합계 (활성 + 대기)
    power_w: Dict[str, float]           # 기기별 예측 전력 (W)
    gate: Dict[str, float]              # 기기별 ON 확률 0~1
    standby_w: float                    # 대기전력 합
    residual_w: float                   # 관측 − 예측 합계

    def on(self, threshold: float = 0.5) -> List[str]:
        """켜졌다고 본 기기 목록."""
        return [k for k, v in self.gate.items() if v > threshold]


class NILMPredictor:
    """수신기 CSV 행을 받아 기기별 전력을 내놓는다.

    **스레드 안전하지 않다.** UI 에서 쓸 때는 한 스레드에서만 `push_row`/`predict`
    를 부르고, 결과를 큐로 넘기는 편이 낫다.
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None,
                 postproc: str = "on", resmatch: float = 0.02,
                 reorder: bool = True):
        """
        Args:
            ckpt_path: 운영점 체크포인트 (`models/adapt_smpsf.pt`)
            device: "cuda" / "cpu". 생략하면 있는 쪽을 쓴다
            postproc: "off" | "on" | "sync" — 물리 전력 상한 후처리.
                운영 기본은 "on" 이다 (README 의 성능 표 참조)
            resmatch: 저항 부하 정합 허용오차. 관측 전력·전압으로 등가저항을
                역산해 저항 조합을 맞바꾼다. 운영 기본 0.02, 0 이면 끔
            reorder: 수신기 CSV 의 순서 뒤바뀜을 t_s 로 보정한다. 끄지 말 것
        """
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.appliances: List[str] = list(ck["appliances"])
        self.model = NILMNet(
            self.appliances, appliance_state_counts(self.appliances),
            width=ck.get("width", 1.0), wide_summary=ck.get("wide_summary", False),
            periodicity=ck.get("periodicity", False),
            fine_dropout=ck.get("fine_dropout", 0.0),
            prior_kappa=ck.get("prior_kappa", 0.0), prior_beta=ck.get("prior_beta", 0.5),
            fine_channels=ck.get("fine_channels", 50)).to(self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.meta = {k: v for k, v in ck.items() if k not in ("model",)}
        self.postproc = postproc
        self.resmatch = float(resmatch)
        self.ring = CycleRing(WINDOW_CYCLES, use_time=reorder)
        self.cols: Optional[Dict[str, int]] = None
        self.target_in_window = target_index(WINDOW_CYCLES)
        self.n_pushed = 0

    # ── 입력 ─────────────────────────────────────────────────────────────
    def set_header(self, header: Sequence[str]) -> None:
        """수신기 CSV 헤더를 한 번 알려 준다."""
        self.cols = csv_columns(list(header))

    def push_row(self, row: Sequence[str], t_s: Optional[float] = None) -> str:
        """CSV 한 행(사이클 1개)을 넣는다. 'new'/'backfill'/'stale'/'seam'.

        `t_s` 를 생략하면 행의 `t_s` 열을 쓴다. 그 값은 보드 seq 로 계산돼
        도착 순서와 무관하게 정확하다 — 링버퍼가 그것으로 자리를 정한다.
        """
        if self.cols is None:
            raise RuntimeError("set_header() 를 먼저 부르십시오.")
        x = row_to_channels(list(row), self.cols)
        if x is None:
            return "stale"
        if t_s is None and "t_s" in self.cols:
            try:
                t_s = float(row[self.cols["t_s"]])
            except (ValueError, IndexError):
                t_s = None
        self.n_pushed += 1
        return self.ring.push(x, t_s)

    def ready(self) -> bool:
        """창(60초)이 찼는가."""
        return self.ring.ready()

    # ── 추론 ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self) -> Optional[PredictionResult]:
        """창이 찼으면 추론 1회. 안 찼으면 None.

        60Hz 로 매 사이클 부를 필요는 없다 — 0.5초(30사이클)마다면 충분하다.
        RTX 2050 에서 약 120~200 추론/초가 나온다.
        """
        if not self.ring.ready():
            return None
        win = self.ring.window()[None]                     # (1, 33, 3600)
        fine, wide = build_inputs(win)
        o = self.model(torch.from_numpy(fine).to(self.dev),
                       torch.from_numpy(wide).to(self.dev))
        gate = torch.sigmoid(o["on_logit"])[0].float().cpu().numpy()
        power = o["power"][0].float().cpu().numpy()
        standby_k = o["standby"][0].float().cpu().numpy()
        standby = float(standby_k.sum())
        p_obs = float(win[0, 30, self.target_in_window])

        if self.postproc != "off":
            power, gate = apply_postproc(power[None, :], gate[None, :], self.appliances,
                                         gate_sync=(self.postproc == "sync"))
            power, gate = power[0], gate[0]
        if self.resmatch > 0:
            # 저항 정합: 등가저항이 기기 고유값이라 조합을 역산할 수 있다.
            #   포트 35.8 / 오븐 40.6 / 드라이기 54.3 / 핫플 101.8 Ω
            obs_h = np.stack([win[0, 0:15, self.target_in_window],
                              win[0, 15:30, self.target_in_window]], axis=-1)
            power, gate = resistive_match(
                power[None, :], gate[None, :], self.appliances,
                np.array([p_obs]), np.array([float(win[0, 32, self.target_in_window])]),
                standby_k[None, :], np.zeros(1),
                obs_harm=obs_h[None], tol=self.resmatch)
            power, gate = power[0], gate[0]

        total = float(power.sum()) + standby
        return PredictionResult(
            t_s=float(self.ring.t_head), observed_w=p_obs, total_w=total,
            power_w={a: float(p) for a, p in zip(self.appliances, power)},
            gate={a: float(g) for a, g in zip(self.appliances, gate)},
            standby_w=standby, residual_w=p_obs - total)

    # ── 상태 ─────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """링버퍼 통계 — 순서 뒤바뀜 비율, 창 채움률 등. UI 진단 패널용."""
        st = self.ring.stats()
        st["n_pushed"] = self.n_pushed
        st["device"] = self.dev
        st["postproc"] = self.postproc
        return st
