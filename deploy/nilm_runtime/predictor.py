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
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import os
import sys
import numpy as np
import torch

#: ⚠⚠ 14.378 — **연구 코드를 그대로 비춘 `deploy/src/` 를 쓴다.** 예전에는 이 묶음이
#: `nilm_runtime/net.py` 같은 **손으로 깎은 사본**을 들고 있었고, 사흘 만에 세밀
#: 채널이 50 대 61 로 갈려 새 체크포인트를 원리상 못 실었다. 사본을 없앤다
#: ([[verify-the-input-path-not-just-the-model]]).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.inputs import (build_inputs, csv_columns,  # noqa: F401,E402
                              csv_has_voltage_harmonics, row_to_channels,
                              target_index, FINE_CYCLES, RAW_CHANNELS,
                              TARGET_LOOKAHEAD, VOLT_ORDERS)
from src.model.build import build_model, check_input_convention  # noqa: E402
from src.model.postproc import (SMPS_GROUP, SNAP_TARGET_W,  # noqa: F401,E402
                                absorb_residual, apply_postproc, resistive_match,
                                snap_power, squelch)

CYCLE_HZ = 60
WINDOW_CYCLES = 3600                 # 60초
HARMONICS = 15

#: 화면 표시용 한글 이름.
APPLIANCE_KO = {
    "electiric_kettle": "전기포트", "oven": "오븐", "hotplate": "핫플레이트",
    "hair_dryer": "드라이기", "minipc": "미니PC", "beam_projector": "프로젝터",
    "laptop_charger": "충전기", "fan": "선풍기", "air_conditioner": "에어컨",
}


def sync_even_median(ck: dict) -> int:
    """체크포인트가 적어 둔 짝수차 이동중앙값으로 **모듈 전역을 맞춘다** (14.164).

    ⚠ `inputs.EVEN_MEDIAN` 은 전역이고 `build_inputs` 가 그것을 읽는다. 안 맞추면
      **창을 짓는 식이 학습과 달라진다** — 죽지 않고 그럴듯한 틀린 수를 낸다.
      `run_gate_check.sync_even_median` 과 같은 일을 한다 (배포는 체크포인트가
      하나뿐이라 목록을 받을 필요가 없다).
    """
    from src.model import inputs as _I
    k = max(int(ck.get("even_median", 0) or 0), 1)
    _I.EVEN_MEDIAN = k
    return k


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

    #: ⚠ 14.378 — **49 다** (33 이 아니다). 33~48 이 단자 전압 고조파 Re/Im 이고
    #  지금 모델은 그것을 입력으로 받는다. 옛 묶음은 33 이라 새 체크포인트를
    #  원리상 못 돌렸다.
    def __init__(self, size: int, channels: int = RAW_CHANNELS,
                 use_time: bool = True,
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
                 postproc: str = "on", resmatch: float = 0.02, rm_snap: bool = True,
                 snap: float = 0.0, squelch: float = 0.1, absorb: float = 1.0,
                 absorb_mode: str = "pq", absorb_wq: float = 1.0,
                 reorder: bool = True):
        """
        Args:
            ckpt_path: 운영점 체크포인트 (`models/adapt_zi_s0.pt`)
            device: "cuda" / "cpu". 생략하면 있는 쪽을 쓴다
            postproc: "off" | "on" | "sync" — 물리 전력 상한 후처리.
                운영 기본은 "on" 이다 (README 의 성능 표 참조)
            rm_snap: 저항 정합이 조합을 확인한 창에서 전력도 V^2/R 로 맞춘다
                (12.117). 끄면 12.117 이전 동작이다.
            resmatch: 저항 부하 정합 허용오차. 관측 전력·전압으로 등가저항을
                역산해 저항 조합을 맞바꾼다. 운영 기본 0.02, 0 이면 끔
            snap: 프로젝터를 격리 참값 W 로 맞추고 차액을 다른 SMPS 로 넘긴다
                (12.129). **2026-09-02 기본 꺼짐** — 복소 Z·I 모델이 프로젝터를
                −0.37W 로 닫아서 못 박을 것이 없고, 흡수와 충돌한다 (12.149.2).
                켜려면 `snap=46.9`.
            squelch: 게이트가 이 값 아래인 기기의 전력을 0 으로 (12.149).
                `P = sigma(on) * p_raw` 라 **꺼졌다고 보고한 기기가 와트를 낸다** —
                에어컨 sigma 0.008 x p_raw 592W = 4.9W. 운영 기본 0.1, 0 이면 끔.
            absorb: 그렇게 빠진 와트를 고조파가 닮은 SMPS 로 되돌린다 (12.104).
                **스켈치와 짝이다** — 하나만 켜면 반쪽이다 (흡수만이면 유령이
                그대로, 스켈치만이면 잔차가 터진다). 운영 기본 1.0, 0 이면 끔.
            absorb_mode: 흡수의 배분 규칙 (12.153). **운영 기본 "pq"** —
                총전력 잔차를 나눌 때 **무효전력 방정식**을 같이 푼다. `Q/P` 는
                SMPS 쌍을 고조파보다 2.2~2.5배 잘 가른다 (12.133).
                "cos" 는 12.104 의 옛 규칙.
            absorb_wq: pq 에서 무효 방정식의 가중. 운영 기본 1.0.
            reorder: 수신기 CSV 의 순서 뒤바뀜을 t_s 로 보정한다. 끄지 말 것
        """
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        #: ★ 14.378 — **연구 채점기와 같은 함수로 짓는다.** 예전에는 여기서 인자를
        #  여섯 개만 넘겨서 dilation·탭·상태상한·**조합 머리**가 통째로 빠진 모델이
        #  섰다. 같은 체크포인트인데 다른 모델이 서는 것을 아무도 못 봤다.
        sync_even_median(ck)
        check_input_convention(ck, ckpt_path)
        self.model, self.appliances = build_model(ck, self.dev, ckpt_path=ckpt_path)
        self.meta = {k: v for k, v in ck.items() if k not in ("model",)}
        #: ★★ Ĝ 추정기 — `comb_tau>0` 이면 **선택이 아니다.** `net.forward` 가
        #: `g_hat` 없이 `ValueError` 로 멈춘다. 조합 머리가 저항 4종의 전력을
        #: **G 표에서** 내는데 그 표의 어느 칸을 쓸지 고르는 것이 Ĝ 다.
        self.comb_tau = float(getattr(self.model, "comb_tau", 0.0) or 0.0)
        self._live = None
        self._sigs_state = None
        self.g_last = float("nan")
        if self.comb_tau > 0:
            z = np.load(Path(__file__).with_name("runtime_tables.npz"), allow_pickle=True)
            if list(z["appliances"]) != list(self.appliances):
                raise ValueError("runtime_tables.npz 의 기기 목록이 체크포인트와 다릅니다: "
                                 "%s != %s" % (list(z["appliances"]), self.appliances))
            self._sigs_state = (z["sig_state"], z["sig_state_used"])
            print("  ** 조합 머리 tau=%.2f · 추론 꺾기 comb_over=%.3f · Ĝ 추정기 켬 **"
                  % (self.comb_tau, float(self.model.comb_over)))
        self.postproc = postproc
        self.resmatch = float(resmatch)
        #: 프로젝터 참값 스냅 (12.128~12.129). 프로젝터 과대예측은 총전력
        #: 부족분의 배출구가 아니라 **충전기와의 제로섬 맞바꿈**(+17.00/−17.01W)
        #: 이고, 참값에 못 박으면 그 17W 가 충전기로 돌아간다.
        self.snap = float(snap)
        #: 저항 정합에서 조합이 이미 맞을 때도 전력을 V^2/R 로 스냅한다 (12.117).
        #: 시드 4개에서 8파일 유령 8.98 -> 5.23W, 폭 6.05 -> 4.15W, 잔차
        #: 10.15 -> 7.30W, F1 과 전이 귀속은 불변. 운영 기본 True.
        self.rm_snap = bool(rm_snap)
        #: 게이트 정합 스켈치 + 잔차 흡수 (12.149 / 12.149.1). **짝이다.**
        #: `sig`/`standby_sig`/`noise_sig` 는 학습 자료에서 뽑아 꾸러미에 구워 뒀다
        #: (`signatures.npz`) — 배포에는 `SegmentPool` 이 없다.
        self.squelch_tau = float(squelch)
        self.absorb = float(absorb)
        self.absorb_mode = str(absorb_mode)
        self.absorb_wq = float(absorb_wq)
        self._sigs: Optional[tuple] = None
        if self.absorb > 0:
            z = np.load(Path(__file__).with_name("signatures.npz"))
            if list(z["appliances"]) != list(self.appliances):
                raise ValueError(
                    f"signatures.npz 의 기기 목록이 체크포인트와 다릅니다: "
                    f"{list(z['appliances'])} != {self.appliances}")
            self._sigs = (z["sig"].astype(np.float64), z["standby_sig"].astype(np.float64),
                          z["noise_sig"].astype(np.float64))
            # 무효전력 지문 (12.133 / 12.153). `signatures.npz` 에 같이 구워 뒀다.
            self._qp = z["qp"].astype(np.float64) if "qp" in z else None
            self._noise_q = float(z["noise_q"]) if "noise_q" in z else 0.0
            if self.absorb_mode == "pq" and self._qp is None:
                raise ValueError("signatures.npz 에 qp 가 없습니다 — 다시 구우십시오")
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

    # ── Ĝ 추정기 (14.378) ────────────────────────────────────────────────
    def _ghat(self, win: np.ndarray) -> float:
        """이 창의 **타깃 사이클에서** `Ĝ` [mS]. 정본은 `gbudget.LiveGhat` 다 —
        `run_live.py` 도 같은 것을 부른다."""
        if self._live is None:
            from src.model.gbudget import LiveGhat
            self._live = LiveGhat(self.appliances, sigs=self._sigs_state)
        self.g_last = self._live.of(win, self.target_in_window)
        return self.g_last

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
        #: ★ 조합 머리는 **Ĝ 가 입력**이다 (14.349). 안 넘기면 `net.forward` 가
        #  멈춘다 — 조용히 반쪽으로 돌지 않는다.
        _kw = {}
        if self.comb_tau > 0:
            _kw["g_hat"] = torch.full((1,), self._ghat(win),
                                      dtype=torch.float32, device=self.dev)
        o = self.model(torch.from_numpy(fine).to(self.dev),
                       torch.from_numpy(wide).to(self.dev), **_kw)
        gate = torch.sigmoid(o["on_logit"])[0].float().cpu().numpy()
        power = o["power"][0].float().cpu().numpy()
        standby_k = o["standby"][0].float().cpu().numpy()
        standby = float(standby_k.sum())
        p_obs = float(win[0, 30, self.target_in_window])

        if self.squelch_tau > 0:
            # **후처리 앞이다** (12.149). 상한/스냅/저항정합이 문턱 아래 유령을
            # 실체로 오인해 재배분하면 안 된다. src/run_live.py 와 같은 순서다.
            power = squelch(power[None, :], gate[None, :], self.squelch_tau)[0]
        if self.postproc != "off":
            power, gate = apply_postproc(power[None, :], gate[None, :], self.appliances,
                                         gate_sync=(self.postproc == "sync"))
            power, gate = power[0], gate[0]
        if self.snap > 0:
            # 프로젝터 참값 스냅 (12.129). **상한 뒤, 저항 정합 앞** —
            # src/run_live.py 와 같은 순서다. 다르면 배포와 채점이 갈린다.
            power, gate = snap_power(power[None, :], gate[None, :], self.appliances,
                                     targets={"beam_projector": float(self.snap)})
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
                obs_harm=obs_h[None], tol=self.resmatch, snap=self.rm_snap)
            power, gate = power[0], gate[0]
        if self.absorb > 0 and self._sigs is not None:
            # 스켈치가 지운 와트를 고조파가 닮은 SMPS 로 되돌린다 (12.104 + 12.149).
            obs_h = np.stack([win[0, 0:15, self.target_in_window],
                              win[0, 15:30, self.target_in_window]], axis=-1)
            kw = {}
            if self.absorb_mode == "pq":
                # 채널 31 이 원시 무효전력이다 (30=P, 32=V 와 같은 규약).
                kw = dict(qp=self._qp, noise_q=self._noise_q, w_q=self.absorb_wq,
                          q_observed=np.array([float(win[0, 31, self.target_in_window])]))
            power = absorb_residual(
                power[None, :], gate[None, :], self.appliances, standby_k[None, :],
                np.zeros(1), np.array([p_obs]), obs_h[None], *self._sigs,
                frac=self.absorb, mode=self.absorb_mode,
                # 스냅으로 못 박은 기기는 잔차를 안 받는다 (12.149.2)
                exclude=["beam_projector"] if self.snap > 0 else None, **kw)[0]

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
