"""
실측 복합 부하 창 로더 (2단계 준지도 적응용)
==============================================
설계 문서 4.2절. **기기별 정답이 없는** 실측 복합 부하에서 창을 뽑는다.
라벨이 필요 없는 두 항(`L_cons`, `L_harm`)만 걸어 sim-to-real 편차를 교정한다.

    from src.model.realdata import RealWindows
    rw = RealWindows()                     # 봉인 파일은 자동 제외
    b = rw.batch(idx)                      # fine, wide, p_observed, obs_harm, p_noise

[봉인]
`test.csv` 는 최종 평가 전용이라 `sealing` 이 막는다. 여기서는 아예 목록에서 뺀다 —
`assert_not_sealed` 를 통과시키는 우회로를 만들지 않는다 (4.3절).

[왜 창을 미리 다 만들어 두는가]
실측 전체가 26.4분(약 95,000 사이클)뿐이라 변환 결과가 stride 60 기준 약 70MB 다.
매 스텝 memmap 을 훑는 것보다 한 번 만들어 RAM 에 두는 편이 단순하고 빠르다.

[p_noise 를 상수로 두는 이유]
합성에서는 `p_noise_w` 가 시점별로 있지만 실측에는 없다. 계측계 자체 소비는
파일 전체에서 거의 일정하다 — `power_features[:,0] - p_denoised_w` 가 세 파일 모두
1.4W 근처다 (`file_registry.NOISE_FLOOR_EXTERNAL_W`). 그 상수를 쓴다.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.evaluation.sealing import assert_not_sealed, is_sealed
from src.model.inputs import build_inputs, target_index
from src.preprocessing import load_nilm_npz
from src.preprocessing.file_registry import NOISE_FLOOR_EXTERNAL_W

DEFAULT_DIR = "processed_data/composite_eval"
WINDOW_CYCLES = 3600
SAMPLING_HZ = 60.0


class RealWindows:
    """실측 복합 부하에서 뽑은 창 묶음. 기기별 라벨은 없다."""

    def __init__(
        self,
        npz_dir: str = DEFAULT_DIR,
        stems: Optional[Sequence[str]] = None,
        window_cycles: int = WINDOW_CYCLES,
        stride: int = 60,
        require_valid: bool = True,
        exclude: Optional[Sequence[str]] = None,
    ):
        """`exclude` 는 적응에서 뺄 파일이다 (leave-one-file-out).

        2단계는 봉인 안 된 실측 3파일 전부로 학습하고 **같은 3파일로 채점**한다.
        도메인 적응이라 transductive 자체는 정당하지만, `w_cons`/`w_harm`/`w_hedge`
        까지 그 점수를 보고 골랐으므로 (12.12.2·12.12.3절) 실측 일반화 증거가 없다.
        `test.csv` 는 봉인되어 있어 쓸 수 없다. 그래서 한 파일을 빼고 적응한 뒤
        그 파일로 채점해 가중치가 튜닝 잔향인지 확인한다.
        """
        d = Path(npz_dir)
        found = sorted(p.stem for p in d.glob("*.npz"))
        if stems is None:
            stems = [s for s in found if not is_sealed(s)]
        else:
            # `assert_not_sealed` 를 쓴다 — **`unseal()` 블록 안에서는 통과한다.**
            # 예전에는 여기서 `is_sealed` 로 무조건 막아, 최종 평가(4.3절)조차
            # 이 경로로는 못 들어왔다. 개봉 여부의 판단은 sealing 한 곳에 둔다.
            for s_ in stems:
                assert_not_sealed(s_)
        if exclude:
            drop = set(exclude)
            unknown = drop - set(found)
            if unknown:
                raise ValueError(f"{npz_dir} 에 없는 파일입니다: {sorted(unknown)}")
            stems = [s for s in stems if s not in drop]
            if not stems:
                raise ValueError(f"전부 제외되어 적응할 파일이 없습니다: exclude={sorted(drop)}")
        self.stems = list(stems)
        self.window_cycles = int(window_cycles)
        self.target_in_window = target_index(self.window_cycles)

        F, W, P, H, S, C, V = [], [], [], [], [], [], []
        self.per_file: Dict[str, int] = {}
        for stem in stems:
            raw = load_nilm_npz(str(d / f"{stem}.npz"))
            x = self._to_33ch(raw)
            n = x.shape[1]
            lo = self.target_in_window
            hi = n - (self.window_cycles - 1 - self.target_in_window)
            if hi <= lo:
                continue
            targets = np.arange(lo, hi, stride, dtype=np.int64)
            if require_valid:
                iv = np.asarray(raw["is_valid"]).astype(bool)
                keep = [t for t in targets
                        if iv[t - self.target_in_window:
                               t - self.target_in_window + self.window_cycles].all()]
                targets = np.asarray(keep, dtype=np.int64)
            if not len(targets):
                continue
            fine, wide = self._windows(x, targets)
            F.append(fine); W.append(wide)
            P.append(np.asarray(raw["power_features"])[targets, 0].astype(np.float32))
            # 단자 전압. 저항 정합 후처리가 P = V^2/R 을 푸는 데 쓴다 (12.112).
            V.append(np.asarray(raw["power_features"])[targets, 4].astype(np.float32))
            H.append(np.asarray(raw["harmonics_ri"])[targets].astype(np.float32))
            S += [stem] * len(targets)
            C.append(targets)
            self.per_file[stem] = len(targets)

        if not F:
            raise RuntimeError(f"실측 창을 하나도 못 만들었습니다: {npz_dir}")
        self.fine = np.concatenate(F)
        self.wide = np.concatenate(W)
        self.p_observed = np.concatenate(P)
        self.v_observed = np.concatenate(V)
        self.obs_harm = np.concatenate(H)
        self.target_cycle = np.concatenate(C)
        self.stem = np.asarray(S)
        self.p_noise = np.full(len(self.fine), NOISE_FLOOR_EXTERNAL_W, np.float32)

    # ── 내부 ────────────────────────────────────────────────────────────
    @staticmethod
    def _to_33ch(raw: dict) -> np.ndarray:
        """npz -> (33, N). 합성기가 주는 배치와 같은 채널 배열로 맞춘다."""
        hr = np.asarray(raw["harmonics_ri"], np.float32)          # (N,15,2)
        pf = np.asarray(raw["power_features"], np.float32)        # (N,6) p,q,s,pf,v,thd
        n = hr.shape[0]
        x = np.empty((33, n), np.float32)
        x[0:15] = hr[:, :, 0].T
        x[15:30] = hr[:, :, 1].T
        x[30], x[31], x[32] = pf[:, 0], pf[:, 1], pf[:, 4]        # P, Q, V
        return x

    def _windows(self, x: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = self.window_cycles
        off = self.target_in_window
        cut = np.stack([x[:, t - off:t - off + w] for t in targets])   # (n,33,W)
        return build_inputs(cut)

    # ── 사용 ────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.fine)

    def batch(self, idx: np.ndarray) -> Tuple[np.ndarray, ...]:
        i = np.asarray(idx)
        return (self.fine[i], self.wide[i], self.p_observed[i],
                self.obs_harm[i], self.p_noise[i])

    def describe(self) -> str:
        rows = [f"  {k:8s} {v:>6,}창" for k, v in sorted(self.per_file.items())]
        return (f"실측 창 {len(self):,}개 (창 {self.window_cycles}, 타깃 {self.target_in_window})\n"
                + "\n".join(rows))


def dense_targets(stem: str, npz_dir: str = DEFAULT_DIR,
                  window_cycles: int = WINDOW_CYCLES, stride: int = 30) -> RealWindows:
    """파일 하나를 촘촘히(기본 0.5초 간격) 잘라 낸다. 실측 채점용."""
    return RealWindows(npz_dir=npz_dir, stems=[stem],
                       window_cycles=window_cycles, stride=stride, require_valid=False)


def upsample_to_cycles(values: np.ndarray, targets: np.ndarray, n_cycles: int) -> np.ndarray:
    """stride 로 띄엄띄엄 낸 예측을 사이클 단위로 펴 준다 (**최근접**).

    `score_on_off` / `score_events` 가 (n_cycles, K) 를 요구하는데, 창을 사이클마다
    만들면 파일당 9만 창이라 비현실적이다.

    [⚠ 앞으로 채우면 안 된다 — 12.52 절]
    처음에는 앞으로 채웠고("타임라인 자체가 초 단위라 잃는 것이 없다") 그 전제가
    **핫플레이트에서 깨진다.** 릴레이 펄스가 1.0~1.3초인데 stride 30(0.5초)을 앞으로
    채우면 모든 펄스가 평균 0.25초씩 **뒤로 밀린다** — 편향된 오차다.

        adapt_ph1 핫플 F1   앞으로 채움 0.838  vs  stride 6(0.1초) 0.965

    다른 기기는 전이가 드물어 영향이 없다 (유령 6.49->6.50, 오븐 0.925 불변).
    **최근접**은 오차가 절반이고 편향이 없다. 창을 더 만들지 않고 고칠 수 있는
    부분은 여기까지다 — 남는 것은 stride 를 줄여야 한다.
    """
    values = np.asarray(values)
    out = np.zeros((n_cycles,) + values.shape[1:], values.dtype)
    if not len(targets):
        return out
    c = np.arange(n_cycles)
    hi = np.searchsorted(targets, c, side="left")
    lo = np.clip(hi - 1, 0, len(targets) - 1)
    hi = np.clip(hi, 0, len(targets) - 1)
    take_hi = np.abs(targets[hi] - c) < np.abs(c - targets[lo])
    return values[np.where(take_hi, hi, lo)]
