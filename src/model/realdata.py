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
import os
import warnings

import numpy as np

from src.evaluation.sealing import assert_not_sealed, is_sealed
from src.model.inputs import build_inputs, target_index
from src.preprocessing import load_nilm_npz
from src.preprocessing.file_registry import NOISE_FLOOR_EXTERNAL_W

DEFAULT_DIR = "processed_data/composite_eval"
WINDOW_CYCLES = 3600
SAMPLING_HZ = 60.0

#: 사람이 스위치를 누르며 적은 라벨만 지도에 쓴다 (`_label_provenance_levels`).
HUMAN_PROVENANCE = "human_switching_log"
SMPS_APPLIANCES = ("beam_projector", "laptop_charger", "minipc")

#: 지도에 쓰는 사람 라벨 파일 — **SMPS 3종이 들어 있는 다섯 개만**.
#:
#: `test_11` / `test_12` 도 `human_switching_log` 지만 **저항 4종뿐이라
#: SMPS 가 하나도 없다.** 그 둘은 규칙 20 의 대조 파일이다 — "이 처방은 SMPS
#: 배분만 겨냥하므로 SMPS 없는 파일은 거의 안 변해야 한다" 가 판정 기준에
#: 들어간다. 그 파일까지 지도하면 대조가 죽는다. 실제로 열어 보면 scope=smps
#: 에서도 두 파일에 846 개 셀이 붙는데 전부 OFF 라벨이라 '유령을 지워라' 라는
#: 강한 감독이 된다 — 대조가 오히려 가장 많이 움직인다.
HUMAN_ON_DEFAULT_STEMS = ("test_5", "test_6", "test_7", "test_8", "test_13")

#: 대조로 남겨야 하는 파일. 여기에 지도가 붙으면 경고한다.
HUMAN_ON_CONTROL_STEMS = ("test_9", "test_11", "test_12")


# ── 장소 전달비 입력 보정 (12.179 / 12.181) ─────────────────────────────────
# 같은 SMPS 가 같은 전력으로 켜져도 콘센트의 전압 고조파에 따라 h9~h15 전류 페이저가
# 돈다 (장소 B: h11 이상 위상 +100°, |h13~h15| 2~3배). 합성은 격리 녹화 장소(= 장소 A)
# 환경만 알고, 모델의 미니PC p_raw 머리와 드라이기 게이트가 그 차수를 읽는다.
# `T_h = median(실측)/median(합성)` 를 단일 SMPS 창에서 재서(`run_site_transfer_probe`)
# 실측 페이저를 `I_h / T_h` 로 되돌린 뒤 지금처럼 입력을 만든다. h1 은 안 건드린다.
#
# **체크포인트가 자기 입력 프레임을 안다.** 2단계가 이 보정으로 적응했으면 그 값을
# 체크포인트에 넣고(`SITE_TRANSFER_KEY`), 채점기·실시간 추론이 같은 보정을 자동으로
# 건다. 프레임이 어긋나면 조용히 틀리므로(12.45.3 과 같은 부류) 여기에 묶는다.
SITE_TRANSFER_KEY = "site_transfer"


def load_site_transfer(path: str) -> np.ndarray:
    """(15,) complex. `run_site_transfer_probe` 가 저장한 `T_h`. h1 은 1 로 고정한다."""
    T = np.asarray(np.load(path), dtype=np.complex128).reshape(-1)
    if T.shape[0] != 15:
        raise ValueError(f"전달비는 15차수여야 합니다: {path} -> {T.shape}")
    T = T.copy()
    T[0] = 1.0 + 0j                     # 전력 라벨의 기준인 기본파는 절대 안 건드린다
    if np.any(np.abs(T) < 1e-6):
        raise ValueError(f"전달비에 0 이 있습니다: {path}")
    return T


def apply_site_transfer(harmonics_ri: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(N,15,2) 실측 페이저를 합성(장소 A) 프레임으로 되돌린다: `I_h <- I_h / T_h`."""
    hr = np.asarray(harmonics_ri, np.float32)
    c = (hr[..., 0].astype(np.complex128) + 1j * hr[..., 1]) / np.asarray(T)[None, :]
    out = np.empty_like(hr)
    out[..., 0], out[..., 1] = c.real, c.imag
    return out


def site_transfer_from_ckpt(ck: dict) -> Optional[Dict[str, np.ndarray]]:
    """체크포인트의 `site_transfer` -> {stem: T}. 없으면 None."""
    st = ck.get(SITE_TRANSFER_KEY)
    if not st:
        return None
    T = np.asarray(st["T"], dtype=np.float64)                      # (15, 2)
    Tc = T[:, 0] + 1j * T[:, 1]
    return {s: Tc for s in st["stems"]}


def site_transfer_to_ckpt(mapping: Dict[str, np.ndarray], path: str) -> dict:
    """{stem: T} -> 체크포인트에 넣을 값. 한 장소(같은 T)만 담는다."""
    stems = sorted(mapping)
    T = mapping[stems[0]]
    if any(not np.allclose(mapping[s], T) for s in stems):
        raise ValueError("한 체크포인트에는 한 장소의 전달비만 담는다")
    return {"npz": path, "stems": stems, "T": np.stack([T.real, T.imag], 1).tolist()}


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
        appliances: Optional[Sequence[str]] = None,
        human_on_scope: str = "off",
        human_on_stems: Optional[Sequence[str]] = None,
        human_on_shuffle: bool = False,
        site_transfer: Optional[Dict[str, np.ndarray]] = None,
    ):
        """`exclude` 는 적응에서 뺄 파일이다 (leave-one-file-out).

        `site_transfer` 는 {stem: T_h (15,) complex} — 그 파일의 페이저를 `1/T_h` 로
        되돌린다 (위 `SITE_TRANSFER_KEY` 주석). 입력 33채널과 `obs_harm` 둘 다에
        건다: `L_harm` 의 지문 `sig` 가 장소 A 프레임이므로 관측도 같은 프레임이어야 한다.

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

        F, W, P, H, S, C, V, Q = [], [], [], [], [], [], [], []
        self.per_file: Dict[str, int] = {}
        self.site_transfer_stems: List[str] = []
        for stem in stems:
            raw = load_nilm_npz(str(d / f"{stem}.npz"))
            if site_transfer and stem in site_transfer:
                raw = {k: raw[k] for k in raw}
                raw["harmonics_ri"] = apply_site_transfer(raw["harmonics_ri"], site_transfer[stem])
                self.site_transfer_stems.append(stem)
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
            # 무효전력. 12.133 이 찾은 **두 번째 판별자** — `L_cons` 의 Q 판이 쓴다.
            # 열 1 이 Q 다 ([p, q, s, pf, vrms, thd_i]).
            Q.append(np.asarray(raw["power_features"])[targets, 1].astype(np.float32))
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
        self.q_observed = np.concatenate(Q)
        self.obs_harm = np.concatenate(H)
        self.target_cycle = np.concatenate(C)
        self.stem = np.asarray(S)
        self.p_noise = np.full(len(self.fine), NOISE_FLOOR_EXTERNAL_W, np.float32)
        self._build_human_on(appliances, human_on_scope, human_on_stems,
                             shuffle=human_on_shuffle)

    # ── 사람 스위칭 로그 지도 (2026-08-31, SMPS_PLAN 4.5절) ─────────────────
    # **이 저장소는 사람이 적은 on/off 정답을 채점에만 썼다.** `run_adapt` 는
    # `crit.unlabeled` 만 부르고, 그 손실에는 기기별 정답이 한 항도 없다 —
    # `--w-hedge` 의 도움말이 *"실측은 라벨이 없어 BCE 가 확신을 강제하지 못한다"*
    # 라고 적은 그 상황이다. 그런데 정답이 있다: `_label_provenance` 가
    # `human_switching_log` 인 파일 다섯 개(test_5/6/7/8/13, 3,499초)다.
    #
    # **전력 정답은 여전히 없다** — 로그는 on/off 만 준다. 그러니 걸 수 있는 것은
    # `on_logit` 하나뿐이고, 마침 지금 무너지는 지표가 정확히 on/off 다
    # (충전기 재현 0.641).
    #
    # [범위를 왜 가르는가 — 규칙 3]
    # 파일에 아예 없는 기기는 정답이 OFF 로 확정이다(`appliances_present`,
    # `score_absent` 가 쓰는 그 필드). 그것까지 지도하면 '유령을 지우는' 효과와
    # 'SMPS 를 가르는' 효과가 섞여 무엇이 움직였는지 못 읽는다. 그래서:
    #     smps     SMPS 3종 열만            <- 가설 그대로. 기본값
    #     present  그 파일에 있던 기기만     <- 유령 억제 없이 검출만
    #     all      9종 전부 (없는 기기=OFF)  <- 유령 억제까지
    #
    # [uncertain 은 양쪽 다 안 센다]
    # `build_on_off_truth` 가 채점에서 쓰는 규칙을 그대로 쓴다. 오븐이 대표적이다
    # (타임라인은 히터 통전만 적혀 있고 팬/조명 구간은 알 수 없다).
    def _build_human_on(self, appliances, scope: str, stems,
                        shuffle: bool = False) -> None:
        """(N,K) 사람 라벨 `human_on` 과 감독 마스크 `human_mask` 를 만든다.

        `human_mask` 가 0 이면 그 (창, 기기) 는 손실에서 빠진다. scope="off" 면
        전부 0 이라 기존 동작과 **글자 그대로 같다.**
        """
        n, k = len(self.fine), (len(appliances) if appliances else 0)
        self.human_on = np.zeros((n, max(k, 1)), np.float32)
        self.human_mask = np.zeros((n, max(k, 1)), np.float32)
        self.human_stems: List[str] = []
        if scope == "off" or not appliances:
            return
        if scope not in ("smps", "present", "all"):
            raise ValueError(f"human_on_scope 는 off|smps|present|all 입니다: {scope}")

        from src.evaluation.real_events import build_on_off_truth, load_events
        ev = load_events()
        allow = set(stems) if stems is not None else set(HUMAN_ON_DEFAULT_STEMS)
        bad = allow & set(HUMAN_ON_CONTROL_STEMS)
        if bad:
            warnings.warn(
                f"대조 파일에 사람 라벨 지도가 걸렸습니다: {sorted(bad)}. "
                "규칙 20 의 대조가 죽습니다 - 판정 기준을 다시 보십시오.",
                RuntimeWarning, stacklevel=2)
        for stem in self.stems:
            spec = ev.get(stem)
            if spec is None:
                continue
            # **사람 로그만 쓴다.** ai_inferred 는 모델이 만든 것이라 자기지도가 된다.
            if spec.get("_label_provenance") != HUMAN_PROVENANCE:
                continue
            if allow is not None and stem not in allow:
                continue
            sel = np.flatnonzero(self.stem == stem)
            if not len(sel):
                continue
            tc = self.target_cycle[sel]
            on, scorable = build_on_off_truth(stem, appliances, int(tc.max()) + 1, ev)
            present = set(spec.get("appliances_present", []))
            cols = np.zeros(k, bool)
            for j, app in enumerate(appliances):
                if scope == "all":
                    cols[j] = True
                elif scope == "present":
                    cols[j] = app in present
                else:                                   # smps
                    cols[j] = app in SMPS_APPLIANCES
            ho = on[tc].astype(np.float32)
            if shuffle:
                # ── 귀무 대조 (규칙 3) ──────────────────────────────────
                # **라벨이 정보를 나르는가, 아니면 BCE 항 자체가 게이트를
                # 규제하는가.** 창 축을 파일 길이의 37% 만큼 순환 이동한다.
                # ON 비율과 구간 길이 분포는 **글자 그대로 보존**되고 시각
                # 대응만 깨진다. 이쪽에서도 같은 이득이 나오면 처방은 라벨이
                # 아니라 정규화이고, 그러면 12.87.3 을 못 건드린다.
                ho = np.roll(ho, int(len(ho) * 0.37), axis=0)
            self.human_on[sel] = ho
            self.human_mask[sel] = (scorable[tc] & cols[None, :]).astype(np.float32)
            self.human_stems.append(stem)

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

    def voltage(self, idx: np.ndarray) -> np.ndarray:
        """관측 단자전압 (n,). h1 지문의 전압 보정이 쓴다 (12.151).

        **`batch()` 의 자리수를 안 바꾼다** — `reactive()` 와 같은 이유다."""
        return self.v_observed[np.asarray(idx)]

    def reactive(self, idx: np.ndarray) -> np.ndarray:
        """관측 무효전력 (n,). **`batch()` 의 자리수를 안 바꾼다** — `human()` 과
        같은 이유다 (`run_gate_check` 등이 5개로 풀고 있다)."""
        return self.q_observed[np.asarray(idx)]

    def human(self, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(human_on, human_mask) 만 따로. **`batch()` 의 자리수를 안 바꾼다** —
        `run_gate_check` / `run_ablation_probe` 가 5개로 풀고 있어서다."""
        i = np.asarray(idx)
        return self.human_on[i], self.human_mask[i]

    def human_coverage(self) -> str:
        """지도에 실제로 들어가는 (창 x 기기) 가 몇 개인지. 판정 전에 찍어 둔다."""
        m = self.human_mask
        if not m.any():
            return "  사람 라벨 지도: 꺼짐"
        rows = []
        for stem in self.human_stems:
            sel = self.stem == stem
            rows.append(f"    {stem:8s} {int(m[sel].sum()):>7,} (창 {int(sel.sum()):,})")
        return (f"  사람 라벨 지도: {len(self.human_stems)}파일, "
                f"{int(m.sum()):,} (창x기기), ON 비율 "
                f"{float((self.human_on * m).sum() / max(m.sum(), 1)):.3f}\n"
                + "\n".join(rows))

    def describe(self) -> str:
        rows = [f"  {k:8s} {v:>6,}창" for k, v in sorted(self.per_file.items())]
        return (f"실측 창 {len(self):,}개 (창 {self.window_cycles}, 타깃 {self.target_in_window})\n"
                + "\n".join(rows))


def dense_targets(stem: str, npz_dir: str = DEFAULT_DIR,
                  window_cycles: int = WINDOW_CYCLES, stride: int = 30,
                  site_transfer: Optional[Dict[str, np.ndarray]] = None) -> RealWindows:
    """파일 하나를 촘촘히(기본 0.5초 간격) 잘라 낸다. 실측 채점용."""
    return RealWindows(npz_dir=npz_dir, stems=[stem],
                       window_cycles=window_cycles, stride=stride, require_valid=False,
                       site_transfer=site_transfer)


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


# ── 교차주파수 어드미턴스 보정 (12.148) ──────────────────────────────────────
def harmonic_offset(stems, target_cycle, coef_npz: str, n_harm: int = 15,
                    z_npz: str = ""):
    """창별 `L_harm` 보정 (n, H, 2). `run_norton_probe --save-coef` 의 산출물.

    `harmonic_signatures` 는 **fixed current injection** 모형이다 — 기기 전류가
    계통 조건과 무관하다고 본다. 문헌이 그 실패를 오래 전에 적었고(attenuation &
    diversity) 표준 처방이 Norton 등가다: `I_h = I_source,h − Y_h·V_h`.

    **전압 위상이 없어도 된다** (12.148.2). `V_h = V_src,h − Z_h·I_h` 에서
    배경 `V_src` 는 그 집의 상수라 회귀 절편이 흡수하고, **변하는 부분 `Z·I`**
    의 위상은 이미 기록된 전류 위상(`ihdeg`)이다. `Z` 는 물리 제약
    `R + j·h·ωL` 로 미지수 둘만 두고 `|V_h|` 와 복소 `I_h` 로 푼다.

        크기만 (12.148)    잔차 8파일 중 6개 악화, 배분 14.3W
        **복소 Z·I**       잔차 7/8 개선 (−21%), 배분 **6.7W**

    ⚠ 원자료 CSV 에서 `ih*`/`ihdeg*` 를 읽는다 — 전처리가 위상을 안 남긴다.
      npz 와 CSV 의 행이 어긋나므로 `vrms` 상관으로 시프트를 찾는다.
    ⚠ 짝수차는 0 이다 (12.72 전류 인공물 + 12.147 전압 짝수차 미결).

    `z_npz` — **현장 임피던스로 갈아끼운다** (`run_fit_impedance --out` 의 산출물).
    `Z` 는 그 집 배선의 값이라 장소가 바뀌면 3.2배까지 다르다 (12.148.2 전이 시험).
    기울기는 물리적으로 `ΣY`(기기들의 어드미턴스 합)라 **기기 구성이 같으면
    전이될 것으로 보지만 확인 안 됐다** — 다른 집 라벨이 없어 못 쟀다.

    ⚠ `Z` 만 갈아도 **상수항은 학습 장소 것이 남는다.** 표준화의 `mu` 가 그 집
       평균이고, 거기에 배경 `V_src` 효과가 섞여 있다 (보정 효과의 47%). `V_src` 가
       장소 간 2.1배 다르므로 그 부분은 전이가 안 된다 (12.150.1).
    """
    import pandas as pd

    from src.preprocessing.raw_csv import read_raw_csv

    z = np.load(coef_npz, allow_pickle=True)
    B = np.asarray(z["coef"], np.float64)
    mu, sd = np.asarray(z["mu"], np.float64), np.asarray(z["sd"], np.float64)
    odd = np.asarray(z["orders"], int)                 # 출력 차수 (0-based)
    zo = np.asarray(z["zi_orders"], int)               # 회귀 차수 (1-based)
    R, X1 = float(z["R"]), float(z["X1"])
    if z_npz:
        zz = np.load(z_npz, allow_pickle=True)
        R, X1 = float(zz["R"]), float(zz["X1"])
    out = np.zeros((len(target_cycle), n_harm, 2), np.float32)
    stems = np.asarray(stems)
    for stem in np.unique(stems):
        csv_p, npz_p = f"data/{stem}.csv", f"{DEFAULT_DIR}/{stem}.npz"
        if not (os.path.exists(csv_p) and os.path.exists(npz_p)):
            continue                                   # 원자료가 없으면 보정 0
        cols = ["vrms"] + [f"ih{h}" for h in zo] + [f"ihdeg{h}" for h in zo]
        # ⚠ **정본 순서로 읽어야 한다** (12.152). 원본 CSV 는 Wi-Fi 재전송 때문에
        #   패킷 순서가 뒤바뀌어 있고(전 파일 3,226곳), npz 는 이미 정렬돼 있다.
        #   정렬 안 하면 아래 vrms 상관이 그만큼 흐려진다 — test.2 0.797 /
        #   test3 0.878 로 11파일 중 최저 둘이었던 것이 정확히 가장 엉킨 둘이다.
        csv, _ = read_raw_csv(csv_p, usecols=cols)
        nv = np.asarray(load_nilm_npz(npz_p)["power_features"])[:, 4]
        best = (-2.0, 0)
        for sh in range(-400, 401, 10):
            a = csv.vrms.values[max(0, -sh):]; b = nv[max(0, sh):]
            n = min(len(a), len(b))
            if n < 100:
                continue
            c = float(np.corrcoef(a[:n], b[:n])[0, 1])
            if c > best[0]:
                best = (c, sh)
        m = stems == stem
        j = np.clip(target_cycle[m] - best[1], 0, len(csv) - 1)
        cc = []
        for h in zo:
            I = (csv[f"ih{h}"].values[j]
                 * np.exp(1j * np.deg2rad(csv[f"ihdeg{h}"].values[j])))
            zi = (R + 1j * h * X1) * I                 # 부하가 만든 전압 왜곡
            cc += [zi.real, zi.imag]
        X = np.c_[np.ones(len(j)), np.array(cc).T]
        y = ((X - mu) / sd) @ B
        k = len(odd)
        buf = np.zeros((int(m.sum()), n_harm, 2), np.float32)
        buf[:, odd, 0] = y[:, :k]; buf[:, odd, 1] = y[:, k:]
        out[m] = buf
    return out
