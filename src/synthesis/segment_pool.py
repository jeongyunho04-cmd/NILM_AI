"""
가전 활성화 세그먼트 풀 및 대기전력 물리 모델 (Segment Pool & Standby Model)
=============================================================================
정제된 .npz 로부터 (1) 가전별 실제 동작 구간, (2) 플러그만 꽂힌 대기 상태의 전기적 지문,
(3) 무부하 배경 노이즈를 분리해 합성 엔진에 공급한다.

[대기전력을 제대로 분리해야 하는 이유]
NILM 모델이 가장 자주 저지르는 오답이 "여러 기기의 대기전력 합"을 "저전력 기기 1대가
켜진 것"으로 오인하는 것이다. 미니PC 아이들이 9.8W, 선풍기 1단이 23.5W 인데
대기전력이 기기당 3W 씩만 잡혀도 9대면 27W 가 되어 실제 기기보다 커진다.

이전 구현은 이 함정에 그대로 빠져 있었다.
  - 대기전력을 power_features[:,0](= 계측 바닥 노이즈가 포함된 p_w)의 '평균'으로 계산
  - 그 결과 계측 보드 자체 소비(1.4W ~ 2.37W)가 기기마다 한 번씩, 총 9번 더해짐
  - 평균이라 상태 전이 구간이 섞여 핫플레이트 대기전력이 8.44W 로 부풀음(실제 중앙값 2.1W)
  - 전 기기 플러그인 시 유령 대기전력 26.7W 발생 -> 미니PC 한 대보다 큰 값

[이 모듈이 적용하는 기준]
계측 보드의 자체 전류/전력은 '기기의 것'이 아니라 '계측계의 것'이다. 따라서
    기기 활성 전류 = 측정 전류(ON)  - 노이즈 기준 전류
    기기 대기 전류 = 측정 전류(OFF) - 노이즈 기준 전류
로 분리하고, 노이즈는 합성 단계에서 딱 한 번만 더한다.
전력도 같은 원리로 p_denoised_w(계측 바닥 제거본)의 '중앙값'을 쓴다.
그 결과 선풍기/전기포트처럼 기계식 스위치를 쓰는 기기는 대기전력 0W 로,
에어컨처럼 대기 회로가 있는 기기는 약 4.7W 로 물리적으로 맞게 잡힌다.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.preprocessing.numpy_exporter import load_nilm_npz
from src.preprocessing.file_registry import FileRole, get_load_class, is_periodic_duty

# 활성화로 인정할 최소 길이 (0.5초)
MIN_ACTIVATION_CYCLES = 30
# 대기 지문을 신뢰하려면 최소 이만큼의 OFF 샘플이 필요하다 (1초)
MIN_STANDBY_SAMPLES = 60
# 대기 지터 풀로 보관할 최대 샘플 수 (메모리 상한)
MAX_JITTER_POOL_SAMPLES = 36000  # 10분
# 이 값을 넘으면 '진짜 대기 회로가 있다'고 본다. 아래는 계측 잔차로 취급한다.
STANDBY_POWER_THRESHOLD_W = 0.5


@dataclass
class ApplianceActivation:
    """가전 1회 연속 동작 구간."""
    appliance_type: str
    source_file: str
    korean_name: str
    duration_cycles: int
    duration_s: float

    # 계측 노이즈를 제거한 기기 순수 전류 고조파: (L, 15, 2) float32
    net_harmonics_ri: np.ndarray
    # 동일 내용의 복소 페이저: (L, 15) complex64
    net_harmonics_complex: np.ndarray
    # 전력 특징 [p, q, s, pf, vrms, thd_i] (L, 6) float32
    net_power_features: np.ndarray

    # 정답 라벨
    is_on: np.ndarray          # (L,) int8 (전부 1)
    state_id: np.ndarray       # (L,) int16
    target_power_w: np.ndarray # (L,) float32 (계측 바닥 제거됨)

    # 돌입 전류(inrush) 구간 길이. 시간 워핑 시 이 구간은 늘이지 않는다.
    inrush_cycles: int

    # 이 파형이 실제로 녹화될 때의 계통 전압. 전압 환산의 기준(kappa = V_bus / v_ref).
    v_ref_v: float = 220.0
    # 서모스탯/릴레이로 주기적 ON-OFF 를 반복하는 부하인가 (시간 워핑 방식이 달라진다)
    periodic_duty: bool = False


@dataclass
class StandbyProfile:
    """플러그는 꽂혀 있고 전원은 꺼진 상태의 전기적 지문.

    여기 담긴 값은 모두 '기기 자신의 것'이다. 계측 보드 자체 소비는 이미 제거되어 있으며,
    합성 단계에서 배경 노이즈로 딱 한 번만 더해진다.
    """
    appliance_type: str
    harmonics_ri: np.ndarray       # (15, 2) float32
    harmonics_complex: np.ndarray  # (15,) complex64
    power_w: float                 # 기기 자체 대기 유효전력 (W)
    reactive_var: float            # 기기 자체 대기 무효전력 (VAR)
    sample_count: int              # 이 값을 만든 OFF 샘플 수 (신뢰도 지표)
    v_ref_v: float = 220.0         # 이 지문이 녹화될 때의 계통 전압 (전압 환산 기준)

    # 대기 상태의 미세 변동. 완전히 일정한 값은 현실에도 없고,
    # 모델에게 "변하지 않으면 대기전력"이라는 가짜 단서를 주게 되므로 실측 잔차를 보관한다.
    jitter_pool: np.ndarray = field(default_factory=lambda: np.zeros((0, 15), dtype=np.complex64))
    jitter_power: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # 잔차 중 계측 노이즈가 아닌 '기기 자신의 흔들림' 비율 (0~1)
    jitter_scale: float = 0.0

    @property
    def is_true_standby(self) -> bool:
        """실제로 대기 회로를 갖고 전력을 소비하는 기기인가.

        선풍기/전기포트/드라이기처럼 기계식 스위치를 쓰면 꺼진 순간 회로가 물리적으로
        끊겨 대기전력이 0 이고, 남는 것은 계측 잔차뿐이다. 실측값도 이 경계에서 갈린다.
            기계식 스위치: 노트북충전기 0.01W, 전기포트 0.18W, 선풍기 0.26W, 드라이기 0.33W
            대기 회로 보유: 오븐 0.64W, 핫플레이트 0.65W, 빔프로젝터 0.88W, 에어컨 4.83W
        """
        return self.power_w > STANDBY_POWER_THRESHOLD_W


@dataclass
class NoiseReference:
    """계측계 자체의 무부하 기준값. 기기의 것이 아니라 계측계의 것이다."""
    name: str
    noise_floor_w: float
    median_phasor: np.ndarray      # (15,) complex64
    harmonics_ri: np.ndarray       # (N, 15, 2)
    harmonics_complex: np.ndarray  # (N, 15)
    power_features: np.ndarray     # (N, 6)
    residual_variance: float       # 잔차 분산 (기기 지터에서 노이즈 몫을 빼는 데 쓴다)


class SegmentPool:
    """가전 활성화 구간, 대기 지문, 배경 노이즈를 관리한다."""

    def __init__(
        self,
        npz_dir: Union[str, Path] = "processed_data/npz",
        strict_role_check: bool = True,
    ):
        self.npz_dir = Path(npz_dir)
        self.strict_role_check = strict_role_check

        self.appliance_activations: Dict[str, List[ApplianceActivation]] = {}
        self.standby_profiles: Dict[str, StandbyProfile] = {}
        self.noise_references: Dict[str, NoiseReference] = {}
        self.rejected_files: List[Tuple[str, str]] = []  # (파일명, 거부 사유)

        # 구버전 호환 속성
        self.noise_pool: Optional[np.ndarray] = None
        self.noise_complex: Optional[np.ndarray] = None
        self.noise_power: Optional[np.ndarray] = None

        self.load_all_npz_files()

    # ── 적재 ────────────────────────────────────────────────────────────────
    def load_all_npz_files(self):
        """.npz 를 전부 읽어 활성화 구간, 대기 지문, 노이즈 기준을 추출한다."""
        npz_files = sorted(self.npz_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {self.npz_dir}. Run preprocessing first!")

        loaded = [(f, load_nilm_npz(f)) for f in npz_files]

        # 1차: 배경 노이즈 기준을 먼저 만든다.
        #      기기의 순수 전류를 뽑으려면 계측계 기준값이 먼저 있어야 한다.
        device_entries = []
        for f, data in loaded:
            meta = data.get("metadata", {})
            role = meta.get("file_role")
            stem = f.stem

            # 복합 부하 실측이 세그먼트 풀에 들어오면 합성 정답이 통째로 망가진다.
            # 전처리 단계에서 이미 분리되지만, 여기서 한 번 더 막는다.
            if role == FileRole.COMPOSITE_EVAL.value:
                self.rejected_files.append((stem, "복합 부하 실측 파일 - 단일 가전이 아님"))
                continue
            if self.strict_role_check and role is None and "noise" not in stem:
                # 역할 메타데이터가 없는 구버전 npz. 이름으로 최소한의 방어만 한다.
                if stem.startswith("test") or stem.startswith("nilm_"):
                    self.rejected_files.append((stem, "역할 메타데이터 없음 + 복합 부하 의심 파일명"))
                    continue

            if role == FileRole.NOISE.value or "noise" in stem:
                self._register_noise_reference(stem, data, meta)
            else:
                device_entries.append((stem, data, meta))

        if not self.noise_references:
            raise ValueError(
                "배경 노이즈 기준 파일을 찾지 못했습니다. "
                "noise_noselfpower / noise_selfpower 를 먼저 전처리하세요."
            )

        # 2차: 기기별 활성화 구간과 대기 지문 추출
        raw_standby: Dict[str, List[StandbyProfile]] = {}
        for stem, data, meta in device_entries:
            appliance_type = meta.get("appliance_type", stem)
            korean_name = meta.get("korean_name", stem)
            v_ref = float(meta.get("v_ref_v", 220.0))
            noise_ref = self._pick_noise_reference(float(meta.get("noise_floor_w", 1.4)))

            profile = self._extract_standby_profile(appliance_type, data, noise_ref, v_ref)
            if profile is not None:
                raw_standby.setdefault(appliance_type, []).append(profile)

            self._extract_activations(
                stem, appliance_type, korean_name, data, noise_ref, v_ref
            )

        # 같은 가전 종류의 여러 파일에서 나온 대기 지문을 합친다.
        for app, profiles in raw_standby.items():
            self.standby_profiles[app] = self._merge_standby_profiles(app, profiles)

        self._build_legacy_noise_view()

    def _register_noise_reference(self, stem: str, data: dict, meta: dict):
        """무부하 노이즈 파일 하나를 계측계 기준값으로 등록한다."""
        hc = data["harmonics_complex"]
        median_phasor = (
            np.median(np.real(hc), axis=0) + 1j * np.median(np.imag(hc), axis=0)
        ).astype(np.complex64)
        residual = hc - median_phasor
        residual_var = float(np.mean(np.abs(residual) ** 2))

        self.noise_references[stem] = NoiseReference(
            name=stem,
            noise_floor_w=float(meta.get("noise_floor_w", np.median(data["power_features"][:, 0]))),
            median_phasor=median_phasor,
            harmonics_ri=data["harmonics_ri"],
            harmonics_complex=hc,
            power_features=data["power_features"],
            residual_variance=residual_var,
        )

    def _pick_noise_reference(self, noise_floor_w: float) -> NoiseReference:
        """이 기기 측정에 적용된 바닥 전력과 가장 가까운 노이즈 기준을 고른다.

        보드가 외부 전원이면 1.4W, 측정 회로에서 자체 전원을 뽑으면 2.37W 로
        계측계 자체 소비가 달라지므로 기준도 달라져야 한다.
        """
        return min(
            self.noise_references.values(),
            key=lambda r: abs(r.noise_floor_w - noise_floor_w),
        )

    # ── 대기 지문 추출 ──────────────────────────────────────────────────────
    def _extract_standby_profile(
        self, appliance_type: str, data: dict, noise_ref: NoiseReference, v_ref: float = 220.0
    ) -> Optional[StandbyProfile]:
        """OFF 구간에서 기기 자신의 대기 전기 지문을 뽑는다."""
        is_on = data["is_on"]
        idle_mask = is_on == 0
        # 품질 게이팅에 걸린 샘플은 제외한다.
        if "is_valid" in data:
            idle_mask = idle_mask & (data["is_valid"] == 1)

        n_idle = int(idle_mask.sum())
        if n_idle < MIN_STANDBY_SAMPLES:
            return None

        hc_idle = data["harmonics_complex"][idle_mask]

        # 평균이 아니라 중앙값을 쓴다. OFF 구간에는 켜지고 꺼지는 순간의 과도 샘플이
        # 반드시 섞여 들어오는데, 평균은 그 꼬리에 끌려간다.
        # (실측: 핫플레이트 평균 14.42W vs 중앙값 2.07W)
        measured_phasor = (
            np.median(np.real(hc_idle), axis=0) + 1j * np.median(np.imag(hc_idle), axis=0)
        ).astype(np.complex64)

        # 계측계 자체 전류를 빼야 '기기의 것'만 남는다.
        own_phasor = (measured_phasor - noise_ref.median_phasor).astype(np.complex64)

        # 전력도 같은 원리. p_denoised_w 는 계측 바닥이 이미 제거된 값이다.
        if "p_denoised_w" in data:
            p_idle = data["p_denoised_w"][idle_mask]
        else:  # 구버전 npz 대비
            p_idle = np.maximum(0.0, data["power_features"][idle_mask, 0] - noise_ref.noise_floor_w)
        own_power = float(np.median(p_idle))
        own_reactive = float(np.median(data["power_features"][idle_mask, 1]))

        # 대기 상태의 미세 변동(잔차). 계측계 잡음 몫을 뺀 나머지가 기기 자신의 흔들림이다.
        residual = (hc_idle - measured_phasor).astype(np.complex64)
        total_var = float(np.mean(np.abs(residual) ** 2))
        own_var = max(0.0, total_var - noise_ref.residual_variance)
        jitter_scale = float(np.sqrt(own_var / total_var)) if total_var > 1e-18 else 0.0

        if len(residual) > MAX_JITTER_POOL_SAMPLES:
            residual = residual[:MAX_JITTER_POOL_SAMPLES]
        p_residual = (p_idle[: len(residual)] - own_power).astype(np.float32)

        ri = np.zeros((15, 2), dtype=np.float32)
        ri[:, 0] = np.real(own_phasor)
        ri[:, 1] = np.imag(own_phasor)

        return StandbyProfile(
            appliance_type=appliance_type,
            harmonics_ri=ri,
            harmonics_complex=own_phasor,
            power_w=max(0.0, own_power),
            reactive_var=own_reactive,
            sample_count=n_idle,
            v_ref_v=v_ref,
            jitter_pool=residual,
            jitter_power=p_residual,
            jitter_scale=jitter_scale,
        )

    def _merge_standby_profiles(
        self, appliance_type: str, profiles: List[StandbyProfile]
    ) -> StandbyProfile:
        """같은 가전 종류의 여러 측정 파일에서 나온 대기 지문을 표본 수 가중으로 합친다."""
        if len(profiles) == 1:
            return profiles[0]

        weights = np.array([p.sample_count for p in profiles], dtype=np.float64)
        weights = weights / weights.sum()

        phasor = np.sum(
            [p.harmonics_complex * w for p, w in zip(profiles, weights)], axis=0
        ).astype(np.complex64)
        power = float(np.sum([p.power_w * w for p, w in zip(profiles, weights)]))
        reactive = float(np.sum([p.reactive_var * w for p, w in zip(profiles, weights)]))

        jitter_pool = np.concatenate([p.jitter_pool for p in profiles], axis=0)
        jitter_power = np.concatenate([p.jitter_power for p in profiles], axis=0)
        if len(jitter_pool) > MAX_JITTER_POOL_SAMPLES:
            jitter_pool = jitter_pool[:MAX_JITTER_POOL_SAMPLES]
            jitter_power = jitter_power[:MAX_JITTER_POOL_SAMPLES]

        ri = np.zeros((15, 2), dtype=np.float32)
        ri[:, 0] = np.real(phasor)
        ri[:, 1] = np.imag(phasor)

        return StandbyProfile(
            appliance_type=appliance_type,
            harmonics_ri=ri,
            harmonics_complex=phasor,
            power_w=max(0.0, power),
            reactive_var=reactive,
            sample_count=int(sum(p.sample_count for p in profiles)),
            v_ref_v=float(np.sum([p.v_ref_v * w for p, w in zip(profiles, weights)])),
            jitter_pool=jitter_pool,
            jitter_power=jitter_power,
            jitter_scale=float(np.sum([p.jitter_scale * w for p, w in zip(profiles, weights)])),
        )

    # ── 활성화 구간 추출 ────────────────────────────────────────────────────
    def _extract_activations(
        self,
        stem: str,
        appliance_type: str,
        korean_name: str,
        data: dict,
        noise_ref: NoiseReference,
        v_ref: float,
    ):
        """is_on == 1 인 연속 구간을 기기 순수 전류로 분리해 저장한다."""
        is_on = data["is_on"]
        on_indices = np.where(is_on == 1)[0]
        if len(on_indices) == 0:
            return

        blocks = np.split(on_indices, np.where(np.diff(on_indices) > 1)[0] + 1)
        bucket = self.appliance_activations.setdefault(appliance_type, [])
        periodic = is_periodic_duty(appliance_type)

        for block in blocks:
            if len(block) < MIN_ACTIVATION_CYCLES:
                continue
            start_i, end_i = block[0], block[-1] + 1

            raw_c = data["harmonics_complex"][start_i:end_i]
            raw_pow = data["power_features"][start_i:end_i]

            # 기기 순수 전류 = 측정 전류 - 계측계 자체 전류.
            # 이전 구현은 여기서 기기 자신의 대기 전류(idle 평균)를 뺐는데, 그것은
            # 기기가 켜져 있을 때도 흐르는 자기 전류이므로 빼면 안 되는 값이었다.
            net_c = (raw_c - noise_ref.median_phasor).astype(np.complex64)

            # 실수부/허수부는 위상에 따라 음수가 정상이다.
            # 이전 구현의 np.maximum(0, ...) 클리핑은 위상 평면의 절반을 잘라내는
            # 물리적으로 잘못된 연산이었다.
            net_ri = np.zeros((len(net_c), 15, 2), dtype=np.float32)
            net_ri[:, :, 0] = np.real(net_c)
            net_ri[:, :, 1] = np.imag(net_c)

            net_pow = raw_pow.copy()
            net_pow[:, 0] = np.maximum(0.0, net_pow[:, 0] - noise_ref.noise_floor_w)

            bucket.append(ApplianceActivation(
                appliance_type=appliance_type,
                source_file=stem,
                korean_name=korean_name,
                duration_cycles=len(block),
                duration_s=round(len(block) / 60.0, 2),
                net_harmonics_ri=net_ri,
                net_harmonics_complex=net_c,
                net_power_features=net_pow.astype(np.float32),
                is_on=data["is_on"][start_i:end_i],
                state_id=data["state_id"][start_i:end_i],
                target_power_w=data["target_power_w"][start_i:end_i],
                inrush_cycles=min(len(block) // 3, 60),
                v_ref_v=v_ref,
                periodic_duty=periodic,
            ))

    def _build_legacy_noise_view(self):
        """구버전 API 호환용 노이즈 배열을 만든다."""
        refs = list(self.noise_references.values())
        self.noise_pool = np.concatenate([r.harmonics_ri for r in refs], axis=0)
        self.noise_complex = np.concatenate([r.harmonics_complex for r in refs], axis=0)
        self.noise_power = np.concatenate([r.power_features for r in refs], axis=0)

    # ── 조회 API ────────────────────────────────────────────────────────────
    def get_appliance_types(self) -> List[str]:
        """사용 가능한 가전 종류 목록."""
        return sorted(self.appliance_activations.keys())

    def sample_activation(self, appliance_type: str) -> ApplianceActivation:
        """지정한 가전의 활성화 구간을 무작위로 하나 뽑는다."""
        acts = self.appliance_activations.get(appliance_type, [])
        if not acts:
            raise ValueError(f"No activations available for appliance type '{appliance_type}'")
        return acts[int(np.random.randint(0, len(acts)))]

    def get_standby_profile(self, appliance_type: str) -> StandbyProfile:
        """가전의 대기 전기 지문을 반환한다 (없으면 0 대기전력)."""
        if appliance_type in self.standby_profiles:
            return self.standby_profiles[appliance_type]
        return StandbyProfile(
            appliance_type=appliance_type,
            harmonics_ri=np.zeros((15, 2), dtype=np.float32),
            harmonics_complex=np.zeros(15, dtype=np.complex64),
            power_w=0.0,
            reactive_var=0.0,
            sample_count=0,
        )

    def sample_standby_series(
        self, appliance_type: str, length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """대기 상태의 전류 페이저/전력 시계열을 생성한다.

        완전히 일정한 값을 쓰면 "전혀 변하지 않는 성분 = 대기전력"이라는,
        합성 데이터에만 존재하는 단서를 모델이 학습해 버린다. 실측 OFF 구간의
        잔차를 이어 붙여 실제와 같은 미세 변동을 준다.
        """
        profile = self.get_standby_profile(appliance_type)
        base_c = np.tile(profile.harmonics_complex, (length, 1)).astype(np.complex64)
        base_p = np.full(length, profile.power_w, dtype=np.float32)

        pool = profile.jitter_pool
        if profile.jitter_scale <= 0.0 or len(pool) < 2 or length <= 0:
            return base_c, base_p

        # 시간 상관을 유지하려면 무작위 샘플이 아니라 연속 구간을 잘라 써야 한다.
        if length >= len(pool):
            reps = int(np.ceil(length / len(pool)))
            jitter_c = np.tile(pool, (reps, 1))[:length]
            jitter_p = np.tile(profile.jitter_power, reps)[:length]
        else:
            start = int(np.random.randint(0, len(pool) - length))
            jitter_c = pool[start:start + length]
            jitter_p = profile.jitter_power[start:start + length]

        base_c = base_c + jitter_c * profile.jitter_scale

        # 전력 지터는 평균을 0 으로 맞춘 뒤 대기전력 크기 안으로 대칭 제한한다.
        # 그냥 max(0, ...) 으로 자르면 음수 쪽 잔차만 잘려 평균이 위로 밀린다.
        # (실측에서 합계 8.2W 대기전력이 12.7W 로 부풀어 오르던 원인)
        jp = (jitter_p * profile.jitter_scale).astype(np.float32)
        jp = jp - jp.mean()
        limit = max(profile.power_w, 1e-6)
        jp = np.clip(jp, -limit, limit)
        base_p = base_p + jp  # limit 제한 덕분에 음수가 될 수 없다

        return base_c.astype(np.complex64), base_p.astype(np.float32)

    def sample_noise_slice(self, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """배경 노이즈 연속 구간을 뽑는다. 합성 전체에서 딱 한 번만 더해야 한다."""
        # 노이즈 파일을 섞어 쓰면 서로 다른 계측 조건이 한 구간 안에서 튀므로
        # 파일 하나를 고른 뒤 그 안에서만 자른다.
        ref = self.noise_references[
            list(self.noise_references.keys())[int(np.random.randint(0, len(self.noise_references)))]
        ]
        total_n = len(ref.harmonics_ri)

        if length >= total_n:
            reps = int(np.ceil(length / total_n))
            return (
                np.tile(ref.harmonics_ri, (reps, 1, 1))[:length],
                np.tile(ref.harmonics_complex, (reps, 1))[:length],
                np.tile(ref.power_features, (reps, 1))[:length],
            )

        start = int(np.random.randint(0, total_n - length))
        end = start + length
        return (
            ref.harmonics_ri[start:end],
            ref.harmonics_complex[start:end],
            ref.power_features[start:end],
        )

    # ── 진단 ────────────────────────────────────────────────────────────────
    def describe(self) -> Dict[str, dict]:
        """풀 구성 요약. 활성화 다양성이 부족한 가전을 확인하는 데 쓴다."""
        out = {}
        for app in self.get_appliance_types():
            acts = self.appliance_activations[app]
            sp = self.get_standby_profile(app)
            out[app] = {
                "activations": len(acts),
                "total_minutes": round(sum(a.duration_s for a in acts) / 60.0, 1),
                "source_files": sorted({a.source_file for a in acts}),
                "v_ref_range": [
                    round(min(a.v_ref_v for a in acts), 1),
                    round(max(a.v_ref_v for a in acts), 1),
                ],
                "load_class": get_load_class(app).value,
                "periodic_duty": is_periodic_duty(app),
                "standby_w": round(sp.power_w, 3),
                "standby_is_real": sp.is_true_standby,
                "standby_samples": sp.sample_count,
            }
        return out
