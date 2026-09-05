# -*- coding: utf-8 -*-
"""전압 텍스처 라이브러리 — 2Hz 녹화의 전압 고조파(vh·vhdeg)에서 (2026-09-06, 12.187).

무엇인가
--------
커패시터 입력 정류기(SMPS)의 전류 고조파는 전압 **파형**에 초선형으로 반응한다 (옛 12.185.16:
h17+ 의 0.27% 가 h9~h15 를 14%·20~37° 흔든다; 12.185.22: 조합-단독 불일치 13.5% 중 텍스처가 2.7% 로
줄이는 몫이 지배적). 그 파형이 '텍스처' 다. 옛 계측기에서는 원시 스냅샷(18개, 삭제)에서만 얻었는데,
새 계측기의 2Hz CSV 는 모든 0.5초 블록에 **위상까지 있는 전압 고조파 15개**를 준다 — 모든 녹화가 텍스처
라이브러리다. 새 회로 모델(v12g)의 규약도 "V15(참 전압) -> sim_harmonics" 라 h15 까지면 된다 (README_v12 §규약 3).

관례
----
npz 의 `voltage_harmonics_complex[:, h-1]` = `vh_h · e^{j·vhdeg_h}`, 펌웨어 관례 `vhdeg_h = arg(V_h) − h·arg(V_1)`
(`vhdeg1 ≡ 0`, |V1| = vrms). 여기서는 **상대 텍스처** `rel = V15 / |V1|` (rel[0] = 1) 로 두고, 합성 환경의
기저 전압 v1 을 곱해 소스 `V15 = rel · v1` 을 만든다. 같은 파일 안에서는 |V3|/|V1| 의 산포가 0.02~0.04% 라
60초에 하나만 뽑아도 충분하고, 파일(세션) 사이에서는 0.6% 와 3.2% 로 갈린다 (READ_ME_FIRST §2).

쓰임
----
    lib = VoltageTextureLibrary.from_npz_dir()            # processed_data/npz 의 device·noise 파일
    tex = lib.sample(rng, vrms_target=229.5)               # 기저 전압에 가까운 세션의 텍스처 하나
    rel_rec = lib.file_rel("laptop_charger_1")             # 그 녹화 자체의 텍스처 (델타의 기준)
    V15 = tex.rel * 229.5                                  # fcm12 소스
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import json

import numpy as np

H = 15
DEFAULT_NPZ_DIR = "processed_data/npz"
#: 텍스처를 뽑는 간격 (초). 파일 안 산포(0.02~0.04%)가 세션 간 차(0.6~3.2%)의 1/50 이라 60초면 넉넉하다.
DEFAULT_STEP_S = 60.0
#: 기저 전압에 가까운 세션을 고를 때의 허용폭 (V). 저녁(216V)/심야(229V) 무리를 가르되 그 안에서는 섞는다.
DEFAULT_TOL_V = 4.0


@dataclass
class Texture:
    id: int
    stem: str
    t_rel_s: float
    vrms: float
    rel: np.ndarray          # (15,) complex, rel[0] = 1+0j

    def v15(self, v1: float) -> np.ndarray:
        return (self.rel * float(v1)).astype(np.complex128)


class VoltageTextureLibrary:
    def __init__(self, textures: Sequence[Texture], file_rel: Dict[str, np.ndarray]):
        self.textures: List[Texture] = list(textures)
        self._file_rel: Dict[str, np.ndarray] = dict(file_rel)
        self._file_ids: Dict[str, int] = {s: i for i, s in enumerate(sorted(self._file_rel))}
        self._vrms = np.array([t.vrms for t in self.textures], dtype=np.float64)

    # ── 구성 ──────────────────────────────────────────────────────────────
    @classmethod
    def from_npz_dir(cls, npz_dir: Union[str, Path] = DEFAULT_NPZ_DIR, step_s: float = DEFAULT_STEP_S,
                     roles: Sequence[str] = ("device", "noise")) -> "VoltageTextureLibrary":
        """device·noise npz 에서 `step_s` 마다 텍스처 하나, 파일마다 중앙 텍스처 하나."""
        textures: List[Texture] = []
        file_rel: Dict[str, np.ndarray] = {}
        d = Path(npz_dir)
        if not d.exists():
            return cls([], {})
        for f in sorted(d.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            if "voltage_harmonics_complex" not in z.files:
                continue
            meta = json.loads(str(z["metadata_json"])) if "metadata_json" in z.files else {}
            if not meta.get("voltage_phase_available", False):
                continue
            if roles and meta.get("file_role") not in roles:
                continue
            V = np.asarray(z["voltage_harmonics_complex"], dtype=np.complex128)
            if V.ndim != 2 or V.shape[1] < H or len(V) == 0:
                continue
            valid = (np.asarray(z["is_valid"]) == 1) if "is_valid" in z.files else np.ones(len(V), bool)
            v1 = np.abs(V[:, 0])
            ok = valid & (v1 > 150.0) & (v1 < 280.0) & np.isfinite(V[:, :H]).all(axis=1)
            if int(ok.sum()) < 60:
                continue
            t = np.asarray(z["t_rel_s"], dtype=np.float64) if "t_rel_s" in z.files else np.arange(len(V)) / 60.0
            rel_all = V[:, :H] / v1[:, None]
            # 파일 중앙 텍스처 — 실수/허수 각각 중앙값 (위상 상쇄를 피한다)
            m = rel_all[ok]
            file_rel[f.stem] = (np.median(m.real, 0) + 1j * np.median(m.imag, 0)).astype(np.complex128)
            file_rel[f.stem][0] = 1.0 + 0j
            # step_s 마다 하나: 그 구간의 중앙값
            step = max(60, int(step_s * 60))
            for a in range(0, len(V), step):
                sel = ok[a:a + step]
                if int(sel.sum()) < 30:
                    continue
                seg = rel_all[a:a + step][sel]
                rel = (np.median(seg.real, 0) + 1j * np.median(seg.imag, 0)).astype(np.complex128)
                rel[0] = 1.0 + 0j
                textures.append(Texture(id=len(textures), stem=f.stem, t_rel_s=float(t[a]),
                                        vrms=float(np.median(v1[a:a + step][sel])), rel=rel))
        return cls(textures, file_rel)

    # ── 조회 ──────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.textures)

    def file_rel(self, stem: str) -> Optional[np.ndarray]:
        """그 녹화(npz stem)의 중앙 상대 텍스처. 모르면 None."""
        return self._file_rel.get(stem)

    def file_id(self, stem: str) -> int:
        """stem -> 정수 id (캐시 키용). 모르면 −1."""
        return self._file_ids.get(stem, -1)

    def file_rel_by_id(self, fid: int) -> Optional[np.ndarray]:
        for s, i in self._file_ids.items():
            if i == fid:
                return self._file_rel[s]
        return None

    def sample(self, rng: np.random.Generator, vrms_target: Optional[float] = None,
               tol_v: float = DEFAULT_TOL_V) -> Optional[Texture]:
        """기저 전압에 가까운(±tol_v) 텍스처 중 하나. 가까운 것이 없으면 전체에서."""
        if not self.textures:
            return None
        if vrms_target is not None:
            near = np.flatnonzero(np.abs(self._vrms - float(vrms_target)) <= tol_v)
            if len(near):
                return self.textures[int(rng.choice(near))]
        return self.textures[int(rng.integers(len(self.textures)))]

    def describe(self) -> str:
        if not self.textures:
            return "텍스처 없음"
        stems = sorted({t.stem for t in self.textures})
        v3 = np.array([abs(t.rel[2]) for t in self.textures]) * 100
        return (f"텍스처 {len(self.textures)}개 / 파일 {len(stems)}개, vrms {self._vrms.min():.1f}~{self._vrms.max():.1f}V, "
                f"|V3|/|V1| {v3.min():.2f}~{v3.max():.2f}%")


_DEFAULT: Optional[VoltageTextureLibrary] = None


def default_library(npz_dir: Union[str, Path] = DEFAULT_NPZ_DIR) -> VoltageTextureLibrary:
    """프로세스 안에서 한 번만 읽는다 (워커마다 한 번)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = VoltageTextureLibrary.from_npz_dir(npz_dir)
    return _DEFAULT
