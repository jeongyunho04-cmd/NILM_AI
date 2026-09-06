# -*- coding: utf-8 -*-
"""전압 텍스처 라이브러리 — 2Hz 녹화의 전압 고조파(vh·vhdeg)에서 (2026-09-06, 13.2).

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
    tex = lib.sample(rng, vrms_target=229.5, site="E")     # 그 자리의 텍스처 하나 (세션 균등)
    rel_rec = lib.file_rel("laptop_charger_1")             # 그 녹화 자체의 텍스처 (델타의 기준)
    V15 = tex.rel * 229.5                                  # fcm12 소스

⚠ 세 가지를 2026-09-06(13.19)에 고쳤다 — 텍스처 **선택**이 고차 전류를 크게 움직이기 때문이다
(같은 자리라도 세션이 다르면 미니PC 의 |I13|/P 가 30~45% 갈린다, 13.19.1):
  ① **자리를 명시로 받는다.** 옛 코드는 기저 전압만 보고 골랐다. 236V 위에서는 ±4V 안에 아무것도
     없어 **라이브러리 전체에서 균등**으로 떨어졌고, 그 절반이 216V 자리(D)의 텍스처였다.
  ② 가까운 것이 없으면 **전압이 가장 가까운 것**을 준다 (균등 난수가 아니라).
  ③ 후보 안에서 **세션을 먼저 균등하게** 고르고 그 안에서 텍스처를 고른다. 텍스처 개수는 녹화
     길이의 부산물이라, 그대로 두면 긴 녹화 한 세션이 그 자리를 대표해 버린다 (E1 82% 대 E2 18%).
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
#: 기저 전압에 가까운 세션을 고를 때의 허용폭 (V). D(216V)/E(229V) 무리를 가르되 그 안에서는 섞는다.
DEFAULT_TOL_V = 4.0


@dataclass
class Texture:
    id: int
    stem: str
    t_rel_s: float
    vrms: float
    rel: np.ndarray          # (15,) complex, rel[0] = 1+0j — **단자** 전압 (녹화 그대로)
    site: str = ""           #: 자리 (`file_registry.site_of`). 모르면 ""
    session: str = ""        #: 세션 (`file_registry.SITE_SESSIONS` 의 키). 모르면 stem
    #: **개방 전압** (13.22). `V_open = V_term + Z_session·I_block` 로 그 녹화 부하의 강하를 벗긴 것.
    #: 합성 창의 소스 전압은 이쪽이다 — `rel` 은 그 녹화의 부하가 이미 얹힌 값이라 소스로 쓰면
    #: 결합 델타가 강하를 두 번 건다. Z 를 모르는 세션이면 `rel` 과 같다.
    rel_open: Optional[np.ndarray] = None

    def source_rel(self) -> np.ndarray:
        """합성의 소스로 쓸 상대 텍스처 (개방 전압이 있으면 그것)."""
        return self.rel if self.rel_open is None else self.rel_open

    def v15(self, v1: float) -> np.ndarray:
        return (self.rel * float(v1)).astype(np.complex128)


#: 개방 전압 복원에 쓰는 선로 리액턴스 [H]. 꼬리 자료로는 정해지지 않아(h1~h5 의 지렛대가 짧다,
#: 13.22.3) 생성기의 `x_grid_range` 중앙(0.085Ω @60Hz)에 해당하는 값을 쓴다. R 이 지배적이라
#: h15 에서도 |jwL| 0.57Ω 대 R 0.42~1.15Ω 다.
DEEMBED_L_H = 225e-6


def _site_session(stem: str) -> tuple:
    """stem -> (자리, 세션, 그 세션의 Z). 등록부가 정본이다."""
    try:
        from src.preprocessing.file_registry import SITE_SESSIONS, site_of
    except Exception:
        return "", stem, None
    for key, spec in SITE_SESSIONS.items():
        if stem in spec.get("stems", ()):
            z = spec.get("z_ohm")
            if z is None:
                site = spec.get("site", "")
                z = {"D": 1.15, "E": 0.42}.get(site)
            return spec.get("site", ""), key, z
    return site_of(stem) or "", stem, None


class VoltageTextureLibrary:
    def __init__(self, textures: Sequence[Texture], file_rel: Dict[str, np.ndarray]):
        self.textures: List[Texture] = list(textures)
        self._file_rel: Dict[str, np.ndarray] = dict(file_rel)
        self._file_ids: Dict[str, int] = {s: i for i, s in enumerate(sorted(self._file_rel))}
        self._vrms = np.array([t.vrms for t in self.textures], dtype=np.float64)
        self._site = np.array([t.site for t in self.textures], dtype=object)

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
            # 개방 전압 복원 (13.22): V_open = V_term + Z(h)·I_block. 그 녹화 부하의 강하를 벗긴다.
            st_, sess_, z_ohm = _site_session(f.stem)
            Icur = (np.asarray(z["harmonics_complex"], dtype=np.complex128)
                    if "harmonics_complex" in z.files else None)
            Zh = (None if (z_ohm is None or Icur is None or Icur.shape[1] < H)
                  else z_ohm + 1j * 2 * np.pi * 60.0 * np.arange(1, H + 1) * DEEMBED_L_H)
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
                rel_open = None
                if Zh is not None:
                    Vseg = V[a:a + step][sel][:, :H]
                    Iseg = Icur[a:a + step][sel][:, :H]
                    vo = (np.median(Vseg.real, 0) + 1j * np.median(Vseg.imag, 0)
                          + Zh * (np.median(Iseg.real, 0) + 1j * np.median(Iseg.imag, 0)))
                    if np.all(np.isfinite(vo)) and abs(vo[0]) > 1.0:
                        rel_open = (vo / abs(vo[0])).astype(np.complex128)
                        rel_open[0] = 1.0 + 0j
                textures.append(Texture(id=len(textures), stem=f.stem, t_rel_s=float(t[a]),
                                        vrms=float(np.median(v1[a:a + step][sel])), rel=rel,
                                        site=st_, session=sess_, rel_open=rel_open))
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
               tol_v: float = DEFAULT_TOL_V, site: Optional[str] = None,
               by_session: bool = True) -> Optional[Texture]:
        """그 환경의 텍스처 하나 (13.19).

        `site` 를 주면 **그 자리의 텍스처만** 본다 (측정된 무리에서 온 창). 안 주면 전 자리에서 고른다
        (탐색 성분 — 미측정 콘센트라 자리를 모른다).
        전압은 `vrms_target ± tol_v` 로 거르되, **비면 가장 가까운 것**으로 간다 (옛 코드는 전체 균등이라
        236V 창이 216V 자리의 텍스처를 절반 확률로 받았다).
        `by_session` 이면 후보 안에서 **세션을 먼저 균등하게** 고른다 — 텍스처 개수는 녹화 길이의
        부산물이라 그대로 두면 긴 녹화가 그 자리를 대표한다.
        """
        if not self.textures:
            return None
        idx = np.arange(len(self.textures))
        if site:
            m = np.flatnonzero(self._site[idx] == site)
            if len(m):
                idx = idx[m]
        if vrms_target is not None and len(idx):
            near = idx[np.abs(self._vrms[idx] - float(vrms_target)) <= tol_v]
            if len(near):
                idx = near
            else:                                   # 가까운 것이 없으면 **가장 가까운 쪽 한 무리**
                d = np.abs(self._vrms[idx] - float(vrms_target))
                idx = idx[d <= d.min() + tol_v]     # 한 개로 좁히면 다양성이 사라진다
        if not len(idx):
            idx = np.arange(len(self.textures))
        if by_session:
            sess = sorted({self.textures[i].session for i in idx})
            if len(sess) > 1:
                pick = sess[int(rng.integers(len(sess)))]
                idx = np.array([i for i in idx if self.textures[i].session == pick])
        return self.textures[int(idx[int(rng.integers(len(idx)))])]

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
