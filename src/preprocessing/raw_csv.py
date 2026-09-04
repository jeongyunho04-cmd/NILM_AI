"""원본 CSV 를 **정본 순서로** 읽는다 (12.155)

왜 필요한가
----------
수신기는 Wi-Fi 재전송 때문에 패킷을 **순서가 뒤바뀐 채** 기록한다. 실측에서 심하다:

```
hotplate_1        역행 210회, 최대 15패킷(7.5초)
beam_projector    역행 123회, 최대 15패킷
electiric_kettle  역행  58회
```

전처리(`DataCleaner`)는 `(session_id, seq, cycle)` 로 정렬하므로 **npz 는 멀쩡하다.**
그런데 **원본 CSV 를 직접 읽는 코드는 정렬을 안 한다** — `harmonic_offset`,
`run_fit_impedance`, `run_norton_probe`, 그리고 전이 지문 탐침들이 전부 그렇다.
그쪽은 행 순서를 시간 순서로 믿고 차분·계단을 잡으므로, 7.5초짜리 순서 뒤바뀜이
없는 계단을 만들고 있는 계단을 지운다.

방증: `harmonic_offset` 의 vrms 정렬 상관이 test.2 0.797 / test3 0.878 로 11파일 중
최저 둘인데, 그 둘이 정확히 seq 가 가장 많이 엉킨 파일이다 (test.2 는 299곳).

한 파일에 녹화가 여럿일 수도 있다 (`test_14`: seq 925 -> 0, host_time +362.8초).
그것도 여기서 가른다. **세션 판정 규칙은 `DataCleaner.assign_sessions` 를 그대로
쓴다** — 여기서 따로 만들면 전처리와 조용히 갈라진다.

쓰는 법
------
    from src.preprocessing.raw_csv import read_raw_csv
    df, info = read_raw_csv("data/test_14.csv", usecols=[...], session=-1)

`session=-1` 은 **가장 긴 세션**이다 (기본). `None` 이면 전부, 정수면 그 번호.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#: `DataCleaner` 와 같은 값이어야 한다. 재전송 역전은 이 이내, 리셋은 이보다 크다.
from src.preprocessing.cleaner import REORDER_TOLERANCE_FRAMES

CYCLES_PER_FRAME = 30


def assign_sessions(seq: np.ndarray) -> np.ndarray:
    """`DataCleaner.assign_sessions` 와 같은 규칙. seq 만으로 판정한다."""
    running_max = np.maximum.accumulate(seq)
    is_reset = seq < (running_max - REORDER_TOLERANCE_FRAMES)
    boundary = is_reset & ~np.concatenate([[False], is_reset[:-1]])
    return np.cumsum(boundary.astype(np.int64))


def read_raw_csv(
    path: Union[str, Path],
    usecols: Optional[Sequence[str]] = None,
    session: Optional[int] = -1,
    phase_fix: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """정본 순서로 읽는다. 반환 인덱스는 0..N-1 로 다시 매긴다.

    `usecols` 를 줘도 `seq`/`cycle` 은 자동으로 포함한다 — 정렬에 필요하다.
    `phase_fix` 가 True 면 등록부 `PHASE_FIX_DEG_PER_ORDER` 의 위상 복원을 건다
    (12.184.3). 원시값이 필요한 검정(지연 검정 등)은 False 로 읽는다.
    """
    path = Path(path)
    need = {"seq", "cycle"}
    if phase_fix:
        need = need | {"range", "host_time"}        # LOW 교정 규약 판정에 필요 (12.184.13)
    cols = None if usecols is None else sorted(set(usecols) | need)
    df = pd.read_csv(path, usecols=cols)
    n0 = len(df)
    fix_deg = 0.0
    low_shift = 0.0
    if phase_fix:
        from src.preprocessing.file_registry import low_cal_shift_deg, phase_fix_of
        from src.preprocessing.raw_phasors import apply_phase_rotation
        fix_deg = phase_fix_of(path.stem)
        if fix_deg:
            df = apply_phase_rotation(df, fix_deg)
        ht = str(df["host_time"].iloc[0]) if "host_time" in df.columns and len(df) else None
        low_shift = low_cal_shift_deg(path.stem, ht)
        if low_shift and "range" in df.columns:
            df = apply_phase_rotation(df, low_shift, mask=df["range"].to_numpy() == 0)
        if usecols is not None:
            keep = [c for c in df.columns if c in set(usecols) | {"seq", "cycle"}]
            df = df[keep]

    sid = assign_sessions(df["seq"].to_numpy(np.int64))
    df = df.assign(_session=sid)
    # 세션을 키에 넣어야 리셋 후 겹치는 seq 가 중복으로 오인되지 않는다.
    df = df.drop_duplicates(subset=["_session", "seq", "cycle"])
    n_dup = n0 - len(df)
    # 정렬 전 순서가 이미 정본이었는지 세어 둔다 (진단용).
    key = df["_session"].to_numpy(np.int64) * (1 << 40) \
        + df["seq"].to_numpy(np.int64) * CYCLES_PER_FRAME + df["cycle"].to_numpy(np.int64)
    n_out_of_order = int((np.diff(key) < 0).sum())
    df = df.sort_values(["_session", "seq", "cycle"], kind="stable").reset_index(drop=True)

    sizes = df.groupby("_session").size()
    if session is not None and len(sizes) > 1:
        pick = int(sizes.idxmax()) if session < 0 else int(session)
        if pick not in sizes.index:
            raise ValueError(f"{path.name} 에 세션 {pick} 이 없습니다 (있는 것: {list(sizes.index)})")
        df = df[df["_session"] == pick].reset_index(drop=True)
    elif session is not None:
        pick = int(sizes.index[0])
    else:
        pick = None

    s = df["seq"].to_numpy(np.int64)
    info = {
        "rows_raw": n0, "rows": len(df), "duplicates": n_dup,
        "out_of_order": n_out_of_order,
        "n_sessions": int(len(sizes)), "session": pick,
        "session_rows": {int(k): int(v) for k, v in sizes.items()},
        "seq_lo": int(s.min()), "seq_hi": int(s.max()),
        "phase_fix_deg_per_order": fix_deg,
        "low_cal_shift_deg_per_order": low_shift,      # range==0 사이클에 건 회전 (12.184.13)
        # npz 의 행 i 는 `(seq - seq_lo)*30 + cycle` 이다 (이음매·결손이 없을 때).
        "contiguous": bool(len(df) == (int(s.max()) - int(s.min()) + 1) * CYCLES_PER_FRAME),
    }
    return df.drop(columns=["_session"]), info


#: 원본 CSV 의 열 구성은 펌웨어/수신기 판마다 다르다. 전부 **이름으로** 읽으므로 순서는
#: 무관하지만, 있는 열이 다르다:
#:   1~2차 (8/25 이전)      ih/ihdeg/vh, off_low 없음
#:   3차   (8/27~9/02)       + off_low 가 **맨 끝**에                      (프로토콜 v4)
#:   4차   (9/04~)           + **vhdeg1~15** (프로토콜 v5). off_low 위치는 수신기 판에 따라
#:                           over_range 뒤(오전 수신기) 또는 vh15 뒤(오후 수신기) — 이름으로 읽으면 무관
#: ⚠ 위상은 열 구성이 아니라 **보드의 교정 상태**를 탄다: 부하 없이 USER 버튼 교정이 눌리면
#: 그 부팅의 LOW 레인지 전체가 −k×h 돈다 (12.184.3, laptop_charger_5C). 등록부의
#: PHASE_FIX_DEG_PER_ORDER 가 그런 파일을 읽을 때 되돌린다.
#: ⚠ 2026-09-04 17:00 부터 펌웨어 LOW 교정값이 0.44 -> 2.62° (정본). 그 전 파일은 읽을 때 `range==0`
#: 사이클만 −2.18°×h 로 돌려 맞춘다 (`file_registry.low_cal_shift_deg`, 첫 host_time 으로 판정). 12.184.13.
def detect_raw_format(path: Union[str, Path]) -> Dict:
    """헤더만 읽어 열 구성을 판정한다. {'off_low', 'vhdeg', 'version'}"""
    cols = set(pd.read_csv(path, nrows=0).columns)
    has_off = "off_low" in cols
    has_vdeg = all(f"vhdeg{h}" in cols for h in range(1, 16))
    version = 4 if has_vdeg else (3 if has_off else 2)
    return {"off_low": has_off, "vhdeg": has_vdeg, "version": version, "n_cols": len(cols)}


def npz_row_of_seq(seq_lo: int, seq: np.ndarray, cycle: int = 0) -> np.ndarray:
    """seq -> npz 행. `info['contiguous']` 가 True 일 때만 정확하다."""
    return (np.asarray(seq, np.int64) - int(seq_lo)) * CYCLES_PER_FRAME + int(cycle)
