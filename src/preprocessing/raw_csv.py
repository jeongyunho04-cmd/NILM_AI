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
) -> Tuple[pd.DataFrame, Dict]:
    """정본 순서로 읽는다. 반환 인덱스는 0..N-1 로 다시 매긴다.

    `usecols` 를 줘도 `seq`/`cycle` 은 자동으로 포함한다 — 정렬에 필요하다.
    """
    path = Path(path)
    need = {"seq", "cycle"}
    cols = None if usecols is None else sorted(set(usecols) | need)
    df = pd.read_csv(path, usecols=cols)
    n0 = len(df)

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
        # npz 의 행 i 는 `(seq - seq_lo)*30 + cycle` 이다 (이음매·결손이 없을 때).
        "contiguous": bool(len(df) == (int(s.max()) - int(s.min()) + 1) * CYCLES_PER_FRAME),
    }
    return df.drop(columns=["_session"]), info


def npz_row_of_seq(seq_lo: int, seq: np.ndarray, cycle: int = 0) -> np.ndarray:
    """seq -> npz 행. `info['contiguous']` 가 True 일 때만 정확하다."""
    return (np.asarray(seq, np.int64) - int(seq_lo)) * CYCLES_PER_FRAME + int(cycle)
