"""
최종 평가 파일 봉인 (Sealed Test Set Guard)
============================================
`test.csv` 는 최종 평가에 **딱 한 번** 쓴다. 실측이 전부 35분뿐이라 반복해서
들여다보면 즉시 오염된다 (설계 문서 4.3절).

사람의 규율에 맡기지 않는다. 코드가 막는다.

    from src.evaluation.sealing import assert_not_sealed, unseal

    assert_not_sealed("test.2")     # 통과
    assert_not_sealed("test")       # SealedDatasetError

    with unseal("최종 평가 1회차 - 2026-08-21"):
        ...                          # 이 블록 안에서만 test.csv 접근 허용

개봉하면 `processed_data/SEAL_BROKEN.json` 에 시각과 사유가 기록된다.
그 파일이 존재한다는 것 자체가 "이 데이터는 더 이상 봉인되어 있지 않다"는 뜻이다.
"""
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Union
import json
import os

# 봉인 대상. 파일 stem 기준.
SEALED_STEMS = frozenset({"test"})

# 개봉 기록이 남는 자리
SEAL_LOG_PATH = Path("processed_data/SEAL_BROKEN.json")

_UNSEALED_DEPTH = 0


class SealedDatasetError(RuntimeError):
    """봉인된 최종 평가 데이터에 접근하려 했을 때."""


def _normalize(stem: Union[str, Path]) -> str:
    """'data/test.csv' -> 'test'. 'test.2.csv' -> 'test.2' (봉인 대상 아님)."""
    name = Path(str(stem)).name
    for suffix in (".npz", ".csv"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def is_sealed(stem: Union[str, Path]) -> bool:
    """이 파일이 봉인 대상인가. `test.2` / `test3` / `test_4` 는 아니다."""
    return _normalize(stem).lower() in {s.lower() for s in SEALED_STEMS}


def assert_not_sealed(stem: Union[str, Path]) -> None:
    """봉인된 파일이면 즉시 실패시킨다. `unseal()` 블록 안에서는 통과한다."""
    name = _normalize(stem)
    if not is_sealed(name) or _UNSEALED_DEPTH > 0:
        return
    raise SealedDatasetError(
        f"'{name}' 은 최종 평가 전용으로 봉인되어 있습니다 (설계 문서 4.3절).\n"
        f"  - 검증에는 test.2 / test3 / test_4 를 쓰십시오.\n"
        f"  - 정말 최종 평가라면 sealing.unseal('사유') 블록 안에서 접근하십시오.\n"
        f"  - 개봉은 되돌릴 수 없습니다. 개봉 후에는 하이퍼파라미터를 바꾸지 마십시오."
    )


def filter_sealed(stems: List[Union[str, Path]]) -> List[Union[str, Path]]:
    """봉인 대상을 걸러낸 목록. 평가 루프에서 조용히 제외할 때 쓴다."""
    return [s for s in stems if not is_sealed(s)]


@contextmanager
def unseal(reason: str, timestamp: Optional[str] = None) -> Iterator[None]:
    """봉인을 열고 그 사실을 디스크에 기록한다.

    Args:
        reason: 왜 열었는지. 기록에 남는다.
        timestamp: 개봉 시각 문자열. 생략하면 기록만 남기고 시각은 비운다
            (이 저장소는 결정성을 위해 코드에서 시계를 읽지 않는다).
    """
    global _UNSEALED_DEPTH
    if not reason or not reason.strip():
        raise ValueError("개봉 사유를 반드시 남기십시오.")

    SEAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = []
    if SEAL_LOG_PATH.exists():
        try:
            log = json.loads(SEAL_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = []
    log.append({"reason": reason.strip(), "timestamp": timestamp, "sealed_stems": sorted(SEALED_STEMS)})
    SEAL_LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    _UNSEALED_DEPTH += 1
    try:
        yield
    finally:
        _UNSEALED_DEPTH -= 1


def seal_status() -> dict:
    """봉인이 아직 유효한지. 실험 기록에 남길 것."""
    broken = SEAL_LOG_PATH.exists()
    entries = []
    if broken:
        try:
            entries = json.loads(SEAL_LOG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            entries = [{"reason": "(기록 파일이 손상됨)"}]
    return {
        "sealed_stems": sorted(SEALED_STEMS),
        "intact": not broken,
        "openings": len(entries),
        "log": entries,
    }
