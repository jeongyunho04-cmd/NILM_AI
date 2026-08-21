"""
학습 프로세스 환경 가드 (Environment Guard)
============================================
**torch 를 import 하기 전에 가장 먼저 import 할 것.**

    from src import env_guard  # noqa: F401  ← 맨 위
    import torch

[왜 필요한가 — 이 장비에서 실제로 발생한 문제]
Anaconda 의 MKL 과 pip 로 설치한 torch 가 각자 OpenMP 런타임을 들고 온다.

    C:\\Users\\...\\anaconda3\\Library\\bin\\libiomp5md.dll          (MKL)
    C:\\Users\\...\\anaconda3\\Lib\\site-packages\\torch\\lib\\libiomp5md.dll  (torch)

둘 다 로드되면 프로세스가 죽는다.

    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
    already initialized.

`import torch, numpy` 든 `import numpy, torch` 든 순서와 무관하게 터진다.

[근본 해결과 임시 조치]
근본은 전용 환경을 만들어 OpenMP 런타임을 한 벌로 맞추는 것이다.

    conda create -n nilm python=3.12
    conda activate nilm
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    pip install numpy pandas scikit-learn matplotlib pytest

여기서 쓰는 `KMP_DUPLICATE_LIB_OK=TRUE` 는 Intel 이 문서에 "unsafe, unsupported,
undocumented" 라고 적어 둔 우회다. 실제로 조용히 틀린 결과를 낼 수 있다.
그래서 `verify_numerics()` 로 매 학습 시작 시 기본 수치 연산을 검산한다.

[워커 BLAS 스레드]
워커 11개 × BLAS 12스레드 = 132스레드가 12코어에서 서로 밀어낸다.
워커 안에서는 BLAS 를 1스레드로 묶어야 한다 (설계 문서 4.1절).
"""
from typing import Optional
import os
import warnings

_APPLIED = False


def apply(single_thread_blas: bool = True, allow_duplicate_omp: bool = True) -> None:
    """torch import 전에 환경 변수를 세팅한다. 두 번 불러도 안전하다.

    Args:
        single_thread_blas: BLAS 를 1스레드로 묶는다. 데이터 워커에서는 반드시 True.
            단일 프로세스로 무거운 선형대수를 돌릴 때만 False 로 둘 것.
        allow_duplicate_omp: OpenMP 중복 로드를 허용한다 (위 설명 참조).
    """
    global _APPLIED
    if single_thread_blas:
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(v, "1")
    if allow_duplicate_omp:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _APPLIED = True


def verify_numerics(tolerance: float = 1e-10) -> None:
    """OpenMP 중복 로드가 수치를 망가뜨리지 않았는지 확인한다.

    `KMP_DUPLICATE_LIB_OK` 의 위험은 크래시가 아니라 **조용한 오답**이다.
    학습 시작 전에 한 번 돌려 두면 최소한 기본 연산은 걸러진다.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.standard_normal((512, 512))
    b = rng.standard_normal((512, 512))

    # 1) GEMM 이 순진한 계산과 일치하는가 (일부 행만 표본 검사)
    c = a @ b
    for i in (0, 137, 511):
        ref = float(np.sum(a[i] * b[:, i]))
        if abs(c[i, i] - ref) > 1e-6 * max(1.0, abs(ref)):
            raise RuntimeError(
                f"BLAS 결과가 어긋납니다 (행 {i}: {c[i, i]} vs {ref}). "
                "OpenMP 중복 로드가 원인일 수 있습니다. 전용 환경을 만드십시오 "
                "(src/env_guard.py 상단 참조)."
            )

    # 2) 대칭 행렬의 고윳값 합 == trace
    s = a + a.T
    if abs(float(np.linalg.eigvalsh(s).sum()) - float(np.trace(s))) > 1e-6 * abs(float(np.trace(s))):
        raise RuntimeError("LAPACK 결과가 어긋납니다. 전용 환경을 만드십시오.")

    # 3) FFT 왕복
    v = rng.standard_normal(4096)
    if float(np.max(np.abs(np.fft.irfft(np.fft.rfft(v), n=len(v)) - v))) > 1e-9:
        raise RuntimeError("FFT 왕복 오차가 큽니다. 전용 환경을 만드십시오.")


def describe() -> dict:
    """현재 환경 요약. 실험 기록에 남길 것."""
    info = {
        "env_guard_applied": _APPLIED,
        "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception as exc:  # torch 가 없어도 진단은 나와야 한다
        info["torch"] = f"불러오지 못함: {exc}"
    return info


# import 만 해도 적용되게 한다. 잊고 호출하지 않는 실수가 더 잦다.
apply()
