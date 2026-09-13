"""NILM 실시간 추론 런타임 — 학습 코드 없이 도는 최소 묶음.

    from nilm_runtime import NILMPredictor

    pred = NILMPredictor("models/adapt_zi_s0.pt")
    pred.push_row(csv_row, header)      # 수신기 CSV 한 행 (사이클 1개)
    out = pred.predict()                # 창이 차면 결과, 아니면 None

자세한 구조는 `deploy/README.md` 참조.
"""
from .predictor import NILMPredictor, PredictionResult, APPLIANCE_KO  # noqa: F401

__all__ = ["NILMPredictor", "PredictionResult", "APPLIANCE_KO"]
