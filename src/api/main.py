"""FastAPI microservice for spacecraft telemetry anomaly detection."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from src.api.schemas import PredictRequest, PredictResponse
from src.config import AppConfig, load_config
from src.data.scaling import load_scaler, transform_windows
from src.models.isolation_forest import load_model, score_window
from src.models.threshold import DynamicThreshold, load_threshold_config

_state: dict[str, Any] = {}


def _build_window_vector(
    values: list[float] | list[list[float]],
    window_size: int,
    expected_features: int,
) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "values must be a numeric 1D or 2D array with consistent "
                f"row lengths (full SMAP model expects {window_size} rows × "
                f"{expected_features // window_size} sensors)."
            ),
        ) from exc

    # 2D input: use last `window_size` rows and flatten.
    if arr.ndim == 2:
        n_sensors = expected_features // window_size
        if arr.shape[1] != n_sensors:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"invalid input shape: got {arr.shape[0]}×{arr.shape[1]}, "
                    f"expected at least {window_size}×{n_sensors} for the full SMAP model"
                ),
            )
        if arr.shape[0] < window_size:
            raise HTTPException(
                status_code=400,
                detail=f"values must contain at least {window_size} timesteps",
            )
        window = arr[-window_size:, :].reshape(1, -1)
    # 1D input: accept either exactly window_size (univariate) or pre-flattened
    # vectors that match the model feature width.
    elif arr.ndim == 1:
        if arr.shape[0] >= expected_features:
            window = arr[-expected_features:].reshape(1, -1)
        elif arr.shape[0] >= window_size:
            window = arr[-window_size:].reshape(1, -1)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"values must contain at least {window_size} readings",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="values must be a 1D or 2D numeric array",
        )

    if window.shape[1] != expected_features:
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid input shape: got {window.shape[1]} features, "
                f"expected {expected_features}"
            ),
        )
    return window


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    if not config.model_path.exists() or not config.scaler_path.exists():
        raise RuntimeError(
            "Artifacts not found. Run: python scripts/train.py"
        )
    _state["config"] = config
    _state["model"] = load_model(config.model_path)
    _state["scaler"] = load_scaler(config.scaler_path)
    if config.threshold_config_path.exists():
        _state["threshold"] = load_threshold_config(config.threshold_config_path)
    else:
        _state["threshold"] = DynamicThreshold(
            rolling_window=config.rolling_window,
            n_sigma=config.n_sigma,
        )
    yield
    _state.clear()


app = FastAPI(
    title="Spacecraft Telemetry Anomaly Detection",
    description="Unsupervised SMAP anomaly scoring with dynamic thresholding",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _finite_threshold(value: float) -> float:
    """JSON-safe threshold (no inf/nan in API responses)."""
    if not np.isfinite(value):
        return 1.0
    return float(value)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    config: AppConfig = _state["config"]
    window_size = config.window_size
    expected_features = int(_state["scaler"].n_features_in_)
    try:
        window = _build_window_vector(request.values, window_size, expected_features)
        scaled = transform_windows(_state["scaler"], window)
        anomaly_score = score_window(_state["model"], scaled)

        threshold_engine: DynamicThreshold = _state["threshold"]
        result = threshold_engine.evaluate(
            anomaly_score,
            recent_scores=request.recent_scores,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"could not score request: {exc}",
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000.0
    return PredictResponse(
        anomaly_score=round(result.score, 6),
        threshold=round(_finite_threshold(result.threshold), 6),
        status=result.status,
        latency_ms=round(latency_ms, 3),
    )
