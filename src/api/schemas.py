"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    channel_id: str = Field(..., description="SMAP channel identifier, e.g. E-1")
    values: list[Any] = Field(
        ...,
        min_length=1,
        description=(
            "Recent telemetry values. Accepts either a 1D list of floats "
            "(univariate/synthetic or pre-flattened multivariate) or a 2D list "
            "(timesteps x features)."
        ),
    )
    recent_scores: list[float] | None = Field(
        default=None,
        description="Optional prior anomaly scores for dynamic thresholding",
    )


class PredictResponse(BaseModel):
    anomaly_score: float
    threshold: float
    status: str
    latency_ms: float
