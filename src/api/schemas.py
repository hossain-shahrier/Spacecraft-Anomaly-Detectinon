"""Pydantic request/response models for the prediction API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_ROW_25 = [0.1, 0.2, 0.15, 0.05, 0.03, 0.08, 0.11, 0.09, 0.07, 0.13] * 2 + [0.1, 0.2, 0.15, 0.05, 0.03]


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "channel_id": "E-1",
                    "values": [_ROW_25] * 50,
                    "recent_scores": [0.4, 0.42, 0.41],
                }
            ]
        }
    )

    channel_id: str = Field(..., description="SMAP channel identifier, e.g. E-1")
    values: list[Any] = Field(
        ...,
        min_length=1,
        description=(
            "Full SMAP model: 50 rows × 25 sensors (or 1250 floats flat). "
            "Each row is one timestep; each row must have exactly 25 numbers."
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
