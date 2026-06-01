"""Sliding-window feature extraction for univariate telemetry streams."""

from __future__ import annotations

import numpy as np


def sliding_windows(
    series: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """
    Build sliding windows from a 1D series.

    Returns shape (n_windows, window_size).
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series.reshape(-1, 1)
    elif series.ndim != 2:
        raise ValueError(f"Expected 1D or 2D series, got shape {series.shape}")

    n_timesteps, n_features = series.shape
    flat_width = window_size * n_features
    if n_timesteps < window_size:
        return np.empty((0, flat_width), dtype=np.float64)

    n_windows = (n_timesteps - window_size) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        series, window_shape=(window_size, n_features)
    )
    windows = windows[::stride]
    return np.asarray(windows, dtype=np.float64).reshape(n_windows, flat_width)


def window_end_indices(window_size: int, n_windows: int, stride: int = 1) -> np.ndarray:
    """Map each window index to its end timestamp in the original series."""
    return np.arange(n_windows, dtype=np.int64) * stride + window_size - 1


def propagate_window_alerts(
    alerts: np.ndarray,
    end_indices: np.ndarray,
    window_size: int,
    series_length: int,
) -> np.ndarray:
    """Label each timestep anomalous if any overlapping window fired."""
    point_preds = np.zeros(series_length, dtype=bool)
    for alert, end_idx in zip(alerts, end_indices, strict=True):
        if not alert:
            continue
        start_idx = int(end_idx) - window_size + 1
        point_preds[max(0, start_idx) : int(end_idx) + 1] = True
    return point_preds
