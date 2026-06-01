"""Isolation Forest training and continuous anomaly scoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import AppConfig, IsolationForestConfig, load_config


def build_isolation_forest(if_config: IsolationForestConfig | None = None) -> IsolationForest:
    if_config = if_config or load_config().isolation_forest
    return IsolationForest(
        n_estimators=if_config.n_estimators,
        max_samples=if_config.max_samples,
        contamination=if_config.contamination,
        random_state=if_config.random_state,
        n_jobs=if_config.n_jobs,
    )


def anomaly_scores(model: IsolationForest, windows: np.ndarray) -> np.ndarray:
    """Higher score means more anomalous."""
    raw = model.score_samples(windows)
    return -np.asarray(raw, dtype=np.float64)


def score_window(model: IsolationForest, window: np.ndarray) -> float:
    """Score a single window of shape (window_size,) or (1, window_size)."""
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return float(anomaly_scores(model, arr)[0])


def save_model(model: IsolationForest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> IsolationForest:
    return joblib.load(path)


def train_global_model(
    train_windows: np.ndarray,
    config: AppConfig | None = None,
) -> IsolationForest:
    config = config or load_config()
    model = build_isolation_forest(config.isolation_forest)
    model.fit(train_windows)
    return model
