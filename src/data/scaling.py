"""StandardScaler fit/transform for window features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler(windows: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(windows)
    return scaler


def transform_windows(scaler: StandardScaler, windows: np.ndarray) -> np.ndarray:
    return scaler.transform(windows)


def save_scaler(scaler: StandardScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path) -> StandardScaler:
    return joblib.load(path)
