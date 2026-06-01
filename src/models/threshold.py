"""Dynamic n-sigma thresholding on anomaly score streams."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ThresholdResult:
    score: float
    threshold: float
    alert: bool
    status: str


class DynamicThreshold:
    """Rolling mean + n*std boundary; alerts when score exceeds threshold."""

    def __init__(self, rolling_window: int = 100, n_sigma: float = 3.0):
        self.rolling_window = rolling_window
        self.n_sigma = n_sigma
        self._history: deque[float] = deque(maxlen=rolling_window)

    def reset(self) -> None:
        self._history.clear()

    def seed(self, scores: list[float] | np.ndarray) -> None:
        self.reset()
        for s in scores:
            self._history.append(float(s))

    def evaluate(self, score: float, recent_scores: list[float] | None = None) -> ThresholdResult:
        if recent_scores is not None:
            history = [float(s) for s in recent_scores]
        else:
            history = list(self._history)
            self._history.append(score)

        threshold = self._compute_threshold(history)
        alert = score > threshold
        status = "hardware_degradation" if alert else "normal"
        return ThresholdResult(score=score, threshold=threshold, alert=alert, status=status)

    def apply_sequence(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (alerts, thresholds) for a score sequence."""
        alerts = np.zeros(len(scores), dtype=bool)
        thresholds = np.zeros(len(scores), dtype=np.float64)
        self.reset()
        for i, score in enumerate(scores):
            result = self.evaluate(float(score))
            alerts[i] = result.alert
            thresholds[i] = result.threshold
        return alerts, thresholds

    def _compute_threshold(self, history: list[float]) -> float:
        if not history:
            return float("inf")
        arr = np.asarray(history, dtype=np.float64)
        if len(arr) == 1:
            return arr[0] + self.n_sigma * 1e-6
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        if std == 0.0:
            std = 1e-6
        return mean + self.n_sigma * std

    def to_dict(self) -> dict:
        return {
            "rolling_window": self.rolling_window,
            "n_sigma": self.n_sigma,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DynamicThreshold:
        return cls(
            rolling_window=int(data.get("rolling_window", 100)),
            n_sigma=float(data.get("n_sigma", 3.0)),
        )


def save_threshold_config(threshold: DynamicThreshold, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(threshold.to_dict(), f, indent=2)


def load_threshold_config(path: Path) -> DynamicThreshold:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return DynamicThreshold.from_dict(data)
