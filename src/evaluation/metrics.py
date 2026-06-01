"""Point-Adjusted precision, recall, and F1 for anomaly detection."""

from __future__ import annotations

import numpy as np


def _extract_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, val in enumerate(labels):
        if val and start is None:
            start = i
        elif not val and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(labels) - 1))
    return segments


def point_adjusted_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """
    Point-adjusted precision and recall (Telemanom-style).

    If any prediction hits inside a true anomaly segment, the whole segment
    counts as detected for recall. For precision, true positives are adjusted
    when a predicted anomaly overlaps a true segment.
    """
    y_true = np.asarray(y_true, dtype=bool).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=bool).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    true_segments = _extract_segments(y_true)
    pred_segments = _extract_segments(y_pred)

    if not true_segments and not pred_segments:
        return 1.0, 1.0
    if not true_segments:
        return 0.0, 1.0
    if not pred_segments:
        return 1.0, 0.0

    tp_recall = 0
    for start, end in true_segments:
        if np.any(y_pred[start : end + 1]):
            tp_recall += 1
    recall = tp_recall / len(true_segments)

    tp_precision = 0
    for p_start, p_end in pred_segments:
        if np.any(y_true[p_start : p_end + 1]):
            tp_precision += 1
    precision = tp_precision / len(pred_segments) if pred_segments else 0.0

    return precision, recall


def point_adjusted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision, recall = point_adjusted_precision_recall(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
