import numpy as np

from src.evaluation.metrics import point_adjusted_f1, point_adjusted_precision_recall


def test_pa_f1_detects_segment():
    y_true = np.zeros(100, dtype=bool)
    y_true[30:40] = True
    y_pred = np.zeros(100, dtype=bool)
    y_pred[35] = True

    precision, recall = point_adjusted_precision_recall(y_true, y_pred)
    assert recall == 1.0
    assert precision == 1.0
    assert point_adjusted_f1(y_true, y_pred) == 1.0


def test_pa_f1_miss():
    y_true = np.zeros(100, dtype=bool)
    y_true[30:40] = True
    y_pred = np.zeros(100, dtype=bool)

    _, recall = point_adjusted_precision_recall(y_true, y_pred)
    assert recall == 0.0
