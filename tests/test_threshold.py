import numpy as np

from src.models.threshold import DynamicThreshold


def test_dynamic_threshold_spike():
    threshold = DynamicThreshold(rolling_window=20, n_sigma=3.0)
    threshold.seed([0.5] * 30)

    result = threshold.evaluate(10.0)
    assert result.alert is True
    assert result.status == "hardware_degradation"


def test_recent_scores_override():
    threshold = DynamicThreshold(rolling_window=100, n_sigma=3.0)
    recent = [0.5, 0.52, 0.48, 0.51]
    result = threshold.evaluate(5.0, recent_scores=recent)
    assert result.alert is True
