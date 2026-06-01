import numpy as np

from src.data.windowing import propagate_window_alerts, sliding_windows, window_end_indices


def test_sliding_windows_shape():
    series = np.arange(100, dtype=np.float64)
    windows = sliding_windows(series, window_size=10, stride=1)
    assert windows.shape == (91, 10)
    assert np.allclose(windows[0], series[:10])


def test_window_end_indices():
    ends = window_end_indices(window_size=10, n_windows=5, stride=2)
    assert list(ends) == [9, 11, 13, 15, 17]


def test_propagate_window_alerts():
    alerts = np.array([False, True, False])
    ends = window_end_indices(5, 3, 1)
    point = propagate_window_alerts(alerts, ends, window_size=5, series_length=10)
    assert point.sum() >= 1
