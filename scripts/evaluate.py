#!/usr/bin/env python3
"""Evaluate SMAP test channels with dynamic thresholding and Point-Adjusted F1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loader import (
    build_point_labels,
    load_available_smap_channels,
    load_channel_series,
    load_smap_channels,
    series_timesteps,
)
from src.data.scaling import load_scaler, transform_windows
from src.data.windowing import propagate_window_alerts, sliding_windows, window_end_indices
from src.evaluation.metrics import point_adjusted_f1, point_adjusted_precision_recall
from src.models.isolation_forest import anomaly_scores, load_model
from src.models.threshold import DynamicThreshold


def evaluate_channel(
    chan_id: str,
    config,
    model,
    scaler,
) -> dict:
    series = load_channel_series(chan_id, "test", config)
    windows = sliding_windows(series, config.window_size, config.stride)
    if len(windows) == 0:
        return {"chan_id": chan_id, "pa_f1": 0.0, "precision": 0.0, "recall": 0.0, "n_windows": 0}

    scaled = transform_windows(scaler, windows)
    scores = anomaly_scores(model, scaled)

    threshold = DynamicThreshold(
        rolling_window=config.rolling_window,
        n_sigma=config.n_sigma,
    )
    alerts, _ = threshold.apply_sequence(scores)

    end_idx = window_end_indices(config.window_size, len(windows), config.stride)
    point_preds = propagate_window_alerts(
        alerts, end_idx, config.window_size, series_timesteps(series)
    )

    channels = {c.chan_id: c for c in load_smap_channels(config)}
    labels = build_point_labels(channels[chan_id])

    precision, recall = point_adjusted_precision_recall(labels, point_preds)
    f1 = point_adjusted_f1(labels, point_preds)

    anomaly_rate = float(labels.sum()) / len(labels) * 100
    return {
        "chan_id": chan_id,
        "pa_f1": f1,
        "precision": precision,
        "recall": recall,
        "n_windows": len(windows),
        "anomaly_rate_pct": anomaly_rate,
    }


def main() -> None:
    config = load_config()
    if not config.model_path.exists():
        print("Model not found. Run: python scripts/train.py")
        sys.exit(1)

    model = load_model(config.model_path)
    scaler = load_scaler(config.scaler_path)
    channels = load_available_smap_channels(config)

    print(f"Evaluating {len(channels)} SMAP channels...\n")
    print(f"{'Channel':<10} {'PA-F1':>8} {'Prec':>8} {'Recall':>8} {'Anom%':>8}")
    print("-" * 48)

    results = []
    for ch in channels:
        r = evaluate_channel(ch.chan_id, config, model, scaler)
        results.append(r)
        print(
            f"{r['chan_id']:<10} {r['pa_f1']:8.4f} {r['precision']:8.4f} "
            f"{r['recall']:8.4f} {r['anomaly_rate_pct']:7.3f}"
        )

    macro_f1 = float(np.mean([r["pa_f1"] for r in results]))
    macro_prec = float(np.mean([r["precision"] for r in results]))
    macro_rec = float(np.mean([r["recall"] for r in results]))

    print("-" * 48)
    print(f"{'MACRO':<10} {macro_f1:8.4f} {macro_prec:8.4f} {macro_rec:8.4f}")
    print(f"\nMacro Point-Adjusted F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
