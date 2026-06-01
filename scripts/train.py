#!/usr/bin/env python3
"""Train global Isolation Forest on pooled SMAP train windows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.loader import (
    load_available_smap_channels,
    load_channel_series,
    series_timesteps,
)
from src.data.scaling import fit_scaler, save_scaler, transform_windows
from src.data.windowing import sliding_windows
from src.models.isolation_forest import save_model, train_global_model
from src.models.threshold import DynamicThreshold, save_threshold_config


def collect_train_windows(config) -> np.ndarray:
    channels = load_available_smap_channels(config)
    all_windows: list[np.ndarray] = []
    for ch in channels:
        series = load_channel_series(ch.chan_id, "train", config)
        windows = sliding_windows(series, config.window_size, config.stride)
        if len(windows):
            all_windows.append(windows)
        print(f"  {ch.chan_id}: {series_timesteps(series)} points -> {len(windows)} windows")

    if not all_windows:
        raise RuntimeError("No train windows found. Download the SMAP dataset first.")
    return np.vstack(all_windows)


def main() -> None:
    config = load_config()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SMAP train windows...")
    raw_windows = collect_train_windows(config)
    print(f"Total train windows: {raw_windows.shape[0]}")

    print("Fitting StandardScaler...")
    scaler = fit_scaler(raw_windows)
    scaled_windows = transform_windows(scaler, raw_windows)
    save_scaler(scaler, config.scaler_path)

    print("Training Isolation Forest...")
    model = train_global_model(scaled_windows, config)
    save_model(model, config.model_path)

    threshold = DynamicThreshold(
        rolling_window=config.rolling_window,
        n_sigma=config.n_sigma,
    )
    save_threshold_config(threshold, config.threshold_config_path)

    print(f"Saved scaler -> {config.scaler_path}")
    print(f"Saved model  -> {config.model_path}")
    print(f"Saved threshold config -> {config.threshold_config_path}")
    print("Done.")


if __name__ == "__main__":
    main()
