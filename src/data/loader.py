"""Load NASA SMAP telemetry channels and anomaly labels."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AppConfig, load_config


@dataclass
class ChannelInfo:
    chan_id: str
    spacecraft: str
    anomaly_sequences: list[list[int]]
    anomaly_class: str
    num_values: int


def _parse_anomaly_sequences(value: str) -> list[list[int]]:
    parsed = ast.literal_eval(value)
    return [[int(start), int(end)] for start, end in parsed]


def load_smap_channels(config: AppConfig | None = None) -> list[ChannelInfo]:
    """Return metadata for SMAP channels (excluding configured channels)."""
    config = config or load_config()
    df = pd.read_csv(config.labels_file)
    df = df[df["spacecraft"] == config.spacecraft]
    df = df[~df["chan_id"].isin(config.exclude_channels)]

    channels: list[ChannelInfo] = []
    for _, row in df.iterrows():
        channels.append(
            ChannelInfo(
                chan_id=str(row["chan_id"]),
                spacecraft=str(row["spacecraft"]),
                anomaly_sequences=_parse_anomaly_sequences(str(row["anomaly_sequences"])),
                anomaly_class=str(row["class"]),
                num_values=int(row["num_values"]),
            )
        )
    return channels


def channel_file_exists(chan_id: str, split: str, config: AppConfig | None = None) -> bool:
    config = config or load_config()
    base = config.train_dir if split == "train" else config.test_dir
    return (Path(base) / f"{chan_id}.npy").exists()


def load_channel_series(chan_id: str, split: str, config: AppConfig | None = None) -> np.ndarray:
    """Load a single channel time series from train or test split."""
    config = config or load_config()
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    base = config.train_dir if split == "train" else config.test_dir
    path = Path(base) / f"{chan_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing channel file: {path}")
    series = np.load(path)
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D series in {path}, got shape {arr.shape}")


def series_timesteps(series: np.ndarray) -> int:
    """Number of time steps in a channel (handles multivariate (T, F) arrays)."""
    arr = np.asarray(series)
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim == 2:
        return int(arr.shape[0])
    raise ValueError(f"Expected 1D or 2D series, got shape {arr.shape}")


def load_available_smap_channels(config: AppConfig | None = None) -> list[ChannelInfo]:
    """SMAP channels that have both train and test files on disk."""
    config = config or load_config()
    return [
        ch
        for ch in load_smap_channels(config)
        if channel_file_exists(ch.chan_id, "train", config)
        and channel_file_exists(ch.chan_id, "test", config)
    ]


def build_point_labels(channel: ChannelInfo) -> np.ndarray:
    """Binary point-level labels for a test channel."""
    labels = np.zeros(channel.num_values, dtype=bool)
    for start, end in channel.anomaly_sequences:
        labels[start : end + 1] = True
    return labels
