"""Load project configuration from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "smap.yaml"


@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    max_samples: str | float = "auto"
    contamination: str | float = "auto"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class AppConfig:
    spacecraft: str = "SMAP"
    exclude_channels: list[str] = field(default_factory=lambda: ["P-2"])
    window_size: int = 50
    stride: int = 1
    rolling_window: int = 100
    n_sigma: float = 3.0
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    labels_file: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "labeled_anomalies.csv")
    train_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "train")
    test_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "test")

    @property
    def scaler_path(self) -> Path:
        return self.artifacts_dir / "scaler.joblib"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "model.joblib"

    @property
    def threshold_config_path(self) -> Path:
        return self.artifacts_dir / "threshold_state.json"


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if_cfg = raw.get("isolation_forest", {})
    paths = raw.get("paths", {})
    data_dir = PROJECT_ROOT / paths.get("data_dir", "data")

    return AppConfig(
        spacecraft=raw.get("spacecraft", "SMAP"),
        exclude_channels=list(raw.get("exclude_channels", ["P-2"])),
        window_size=int(raw.get("window_size", 50)),
        stride=int(raw.get("stride", 1)),
        rolling_window=int(raw.get("rolling_window", 100)),
        n_sigma=float(raw.get("n_sigma", 3.0)),
        isolation_forest=IsolationForestConfig(
            n_estimators=int(if_cfg.get("n_estimators", 200)),
            max_samples=if_cfg.get("max_samples", "auto"),
            contamination=if_cfg.get("contamination", "auto"),
            random_state=int(if_cfg.get("random_state", 42)),
            n_jobs=int(if_cfg.get("n_jobs", -1)),
        ),
        data_dir=data_dir,
        artifacts_dir=PROJECT_ROOT / paths.get("artifacts_dir", "artifacts"),
        labels_file=PROJECT_ROOT / paths.get("labels_file", "data/labeled_anomalies.csv"),
        train_dir=PROJECT_ROOT / paths.get("train_dir", "data/train"),
        test_dir=PROJECT_ROOT / paths.get("test_dir", "data/test"),
    )
