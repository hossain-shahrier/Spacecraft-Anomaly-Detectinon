#!/usr/bin/env python3
"""Generate minimal synthetic SMAP-like data for tests and smoke runs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHAN_ID = "T-1"
NUM_TRAIN = 500
NUM_TEST = 300
WINDOW_ANOMALY = (200, 230)


def main() -> None:
    data_dir = ROOT / "data"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    train = rng.normal(0.0, 1.0, NUM_TRAIN)
    test = rng.normal(0.0, 1.0, NUM_TEST)
    test[WINDOW_ANOMALY[0] : WINDOW_ANOMALY[1] + 1] += 8.0

    np.save(train_dir / f"{CHAN_ID}.npy", train.astype(np.float64))
    np.save(test_dir / f"{CHAN_ID}.npy", test.astype(np.float64))

    # Test fixture: single-channel labels file only
    labels_path = data_dir / "labeled_anomalies.csv"
    df = pd.DataFrame(
        [
            {
                "chan_id": CHAN_ID,
                "spacecraft": "SMAP",
                "anomaly_sequences": str([list(WINDOW_ANOMALY)]),
                "class": "contextual",
                "num_values": NUM_TEST,
            }
        ]
    )
    df.to_csv(labels_path, index=False)

    print(f"Synthetic channel {CHAN_ID} written to {data_dir}")


if __name__ == "__main__":
    main()
