"""End-to-end train, evaluate, and API latency smoke tests."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from joblib import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_artifacts():
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py")], check=True, cwd=ROOT)
    return ROOT


def test_train_and_evaluate(trained_artifacts):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0
    assert "Macro Point-Adjusted F1" in result.stdout


def test_predict_latency_under_100ms(trained_artifacts):
    from src.api.main import app

    scaler = load(ROOT / "artifacts" / "scaler.joblib")
    feature_width = int(scaler.n_features_in_)
    n_dims = max(1, feature_width // 50)
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, (50, n_dims)).tolist()

    latencies = []
    with TestClient(app) as client:
        for _ in range(20):
            t0 = time.perf_counter()
            resp = client.post(
                "/predict",
                json={
                    "channel_id": "T-1",
                    "values": values,
                    "recent_scores": [0.4, 0.42, 0.41],
                },
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            assert resp.status_code == 200

    p95 = float(np.percentile(latencies, 95))
    assert p95 < 100.0, f"p95 latency {p95:.1f}ms exceeds 100ms budget"
