"""Pytest fixtures: synthetic SMAP-like telemetry."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def synthetic_data() -> None:
    script = ROOT / "scripts" / "generate_synthetic_data.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
