#!/usr/bin/env bash
# Download NASA SMAP/MSL telemetry dataset from Kaggle (Telemanom layout).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v kaggle &> /dev/null; then
  echo "Kaggle CLI not found. Install: pip install kaggle"
  echo "Configure API key: https://www.kaggle.com/docs/api"
  exit 1
fi

echo "Downloading nasa-anomaly-detection-dataset-smap-msl..."
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl -p "$ROOT" --unzip

# Normalize layout to data/train, data/test, data/labeled_anomalies.csv
if [ -d "$ROOT/data/data" ]; then
  mv "$ROOT/data/data/train" "$ROOT/data/train" 2>/dev/null || true
  mv "$ROOT/data/data/test" "$ROOT/data/test" 2>/dev/null || true
  rm -rf "$ROOT/data/data"
fi

if [ ! -f "$ROOT/data/labeled_anomalies.csv" ]; then
  echo "Fetching labeled_anomalies.csv from Telemanom..."
  curl -fsSL -o "$ROOT/data/labeled_anomalies.csv" \
    "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
fi

echo "Dataset ready under $ROOT/data/"
