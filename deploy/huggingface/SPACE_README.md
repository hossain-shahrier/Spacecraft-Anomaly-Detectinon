---
title: SMAP Spacecraft Anomaly Detection
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SMAP Spacecraft Anomaly Detection API

Unsupervised anomaly detection on **full NASA SMAP telemetry** (50 timesteps × 25 sensors per window) using an **Isolation Forest** and dynamic thresholding.

**Space SDK:** Docker (FastAPI backend).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Anomaly score for one window |
| `GET` | `/docs` | Interactive API docs (Swagger) |

## Predict input (full model)

- **`values`:** JSON array of **50 rows**, each row has **25** sensor readings (last 50 timesteps).
- Or a **flat** list of **1,250** floats (row-major).
- **`recent_scores`:** optional list of prior scores for dynamic thresholding.

```json
{
  "channel_id": "E-1",
  "values": [[... 25 floats ...], "... 50 rows total ..."],
  "recent_scores": [0.4, 0.42, 0.41]
}
```

## Example response

```json
{
  "anomaly_score": 0.52,
  "threshold": 0.65,
  "status": "normal",
  "latency_ms": 12.3
}
```

`status` is `normal` or `hardware_degradation`.

## Try it

Open **`/docs`** on this Space and run **POST /predict** from the browser.

## Source

Trained locally on the [NASA SMAP/MSL Kaggle dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl); this Space ships **inference artifacts only** (no raw training data).
