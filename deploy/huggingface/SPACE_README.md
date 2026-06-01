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

Unsupervised anomaly detection for NASA **SMAP** telemetry using an **Isolation Forest** and dynamic thresholding.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | Score a telemetry window |
| `GET` | `/docs` | Swagger UI |

## Example: health

```bash
curl https://YOUR_USER-YOUR_SPACE.hf.space/health
```

## Example: predict

Send the last **50 timesteps** (synthetic demo model) or **50×25** grid (full SMAP model).

```json
{
  "channel_id": "T-1",
  "values": [0.1, 0.2, "..."],
  "recent_scores": [0.4, 0.42, 0.41]
}
```

Response:

```json
{
  "anomaly_score": 0.52,
  "threshold": 0.65,
  "status": "normal",
  "latency_ms": 12.3
}
```

`status` is `normal` or `hardware_degradation`.

## Source

GitHub: link your repository here.

Train locally with `python scripts/train.py`, then deploy artifacts via Docker Space.
