# Deploy on Hugging Face Spaces (full SMAP model)

## Which Space SDK to choose?

| SDK | Use for this project? |
|-----|------------------------|
| **Docker** | **Yes — use this.** You already have FastAPI + `Dockerfile` + trained `artifacts/`. |
| Gradio | No (unless you build a separate UI). This repo is an API, not a Gradio app. |
| Streamlit | No (same reason). |
| Static | No. There is no static frontend; inference runs in Python. |

When creating the Space, select **Docker** and **app port 7860**.

---

## What gets deployed

| On the Space | On your machine only |
|--------------|----------------------|
| `Dockerfile`, `src/`, `config/`, `requirements.txt` | Full `data/` (train/test `.npy`) |
| Trained `artifacts/` (from full SMAP training) | `scripts/train.py` at deploy time |

The Space serves **inference only**. Train once locally on the full NASA dataset, then push **artifacts**.

---

## Before you start

### 1. Full dataset locally

```powershell
# Windows
.\scripts\download_data.ps1
```

```bash
# Linux/macOS
bash scripts/download_data.sh
```

Restore real labels if needed (file should be ~4 KB, not ~95 bytes):

```bash
curl -fsSL -o data/labeled_anomalies.csv \
  https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

### 2. Train on full SMAP

```bash
python scripts/train.py
python scripts/evaluate.py
```

Expect evaluation on **~53 channels**. Artifacts use **1,250 features** per window (50 timesteps × 25 sensors).

### 3. Commit artifacts to Git

Artifacts are gitignored by default. Hugging Face must receive them in the repo (or via Git LFS):

```bash
git add -f artifacts/model.joblib artifacts/scaler.joblib artifacts/threshold_state.json
git commit -m "Add full SMAP model artifacts for HF Space"
git push
```

---

## Create the Space

1. Open [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Owner / name:** e.g. `your-username/smap-anomaly-api`.
3. **Space SDK:** **Docker** (not Gradio, not Static).
4. **Space hardware:** CPU basic (free).
5. **Visibility:** Public (required for free tier).
6. **Repository:** connect this GitHub repo (or push the same files into the Space repo).

### Build settings

| Field | Value |
|-------|--------|
| SDK | `docker` |
| App port | `7860` |
| Dockerfile path | `Dockerfile` (repository root) |

### Space README

Copy [`SPACE_README.md`](./SPACE_README.md) into the Space **README** on Hugging Face (YAML frontmatter at the top is required for Docker Spaces).

---

## Test after deploy

Replace `YOUR_USER` and `YOUR_SPACE`:

```bash
curl https://YOUR_USER-YOUR_SPACE.hf.space/health
```

**Predict** requires **50 timesteps × 25 sensors** as a JSON 2D array (`values`), or a flat list of **1,250** floats.

Easiest way to try requests: open

```text
https://YOUR_USER-YOUR_SPACE.hf.space/docs
```

Use **POST /predict** in Swagger with a body like:

```json
{
  "channel_id": "E-1",
  "values": [
    [0.1, 0.2, 0.15, 0.05, 0.03, 0.08, 0.11, 0.09, 0.07, 0.13, 0.1, 0.2, 0.15, 0.05, 0.03, 0.08, 0.11, 0.09, 0.07, 0.13, 0.1, 0.2, 0.15, 0.05, 0.03],
    "... repeat for 50 rows total ..."
  ],
  "recent_scores": [0.4, 0.42, 0.41]
}
```

Each inner array must have **25** numbers (one row per timestep).

---

## Local test (same Docker image as HF)

```bash
docker build -t smap-hf .
docker run -p 7860:7860 smap-hf
curl http://127.0.0.1:7860/health
```

---

## Free tier notes

- CPU only (enough for Isolation Forest).
- Space may **sleep** when idle; first request after sleep can be slow.
- Do **not** upload `data/` — only code + `artifacts/`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build: artifacts missing | Train on full data; `git add -f artifacts/*` and push |
| `/predict` 400 invalid shape | Send 50×25 `values`, not 50 floats |
| Only 1 channel at train time | Restore full `labeled_anomalies.csv` before `train.py` |
| App not ready | Logs tab → confirm listen on `0.0.0.0:7860` |
