# Deploy on Hugging Face Spaces (free)

This guide deploys **only the API** (`/health`, `/predict`). Train locally first, then push artifacts with your repo.

## Before you start

1. Train and save artifacts:

```bash
# Demo Space (simple 50-number /predict requests)
python scripts/generate_synthetic_data.py
python scripts/train.py

# OR full SMAP (50×25 /predict requests — see README)
python scripts/train.py
```

2. Confirm these files exist:

- `artifacts/model.joblib`
- `artifacts/scaler.joblib`
- `artifacts/threshold_state.json`

3. **Include artifacts in Git** (they are gitignored by default). For HF Docker build you must either:

- Temporarily allow artifacts in git (recommended for a small demo model), or  
- Use [Git LFS](https://git-lfs.github.com/) for `*.joblib`, or  
- Upload artifacts in a private release and `curl` them in `Dockerfile` (advanced).

Example to track artifacts once:

```bash
git add -f artifacts/model.joblib artifacts/scaler.joblib artifacts/threshold_state.json
git commit -m "Add trained model artifacts for HF Space"
git push
```

## Step 1 — Push code to GitHub

Hugging Face pulls from GitHub (or you can upload files manually). Your repo root must contain:

- `Dockerfile` (listens on port **7860** by default for HF)
- `requirements.txt`, `src/`, `config/`, `artifacts/`
- Space README with Docker SDK header (see `SPACE_README.md`)

## Step 2 — Create a Docker Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Space SDK:** Docker.
3. **Hardware:** CPU basic (free).
4. **Visibility:** Public (required for free tier).
5. Connect your GitHub repo (or duplicate files into the Space repo).

## Step 3 — Space README (card)

On Hugging Face, open the Space **README** editor and paste the contents of [`SPACE_README.md`](./SPACE_README.md) (YAML frontmatter + short description).

Or merge that YAML block at the top of your GitHub `README.md` if the Space uses the same file.

## Step 4 — Build settings

| Setting | Value |
|---------|--------|
| SDK | Docker |
| App port | `7860` (default; matches `Dockerfile`) |
| Dockerfile | `Dockerfile` (repo root) |

Click **Create Space** and wait for the Docker build. First build may take several minutes.

## Step 5 — Test the live API

Replace `YOUR_USER` and `YOUR_SPACE`:

```bash
curl https://YOUR_USER-YOUR_SPACE.hf.space/health
```

**Synthetic model** (50 floats):

```bash
curl -X POST "https://YOUR_USER-YOUR_SPACE.hf.space/predict" \
  -H "Content-Type: application/json" \
  -d "{\"channel_id\":\"T-1\",\"values\":[0.1,0.2,0.15,0.13,0.12,0.11,0.1,0.2,0.15,0.13,0.12,0.11,0.1,0.2,0.15,0.13,0.12,0.11,0.1,0.2,0.15,0.13,0.12,0.11,0.1,0.2,0.15,0.13,0.12,0.11,0.1,0.2,0.15,0.13,0.12,0.11,0.1,0.2,0.15,0.13,0.12,0.11,0.1,0.2,0.15,0.13,0.12,0.11,0.1,0.2],\"recent_scores\":[0.4,0.42,0.41]}"
```

**Full SMAP model** (50 rows × 25 sensors): send `values` as a JSON 2D array (see main README).

Interactive docs: `https://YOUR_USER-YOUR_SPACE.hf.space/docs`

## Local test (same image as HF)

```bash
docker build -t smap-hf .
docker run -p 7860:7860 smap-hf
curl http://127.0.0.1:7860/health
```

## Free tier notes

- CPU only; no GPU needed for Isolation Forest.
- Space sleeps when idle; first request after sleep can be slow (cold start).
- Public Spaces only on the free plan.
- Do **not** upload the full `data/` folder — only `artifacts/`.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails: `Artifacts not found` | Train locally; `git add -f artifacts/*.joblib` and push |
| `/predict` returns 400 invalid shape | Model trained on full data needs 50×25 input; synthetic needs 50 floats |
| Build fails: missing `artifacts/` in context | `.dockerignore` must not exclude `artifacts/` |
| App never becomes ready | Check Logs tab; ensure app listens on `0.0.0.0:7860` |
