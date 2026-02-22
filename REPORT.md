# Assignment 2 Report — Cats vs Dogs End-to-End MLOps Pipeline (50 Marks)

## Use case
Binary image classification (Cats vs Dogs) for a pet adoption platform.

## Dataset
Kaggle Cats vs Dogs (PetImages). The pipeline downloads a public mirror (`kagglecatsanddogs_5340.zip`) and extracts to:
`data/raw/PetImages/{Cat,Dog}/*.jpg`.

> Note: The PetImages dataset contains known corrupted images. The preprocessing stage automatically filters unreadable files.

## M1 — Model Development & Experiment Tracking (10M)
**Data & code versioning**
- Git is used for source versioning.
- DVC is supported via `dvc.yaml` (stages: ingest → preprocess → train). Dataset artifacts are intended to be tracked by DVC.

**Preprocessing**
- Resizes to **224×224 RGB**
- Splits into **train/val/test = 80/10/10**
- Saves split manifest: `data/processed/splits.csv`
- Removes corrupted images automatically

**Model**
- Baseline **SimpleCNN** (PyTorch)
- Saved artifact: `models/catsdogs_cnn.pt`

**Experiment tracking**
- MLflow logging enabled (params, metrics, artifacts)
- Artifacts include confusion matrix + loss curve

**Generated artifacts**
- `artifacts/plots/confusion_matrix.png`
- `artifacts/plots/loss_curve.png`
- `artifacts/metrics/metrics.json`
- `artifacts/metrics/classification_report.txt`

## M2 — Packaging & Containerization (10M)
**FastAPI inference service**
- `GET /health` — service + model status
- `POST /predict` — image upload, returns label + probabilities
- Swagger UI at `GET /docs`
- Prometheus metrics at `GET /metrics`

**Reproducible environment**
- Pinned dependencies in `requirements.txt`

**Docker**
- Dockerfile: `deployment/docker/Dockerfile`
- Container runs: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

## M3 — CI Pipeline (10M)
GitHub Actions workflow: `.github/workflows/ci.yml`
- Installs dependencies
- Runs `ruff` and `pytest`
- Runs pipeline smoke test using a **tiny synthetic PetImages** dataset (no external downloads)
- Builds the Docker image

## M4 — CD Pipeline & Deployment (10M)
GitHub Actions workflow: `.github/workflows/cd.yml`
- Builds and pushes image to GHCR
- On `main`, deploys via docker-compose and runs a post-deploy smoke test:
  - health check
  - one prediction call

Deployment options included:
- Docker Compose: `deployment/docker-compose.yml`
- Kubernetes manifests: `deployment/k8s/`

## M5 — Monitoring & Logs (10M)
- Request logging middleware (no image bytes logged)
- Prometheus metrics (`/metrics`) + Docker-compose Prometheus scrape
- Grafana provisioning included under `deployment/monitoring/grafana/`

## How to run (local)
```bash
uv venv
uv pip install -r requirements.txt
python scripts/run_pipeline.py
export MODEL_WEIGHTS_PATH=models/catsdogs_cnn.pt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## How to run (docker-compose)
```bash
docker compose -f deployment/docker-compose.yml up --build
# API:      http://localhost:8000/docs
# MLflow:   http://localhost:5000
# Prom:     http://localhost:9090
# Grafana:  http://localhost:3000
```

## Submission
The submission ZIP contains code + configs + model artifact and excludes the full raw dataset.
