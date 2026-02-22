# MLOps Assignment 2 (Cats vs Dogs) — End-to-End Pipeline (50 Marks)

**Use case:** Binary image classification (Cats vs Dogs) for a pet adoption platform  
**Dataset:** Kaggle Cats vs Dogs (PetImages)  
**Preprocessing:** 224×224 RGB, train/val/test split **80/10/10**, data augmentation in training  
**Experiment tracking:** MLflow  
**Serving:** FastAPI + Swagger UI (`/docs`)  
**CI/CD:** GitHub Actions (CI: tests + build; CD: build/push image + deploy smoke test)  
**Monitoring:** Prometheus metrics (`/metrics`) + Grafana (optional via docker-compose)

---

## 1) Project structure (key folders)

- `src/catsdogs/` – data ingest, preprocessing, EDA, training, inference utilities
- `api/` – FastAPI inference service (`/health`, `/predict`, `/metrics`, Swagger `/docs`)
- `deployment/` – Dockerfile, docker-compose, k8s manifests, Prometheus/Grafana configs
- `.github/workflows/` – CI/CD pipelines
- `tests/` – unit tests for preprocessing + inference

---

## 2) Environment setup (uv recommended)

> Assignment requires `requirements.txt` (included), but **uv** is also supported for faster setup.

### Option A — uv
```bash
uv venv
uv pip install -r requirements.txt
```

### Option B — pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3) Run the full pipeline locally (end-to-end)

```bash
python scripts/run_pipeline.py
```

### Stage-wise (optional)
```bash
python -m src.catsdogs.data_ingest
python -m src.catsdogs.preprocess
python -m src.catsdogs.eda
python -m src.catsdogs.train
```

Outputs:
- `data/processed/` – resized/split dataset
- `artifacts/plots/` – EDA + confusion matrix + loss curve
- `artifacts/metrics/` – metrics.json + classification_report.txt
- `models/catsdogs_cnn.pt` – trained model artifact

---

## 4) MLflow UI (experiment tracking)

Local UI (file store):
```bash
mlflow ui --backend-store-uri mlruns --host 127.0.0.1 --port 5000
```

With docker-compose (MLflow + MinIO + Postgres) use the service:
- MLflow UI: http://localhost:5000

---

## 5) Run the API (Swagger + simple UI)

```bash
export MODEL_WEIGHTS_PATH=models/catsdogs_cnn.pt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open:
- **Swagger UI:** http://localhost:8000/docs
- **Simple upload UI:** http://localhost:8000/
- **Prometheus metrics:** http://localhost:8000/metrics

### Predict using curl
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path/to/image.jpg"
```

---

## 6) Docker & Deployment

### Build/run inference image
```bash
docker build -f deployment/docker/Dockerfile -t catsdogs-api:latest .
docker run -p 8000:8000 catsdogs-api:latest
```

### Docker compose (includes MLflow + monitoring)
```bash
docker compose -f deployment/docker-compose.yml up --build
```

---

## 7) DVC (dataset versioning)

This repo includes a starter `dvc.yaml` showing how to track:
- `data/raw/` (downloaded zip + extracted PetImages)
- `data/processed/` (224×224 split outputs)

Typical setup:
```bash
dvc init
dvc add data/raw/kagglecatsanddogs_5340.zip
dvc add data/processed
git add data/*.dvc dvc.yaml .dvc/config
git commit -m "Add DVC tracking for dataset"
```

---

## 8) CI/CD (GitHub Actions)

- **CI**: installs dependencies, runs unit tests, builds Docker image
- **CD**: pushes image to registry and runs a smoke test (health + predict) using docker-compose

---

## 9) Submission deliverables checklist (Assignment 2)

✅ Source code + configs (Docker, CI/CD, deployment, monitoring)  
✅ Trained model artifact (`models/catsdogs_cnn.pt`)  
✅ Experiment logs + artifacts (MLflow + plots)  
✅ API with Swagger UI and prediction endpoint  
✅ Unit tests (preprocess + inference)  
✅ Smoke test script (`scripts/smoke_test.py`)  
✅ Final ZIP (excluding huge raw dataset)

---

## Troubleshooting

### Windows: "RuntimeError: Numpy is not available" (PyTorch↔NumPy bridge)

On some Windows setups, `import torch` may emit a warning about NumPy and then
fail at runtime if code calls `torch.from_numpy()` or `Tensor.numpy()`.

This project avoids those calls in training/inference. If you still hit this in
your environment (due to other packages), the safest fix is to **delete the
virtual environment** and reinstall dependencies from scratch.



## Windows note (PyTorch + NumPy)
On Windows, PyTorch wheels may not be compatible with NumPy 2.x for `.from_numpy()`/`.numpy()`; this repo pins `numpy==1.26.4` to avoid `RuntimeError: Numpy is not available`.
