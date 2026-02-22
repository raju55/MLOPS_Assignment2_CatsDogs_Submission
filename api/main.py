"""
api.main

FastAPI inference service for Cats vs Dogs classification (Assignment 2).

Requirements mapping:
- Inference Service with REST API (FastAPI)
- Endpoints:
    GET  /health   -> service + model status
    POST /predict  -> accepts an image upload, returns class probabilities + label
    GET  /metrics  -> Prometheus scrape endpoint
- Basic request logging (excluding sensitive data)
- Basic metrics: request count + latency

Swagger/OpenAPI:
- Swagger UI:  /docs
- ReDoc UI:    /redoc
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from PIL import Image

from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.catsdogs.logging_config import setup_logging
from src.catsdogs.predict import load_model, predict_image

log = logging.getLogger("catsdogs.api")

# -----------------------------
# Prometheus metrics
# -----------------------------
MODEL_LOADED = Gauge("model_loaded", "Whether the model is loaded (1=yes, 0=no)")
PREDICTIONS_TOTAL = Counter("predictions_total", "Total number of predictions made")
PREDICTION_ERRORS_TOTAL = Counter("prediction_errors_total", "Total number of prediction errors")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference latency in seconds")

# -----------------------------
# Model state
# -----------------------------
_model = None  # type: ignore
_device = None  # type: ignore
_model_lock = threading.Lock()


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    model_source: str | None = None


class PredictResponse(BaseModel):
    label: str = Field(..., examples=["Cat"])
    prob_cat: float = Field(..., ge=0.0, le=1.0)
    prob_dog: float = Field(..., ge=0.0, le=1.0)
    inference_ms: float


def _load_local_weights() -> Tuple[Any, Any, str]:
    """
    Load model weights from a local .pt file.
    """
    # Allow override for grading/demo environments
    weights_path = os.getenv("MODEL_WEIGHTS_PATH", "models/catsdogs_cnn.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"MODEL_WEIGHTS_PATH not found: {weights_path}")
    model, device = load_model(PathLike(weights_path))
    return model, device, f"local:{weights_path}"


def PathLike(path_str: str):
    # small helper to avoid importing pathlib in hot-path
    from pathlib import Path
    return Path(path_str)


def _try_load_model() -> str:
    """
    Best-effort model loading (never crash the API).
    Returns model_source string when loaded.
    """
    global _model, _device
    with _model_lock:
        # already loaded?
        if _model is not None and _device is not None:
            return os.getenv("MODEL_SOURCE", "loaded")

        # 1) Local weights (default for assignment submission zip)
        try:
            from pathlib import Path
            weights_path = Path(os.getenv("MODEL_WEIGHTS_PATH", "models/catsdogs_cnn.pt"))
            model, device = load_model(weights_path)
            _model, _device = model, device
            MODEL_LOADED.set(1)
            source = f"local:{weights_path}"
            log.info("Model loaded from %s", source)
            os.environ["MODEL_SOURCE"] = source
            return source
        except Exception as e:
            log.warning("Local model load failed: %s", e)

        # 2) Optional: MLflow registry (if used in docker-compose)
        try:
            import mlflow
            import mlflow.pytorch

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            model_name = os.getenv("MLFLOW_MODEL_NAME", "cats-dogs")
            stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
            uri = f"models:/{model_name}/{stage}"
            log.info("Attempting to load MLflow model from %s", uri)
            model = mlflow.pytorch.load_model(uri)
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            _model, _device = model, device
            MODEL_LOADED.set(1)
            source = f"mlflow:{uri}"
            log.info("Model loaded from %s", source)
            os.environ["MODEL_SOURCE"] = source
            return source
        except Exception as e:
            log.warning("MLflow model load failed: %s", e)

        MODEL_LOADED.set(0)
        raise RuntimeError("Model could not be loaded (local weights and MLflow both failed).")


def _background_retry_loop(retry_seconds: int) -> None:
    while True:
        try:
            _try_load_model()
            return
        except Exception:
            log.info("Retrying model load in %ds ...", retry_seconds)
            time.sleep(retry_seconds)


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title="Cats vs Dogs Inference API",
        description=(
            "Binary image classification service for a pet adoption platform.\n\n"
            "Use **/docs** for Swagger UI."
        ),
        version=os.getenv("API_VERSION", "1.0.0"),
        openapi_tags=[
            {"name": "health", "description": "Health checks & readiness"},
            {"name": "inference", "description": "Prediction endpoints"},
            {"name": "monitoring", "description": "Metrics for Prometheus"},
        ],
    )

    Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            dur_ms = (time.time() - start) * 1000.0
            status = getattr(response, "status_code", "NA")
            log.info("%s %s -> %s (%.1fms)", request.method, request.url.path, status, dur_ms)

    @app.on_event("startup")
    def startup() -> None:
        # attempt load; if configured, retry in background
        retry = os.getenv("MODEL_LOAD_RETRY", "true").lower() in {"1", "true", "yes"}
        retry_seconds = int(os.getenv("MODEL_LOAD_RETRY_SECONDS", "20"))
        try:
            _try_load_model()
        except Exception:
            log.warning("Model not loaded at startup. Service will still run.")
            if retry:
                t = threading.Thread(target=_background_retry_loop, args=(retry_seconds,), daemon=True)
                t.start()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        # Simple HTML UI for quick manual grading/demo
        return """
        <html>
          <head><title>Cats vs Dogs API</title></head>
          <body style="font-family: Arial; max-width: 720px; margin: 40px auto;">
            <h2>Cats vs Dogs - Inference API</h2>
            <p>Swagger UI: <a href="/docs">/docs</a> • ReDoc: <a href="/redoc">/redoc</a> • Metrics: <a href="/metrics">/metrics</a></p>
            <form id="f">
              <input type="file" id="file" accept="image/*" />
              <button type="submit">Predict</button>
            </form>
            <pre id="out" style="background:#f6f6f6; padding:12px; border-radius:8px;"></pre>
            <script>
              const f = document.getElementById('f');
              f.addEventListener('submit', async (e) => {
                e.preventDefault();
                const file = document.getElementById('file').files[0];
                if(!file){ alert('Choose an image'); return; }
                const fd = new FormData();
                fd.append('file', file);
                const res = await fetch('/predict', { method: 'POST', body: fd });
                const txt = await res.text();
                document.getElementById('out').textContent = txt;
              });
            </script>
          </body>
        </html>
        """

    @app.get("/health", tags=["health"], response_model=HealthResponse, summary="Health check")
    def health() -> HealthResponse:
        loaded = _model is not None
        return HealthResponse(status="ok", model_loaded=loaded, model_source=os.getenv("MODEL_SOURCE"))

    @app.post(
        "/predict",
        tags=["inference"],
        response_model=PredictResponse,
        summary="Predict Cats vs Dogs",
        description="Upload an image file; returns predicted label and class probabilities.",
    )
    async def predict(file: UploadFile = File(...)) -> PredictResponse:
        global _model, _device

        if _model is None or _device is None:
            try:
                _try_load_model()
            except Exception as e:
                MODEL_LOADED.set(0)
                raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")

        # Read file bytes (do NOT log content)
        try:
            data = await file.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            PREDICTION_ERRORS_TOTAL.inc()
            raise HTTPException(status_code=400, detail="Invalid image upload")

        t0 = time.time()
        try:
            with INFERENCE_LATENCY.time():
                out = predict_image(_model, _device, img)
            PREDICTIONS_TOTAL.inc()
        except Exception as e:
            PREDICTION_ERRORS_TOTAL.inc()
            log.error("Prediction failed: %s\n%s", e, traceback.format_exc())
            raise HTTPException(status_code=500, detail="Prediction failed")

        inference_ms = (time.time() - t0) * 1000.0
        return PredictResponse(
            label=str(out["label"]),
            prob_cat=float(out["prob_cat"]),
            prob_dog=float(out["prob_dog"]),
            inference_ms=float(inference_ms),
        )

    return app


app = create_app()
