# deployment/docker/mlflow/Dockerfile
# Custom MLflow image (no GHCR dependency)

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow + DB + S3 support
RUN pip install --no-cache-dir \
    mlflow==2.11.3 \
    psycopg2-binary \
    boto3

EXPOSE 5000

# MLflow server startup
CMD mlflow server \
  --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow \
  --default-artifact-root s3://mlflow \
  --host 0.0.0.0 \
  --port 5000