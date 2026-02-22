# deployment/docker/Dockerfile
# Inference service container for Cats vs Dogs (FastAPI)

FROM python:3.11-slim

ARG APP_ENV=nonprod
ARG GIT_SHA=unknown
ARG GIT_REF=unknown

ENV APP_ENV=${APP_ENV}
ENV GIT_SHA=${GIT_SHA}
ENV GIT_REF=${GIT_REF}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install curl (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY src /app/src
COPY models /app/models

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck (optional if using in compose)
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]