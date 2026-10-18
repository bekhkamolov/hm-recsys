# hm-recsys recommendation API: ranks precomputed candidates with the LightGBM ranker and falls
# back to popularity for unknown customers.
#   docker build -t hm-recsys .
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" hm-recsys
# The image has only the serving dependencies (no duckdb, implicit or MLflow). Real model artifacts
# are not part of the image: build them with scripts/build_serving.py (H&M data) and mount the
# folder at /app/artifacts. A synthetic demo model is baked in at /app/demo-artifacts.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ARTIFACT_DIR=/app/artifacts

# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The package is installed with only the `serve` extra (simple, small project; a code change
# rebuilds this layer).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# A small synthetic demo model, so the image can serve out of the box (that is what runs on
# Render: ARTIFACT_DIR=/app/demo-artifacts). It is not H&M data.
COPY deploy/demo-artifacts ./demo-artifacts

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 app && mkdir -p /app/artifacts && chown app /app/artifacts
USER app

# Hosting platforms (e.g. Render) pass the port in $PORT; 8000 otherwise.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/health', timeout=4)"

# exec: uvicorn replaces the shell, so it receives stop signals directly and shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn hm_recsys.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
