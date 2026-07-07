# Description Matching System - Model Score Extraction API
# Multi-stage-free, minimal image: install deps, copy the model artifact
# and code, run as a non-root user, expose a health check Docker itself
# can use to decide if the container is actually ready.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached across code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code and the versioned model artifact it serves
COPY src/ml_utils.py src/score_extractor.py src/api.py ./src/
COPY run_artifacts/model_v1.0.0.pkl ./

# Run as a non-root user - a container running as root is a needless
# privilege-escalation risk if the API process is ever compromised
RUN useradd --create-home --shell /bin/bash apiuser
USER apiuser

EXPOSE 8000

# Docker's own healthcheck calls the API's /health endpoint - if the
# model failed to load at startup, /health returns 503 and Docker will
# correctly mark the container unhealthy rather than "running" and silently broken
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
