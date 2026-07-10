# Containerized Task 13 scoring service.
FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY run_artifacts/model_package.joblib ./model_package.joblib

ENV MODEL_PATH=/app/model_package.joblib
ENV CONTAINER_IMAGE_DIGEST=unknown

RUN useradd --create-home --shell /bin/bash apiuser
USER apiuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
