"""
REST API for the Description Matching System model serving.

Wraps ScoreExtractor behind FastAPI so the model is callable over HTTP.
Based on Task13 reference implementation with production-ready features.

Endpoints:
  GET  /health              - liveness/readiness, reports model_version
  POST /score/single        - one record -> one score
  POST /score/batch         - many records -> many scores
"""

import logging
import time
import uuid
import os
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ml_utils import check_dependencies, log
from score_extractor import ScoreExtractor, ScoringError

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "run_artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_v1.0.0.pkl")
MAX_BATCH_SIZE = 5000

app = FastAPI(title="Description Matching System API", version="1.0.0")

# -----------------------------------------------------------------
# Startup: dependency check + model load happen ONCE, at process
# startup, not per request. If either fails, the failure is logged
# clearly and _model stays None so /health can report "not_ready"
# instead of the process crashing on the first request with a
# confusing NoneType error deep in the handler.
# -----------------------------------------------------------------
_model: Optional[ScoreExtractor] = None
_startup_error: Optional[str] = None


@app.on_event("startup")
def load_model():
    global _model, _startup_error
    try:
        check_dependencies()
        _model = ScoreExtractor(MODEL_PATH)
        log.info("Startup complete: model_version=%s loaded from %s", _model.model_version, MODEL_PATH)
    except Exception as e:
        _startup_error = str(e)
        log.error("STARTUP FAILURE: %s", e)
        # Deliberately do not re-raise: the process stays up so /health
        # can report the failure over HTTP instead of the container
        # exiting with no diagnosable signal at all.


# -----------------------------------------------------------------
# Request-level logging middleware: every request gets a short id and
# its latency logged, live, regardless of whether it succeeds or fails.
# -----------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    log.info("[%s] %s %s -> %d (%.2f ms)", request_id, request.method, request.url.path,
              response.status_code, duration_ms)
    response.headers["X-Request-ID"] = request_id
    return response


# -----------------------------------------------------------------
# Structured error responses: a ScoringError (bad input) becomes a 400
# with a plain-English message; anything unexpected becomes a 500 that
# never leaks an internal stack trace to the caller, but still logs the
# full detail server-side for debugging.
# -----------------------------------------------------------------
@app.exception_handler(ScoringError)
async def scoring_error_handler(request: Request, exc: ScoringError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "invalid_input", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    log.error("Unhandled exception on %s: %r", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "detail": "an unexpected error occurred; this has been logged"},
    )


class SingleRecordRequest(BaseModel):
    model_config = {"extra": "allow"}   # extra fields tolerated, forwarded to ScoreExtractor's own contract
    id: Optional[str] = None


class BatchRequest(BaseModel):
    records: list = Field(..., description="List of records, each shaped like SingleRecordRequest")


@app.get("/health")
def health():
    """Liveness + readiness in one endpoint: 200 only if the model is
    actually loaded and ready to score, not just if the process is up."""
    if _model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": _startup_error or "model not loaded"},
        )
    return {"status": "ok", "model_version": _model.model_version, "score_meaning": _model.score_meaning}


@app.post("/score/single")
def score_single(record: SingleRecordRequest):
    if _model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "model_not_ready", "detail": _startup_error or "model not loaded"},
        )
    return _model.score_single(record.model_dump())


@app.post("/score/batch")
def score_batch(payload: BatchRequest):
    if _model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "model_not_ready", "detail": _startup_error or "model not loaded"},
        )
    if len(payload.records) > MAX_BATCH_SIZE:
        return JSONResponse(
            status_code=413,  # Content Too Large
            content={"error": "batch_too_large",
                     "detail": f"batch has {len(payload.records)} records; max is {MAX_BATCH_SIZE}"},
        )
    return _model.score_batch(payload.records)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
