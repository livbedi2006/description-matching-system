"""FastAPI score service for the description-matching model.

Endpoints follow the Task 13 contract:
  GET  /health
  POST /score
  POST /score-batch
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator

try:  # Supports both `python src/...py` and `uvicorn src.api:app`.
    from .ml_utils import check_dependencies, log
    from .score_extractor import ScoreExtractor, ScoringError, validate_features
except ImportError:  # pragma: no cover - direct-script compatibility
    from ml_utils import check_dependencies, log
    from score_extractor import ScoreExtractor, ScoringError, validate_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "run_artifacts" / "model_package.joblib"))
CONTAINER_IMAGE_DIGEST = os.getenv("CONTAINER_IMAGE_DIGEST", "unknown")
MAX_BATCH_SIZE = 5_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate dependencies and load the model once before binding traffic."""
    check_dependencies()
    app.state.scorer = ScoreExtractor(
        MODEL_PATH, container_image_digest=CONTAINER_IMAGE_DIGEST
    )
    yield
    app.state.scorer = None


app = FastAPI(
    title="Description Matching Scoring API",
    version="1.0.0",
    description="Validated, versioned single-record and batch scoring.",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    started_at = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started_at) * 1_000
    response.headers["X-Request-ID"] = request_id
    log.info(
        "[%s] %s %s -> %s (%.2f ms)",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


class ScoreRequest(BaseModel):
    """HTTP input contract: all model inputs live under `features`."""

    features: dict[str, Any]
    record_id: str | None = None

    @field_validator("features")
    @classmethod
    def validate_required_features(cls, features: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate_features(features)
        except ScoringError as exc:
            raise ValueError(str(exc)) from exc


def _scorer(request: Request) -> ScoreExtractor:
    scorer = getattr(request.app.state, "scorer", None)
    if scorer is None:  # Defensive guard; a failed lifespan prevents startup.
        raise HTTPException(status_code=503, detail="model is not loaded")
    return scorer


@app.get("/health")
def health(request: Request):
    scorer = _scorer(request)
    return {
        "status": "ok",
        "model_version": scorer.model_version,
        "model_loaded": True,
        "container_image_digest": scorer.container_image_digest,
    }


@app.post("/score")
def score(payload: ScoreRequest, request: Request):
    try:
        return _scorer(request).score_single(payload.features, payload.record_id)
    except ScoringError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/score-batch")
def score_batch(payloads: list[ScoreRequest], request: Request):
    if not payloads:
        raise HTTPException(status_code=422, detail="Empty batch: 0 records received")
    if len(payloads) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"batch has {len(payloads)} records; maximum is {MAX_BATCH_SIZE}",
        )
    try:
        scorer = _scorer(request)
        results = scorer.score_batch(
            [payload.features for payload in payloads],
            [payload.record_id for payload in payloads],
        )
        log.info("Batch scoring complete: %d records", len(results))
        return results
    except ScoringError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
