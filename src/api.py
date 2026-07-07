"""
REST API for the Description Matching System model serving.

Provides endpoints for:
- Single record scoring
- Batch scoring
- Health checks
- Model information

Uses FastAPI for high-performance async API with automatic OpenAPI documentation.
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from score_extractor import ScoreExtractor, ScoringError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Initialize FastAPI app
app = FastAPI(
    title="Description Matching System API",
    description="REST API for scoring job candidate matches using trained ML models",
    version="1.0.0",
)

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "run_artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_v1.0.0.pkl")

# Global model instance (loaded once at startup)
score_extractor: Optional[ScoreExtractor] = None


# Pydantic models for API requests/responses
class SingleRecordRequest(BaseModel):
    """Request model for single record scoring."""
    id: Optional[str] = Field(None, description="Optional record identifier")
    exp_required_years: float = Field(..., description="Years of experience required")
    salary_offered_inr: float = Field(..., description="Salary offered in INR")
    python_required: float = Field(..., description="Python skill requirement score")
    sql_required: float = Field(..., description="SQL skill requirement score")
    ml_required: float = Field(..., description="ML skill requirement score")
    javascript_required: float = Field(..., description="JavaScript skill requirement score")
    data_structures_required: float = Field(..., description="Data Structures skill requirement score")
    statistics_required: float = Field(..., description="Statistics skill requirement score")
    years_experience: float = Field(..., description="Candidate years of experience")
    python_score: float = Field(..., description="Candidate Python skill score")
    sql_score: float = Field(..., description="Candidate SQL skill score")
    ml_score: float = Field(..., description="Candidate ML skill score")
    javascript_score: float = Field(..., description="Candidate JavaScript skill score")
    data_structures_score: float = Field(..., description="Candidate Data Structures skill score")
    statistics_score: float = Field(..., description="Candidate Statistics skill score")
    exam_time_seconds: float = Field(..., description="Exam time in seconds")
    self_reported_confidence: float = Field(..., description="Self-reported confidence")
    retake_count: float = Field(..., description="Number of exam retakes")
    expected_salary_inr: float = Field(..., description="Expected salary in INR")
    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    location_job: str = Field(..., description="Job location")
    edu_minimum: str = Field(..., description="Minimum education required")
    education_level: str = Field(..., description="Candidate education level")
    location_student: str = Field(..., description="Candidate location")


class BatchRecordRequest(BaseModel):
    """Request model for batch scoring."""
    records: List[dict] = Field(..., description="List of records to score")


class ScoreResponse(BaseModel):
    """Response model for scoring results."""
    score: float = Field(..., description="Probability of good match")
    score_meaning: str = Field(..., description="Explanation of the score")
    model_version: str = Field(..., description="Version of the model used")
    input_record_id: Optional[str] = Field(None, description="Record identifier from input")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version if loaded")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_version: str = Field(..., description="Model version")
    score_meaning: str = Field(..., description="Explanation of scores")
    feature_names: List[str] = Field(..., description="Feature names expected by model")


# Startup event to load the model
@app.on_event("startup")
async def startup_event():
    """Load the model on startup."""
    global score_extractor
    try:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model not found at {MODEL_PATH}. API will run but scoring will fail.")
            logger.warning("Run 'python src/export_model.py' to create the model artifact.")
        else:
            score_extractor = ScoreExtractor(MODEL_PATH)
            logger.info(f"Model loaded successfully: {score_extractor.model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't fail startup - allow API to run for health checks


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    model_loaded = score_extractor is not None
    model_version = score_extractor.model_version if model_loaded else None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information."""
    if score_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check health endpoint."
        )
    
    return ModelInfoResponse(
        model_version=score_extractor.model_version,
        score_meaning=score_extractor.score_meaning,
        feature_names=score_extractor.feature_names,
    )


@app.post("/score/single", response_model=ScoreResponse, tags=["Scoring"])
async def score_single_record(request: SingleRecordRequest):
    """Score a single record."""
    if score_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check health endpoint."
        )
    
    try:
        record = request.model_dump()
        result = score_extractor.score_single(record)
        return ScoreResponse(**result)
    except ScoringError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input validation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {str(e)}"
        )


@app.post("/score/batch", response_model=List[ScoreResponse], tags=["Scoring"])
async def score_batch_records(request: BatchRecordRequest):
    """Score multiple records in batch."""
    if score_extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check health endpoint."
        )
    
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty batch - no records provided"
        )
    
    try:
        results = score_extractor.score_batch(request.records)
        return [ScoreResponse(**r) for r in results]
    except ScoringError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input validation failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch scoring failed: {str(e)}"
        )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Description Matching System API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "score_single": "/score/single",
            "score_batch": "/score/batch",
            "docs": "/docs",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
