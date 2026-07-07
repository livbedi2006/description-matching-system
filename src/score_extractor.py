"""
Model Score Extraction: the scoring interface for the Description Matching System.

Wraps a trained model behind a clean predict interface (ScoreExtractor)
with:
  - an explicit, pydantic-validated input contract (exactly which
    features, what shape, what type)
  - a standardised output: score + score_meaning + model_version +
    input_record_id, for both single and batch calls
  - graceful, structured errors instead of raw stack traces
  - versioning that ties every output back to a specific model artifact
"""

import time
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from pydantic import BaseModel, ConfigDict, ValidationError as PydanticValidationError, field_validator

from ml_utils import log

# Feature names from our preprocessing pipeline
NUMERIC_COLS = [
    "exp_required_years", "salary_offered_inr",
    "python_required", "sql_required", "ml_required",
    "javascript_required", "data_structures_required", "statistics_required",
    "years_experience", "python_score", "sql_score", "ml_score",
    "javascript_score", "data_structures_score", "statistics_score",
    "exam_time_seconds", "self_reported_confidence",
    "retake_count", "expected_salary_inr",
]

CATEGORICAL_COLS = [
    "company", "title", "location_job",
    "edu_minimum", "education_level", "location_student",
]

# All required features for scoring
FEATURE_NAMES = NUMERIC_COLS + CATEGORICAL_COLS


class ScoringError(Exception):
    """Raised for any input that fails the contract, with a plain-English
    message a non-ML consumer can act on without reading a stack trace."""
    pass


class _RecordContract(BaseModel):
    """The input contract (pydantic model): exactly which fields are
    required, their type, and that extra fields are tolerated (dropped),
    not rejected - a deliberate choice for flexibility."""
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _reject_non_finite(cls, v, info):
        if info.field_name in NUMERIC_COLS:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"non-numeric value for {info.field_name}: {v!r}")
            if not np.isfinite(fv):
                raise ValueError(f"non-finite (NaN/Inf) value for {info.field_name}: {v!r}")
            return fv
        return v


# Dynamically require all feature columns
from pydantic import create_model

RecordContract = create_model(
    "RecordContract",
    __base__=_RecordContract,
    **{name: (str if name in CATEGORICAL_COLS else float, ...) for name in FEATURE_NAMES},
)


class ScoreExtractor:
    """Loads a versioned model artifact once, then serves both single-
    record and batch scoring through a validated, standardised interface."""

    def __init__(self, model_path: str):
        artifact = joblib.load(model_path)   # loaded once at init, not per call
        self.model = artifact["model"]
        self.model_version = artifact["model_version"]
        self.score_meaning = artifact["score_meaning"]
        self.feature_names = artifact["feature_names"]
        log.info("ScoreExtractor loaded model_version=%s from %s", self.model_version, model_path)

    def _validate_and_build_frame(self, records: list) -> tuple:
        """Runs the pydantic contract over every record before any model
        call. Returns (dataframe_of_valid_records, list_of_ids) or raises
        ScoringError with every problem found, not just the first one."""
        if len(records) == 0:
            raise ScoringError("empty input, 0 records received")

        validated_rows = []
        ids = []
        errors = []
        for i, record in enumerate(records):
            try:
                parsed = RecordContract(**record)
                row = {name: getattr(parsed, name) for name in self.feature_names}
                validated_rows.append(row)
                ids.append(parsed.id if parsed.id is not None else record.get("id"))
            except PydanticValidationError as e:
                errors.append(f"record {i}: {_summarize_pydantic_error(e)}")

        if errors:
            raise ScoringError("; ".join(errors))

        df = pd.DataFrame(validated_rows, columns=self.feature_names)
        return df, ids

    def score_single(self, record: dict) -> dict:
        df, ids = self._validate_and_build_frame([record])
        proba = self.model.predict_proba(df)[0, 1]
        return {
            "score": round(float(proba), 4),
            "score_meaning": self.score_meaning,
            "model_version": self.model_version,
            "input_record_id": ids[0],
        }

    def score_batch(self, records: list) -> list:
        df, ids = self._validate_and_build_frame(records)
        probas = self.model.predict_proba(df)[:, 1]
        return [
            {
                "score": round(float(p), 4),
                "score_meaning": self.score_meaning,
                "model_version": self.model_version,
                "input_record_id": rid,
            }
            for p, rid in zip(probas, ids)
        ]


def _summarize_pydantic_error(e: PydanticValidationError) -> str:
    """Turns pydantic's (correct but verbose) error object into a single
    human-readable line, per the study guide's 'graceful errors' step."""
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        if err["type"] == "missing":
            parts.append(f"missing required field '{loc}'")
        else:
            parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)
