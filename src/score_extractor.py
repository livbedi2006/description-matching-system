"""Validated, versioned score extraction for the description-matching model."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, create_model, field_validator

try:  # Supports both `python src/...py` and `uvicorn src.api:app`.
    from .ml_utils import log
except ImportError:  # pragma: no cover - direct-script compatibility
    from ml_utils import log

NUMERIC_COLS = [
    "exp_required_years", "salary_offered_inr",
    "python_required", "sql_required", "ml_required",
    "javascript_required", "data_structures_required", "statistics_required",
    "years_experience", "python_score", "sql_score", "ml_score",
    "javascript_score", "data_structures_score", "statistics_score",
    "exam_time_seconds", "self_reported_confidence", "retake_count",
    "expected_salary_inr",
]

CATEGORICAL_COLS = [
    "company", "title", "location_job", "edu_minimum", "education_level",
    "location_student",
]

FEATURE_NAMES = NUMERIC_COLS + CATEGORICAL_COLS
DEFAULT_THRESHOLD = 0.5
DEFAULT_SCORE_MEANING = "probability that the application is a good match (positive class)"


class ScoringError(ValueError):
    """A safe, consumer-facing validation or inference error."""


class _FeatureContract(BaseModel):
    """Base validators shared by the dynamically-created feature contract."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def reject_invalid_numeric_values(cls, value: Any, info):
        if info.field_name not in NUMERIC_COLS:
            return value
        if isinstance(value, bool):
            raise ValueError("must be a numeric value, not a boolean")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"must be numeric; received {value!r}") from exc
        if not np.isfinite(numeric_value):
            raise ValueError("must be finite (not NaN or infinity)")
        return numeric_value


FeatureContract = create_model(
    "FeatureContract",
    __base__=_FeatureContract,
    **{name: (str if name in CATEGORICAL_COLS else float, ...) for name in FEATURE_NAMES},
)


def _summarize_validation_error(error: ValidationError) -> str:
    messages = []
    for item in error.errors():
        field = ".".join(str(part) for part in item["loc"])
        if item["type"] == "missing":
            messages.append(f"missing required feature '{field}'")
        elif item["type"] == "extra_forbidden":
            messages.append(f"unexpected feature '{field}'")
        else:
            messages.append(f"{field}: {item['msg']}")
    return "; ".join(messages)


def validate_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and order one raw model-input record using Pydantic."""
    if not isinstance(features, Mapping):
        raise ScoringError("features must be a JSON object")
    try:
        validated = FeatureContract.model_validate(dict(features))
    except ValidationError as exc:
        raise ScoringError(_summarize_validation_error(exc)) from exc
    return validated.model_dump()


class ScoreExtractor:
    """Load a packaged model once and expose safe single and batch scoring."""

    def __init__(self, model_path: str | Path, *, container_image_digest: str | None = None):
        artifact_path = Path(model_path)
        artifact = joblib.load(artifact_path)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ValueError(
                f"{artifact_path} is not a model package. Run src/export_model.py to create one."
            )

        self.model = artifact["model"]
        self.threshold = float(artifact.get("threshold", DEFAULT_THRESHOLD))
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"model threshold must be between 0 and 1, got {self.threshold}")

        self.model_version = str(artifact.get("model_version", "v1.0.0"))
        self.score_meaning = str(artifact.get("score_meaning", DEFAULT_SCORE_MEANING))
        self.feature_names = list(artifact.get("feature_names", FEATURE_NAMES))
        if self.feature_names != FEATURE_NAMES:
            raise ValueError(
                "model package feature schema does not match the deployed API schema; "
                "re-export the model with src/export_model.py"
            )

        self.container_image_digest = (
            container_image_digest
            if container_image_digest is not None
            else os.getenv("CONTAINER_IMAGE_DIGEST", "unknown")
        )
        log.info(
            "Model package loaded: version=%s threshold=%.3f path=%s",
            self.model_version,
            self.threshold,
            artifact_path,
        )

    def _build_frame(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        if not records:
            raise ScoringError("empty batch: 0 records received")
        validated = []
        errors = []
        for index, record in enumerate(records):
            try:
                validated.append(validate_features(record))
            except ScoringError as exc:
                errors.append(f"record {index}: {exc}")
        if errors:
            raise ScoringError("; ".join(errors))
        return pd.DataFrame(validated, columns=self.feature_names)

    def _response(self, probability: float, record_id: str | None) -> dict[str, Any]:
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ScoringError("model returned an invalid probability outside the 0–1 range")
        return {
            "match_score": round(probability, 4),
            "decision": int(probability >= self.threshold),
            "score_meaning": self.score_meaning,
            "model_version": self.model_version,
            "record_id": record_id,
            "container_image_digest": self.container_image_digest,
        }

    def score_single(
        self, features: Mapping[str, Any], record_id: str | None = None
    ) -> dict[str, Any]:
        frame = self._build_frame([features])
        probability = float(self.model.predict_proba(frame)[0, 1])
        return self._response(probability, record_id)

    def score_batch(
        self,
        feature_records: Sequence[Mapping[str, Any]],
        record_ids: Sequence[str | None] | None = None,
    ) -> list[dict[str, Any]]:
        frame = self._build_frame(feature_records)
        if record_ids is None:
            record_ids = [None] * len(feature_records)
        if len(record_ids) != len(feature_records):
            raise ScoringError("record_ids must contain one value per feature record")
        probabilities = self.model.predict_proba(frame)[:, 1]
        return [
            self._response(float(probability), record_id)
            for probability, record_id in zip(probabilities, record_ids)
        ]
