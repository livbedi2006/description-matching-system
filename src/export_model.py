"""
Export the trained model as a versioned artifact for the scoring interface.

This script loads the best model from the training pipeline and exports it
in the format expected by score_extractor.py, including:
- The fitted model
- Model version
- Feature names
- Score meaning
"""

import json
import joblib
import os

from ml_utils import check_dependencies, log

# Check dependencies
dependency_versions = check_dependencies()

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "run_artifacts")
MODEL_VERSION = "v1.0.0"

# Load the best model from training
best_model_path = os.path.join(ARTIFACTS_DIR, "best_pipeline.joblib")
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Best model not found at {best_model_path}. Run train_advanced.py first.")

# Load the model
best_model = joblib.load(best_model_path)
log.info("Loaded best model from %s", best_model_path)

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

FEATURE_NAMES = NUMERIC_COLS + CATEGORICAL_COLS

# Export as versioned artifact
artifact_path = os.path.join(ARTIFACTS_DIR, f"model_{MODEL_VERSION}.pkl")
joblib.dump({
    "model": best_model,
    "model_version": MODEL_VERSION,
    "feature_names": FEATURE_NAMES,
    "score_meaning": "probability of good match between candidate and job",
}, artifact_path)
log.info("Model artifact written: %s", artifact_path)

# Export metadata
metadata = {
    "model_version": MODEL_VERSION,
    "feature_names": FEATURE_NAMES,
    "score_meaning": "probability of good match between candidate and job",
    "dependency_versions": dependency_versions,
    "source_model": "best_pipeline.joblib",
}

metadata_path = os.path.join(ARTIFACTS_DIR, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2, default=str)
log.info("Model metadata written: %s", metadata_path)

print(f"\nModel exported successfully!")
print(f"Artifact: {artifact_path}")
print(f"Metadata: {metadata_path}")
print(f"Model version: {MODEL_VERSION}")
print(f"Features: {len(FEATURE_NAMES)}")
