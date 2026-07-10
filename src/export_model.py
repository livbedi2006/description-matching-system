"""Export a deployable, versioned model package for the Task 13 API."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

try:  # Supports both `python src/export_model.py` and package imports.
    from .ml_utils import check_dependencies, log
    from .score_extractor import FEATURE_NAMES, DEFAULT_SCORE_MEANING, DEFAULT_THRESHOLD
except ImportError:  # pragma: no cover - direct-script compatibility
    from ml_utils import check_dependencies, log
    from score_extractor import FEATURE_NAMES, DEFAULT_SCORE_MEANING, DEFAULT_THRESHOLD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "run_artifacts"
MODEL_VERSION = "v1.0.0"
PACKAGE_PATH = ARTIFACTS_DIR / "model_package.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


def _select_source_artifact() -> Path:
    """Return the calibrated Task 12 production package required for serving."""
    candidate = ARTIFACTS_DIR / "production_model_package.joblib"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Calibrated production package not found. Run src/train_advanced.py before "
        "exporting the Task 13 API package."
    )


def main() -> None:
    dependency_versions = check_dependencies()
    source_path = _select_source_artifact()
    source_artifact = joblib.load(source_path)

    if isinstance(source_artifact, dict) and "model" in source_artifact:
        model = source_artifact["model"]
        threshold = float(source_artifact.get("threshold", DEFAULT_THRESHOLD))
        calibration_method = source_artifact.get("calibration_method")
        cost_assumptions = source_artifact.get("cost_assumptions")
    else:
        model = source_artifact
        threshold = DEFAULT_THRESHOLD
        calibration_method = None
        cost_assumptions = None

    package = {
        "model": model,
        "threshold": threshold,
        "model_version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
        "score_meaning": DEFAULT_SCORE_MEANING,
    }
    joblib.dump(package, PACKAGE_PATH)

    metadata = {
        "model_version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
        "score_meaning": DEFAULT_SCORE_MEANING,
        "threshold": threshold,
        "source_model": source_path.name,
        "calibration_method": calibration_method,
        "cost_assumptions": cost_assumptions,
        "dependency_versions": dependency_versions,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    log.info("Model package written: %s", PACKAGE_PATH)
    log.info("Model metadata written: %s", METADATA_PATH)


if __name__ == "__main__":
    main()
