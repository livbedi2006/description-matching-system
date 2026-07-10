"""Shared dependency, validation, and reproducibility utilities."""

import datetime
import hashlib
import importlib
import json
import logging
from pathlib import Path

import numpy as np
from packaging import version

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("ml_utils")

REQUIRED_PACKAGES = {
    "pandas": "1.0.0",
    "numpy": "1.18.0",
    "sklearn": "0.24.0",
    "joblib": "0.14.0",
    "fastapi": "0.68.0",
    "uvicorn": "0.15.0",
}


def check_dependencies():
    """Fail fast when a package needed by training or API serving is missing."""
    problems = []
    installed_versions = {}
    for package, minimum_version in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(package)
            installed = getattr(module, "__version__", "unknown")
            installed_versions[package] = installed
            if installed != "unknown" and version.parse(installed) < version.parse(minimum_version):
                problems.append(f"{package} {installed} < required {minimum_version}")
        except ImportError:
            problems.append(f"{package} is not installed")
    if problems:
        for problem in problems:
            log.error("Dependency check failed: %s", problem)
        raise RuntimeError("Missing/incompatible dependencies: " + "; ".join(problems))
    log.info("Dependency check passed: %s", installed_versions)
    return installed_versions


def validate_data(X, y, min_samples_per_class=10):
    """Validate the training arrays before fitting a model."""
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN/Inf values; impute or drop before fitting")
    if X.shape[0] != len(y):
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {len(y)}")
    if len(y) < min_samples_per_class * 2:
        raise ValueError(f"Too few samples: {len(y)} < {min_samples_per_class * 2}")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes, found {len(classes)}")
    for label, count in zip(classes, counts):
        if count < min_samples_per_class:
            raise ValueError(f"Class {label} has only {count} samples < {min_samples_per_class}")
    return True


def safe_fit(model, X, y, fallback=None, model_name="model"):
    """Fit a model, optionally returning a declared fallback on failure."""
    try:
        model.fit(X, y)
        return model
    except Exception as exc:
        log.warning("%s fit failed: %s", model_name, exc)
        if fallback is not None:
            return fallback
        raise


def safe_search(search, X, y, fallback=None, search_name="search"):
    """Fit a hyperparameter search, optionally returning a declared fallback."""
    try:
        search.fit(X, y)
        return search
    except Exception as exc:
        log.warning("%s failed: %s", search_name, exc)
        if fallback is not None:
            return fallback
        raise


def log_execution(log_path, run_info):
    """Append timestamped JSON execution metadata."""
    record = {**run_info, "timestamp": datetime.datetime.now().isoformat()}
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def compute_file_checksum(filepath):
    """Return the SHA-256 checksum used for reproducibility evidence."""
    digest = hashlib.sha256()
    with Path(filepath).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    print(check_dependencies())
