"""
ml_utils.py
-----------
Utility functions for ML pipeline validation and dependency checking.
Provides failure fallback, dependency validation, and data quality checks.
"""

import importlib
import numpy as np
from packaging import version

REQUIRED_PACKAGES = {
    "pandas": "1.0.0",
    "numpy": "1.18.0",
    "sklearn": "0.24.0",  # scikit-learn is imported as sklearn
    "joblib": "0.14.0",
}


def check_dependencies():
    """
    Fails fast with a clear message if a required package is missing
    or too old, instead of a confusing stack trace mid-pipeline.
    """
    problems = []
    for pkg, min_version in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(pkg)
            installed = getattr(module, "__version__", "unknown")
            if installed != "unknown" and version.parse(installed) < version.parse(min_version):
                problems.append(f"{pkg} {installed} < required {min_version}")
        except ImportError:
            problems.append(f"{pkg} is not installed")
    
    if problems:
        raise RuntimeError("Missing/incompatible dependencies: " + "; ".join(problems))
    
    return True


def validate_data(X, y, min_samples_per_class=10):
    """
    Rejects the input shapes most likely to silently corrupt a fit:
    NaN/Inf, mismatched lengths, too few classes or samples.
    """
    X = np.asarray(X, dtype=float)
    
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN/Inf values; impute or drop before fitting")
    
    if X.shape[0] != len(y):
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {len(y)}")
    
    if len(y) < min_samples_per_class * 2:
        raise ValueError(f"Too few samples: {len(y)} < {min_samples_per_class * 2}")
    
    # Check class balance
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(unique_classes) < 2:
        raise ValueError(f"Need at least 2 classes, found {len(unique_classes)}")
    
    for cls, count in zip(unique_classes, counts):
        if count < min_samples_per_class:
            raise ValueError(f"Class {cls} has only {count} samples < {min_samples_per_class}")
    
    return True


def safe_fit(model, X, y, fallback=None, model_name="model"):
    """
    Wraps model.fit() with try/except to provide graceful fallback
    instead of crashing the entire pipeline.
    """
    try:
        model.fit(X, y)
        return model
    except Exception as e:
        print(f"Warning: {model_name} fit failed with error: {e}")
        if fallback is not None:
            print(f"Falling back to {fallback}")
            return fallback
        raise


def safe_search(search, X, y, fallback=None, search_name="search"):
    """
    Wraps GridSearchCV/RandomizedSearchCV with try/except for graceful fallback.
    """
    try:
        search.fit(X, y)
        return search
    except Exception as e:
        print(f"Warning: {search_name} failed with error: {e}")
        if fallback is not None:
            print(f"Falling back to {fallback}")
            return fallback
        raise


def log_execution(log_path, run_info):
    """
    Logs execution metadata with timestamp for reproducibility.
    """
    import json
    import datetime
    import os
    
    run_info["timestamp"] = datetime.datetime.now().isoformat()
    
    if os.path.exists(log_path):
        with open(log_path, "a") as f:
            f.write(json.dumps(run_info) + "\n")
    else:
        with open(log_path, "w") as f:
            f.write(json.dumps(run_info) + "\n")
    
    return run_info


def compute_file_checksum(filepath):
    """
    Computes MD5 checksum of a file for reproducibility verification.
    """
    import hashlib
    
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest()


if __name__ == "__main__":
    # Test dependency checking
    print("Checking dependencies...")
    check_dependencies()
    print("All dependencies OK!")
