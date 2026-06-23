"""
model.py
--------
Step 1 & 4 of Day 3: this file's ONLY job is defining models.

Every model here follows the SAME interface (Step 5's "interfaces" concept):
  - a .fit(data) method (even if it does nothing, like the baseline)
  - a .predict(data) method that returns a column of 0/1 predictions

This is what lets the harness in evaluate.py train/evaluate ANY model
the same way, without needing to know its internals.
"""

import pandas as pd
from config import CONFIG


class BaselineSkillOverlapModel:
    """
    The Day 2 trivial baseline, now wrapped behind a clean interface.
    No learning happens — it's a fixed rule — but it honors the same
    .fit()/.predict() contract a real ML model would, so it plugs into
    the same harness without special-casing.
    """

    def __init__(self, threshold: float = None):
        # Reads from config by default, but allows override for experimentation —
        # this is exactly what "config-driven, not hard-coded" means in practice.
        self.threshold = threshold if threshold is not None else CONFIG["baseline_match_threshold"]

    def fit(self, data: pd.DataFrame):
        # The baseline has no parameters to learn, so this is a no-op.
        # It still exists so the harness can call .fit() on ANY model
        # without checking "is this a real model or the baseline?" first.
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        return (data["skill_met_fraction"] >= self.threshold).astype(int)


# Future models (Day 4+) get added here, following the exact same shape:
#
# class LogisticRegressionMatchModel:
#     def fit(self, data): ...
#     def predict(self, data): ...
#
# See README.md "Adding a new model" section for the full checklist.
