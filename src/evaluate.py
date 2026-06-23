"""
evaluate.py
-----------
Step 1 of Day 3: this file's ONLY job is calculating metrics from
predictions vs true labels. This guarantees every model is judged by
the exact same formulas — the guide's pitfall warning is "eval logic
duplicated and inconsistent," which this file exists to prevent.
"""

import pandas as pd


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Returns accuracy, precision, recall, and the base rate, computed
    the exact same way every single time, for any model.
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    base_rate = y_true.value_counts(normalize=True).max()

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "base_rate": round(base_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
