"""
features.py
-----------
Step 1 of Day 3: this file's ONLY job is turning raw joined data into
clean, model-ready features. It does not load data and does not train
models — separation of concerns.
"""

import numpy as np
import pandas as pd
from config import CONFIG


def compute_skill_gap_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    For each application row, compute the average gap between a
    student's score and the job's required score, across every skill
    that is actually comparable (both values present).

    This is the feature-engineering core carried over from Day 2's
    "is there real signal here" baseline check.
    """
    data = data.copy()
    skills = CONFIG["skills"]

    gaps = []
    met_fractions = []
    for _, row in data.iterrows():
        skill_gaps = []
        met, total = 0, 0
        for sk in skills:
            req = row.get(f"{sk}_required")
            sc = row.get(f"{sk}_score")
            if pd.notna(req) and pd.notna(sc):
                skill_gaps.append(sc - req)
                total += 1
                if sc >= req:
                    met += 1
        gaps.append(np.mean(skill_gaps) if skill_gaps else 0.0)
        met_fractions.append(met / total if total > 0 else 0.0)

    data["avg_skill_gap"] = gaps
    data["skill_met_fraction"] = met_fractions
    data["experience_gap"] = data["years_experience"] - data["exp_required_years"]

    return data


def build_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Single entry point: takes the joined raw table and returns a table
    with all model-ready features attached. Any new feature you invent
    later gets added here, in one place, instead of scattered across
    notebook cells.
    """
    data = compute_skill_gap_features(data)
    return data
