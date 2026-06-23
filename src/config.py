"""
config.py
---------
Step 1 & 2 of Day 3: ONE place for every path, parameter, and seed.

Nothing in any other file should ever hard-code a file path, a threshold,
or the random seed directly. Every other module imports this file and
reads its settings from here. Change a setting once, here, and it applies
everywhere automatically — that's the entire point of "config-driven runs."
"""

import os

# Project root = one level up from this file (src/), regardless of which
# directory you happen to run a script FROM. This is exactly the kind of
# "config drift" bug Section 8 of the study guide warns about — a relative
# path like "data/x.csv" silently breaks the moment someone runs the script
# from a different folder. Anchoring to this file's own location fixes it
# for everyone, permanently, in one place.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(relative_path: str) -> str:
    """Turn a path relative to the project root into an absolute path."""
    return os.path.join(PROJECT_ROOT, relative_path)


CONFIG = {
    # --- reproducibility (Day 1 lesson, carried forward) ---
    "seed": 42,

    # --- data paths (always resolved relative to the project root) ---
    "students_path": _p("data/students_exam_scores.csv"),
    "jobs_path": _p("data/job_descriptions.csv"),
    "applications_path": _p("data/applications.csv"),

    # --- known-leaky column from Day 2, removed centrally here ---
    "leaky_columns": ["recruiter_notes_post_decision"],

    # --- feature engineering parameters ---
    "skills": ["python", "sql", "ml", "javascript", "data_structures", "statistics"],

    # --- baseline model parameter ---
    "baseline_match_threshold": 0.6,   # >=60% of required skills met -> predicted match

    # --- train/test split ---
    "test_size": 0.2,

    # --- experiment log location ---
    "experiment_log_path": _p("experiment_log.csv"),
}
