"""
data.py
-------
Step 1 of Day 3: this file's ONLY job is loading and joining raw CSVs.
It does not know about features, models, or metrics — separation of concerns.
Any other module that needs the joined table calls load_joined_data().
"""

import pandas as pd
from config import CONFIG


def load_raw_tables():
    """Load the three raw CSVs exactly as they are, no transformation."""
    students = pd.read_csv(CONFIG["students_path"])
    jobs = pd.read_csv(CONFIG["jobs_path"])
    applications = pd.read_csv(CONFIG["applications_path"])
    return students, jobs, applications


def load_joined_data():
    """
    Join students + jobs + applications into one modelling table,
    and drop any column already known to be leaky (Day 2 finding),
    using the central list in config.py — not a hard-coded name here.
    """
    students, jobs, applications = load_raw_tables()

    data = (
        applications
        .merge(students, on="student_id")
        .merge(jobs, on="job_id", suffixes=("_student", "_job"))
    )

    data = data.drop(columns=CONFIG["leaky_columns"], errors="ignore")
    return data


if __name__ == "__main__":
    # Lets you run "python data.py" directly to sanity-check loading works.
    df = load_joined_data()
    print(f"Loaded joined table: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Leaky columns removed: {CONFIG['leaky_columns']}")
