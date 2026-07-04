import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, auc, average_precision_score
)
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Set paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "clean_modelling_table.csv")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "run_artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Target balance:\n{df['is_good_match'].value_counts(normalize=True).round(3)}")

# ── 2. Define columns ────────────────────────────────────────
TARGET = "is_good_match"

DROP_COLS = [
    "application_id", "student_id", "job_id",
    "application_date", "name",
    "recruiter_notes_post_decision",
    TARGET
]

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

# ── 3. Split: 70% train / 15% val / 15% test ────────────────
X = df[NUMERIC_COLS + CATEGORICAL_COLS]
y = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"\nSplit sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"Train target balance: {y_train.mean():.2%} good matches")

# ── 4. Build preprocessing pipeline ──────────────────────────
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline,      NUMERIC_COLS),
    ("cat", categorical_pipeline,  CATEGORICAL_COLS),
], remainder="drop")

X_train_t = preprocessor.fit_transform(X_train)
X_val_t   = preprocessor.transform(X_val)
X_test_t  = preprocessor.transform(X_test)

print(f"\nTransformed shapes:")
print(f"  Train: {X_train_t.shape}")
print(f"  Val:   {X_val_t.shape}")
print(f"  Test:  {X_test_t.shape}")

joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "fitted_preprocessor.joblib"))
print("Saved: fitted_preprocessor.joblib")

# ── 5. Train Logistic Regression baseline ────────────────────
def evaluate(y_true, y_pred, y_proba, name):
    print(f"\n-- {name} --")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_t, y_train)

val_pred  = model.predict(X_val_t)
val_proba = model.predict_proba(X_val_t)[:, 1]
print("\nMODEL — Logistic Regression:")
evaluate(y_val, val_pred, val_proba, "Model on Validation")

# ── 6. Binary decision analysis ───────────────────────────────
cm = confusion_matrix(y_val, val_pred)
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix at threshold=0.50:")
print(f"  TP={tp}  FP={fp}")
print(f"  FN={fn}  TN={tn}")

# Plot ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

fpr, tpr, _ = roc_curve(y_val, val_proba)
roc_auc_val  = auc(fpr, tpr)
axes[0].plot(fpr, tpr, color="#1F3864", lw=2, label=f"ROC AUC = {roc_auc_val:.3f}")
axes[0].plot([0,1], [0,1], "--", color="#AAAAAA", lw=1, label="Chance")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve (validation)")
axes[0].legend()

prec_arr, rec_arr, _ = precision_recall_curve(y_val, val_proba)
pr_auc_val = average_precision_score(y_val, val_proba)
axes[1].plot(rec_arr, prec_arr, color="#A6192E", lw=2, label=f"PR AUC = {pr_auc_val:.3f}")
axes[1].axhline(y_val.mean(), color="#AAAAAA", lw=1, linestyle="--",
                label=f"Base rate = {y_val.mean():.2f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve (validation)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "roc_pr_curves.png"), dpi=150)
plt.show()
print("Saved: roc_pr_curves.png")

# Find cost-optimal threshold
COST_FP = 1.0
COST_FN = 5.0

best_threshold = 0.5
best_cost = float("inf")

for threshold in np.unique(np.round(val_proba, 3)):
    preds = (val_proba >= threshold).astype(int)
    _, fp_t, fn_t, _ = confusion_matrix(y_val, preds, labels=[0,1]).ravel()
    cost = COST_FP * fp_t + COST_FN * fn_t
    if cost < best_cost:
        best_cost = cost
        best_threshold = threshold

print(f"\nDefault threshold (0.50): cost = {COST_FP*fp + COST_FN*fn:.0f}")
print(f"Best threshold ({best_threshold:.3f}): cost = {best_cost:.0f}")
print(f"Cost reduction: {(1 - best_cost/(COST_FP*fp + COST_FN*fn))*100:.0f}%")

# ── 7. Feature engineering ────────────────────────────────────
EDU_ORDER     = {"Bachelors": 1, "Masters": 2, "PhD": 3}
EDU_MIN_ORDER = {"Any": 0, "Bachelors": 1, "Masters": 2}
SKILL_PAIRS   = [
    ("python_score", "python_required"),
    ("sql_score",    "sql_required"),
    ("ml_score",     "ml_required"),
    ("javascript_score", "javascript_required"),
    ("data_structures_score", "data_structures_required"),
    ("statistics_score", "statistics_required"),
]

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives domain features from raw columns."""
    
    EDU_ORDER     = {"Bachelors": 1, "Masters": 2, "PhD": 3}
    EDU_MIN_ORDER = {"Any": 0, "Bachelors": 1, "Masters": 2}
    SKILL_PAIRS   = [
        ("python_score", "python_required"),
        ("sql_score",    "sql_required"),
        ("ml_score",     "ml_required"),
        ("javascript_score", "javascript_required"),
        ("data_structures_score", "data_structures_required"),
        ("statistics_score", "statistics_required"),
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        gaps = np.array([X[s] - X[j] for s, j in self.SKILL_PAIRS])
        X["avg_skill_gap"]    = gaps.mean(axis=0)
        X["exp_gap"]          = X["years_experience"] - X["exp_required_years"]
        X["edu_adequate"]     = (
            X["education_level"].map(self.EDU_ORDER).fillna(0) >=
            X["edu_minimum"].map(self.EDU_MIN_ORDER).fillna(0)
        ).astype(float)
        X["salary_overreach"] = np.maximum(
            X["expected_salary_inr"] -
            X["salary_offered_inr"].fillna(X["expected_salary_inr"]),
            0
        )
        return X

ALL_NUMERIC_COLS = NUMERIC_COLS + ["avg_skill_gap", "exp_gap", "edu_adequate", "salary_overreach"]

def build_full_pipeline(classifier=LogisticRegression(max_iter=1000, random_state=42)):
    """Returns an unfitted end-to-end pipeline."""
    preprocessor_inner = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), ALL_NUMERIC_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_COLS),
    ], remainder="drop")

    return Pipeline(steps=[
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor",     preprocessor_inner),
        ("classifier",       classifier),
    ])

# Re-split with raw features
X_raw = df[[c for c in df.columns if c not in DROP_COLS]]
y_raw = df[TARGET]

X_tr, X_temp2, y_tr, y_temp2 = train_test_split(
    X_raw, y_raw, test_size=0.30, stratify=y_raw, random_state=42)
X_v, X_te, y_v, y_te = train_test_split(
    X_temp2, y_temp2, test_size=0.50, stratify=y_temp2, random_state=42)

pipeline = build_full_pipeline()
pipeline.fit(X_tr, y_tr)

pipe_preds  = pipeline.predict(X_v)
pipe_probas = pipeline.predict_proba(X_v)[:, 1]
pipe_f1     = f1_score(y_v, pipe_preds, zero_division=0)
pipe_auc    = roc_auc_score(y_v, pipe_probas)
print(f"\nEnd-to-end pipeline results:")
print(f"  Val F1:      {pipe_f1:.4f}")
print(f"  Val ROC-AUC: {pipe_auc:.4f}")

joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "pipeline.joblib"))

metrics = {
    "val_f1": round(float(pipe_f1), 4),
    "val_roc_auc": round(float(pipe_auc), 4),
    "pipeline_stages": [name for name, _ in pipeline.steps],
}
with open(os.path.join(ARTIFACTS_DIR, "run_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved: pipeline.joblib")
print("Saved: run_metrics.json")

# ── 8. Hyperparameter tuning ───────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV on Logistic Regression
lr_param_grid = [
    {
        "classifier__C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l2"],
        "classifier__solver":  ["lbfgs", "liblinear"],
    },
    {
        "classifier__C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__penalty": ["l1"],
        "classifier__solver":  ["liblinear"],
    },
]

lr_search = GridSearchCV(
    build_full_pipeline(),
    lr_param_grid,
    scoring="f1",
    cv=cv,
    refit=True,
    n_jobs=-1,
    verbose=1,
)
lr_search.fit(X_tr, y_tr)
print(f"\nBest LR config: {lr_search.best_params_}")
print(f"Best LR CV F1:  {lr_search.best_score_:.4f}")

# RandomizedSearchCV on Random Forest
rf_param_dist = {
    "classifier__n_estimators":     [50, 100, 200, 300],
    "classifier__max_depth":        [3, 4, 5, 6, 8, None],
    "classifier__min_samples_leaf": [1, 2, 4, 8],
    "classifier__max_features":     ["sqrt", "log2", 0.3, 0.5],
}

def build_rf_pipeline():
    preprocessor_inner = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), ALL_NUMERIC_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_COLS),
    ], remainder="drop")
    return Pipeline(steps=[
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor",     preprocessor_inner),
        ("classifier",       RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])

rf_search = RandomizedSearchCV(
    build_rf_pipeline(),
    rf_param_dist,
    n_iter=40,
    scoring="f1",
    cv=cv,
    refit=True,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
rf_search.fit(X_tr, y_tr)
print(f"\nBest RF config: {rf_search.best_params_}")
print(f"Best RF CV F1:  {rf_search.best_score_:.4f}")

# Test set evaluation
default_test_f1 = f1_score(y_te, pipeline.predict(X_te), zero_division=0)
lr_test_f1      = f1_score(y_te, lr_search.best_estimator_.predict(X_te), zero_division=0)
rf_test_f1      = f1_score(y_te, rf_search.best_estimator_.predict(X_te), zero_division=0)

print(f"\n-- TEST SET RESULTS --")
print(f"  Default LR:  F1 = {default_test_f1:.4f}")
print(f"  Tuned LR:    F1 = {lr_test_f1:.4f}  (gain: {lr_test_f1-default_test_f1:+.4f})")
print(f"  Tuned RF:    F1 = {rf_test_f1:.4f}  (gain: {rf_test_f1-default_test_f1:+.4f})")

# Save the best model
winner = lr_search.best_estimator_ if lr_test_f1 >= rf_test_f1 else rf_search.best_estimator_
winner_name = "tuned_lr" if lr_test_f1 >= rf_test_f1 else "tuned_rf"
joblib.dump(winner, os.path.join(ARTIFACTS_DIR, "best_pipeline.joblib"))

final_metrics = {
    "default_lr_test_f1": round(float(default_test_f1), 4),
    "tuned_lr_test_f1": round(float(lr_test_f1), 4),
    "tuned_rf_test_f1": round(float(rf_test_f1), 4),
    "winner": winner_name,
    "best_threshold": round(float(best_threshold), 3),
}
with open(os.path.join(ARTIFACTS_DIR, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)

print(f"\nWinner: {winner_name}")
print(f"Saved: best_pipeline.joblib")
print(f"Saved: final_metrics.json")
print(f"\nAll artifacts saved to: {ARTIFACTS_DIR}")
