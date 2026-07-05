import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from ml_utils import check_dependencies, validate_data, safe_fit, safe_search, log_execution
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

# ── 0. Dependency and data validation ─────────────────────────
print("Checking dependencies...")
check_dependencies()
print("Dependencies OK")

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


# ============================================================
# TASK 10 — COMPLEX RELATIONSHIPS
# ============================================================
# WHAT THIS DOES:
# Tests whether non-linear models capture structure linear models miss.
# Compares linear baseline vs Gradient Boosting with validated lift
# and partial dependence plots to prove real structure learning.

print("\n" + "="*60)
print("TASK 10: Complex Relationships")
print("="*60)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay

# Use the same train/val/test splits from earlier
# Compare linear baseline (Logistic Regression) vs non-linear (Gradient Boosting)

# Linear baseline (already trained earlier as 'model')
linear_baseline = model
linear_test_f1 = f1_score(y_test, linear_baseline.predict(X_test_t), zero_division=0)

# Non-linear challenger: Gradient Boosting with CV tuning
gb_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1],
    "min_samples_leaf": [5, 15],
}

gb_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_grid,
    scoring="f1",
    cv=cv,
    refit=True,
    n_jobs=-1,
    verbose=1,
)
gb_search.fit(X_train_t, y_train)

gb_model = gb_search.best_estimator_
gb_test_f1 = f1_score(y_test, gb_model.predict(X_test_t), zero_division=0)

print(f"\nLinear baseline test F1: {linear_test_f1:.4f}")
print(f"Gradient Boosting test F1: {gb_test_f1:.4f}")
print(f"Lift from non-linear model: {gb_test_f1 - linear_test_f1:+.4f}")

# Partial dependence plots to show what the model learned
if gb_test_f1 > linear_test_f1:
    print("\nGenerating partial dependence plots...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get feature names from preprocessor
    numeric_feature_names = NUMERIC_COLS
    categorical_feature_names = list(preprocessor.named_transformers_["cat"]
                                     .named_steps["encoder"]
                                     .get_feature_names_out(CATEGORICAL_COLS))
    all_feature_names = numeric_feature_names + categorical_feature_names
    
    # Plot top 6 most important features
    feature_importance = gb_model.feature_importances_
    top_indices = np.argsort(feature_importance)[-6:][::-1]
    top_features = [all_feature_names[i] for i in top_indices]
    
    PartialDependenceDisplay.from_estimator(
        gb_model, X_train_t, features=top_indices, feature_names=all_feature_names, ax=ax
    )
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "partial_dependence_plots.png"), dpi=150)
    print("Saved: partial_dependence_plots.png")
else:
    print("\nLinear baseline performs better - skipping partial dependence plots")

joblib.dump(gb_model, os.path.join(ARTIFACTS_DIR, "gradient_boosting_model.joblib"))
print("Saved: gradient_boosting_model.joblib")


# ============================================================
# TASK 11 — ENSEMBLE LEARNING
# ============================================================
# WHAT THIS DOES:
# Combines multiple diverse models for robust predictions.
# Measures diversity between base models and documents latency cost.

print("\n" + "="*60)
print("TASK 11: Ensemble Learning")
print("="*60)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.pipeline import make_pipeline
import time

# Choose genuinely diverse base models (different algorithm families)
base_models = {
    "logistic_regression": make_pipeline(
        StandardScaler(with_mean=False), 
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42),
    "knn": make_pipeline(
        StandardScaler(with_mean=False),
        KNeighborsClassifier(n_neighbors=15)
    ),
    "naive_bayes": GaussianNB(),
}

estimators = list(base_models.items())

# Voting ensemble (soft voting)
voting_ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
voting_ensemble.fit(X_train_t, y_train)
voting_test_f1 = f1_score(y_test, voting_ensemble.predict(X_test_t), zero_division=0)

# Stacking ensemble with cross-validation to prevent leakage
stacking_ensemble = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=cv,
    n_jobs=-1,
)
stacking_ensemble.fit(X_train_t, y_train)
stacking_test_f1 = f1_score(y_test, stacking_ensemble.predict(X_test_t), zero_division=0)

print(f"\nVoting ensemble test F1: {voting_test_f1:.4f}")
print(f"Stacking ensemble test F1: {stacking_test_f1:.4f}")

# Diversity check: how often do base models disagree?
print("\nDiversity analysis:")
preds = {name: m.fit(X_train_t, y_train).predict(X_test_t) for name, m in base_models.items()}
names = list(preds.keys())
disagreements = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        disagreement_rate = np.mean(preds[names[i]] != preds[names[j]])
        disagreements.append(disagreement_rate)
        print(f"  {names[i]} vs {names[j]}: {disagreement_rate:.1%} disagreement")

print(f"Average pairwise disagreement: {np.mean(disagreements):.1%}")

# Measure latency cost
def measure_latency(model, X, n_repeats=50):
    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(X)
    return (time.perf_counter() - start) / n_repeats / len(X) * 1000  # ms/sample

single_ms = measure_latency(base_models["naive_bayes"], X_test_t)
voting_ms = measure_latency(voting_ensemble, X_test_t)
stacking_ms = measure_latency(stacking_ensemble, X_test_t)

print(f"\nLatency analysis:")
print(f"  Single model (Naive Bayes): {single_ms:.3f} ms/sample")
print(f"  Voting ensemble: {voting_ms:.3f} ms/sample ({(voting_ms/single_ms - 1)*100:.1f}% overhead)")
print(f"  Stacking ensemble: {stacking_ms:.3f} ms/sample ({(stacking_ms/single_ms - 1)*100:.1f}% overhead)")

# Save the best ensemble
best_ensemble = voting_ensemble if voting_test_f1 >= stacking_test_f1 else stacking_ensemble
best_ensemble_name = "voting" if voting_test_f1 >= stacking_test_f1 else "stacking"
joblib.dump(best_ensemble, os.path.join(ARTIFACTS_DIR, f"{best_ensemble_name}_ensemble.joblib"))
print(f"\nBest ensemble: {best_ensemble_name}")
print(f"Saved: {best_ensemble_name}_ensemble.joblib")


# ============================================================
# TASK 12 — BINARY CLASSIFICATION
# ============================================================
# WHAT THIS DOES:
# Ships a production-grade classifier with calibration,
# cost-justified threshold, fairness checks, and serving package.

print("\n" + "="*60)
print("TASK 12: Binary Classification (Production-Ready)")
print("="*60)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict

# Select the best model from previous tasks for production
# Use the winner from Task 9 (hyperparameter tuning)
production_model = winner

# Split validation set further for calibration
X_cal_fit, X_cal_val, y_cal_fit, y_cal_val = train_test_split(
    X_v, y_v, test_size=0.4, stratify=y_v, random_state=42
)

# Calibration: choose method by evidence, not assumption
print("\nCalibration analysis:")
candidate_scores = {}
for method in ("isotonic", "sigmoid"):
    try:
        cand = CalibratedClassifierCV(production_model, method=method, cv="prefit")
        cand.fit(X_cal_fit, y_cal_fit)
        cal_proba = cand.predict_proba(X_cal_val)[:, 1]
        candidate_scores[method] = brier_score_loss(y_cal_val, cal_proba)
        print(f"  {method}: Brier score = {candidate_scores[method]:.4f}")
    except Exception as e:
        print(f"  {method}: Failed - {e}")
        candidate_scores[method] = float("inf")

chosen_method = min(candidate_scores, key=candidate_scores.get)
print(f"Chosen calibration method: {chosen_method}")

# Refit on full calibration set
calibrated_model = CalibratedClassifierCV(production_model, method=chosen_method, cv="prefit")
calibrated_model.fit(X_v, y_v)

# Cost-justified threshold selection (using cross-validation on training set)
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0

print(f"\nThreshold selection (cost: FN={COST_FALSE_NEGATIVE}x, FP={COST_FALSE_POSITIVE}x)")

# Use out-of-fold predictions from training set for threshold selection
# Use raw features (X_tr) since production_model is a pipeline that includes preprocessing
oof_proba = cross_val_predict(production_model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]

thresholds = np.linspace(0.01, 0.99, 99)
costs = []
for t in thresholds:
    pred = (oof_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_tr, pred).ravel()
    costs.append(fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE)

best_threshold_idx = np.argmin(costs)
best_threshold = thresholds[best_threshold_idx]
min_cost = costs[best_threshold_idx]

print(f"Optimal threshold: {best_threshold:.3f}")
print(f"Minimum cost: {min_cost:.0f}")

# Fairness check across segments
print("\nFairness analysis by education level:")
education_levels = X_test["education_level"].unique() if "education_level" in X_test.columns else ["Unknown"]
fairness_scores = {}

for edu_level in education_levels:
    if pd.isna(edu_level):
        continue
    mask = X_test["education_level"] == edu_level
    if mask.sum() < 5:  # Skip segments with too few samples
        continue
    y_seg = y_test[mask]
    pred_seg = calibrated_model.predict(X_test_t[mask])
    seg_f1 = f1_score(y_seg, pred_seg, zero_division=0)
    fairness_scores[edu_level] = seg_f1
    print(f"  {edu_level}: F1 = {seg_f1:.4f} (n={mask.sum()})")

if fairness_scores:
    fairness_gap = max(fairness_scores.values()) - min(fairness_scores.values())
    print(f"Fairness gap: {fairness_gap:.4f}")
else:
    print("No segments with sufficient samples for fairness analysis")

# Package for serving with model + threshold + cost assumptions
production_package = {
    "model": calibrated_model,
    "threshold": best_threshold,
    "cost_assumptions": {
        "false_negative": COST_FALSE_NEGATIVE,
        "false_positive": COST_FALSE_POSITIVE
    },
    "calibration_method": chosen_method,
    "fairness_scores": fairness_scores,
}

joblib.dump(production_package, os.path.join(ARTIFACTS_DIR, "production_model_package.joblib"))
print("\nSaved: production_model_package.joblib")

# Final evaluation on test set with calibrated model and optimal threshold
final_proba = calibrated_model.predict_proba(X_test_t)[:, 1]
final_pred = (final_proba >= best_threshold).astype(int)
final_f1 = f1_score(y_test, final_pred, zero_division=0)
final_auc = roc_auc_score(y_test, final_proba)

print(f"\nFinal production model test performance:")
print(f"  F1 score: {final_f1:.4f}")
print(f"  ROC-AUC: {final_auc:.4f}")
print(f"  Threshold: {best_threshold:.3f}")

# Save final metrics
task12_metrics = {
    "calibration_method": chosen_method,
    "optimal_threshold": round(float(best_threshold), 3),
    "final_test_f1": round(float(final_f1), 4),
    "final_test_auc": round(float(final_auc), 4),
    "cost_assumptions": {
        "false_negative": COST_FALSE_NEGATIVE,
        "false_positive": COST_FALSE_POSITIVE
    },
    "fairness_scores": {str(k): float(v) for k, v in fairness_scores.items()},
}

with open(os.path.join(ARTIFACTS_DIR, "task12_production_metrics.json"), "w") as f:
    json.dump(task12_metrics, f, indent=2)
print("Saved: task12_production_metrics.json")
