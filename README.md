# Description Matching System

A machine learning-powered system that matches job applicants to job descriptions based on skill compatibility and requirements.

## Overview

This system analyzes student profiles and job descriptions to determine optimal matches for recruitment purposes. It uses skill-based scoring to evaluate the fit between candidates and job requirements, employing advanced ML techniques including:

- **Preprocessing Pipeline**: Data leakage prevention with proper train/val/test splits
- **Feature Engineering**: Domain-driven features (skill gaps, experience gaps, education adequacy)
- **Model Selection**: Logistic Regression, Random Forest, Gradient Boosting with hyperparameter tuning
- **Complex Relationships**: Non-linear modeling with partial dependence analysis
- **Ensemble Learning**: Voting and Stacking ensembles with diversity analysis
- **Production-Ready Classification**: Model calibration, cost-justified thresholds, fairness checks
- **Binary Decision Analysis**: Cost-optimal threshold selection for business metrics
- **End-to-End Pipeline**: Complete sklearn pipeline for production deployment

## Dataset

The project contains four main datasets in the `data/` directory:

### 1. Job Descriptions (`job_descriptions.csv`)
Contains job listings with detailed requirements:
- **Job Details**: Company, title, location, experience required, salary offered
- **Skill Requirements**: Scores (0-100) for Python, SQL, ML, JavaScript, Data Structures, Statistics
- **Education**: Minimum education level (UG, PG, PhD)

### 2. Student Exam Scores (`students_exam_scores.csv`)
Contains student profiles with assessment results:
- **Student Info**: Name, education level, years of experience, location
- **Skill Scores**: Exam scores (0-100) in all technical skills
- **Additional Metrics**: Exam time, self-reported confidence, retake count, expected salary

### 3. Applications (`applications.csv`)
Records job applications and outcomes:
- **Application Details**: Application ID, student ID, job ID, application date
- **Match Status**: Whether the application was deemed a good match (1 = yes, 0 = no)
- **Recruiter Notes**: Feedback on the application decision (excluded from training to prevent leakage)

### 4. Clean Modelling Table (`clean_modelling_table.csv`)
Processed dataset ready for machine learning model training, with joined student, job, and application data.

## Project Structure

```
.
├── README.md                           # Project documentation
├── Read.md                             # Detailed data documentation
├── data/                               # Dataset directory
│   ├── applications.csv              # Application records
│   ├── clean_modelling_table.csv      # Processed modelling data
│   ├── job_descriptions.csv          # Job listings with requirements
│   └── students_exam_scores.csv      # Student assessment data
├── src/                               # Source code directory
│   ├── config.py                     # Configuration settings
│   ├── data.py                       # Data loading utilities
│   ├── evaluate.py                   # Evaluation metrics
│   ├── features.py                   # Feature engineering
│   ├── learn.ipynb                   # Jupyter notebook for analysis
│   ├── ml_utils.py                   # ML utilities (dependency checking, validation)
│   ├── model.py                      # Model definitions
│   ├── train.py                      # Basic training harness
│   └── train_advanced.py             # Advanced ML pipeline (tasks 4-12)
└── run_artifacts/                     # Trained model artifacts
    ├── best_pipeline.joblib          # Best performing model
    ├── fitted_preprocessor.joblib    # Preprocessing pipeline
    ├── pipeline.joblib               # End-to-end pipeline
    ├── gradient_boosting_model.joblib # Task 10 non-linear model
    ├── stacking_ensemble.joblib      # Task 11 ensemble model
    ├── production_model_package.joblib # Task 12 production-ready model
    ├── roc_pr_curves.png             # ROC/PR curve visualizations
    ├── partial_dependence_plots.png  # Task 10 feature importance plots
    ├── run_metrics.json              # Validation metrics
    ├── final_metrics.json            # Test set comparison
    └── task12_production_metrics.json # Production model metrics
```

## Installation

### Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib

### Setup

1. Clone the repository:
```bash
git clone https://github.com/livbedi2006/description-matching-system.git
cd description-matching-system
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib matplotlib
```

## Usage

### Basic Training

Run the basic training harness:
```bash
python src/train.py
```

This runs the baseline skill overlap model and logs results to `experiment_log.csv`.

### Advanced ML Pipeline

Run the complete ML pipeline with preprocessing, feature engineering, hyperparameter tuning, and production-ready modeling:
```bash
python src/train_advanced.py
```

This will execute the complete pipeline (Tasks 4-12):
1. **Tasks 4-6**: Data preprocessing, Logistic Regression baseline, binary decision analysis
2. **Task 7**: Feature engineering with domain knowledge
3. **Task 8**: End-to-end pipeline construction
4. **Task 9**: Hyperparameter tuning (Logistic Regression + Random Forest)
5. **Task 10**: Complex relationships analysis (Gradient Boosting vs linear baseline)
6. **Task 11**: Ensemble learning (Voting + Stacking with diversity analysis)
7. **Task 12**: Production-ready classification (calibration, cost-justified thresholds, fairness checks)
8. Save all artifacts to `run_artifacts/`

### Using Trained Models

Load and use the trained pipeline:
```python
import joblib
import pandas as pd

# Load the best pipeline
pipeline = joblib.load("run_artifacts/best_pipeline.joblib")

# Prepare your data (same structure as training data)
new_data = pd.DataFrame({
    "exp_required_years": [2.0],
    "salary_offered_inr": [350000],
    "python_required": [62.8],
    # ... other columns
})

# Make predictions
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)[:, 1]
```

### Using Production-Ready Model (Task 12)

Load the production package with calibrated model and optimal threshold:
```python
import joblib

# Load production package
pkg = joblib.load("run_artifacts/production_model_package.joblib")

# Get calibrated probabilities and cost-optimized decisions
proba = pkg["model"].predict_proba(new_data)[:, 1]
decision = (proba >= pkg["threshold"]).astype(int)

# Access cost assumptions and fairness metrics
print(f"Threshold: {pkg['threshold']}")
print(f"Cost assumptions: {pkg['cost_assumptions']}")
print(f"Fairness scores: {pkg['fairness_scores']}")
```

## Model Performance

### Test Set Results (Tasks 4-9)

- **Default Logistic Regression**: F1 = 0.8000
- **Tuned Logistic Regression**: F1 = 0.6897
- **Tuned Random Forest**: F1 = 0.6792

### Task 10 - Complex Relationships

- **Linear Baseline**: F1 = 0.8000
- **Gradient Boosting**: F1 = 0.6792
- **Finding**: Linear baseline performed better (no significant non-linear relationships in this dataset)

### Task 11 - Ensemble Learning

- **Voting Ensemble**: F1 = 0.7333
- **Stacking Ensemble**: F1 = 0.7636
- **Diversity**: 23.3% average pairwise disagreement between base models
- **Latency Overhead**: ~7000% for ensembles vs single model

### Task 12 - Production-Ready Classification

- **Calibration Method**: Isotonic (Brier score: 0.0954)
- **Cost-Optimized Threshold**: Selected via cross-validation on training set
- **Fairness Analysis**: Evaluated across education segments

### Key Findings

- Default Logistic Regression performed best on test set (hyperparameter tuning overfit to validation)
- Cost-optimal threshold: 0.082 (39% cost reduction vs default 0.50)
- Validation F1: 0.7742, ROC-AUC: 0.8733
- Ensemble methods provide robustness but with significant latency overhead
- Model calibration ensures reliable probability estimates

### Business Metrics

The model uses cost-sensitive thresholding:
- **False Positive Cost**: 1.0 (wasted recruiter time)
- **False Negative Cost**: 5.0 (missed good candidate)
- **Cost Reduction**: 39% with optimized threshold

## Key Features

- **Skill-Based Matching**: Compares student skill scores against job requirements
- **Multi-Dimensional Analysis**: Considers education, experience, location, and salary expectations
- **Recruitment Insights**: Tracks application outcomes and recruiter feedback
- **Data-Driven Decisions**: Uses historical application data to improve matching accuracy
- **Data Leakage Prevention**: Proper train/val/test splits with preprocessing fit only on training data
- **Feature Engineering**: Domain knowledge encoded into skill gaps, experience gaps, and education adequacy
- **Hyperparameter Tuning**: Systematic search for optimal model configurations
- **Non-Linear Modeling**: Gradient Boosting for capturing complex relationships
- **Ensemble Methods**: Voting and Stacking ensembles with diversity analysis
- **Model Calibration**: Isotonic and sigmoid calibration for reliable probability estimates
- **Cost-Optimized Thresholds**: Business-driven decision thresholds
- **Fairness Analysis**: Segment-level performance evaluation
- **Production Packaging**: Complete model packages with threshold and metadata
- **Dependency Validation**: Automated checking of package versions and data quality

## Potential Use Cases

- Automated candidate screening
- Job recommendation system
- Recruitment analytics and reporting
- Skill gap analysis
- Salary benchmarking
- Resume matching automation

## Data Schema

### Job Descriptions
- `job_id`: Unique job identifier
- `company`: Company name
- `title`: Job title
- `location_job`: Job location
- `exp_required_years`: Years of experience required
- `salary_offered_inr`: Salary offered in INR
- `*_required`: Skill requirement scores (Python, SQL, ML, JavaScript, Data Structures, Statistics)
- `edu_minimum`: Minimum education level

### Student Scores
- `student_id`: Unique student identifier
- `name`: Student name
- `education_level`: Education level (UG, PG, PhD)
- `years_experience`: Years of work experience
- `location_student`: Student location
- `*_score`: Skill assessment scores
- `exam_time_seconds`: Time taken for exam
- `self_reported_confidence`: Confidence level (1-5)
- `retake_count`: Number of exam retakes
- `expected_salary_inr`: Expected salary in INR

### Applications
- `application_id`: Unique application identifier
- `student_id`: Student identifier
- `job_id`: Job identifier
- `application_date`: Date of application
- `is_good_match`: Match quality flag
- `recruiter_notes_post_decision`: Recruiter feedback (excluded from training)

## Configuration

All configuration settings are centralized in `src/config.py`:
- Random seed for reproducibility
- Data paths
- Feature engineering parameters
- Model hyperparameters
- Experiment logging settings

## ML Utilities (ml_utils.py)

The `src/ml_utils.py` module provides essential utilities for robust ML pipeline development:

### Dependency Checking
- Validates required package versions before pipeline execution
- Fails fast with clear error messages for missing/incompatible dependencies
- Supports pandas, numpy, scikit-learn, and joblib

### Data Validation
- Checks for NaN/Inf values in input data
- Validates X/y length matching
- Ensures minimum samples per class
- Verifies class balance

### Safe Fitting
- Wraps model fitting with try/except for graceful failure handling
- Provides fallback options when primary models fail
- Prevents pipeline crashes from individual component failures

### Execution Logging
- Timestamps all pipeline runs for reproducibility
- Logs execution metadata to track experiments
- Supports file checksum verification for data integrity

## Contributing

To add a new model:

1. Create a new class in `src/model.py` following the interface:
   ```python
   class YourModel:
       def fit(self, data): ...
       def predict(self, data): ...
   ```

2. Run experiments through the harness in `src/train.py`:
   ```python
   from model import YourModel
   from train import run_experiment
   
   model = YourModel()
   results = run_experiment(model, model_name="your_model")
   ```

## License

This project is part of a recruitment analytics initiative.

## Acknowledgments

- Built with scikit-learn for ML pipeline components
- Uses pandas for data manipulation
- Implements best practices for ML model development (data leakage prevention, proper validation, cost-sensitive learning)
