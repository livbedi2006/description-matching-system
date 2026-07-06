# Description Matching System

A machine learning-powered system that matches job applicants to job descriptions based on skill compatibility and requirements.

## Overview

This system analyzes student profiles and job descriptions to determine optimal matches for recruitment purposes. It uses skill-based scoring to evaluate the fit between candidates and job requirements, employing advanced ML techniques including:

- **Preprocessing Pipeline**: Data leakage prevention with proper train/val/test splits and fitted preprocessing on training data only
- **Feature Engineering**: Domain-driven features including skill gaps, experience gaps, and education adequacy
- **Model Selection**: Multiple model types including Logistic Regression, Random Forest, and Gradient Boosting with hyperparameter tuning
- **Non-Linear Modeling**: Gradient Boosting to capture complex relationships that linear models might miss
- **Ensemble Methods**: Voting and Stacking ensembles combining diverse model families for robust predictions
- **Model Calibration**: Isotonic and sigmoid calibration to ensure reliable probability estimates
- **Cost-Optimized Decision Thresholds**: Business-driven threshold selection based on false positive/negative cost ratios
- **Fairness Analysis**: Segment-level performance evaluation to ensure consistent model behavior across groups
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
│   └── train_advanced.py             # Advanced ML pipeline with all techniques
└── run_artifacts/                     # Trained model artifacts
    ├── best_pipeline.joblib          # Best performing model
    ├── fitted_preprocessor.joblib    # Preprocessing pipeline
    ├── pipeline.joblib               # End-to-end pipeline
    ├── gradient_boosting_model.joblib # Non-linear model
    ├── stacking_ensemble.joblib      # Ensemble model
    ├── production_model_package.joblib # Production-ready model with calibration
    ├── roc_pr_curves.png             # ROC/PR curve visualizations
    ├── partial_dependence_plots.png  # Feature importance plots
    ├── run_metrics.json              # Validation metrics
    ├── final_metrics.json            # Test set comparison
    └── production_metrics.json       # Production model metrics
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

Run the complete ML pipeline with preprocessing, feature engineering, hyperparameter tuning, ensemble methods, and production-ready modeling:
```bash
python src/train_advanced.py
```

This will execute the complete pipeline:
1. Data preprocessing with train/val/test splits and fitted preprocessing on training data only
2. Logistic Regression baseline model training
3. Binary decision analysis with cost-optimal thresholds
4. Feature engineering with domain knowledge (skill gaps, experience gaps, education adequacy)
5. End-to-end pipeline construction
6. Hyperparameter tuning for Logistic Regression and Random Forest
7. Non-linear modeling with Gradient Boosting and partial dependence analysis
8. Ensemble learning with Voting and Stacking methods, including diversity analysis
9. Model calibration (isotonic and sigmoid methods)
10. Cost-justified threshold selection
11. Fairness analysis across segments
12. Production model packaging with calibration and threshold metadata
13. Save all artifacts to `run_artifacts/`

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

### Using Production-Ready Model

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

### Test Set Results

- **Default Logistic Regression**: F1 = 0.8000
- **Tuned Logistic Regression**: F1 = 0.6897
- **Tuned Random Forest**: F1 = 0.6792

### Non-Linear Modeling Results

- **Linear Baseline**: F1 = 0.8000
- **Gradient Boosting**: F1 = 0.6792
- **Finding**: Linear baseline performed better (no significant non-linear relationships in this dataset)

### Ensemble Learning Results

- **Voting Ensemble**: F1 = 0.7333
- **Stacking Ensemble**: F1 = 0.7636
- **Diversity**: 23.3% average pairwise disagreement between base models
- **Latency Overhead**: ~7000% for ensembles vs single model

### Production-Ready Model Results

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
