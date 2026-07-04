# Description Matching System

A machine learning-powered system that matches job applicants to job descriptions based on skill compatibility and requirements.

## Overview

This system analyzes student profiles and job descriptions to determine optimal matches for recruitment purposes. It uses skill-based scoring to evaluate the fit between candidates and job requirements, employing advanced ML techniques including:

- **Preprocessing Pipeline**: Data leakage prevention with proper train/val/test splits
- **Feature Engineering**: Domain-driven features (skill gaps, experience gaps, education adequacy)
- **Model Selection**: Logistic Regression and Random Forest with hyperparameter tuning
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
│   ├── features.py                   # Feature工程
│   ├── learn.ipynb                   # Jupyter notebook for analysis
│   ├── model.py                      # Model definitions
│   ├── train.py                      # Basic training harness
│   └── train_advanced.py             # Advanced ML pipeline (tasks 4-9)
└── run_artifacts/                     # Trained model artifacts
    ├── best_pipeline.joblib          # Best performing model
    ├── fitted_preprocessor.joblib    # Preprocessing pipeline
    ├── pipeline.joblib               # End-to-end pipeline
    ├── roc_pr_curves.png             # ROC/PR curve visualizations
    ├── run_metrics.json              # Validation metrics
    └── final_metrics.json            # Test set comparison
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

Run the complete ML pipeline with preprocessing, feature engineering, and hyperparameter tuning:
```bash
python src/train_advanced.py
```

This will:
1. Load and preprocess the data
2. Train Logistic Regression baseline
3. Perform binary decision analysis with cost-optimal thresholds
4. Engineer domain features
5. Build end-to-end pipeline
6. Tune hyperparameters (Logistic Regression + Random Forest)
7. Evaluate on test set
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

## Model Performance

### Test Set Results

- **Default Logistic Regression**: F1 = 0.8000
- **Tuned Logistic Regression**: F1 = 0.6897
- **Tuned Random Forest**: F1 = 0.6792

### Key Findings

- Default Logistic Regression performed best on test set (hyperparameter tuning overfit to validation)
- Cost-optimal threshold: 0.082 (39% cost reduction vs default 0.50)
- Validation F1: 0.7742, ROC-AUC: 0.8733

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
