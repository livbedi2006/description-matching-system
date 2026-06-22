# Description Matching System

A machine learning-powered system that matches job applicants to job descriptions based on skill compatibility and requirements.

## Overview

This system analyzes student profiles and job descriptions to determine optimal matches for recruitment purposes. It uses skill-based scoring to evaluate the fit between candidates and job requirements.

## Dataset

The project contains three main datasets:

### 1. Job Descriptions (`job_descriptions.csv`)
Contains job listings with detailed requirements:
- **Job Details**: Company, title, location, experience required, salary offered
- **Skill Requirements**: Scores (0-100) for required skills:
  - Python
  - SQL
  - Machine Learning (ML)
  - JavaScript
  - Data Structures
  - Statistics
- **Education**: Minimum education level (UG, PG, PhD)

### 2. Student Exam Scores (`students_exam_scores.csv`)
Contains student profiles with assessment results:
- **Student Info**: Name, education level, years of experience, location
- **Skill Scores**: Exam scores (0-100) in:
  - Python
  - SQL
  - Machine Learning (ML)
  - JavaScript
  - Data Structures
  - Statistics
- **Additional Metrics**: Exam time, self-reported confidence, retake count, expected salary

### 3. Applications (`applications.csv`)
Records job applications and outcomes:
- **Application Details**: Application ID, student ID, job ID, application date
- **Match Status**: Whether the application was deemed a good match (1 = yes, 0 = no)
- **Recruiter Notes**: Feedback on the application decision

### 4. Clean Modelling Table (`clean_modelling_table.csv`)
Processed dataset ready for machine learning model training.

## Project Structure

```
.
├── Read.md                           # Project documentation
├── learn.ipynb                       # Jupyter notebook for analysis
├── applications.csv                  # Application records
├── clean_modelling_table.csv         # Processed modelling data
├── job_descriptions.csv             # Job listings with requirements
└── students_exam_scores.csv         # Student assessment data
```

## Key Features

- **Skill-Based Matching**: Compares student skill scores against job requirements
- **Multi-Dimensional Analysis**: Considers education, experience, location, and salary expectations
- **Recruitment Insights**: Tracks application outcomes and recruiter feedback
- **Data-Driven Decisions**: Uses historical application data to improve matching accuracy

## Potential Use Cases

- Automated candidate screening
- Job recommendation system
- Recruitment analytics and reporting
- Skill gap analysis
- Salary benchmarking

## Data Schema

### Job Descriptions
- `job_id`: Unique job identifier
- `company`: Company name
- `title`: Job title
- `location`: Job location
- `exp_required_years`: Years of experience required
- `salary_offered_inr`: Salary offered in INR
- `*_required`: Skill requirement scores (Python, SQL, ML, JavaScript, Data Structures, Statistics)
- `edu_minimum`: Minimum education level

### Student Scores
- `student_id`: Unique student identifier
- `name`: Student name
- `education_level`: Education level (UG, PG, PhD)
- `years_experience`: Years of work experience
- `location`: Student location
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
- `recruiter_notes_post_decision`: Recruiter feedback

## Getting Started

To analyze this data:

1. Load the datasets using pandas:
```python
import pandas as pd

job_descriptions = pd.read_csv('job_descriptions.csv')
student_scores = pd.read_csv('students_exam_scores.csv')
applications = pd.read_csv('applications.csv')
```

2. Explore the data and build matching models using the `learn.ipynb` notebook.

## License

This project is part of a recruitment analytics initiative.