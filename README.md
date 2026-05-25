<h1 align="center">EasyVisa: Predicting Visa Approval Outcomes</h1>

<p align="center"><i>Using Advanced Machine Learning Techniques</i></p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-2.2-150458?logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-2.0-013243?logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="XGBoost" src="https://img.shields.io/badge/XGBoost-3.0-006ACC">
  <img alt="imbalanced-learn" src="https://img.shields.io/badge/imbalanced--learn-SMOTE-2E8B57">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.10-11557C">
  <img alt="seaborn" src="https://img.shields.io/badge/seaborn-0.13-4C72B0">
</p>

## Overview

Businesses across the United States compete for skilled talent, and the Immigration and Nationality Act allows employers to hire foreign workers when there are not enough qualified workers available locally. The Office of Foreign Labor Certification reviews these applications, but the volume keeps rising. In FY 2016 it processed 775,979 employer applications covering nearly 1.7 million positions, a nine percent increase over the prior year. Reviewing every case manually has become slow and resource intensive. This project builds a machine learning classifier for the firm EasyVisa that predicts whether a visa application is likely to be certified or denied, and identifies the factors that most influence the outcome so reviewers can shortlist applicants more efficiently.

## Problem Statement

The objective is to analyze historical visa application data and build a classification model that facilitates the approval process and recommends a suitable applicant profile for certification or denial. By surfacing the drivers that significantly influence case status, the model helps prioritize cases, reduce manual review time, and improve the overall efficiency of the certification process.

## Dataset

The dataset contains **25,480 visa applications** across **12 columns**. The target variable is `case_status`, which indicates whether a visa was certified or denied, and the classes are imbalanced toward certified applications.

| Column | Description |
| --- | --- |
| `case_id` | ID of each visa application |
| `continent` | Continent of the employee |
| `education_of_employee` | Education level of the employee |
| `has_job_experience` | Whether the employee has job experience (Y or N) |
| `requires_job_training` | Whether the employee requires job training (Y or N) |
| `no_of_employees` | Number of employees in the employer company |
| `yr_of_estab` | Year the employer company was established |
| `region_of_employment` | Intended region of employment in the US |
| `prevailing_wage` | Average wage paid to similarly employed workers in the area |
| `unit_of_wage` | Unit of the prevailing wage (Hourly, Weekly, Monthly, Yearly) |
| `full_time_position` | Whether the position is full time (Y or N) |
| `case_status` | Target. Whether the visa was certified or denied |

## Approach

1. **Data understanding.** Reviewed the shape, data types, and summary statistics across the 25,480 applications and 12 columns.
2. **Exploratory data analysis.** Studied univariate distributions and bivariate relationships between applicant and employer features and the case status.
3. **Data preparation.** Engineered and encoded categorical features and split the data into training, validation, and test sets.
4. **Class balancing.** Addressed the imbalance in the target using both oversampling with SMOTE and random undersampling, then compared their effect on model performance.
5. **Modelling.** Trained and compared a Decision Tree alongside ensemble methods including Bagging, Random Forest, AdaBoost, Gradient Boosting, and XGBoost.
6. **Hyperparameter tuning.** Tuned the strongest candidates using Grid Search and Randomized Search to reduce overfitting and improve generalization.
7. **Evaluation and recommendations.** Compared models on accuracy, precision, recall, and F1 score, selected the final model, and translated the findings into business recommendations.

## Models Used

The project compares a range of classification algorithms before and after tuning:

- **Decision Tree** as a single tree baseline.
- **Bagging** and **Random Forest** as bagging based ensembles.
- **AdaBoost**, **Gradient Boosting**, and **XGBoost** as boosting based ensembles.

Each model was trained on the original data, on oversampled data, and on undersampled data, and the best candidates were tuned with Grid Search and Randomized Search.

## Key Findings

- **Oversampling outperformed undersampling.** Oversampling improved recall and F1 score while preserving all training information, whereas undersampling reduced the dataset size and caused some loss of useful patterns.
- **Boosting models generalized better than bagging models.** Before tuning, Random Forest and Bagging showed clear overfitting, with Random Forest reaching perfect training scores but lower validation scores. Boosting models such as AdaBoost, Gradient Boosting, and XGBoost produced more balanced results.
- **Tuning stabilized performance** by narrowing the gap between training and validation scores.
- **XGBoost was the best model.** The tuned XGBoost achieved validation accuracy of 0.7474, precision of 0.7760, recall of 0.8740, and F1 score of 0.8221. On the test set it reached accuracy of 0.7425, precision of 0.7715, recall of 0.8731, and F1 score of 0.8192, which confirms that it generalizes well.
- **Strong recall, room on precision.** The model is effective at identifying certified applications, though it still misclassifies some denied applications as certified.
- **Education was the most influential feature**, especially the High School level. Job experience, wage unit, region of employment, continent, prevailing wage, full time position status, and job training requirement also contributed meaningfully.

## Recommendations

- Select the tuned XGBoost classifier as the final model, given its best validation F1 score and stable test performance.
- Prefer oversampling over undersampling, since it improved performance while retaining useful information.
- Use the model as an early screening and decision support tool. Fast track applications with a high probability of certification, and flag lower probability cases for additional document review or applicant guidance.
- Focus consultant attention on the key drivers, including education level, prior job experience, prevailing wage, wage unit, full time position status, and region of employment.

## Repository Structure

```
easyvisa-approval-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── EasyVisa.csv                     # not committed (see .gitignore)
├── notebooks/
│   └── easyvisa_approval_prediction.ipynb
└── reports/
    └── easyvisa_approval_prediction.html  # exported notebook
```

## Getting Started

```bash
# clone
git clone https://github.com/<your-username>/easyvisa-approval-prediction.git
cd easyvisa-approval-prediction

# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# launch the notebook
jupyter notebook notebooks/easyvisa_approval_prediction.ipynb
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, Matplotlib, seaborn, Jupyter
