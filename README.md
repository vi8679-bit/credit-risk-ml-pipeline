## Credit Risk Prediction using Machine Learning

## Overview
This project builds a machine learning pipeline to predict whether a borrower will default on a loan using financial and credit history features.

Credit risk prediction is an important problem in banking and financial services because it helps lenders identify high-risk borrowers and reduce potential losses.

This project demonstrates a full data science workflow including exploratory data analysis, feature engineering, model training, hyperparameter tuning, and model evaluation.

---

## Problem Statement
Financial institutions face significant losses when borrowers fail to repay loans.

The goal of this project is to develop machine learning models that can classify borrowers as likely to **default** or **not default** based on demographic and financial information.

---

## Dataset
Source: Kaggle Credit Risk Dataset

Records: ~32,000 borrowers

Target variable:
loan_status  
0 = No Default  
1 = Default

### Key Features
- person_age
- person_income
- person_home_ownership
- person_emp_length
- loan_intent
- loan_percent_income
- loan_int_rate
- credit history length

---

## Exploratory Data Analysis
EDA was performed to understand relationships between borrower characteristics and default risk.

Analysis included:

- missing value analysis
- numerical feature distributions
- categorical feature distributions
- correlation heatmap
- default rate by home ownership
- income vs loan default analysis

---

## Feature Engineering
A new feature was created:

credit_history_to_age_ratio

This feature measures the maturity of a borrower's credit history relative to their age.

---

## Machine Learning Pipeline
The modeling workflow uses a structured scikit-learn pipeline.

Steps included:

1. Train-test split with stratification
2. Missing value imputation
3. One-hot encoding for categorical variables
4. Feature scaling for numerical features
5. Model training
6. Hyperparameter tuning using GridSearchCV

Models implemented:

- Logistic Regression
- Random Forest


## Model Evaluation
Models were evaluated using:

- Accuracy
- F1 Score
- ROC-AUC

### Best Model
Random Forest

Performance:

Accuracy: 0.93  
F1 Score: 0.82  
ROC-AUC: 0.93

## Threshold Optimization
The classification threshold was optimized to improve performance on the minority class (loan defaults).

Best threshold ≈ 0.55

This improved recall while maintaining high precision.

## Feature Importance
The most influential predictors identified by the Random Forest model were:

- loan_percent_income
- person_income
- loan_int_rate
- loan_amnt
- person_emp_length

These variables strongly influence the likelihood of loan default.


## Key Insights
Borrowers with higher loan-to-income ratios have a significantly higher probability of default.

Interest rate and borrower income are also strong predictors of credit risk.

These insights can help financial institutions improve lending decisions and risk management strategies.


## Technologies Used
Python  
pandas  
NumPy  
scikit-learn  
Matplotlib  
Seaborn  

## Future Improvements
Possible extensions of this project include:

- XGBoost and LightGBM models
- SHAP model explainability
- deployment using Streamlit
- automated credit scoring system



## Author
Indraneel Mannava
