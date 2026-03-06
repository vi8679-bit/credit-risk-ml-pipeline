# Credit Risk Prediction using Machine Learning

## Overview
This project develops a machine learning pipeline to predict whether a borrower will default on a loan using financial and credit history features. Credit risk prediction is a critical task in the financial industry because it helps lenders identify high-risk borrowers and reduce potential financial losses.

The project demonstrates a complete machine learning workflow including exploratory data analysis, feature engineering, supervised machine learning, hyperparameter tuning, and model evaluation.

## Problem Statement
Financial institutions face significant losses when borrowers fail to repay loans. The objective of this project is to build predictive models that classify whether a borrower is likely to default on a loan based on demographic and financial attributes.


## Dataset
Source: Kaggle Credit Risk Dataset   

Target Variable:  
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
- cb_person_cred_hist_length  


## Exploratory Data Analysis
Exploratory data analysis was performed to understand relationships between borrower characteristics and loan default risk.

Key analyses included:

- Missing value analysis  
- Distribution of numerical variables  
- Categorical feature distributions  
- Loan default distribution  
- Income vs loan default analysis  
- Loan intent vs default analysis  
- Correlation heatmap  
- Default rate by home ownership  


## Feature Engineering
A new feature was created:

credit_history_to_age_ratio

This feature measures the maturity of a borrower’s credit history relative to their age.


## Machine Learning Pipeline
A structured scikit-learn pipeline was implemented to ensure reproducibility and prevent data leakage.

Pipeline steps:

1. Train-test split with stratification  
2. Missing value imputation  
3. One-hot encoding for categorical variables  
4. Feature scaling for numerical variables  
5. Model training  
6. Hyperparameter tuning using GridSearchCV  

Models implemented:

- Logistic Regression  
- Random Forest  

## Model Performance

| Model | Accuracy | F1 Score | ROC-AUC |
|------|------|------|------|
| Logistic Regression | 0.87 | 0.65 | 0.87 |
| Random Forest | 0.93 | 0.82 | 0.93 |

Random Forest achieved the best overall performance.


## Threshold Optimization
To improve prediction performance for loan defaults, classification thresholds were evaluated.

Best threshold ≈ **0.55**

This threshold produced the best balance between precision and recall for detecting default cases.

## Feature Importance
The most influential predictors identified by the Random Forest model were:

- loan_percent_income  
- person_income  
- loan_int_rate  
- loan_amnt  
- person_emp_length  

These variables strongly influence the likelihood of borrower default.


## Key Insights
Borrowers with a higher loan-to-income ratio are more likely to default.

Interest rates and borrower income also play a major role in determining credit risk. These insights can help financial institutions improve lending decisions and risk management strategies.


## Technologies Used
- Python  
- pandas  
- NumPy  
- scikit-learn  
- Matplotlib  
- Seaborn  



## Future Improvements
Possible extensions include:

- Gradient boosting models such as XGBoost or LightGBM  
- Model explainability using SHAP  
- Deployment using Streamlit  
- Automated credit scoring system  



## Author
Indraneel Mannava
