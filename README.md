# Credit Risk Prediction using Machine Learning

# Overview

This project develops a machine learning pipeline to predict whether a borrower will default on a loan using financial and credit history features. Predicting default risk is an important task for financial institutions because it enables lenders to make informed lending decisions and manage credit exposure effectively.The project implements a full data science workflow including exploratory data analysis, feature engineering, supervised machine learning, hyperparameter tuning, and model evaluation.

# Problem Statement

Financial institutions face significant losses due to loan defaults. The objective of this project is to build predictive models that classify whether a borrower will default on a loan based on demographic and financial attributes.

# Dataset

Dataset: Credit Risk Dataset (Kaggle)
Records: 32,581
Target variable: loan_status
Key features include:
person_age
person_income
person_home_ownership
person_emp_length
loan_intent
loan_percent_income
loan_int_rate
cb_person_cred_hist_length

# Project Workflow

The project follows a structured machine learning pipeline.
1. Data Loading
The dataset was downloaded from Kaggle and loaded into a pandas DataFrame.
2. Exploratory Data Analysis
EDA was performed to understand distributions, relationships, and potential predictors.
Analyses included:
missing value analysis
distribution plots for numerical variables
categorical feature distributions
correlation heatmap
default rate analysis
3. Feature Engineering
A new feature was created:
credit_history_to_age_ratio
This feature measures the proportion of credit history relative to borrower age and provides insight into credit maturity.

# Machine Learning Models

The following models were implemented:
Logistic Regression
Random Forest

Hyperparameter tuning was applied using GridSearchCV.

The modeling pipeline includes:
missing value imputation
feature scaling
one hot encoding for categorical variables
model training

# Model Evaluation

Models were evaluated using:
Accuracy
F1 Score
ROC-AUC
Random Forest produced the best performance.
Results:
Accuracy: 0.93
F1 Score: 0.82
ROC-AUC: 0.93

# Threshold Optimization

The classification threshold was optimized to improve performance on the minority class (loan defaults).
The best threshold was approximately:
0.55
This improved the balance between precision and recall.

# Feature Importance

The most important predictors identified by the Random Forest model were:
loan_percent_income
person_income
loan_int_rate
loan_amnt
person_emp_length
These variables strongly influence the likelihood of loan default.

# Key Insights

Borrowers with a higher loan burden relative to income are more likely to default.
Interest rates and borrower income also play a major role in determining default risk.
These insights can assist financial institutions in improving risk assessment and credit approval strategies.
