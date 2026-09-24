# Credit Risk Prediction: Loan Default Model

An end-to-end scikit-learn pipeline that predicts whether a borrower will default, using 32,581 loan records with borrower income, employment, credit history, and loan terms. About 22% of loans in the data defaulted.

<img width="989" height="590" alt="feature_importance" src="https://github.com/user-attachments/assets/dcd159e9-4edb-49bf-bd9a-0255f3c4a61e" />

## Key findings

- **Loan-to-income ratio is the strongest predictor of default.** Borrowers committing a larger share of their income to the loan default far more often.
- **Income and interest rate come next.** Lower-income borrowers and higher-rate loans (which already reflect higher assessed risk) default more.
- **Renters default at more than 4× the rate of homeowners:** about 32% of renters vs. 7% of borrowers who own their home outright.

**Business takeaway:** a lender can tighten approval rules around loan-to-income ratio before anything else. It's the single most informative, easy-to-verify signal in the data.

## Model results (held-out test set, 6,517 loans, 1,422 defaults)

| Model | Accuracy | ROC-AUC | Default precision | Default recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.87 | 0.87 | 0.77 | 0.56 | 0.65 |
| **Random Forest** | **0.93** | **0.93** | **0.97** | **0.71** | **0.82** |

When Random Forest flags a borrower as a likely defaulter it is right 97% of the time, and it catches about 71% of actual defaults. Logistic Regression was tuned with `GridSearchCV` (regularization strength, solver, class weighting) but improved very little over its baseline, which suggests the relationships here are non-linear, where tree models have the edge.

A threshold analysis on the Random Forest showed a flat trade-off between 0.4 and 0.6. Lowering the threshold to catch more defaulters costs precision fairly quickly, which is the kind of decision a lender would set based on its own loss vs. approval-rate targets.

## Pipeline

Built as a single scikit-learn `Pipeline` so every preprocessing step is learned from training data only (no leakage into the test set):

1. **Stratified 80/20 train/test split** first.
2. **Feature engineering** (inside the pipeline via `FunctionTransformer`): capped impossible values (ages up to 144, employment length up to 123 years) and created `credit_history_to_age_ratio`.
3. **Preprocessing** (`ColumnTransformer`): median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical ones.
4. **Models:** Logistic Regression and Random Forest; hyperparameter tuning with 5-fold `GridSearchCV`.
5. **Evaluation:** accuracy, ROC-AUC, precision/recall/F1 on the default class, confusion matrix, and feature importances.

## Dataset

[Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset): 32,581 loans with 11 features including `person_income`, `person_home_ownership`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, and credit bureau history.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook credit_risk_ml_pipeline.ipynb
```

The notebook downloads the data automatically with `kagglehub`.

## Tech stack

Python · pandas · NumPy · scikit-learn (Pipeline, ColumnTransformer, GridSearchCV) · Matplotlib · Seaborn

## Next steps

- Gradient boosting (XGBoost / LightGBM) and SHAP explanations for individual loan decisions.
- An expected-loss view (probability of default × loan amount) to rank the portfolio by dollar risk, not just default probability.
- Check model behavior without `loan_grade` and `loan_int_rate`, since both are set by the lender and already encode its own risk assessment.

---
**Author:** Indraneel Mannava · [LinkedIn](https://www.linkedin.com/in/indraneel-sarma-mannava/)
