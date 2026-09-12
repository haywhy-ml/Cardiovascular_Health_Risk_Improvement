# Cardiovascular Disease Risk Prediction

Binary classification project predicting cardiovascular disease (CVD) risk from lifestyle, demographic, and comorbidity data, using the `CVD_cleaned.csv` dataset (~308,854 rows, ~8% positive rate).

## Problem

Heart disease prediction on real-world survey data is a heavily imbalanced classification problem — roughly 92% of records are negative. The goal is not just to fit a model, but to handle that imbalance correctly and evaluate it in a way that reflects real deployment behavior, not just training-time metrics.

## Approach

1. **EDA** — target imbalance, demographics (age, sex), general health and checkup behavior, body metrics (BMI, height/weight), lifestyle factors (exercise, smoking, alcohol, diet), comorbidities (diabetes, arthritis, depression, skin cancer, other cancer), correlation and multicollinearity checks, and engineered features (`Comorbidity_Count`, `Age_Mid`, `Age_Bin`).
2. **Preprocessing** — ordinal and categorical encoding for lifestyle/health fields, one-hot encoding for `Diabetes`, label encoding for the target, and `StandardScaler` fit only on the training split (no leakage into the test set).
3. **Modeling** — Optuna hyperparameter search (50 trials) choosing between `LGBMClassifier` and `HistGradientBoostingClassifier`, with `SMOTE` applied *inside* an `imblearn` pipeline so oversampling only ever touches training folds, selected by 5-fold stratified cross-validation on average precision.
4. **Evaluation** — held-out test set scoring (average precision, ROC-AUC, confusion matrix, classification report), a precision-recall curve, and threshold tuning to hit a target recall for the positive class.
5. **Deployment artifact** — a single bundle containing the fitted pipeline, scaler, label encoder, feature column order, and the chosen decision threshold, so the model can be reused outside the notebook.

## Results

| Metric | CV (train) | Test set |
|---|---|---|
| Average Precision | 0.298 | 0.302 |
| ROC-AUC | — | 0.833 |

Best model: `LGBMClassifier` (via Optuna search).

At the default 0.5 threshold, recall on the positive class is low (~7%), which is expected given the ~8% base rate. Tuning the threshold to hit ~80% recall trades precision down to ~0.20 — a deliberate trade-off for a screening use case, where missing true positives is costlier than raising false alarms.

## Improvements over the initial version

The first version of this notebook stopped after fitting and saving the best pipeline, with no evaluation on unseen data. This version adds:

- **Held-out test-set evaluation** — confusion matrix, classification report, ROC-AUC, and a precision-recall curve, instead of relying on CV scores alone
- **Threshold tuning** — an explicit precision/recall trade-off analysis instead of assuming the default 0.5 cutoff is appropriate for an imbalanced problem
- **A complete deployment bundle** — the previous version only saved the raw pipeline; this version bundles the pipeline together with the `StandardScaler`, `LabelEncoder`, feature column order, and chosen threshold, so predictions on new data can be reproduced correctly outside the notebook

## Repo structure

```
.
├── cardiovascular_disease_risk_prediction.ipynb
├── requirements.txt
├── bundle
├── render.yaml
├── webapp
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```
