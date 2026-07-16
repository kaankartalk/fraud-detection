# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using XGBoost, built as a hands-on introduction to fraud detection and AML-adjacent ML techniques.

## Overview

This project explores a simulated credit card transaction dataset (2019-2020) to identify patterns that distinguish fraudulent transactions from legitimate ones, then trains a classification model to detect fraud in unseen data.

## Dataset

- Source: [Kaggle - Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- ~1.85 million transactions (1.3M train, 555K test)
- Highly imbalanced: only ~0.58% of transactions are fraudulent

## Approach

1. **Exploratory Data Analysis** — investigated fraud patterns across transaction amount, category, time of day, and customer age
2. **Preprocessing** — dropped identifier columns (names, addresses, card numbers), engineered new features (`age`, `hour`, haversine `distance_km` between customer and merchant), encoded categorical variables
3. **Model Training** — trained a baseline XGBoost classifier, tuned it with grid search and `scale_pos_weight` class-imbalance correction, and trained a Random Forest classifier for comparison
4. **Evaluation** — assessed all models using precision, recall, and F1-score on the fraud class (not accuracy, due to class imbalance), and inspected feature importances to understand each model's decisions

## Key Findings

- Fraudulent transactions average **~8x higher** in amount than legitimate ones — the single strongest predictor across every model tested
- Fraud is heavily concentrated between **10 PM and 3 AM**
- Online shopping categories show higher fraud rates than in-person categories
- Customer-to-merchant distance (engineered via the haversine formula) adds a modest but real signal — around 4% feature importance, ranking 7th of 22 features in the Random Forest model

## Results

| Model | Precision | Recall | F1 | False Positives | False Negatives |
|---|---|---|---|---|---|
| XGBoost (baseline) | 0.91 | 0.76 | 0.83 | 164 | 517 |
| Random Forest + distance | 0.96 | 0.74 | 0.83 | 64 | 568 |
| XGBoost (`scale_pos_weight=5`, tuned) | 0.81 | 0.79 | 0.80 | 396 | 447 |
| XGBoost (`scale_pos_weight=172`, first attempt) | 0.48 | 0.88 | 0.62 | 2233 | 261 |

*(Fraud class metrics; false positive/negative counts are out of 555,719 test transactions, 2,145 of which are fraud.)*

Random Forest with the engineered distance feature turned out to be the strongest model: it matched the baseline XGBoost's F1 score while cutting false alarms by more than half, without any class-imbalance correction. Aggressively reweighting XGBoost (`scale_pos_weight=172` to match the true class ratio) pushed recall to 0.88 but collapsed precision to 0.48. A lighter weight (`scale_pos_weight=5`) struck a middle ground — 121 more caught frauds than Random Forest, at the cost of 332 more false alarms — which is a reasonable choice if the business would rather over-flag than miss fraud.

The practical takeaway: reweighting isn't the only way to handle class imbalance. Averaging across many trees (bagging, as Random Forest does) produced a naturally more robust result here than forcing a single model to compensate through aggressive weighting.

## Tools

Python, pandas, XGBoost, scikit-learn, matplotlib, Jupyter

## Project Structure

```
fraud-detection/
├── data/              # (not included, see Dataset section)
├── notebooks/
│   └── 01_exploration.ipynb
└── README.md
```