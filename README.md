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
2. **Preprocessing** — dropped identifier columns (names, addresses, card numbers), engineered new features (`age`, `hour`), encoded categorical variables
3. **Model Training** — trained an XGBoost classifier on the cleaned dataset
4. **Evaluation** — assessed performance using precision, recall, and F1-score (not accuracy, due to class imbalance)

## Key Findings

- Fraudulent transactions average **~8x higher** in amount than legitimate ones
- Fraud is heavily concentrated between **10 PM and 3 AM**
- Online shopping categories show higher fraud rates than in-person categories
- Customer-to-merchant distance showed no meaningful correlation with fraud

## Results

| Metric | Fraud Class (1) |
|---|---|
| Precision | 0.91 |
| Recall | 0.76 |
| F1-score | 0.83 |

The model correctly flags fraud with high precision (few false alarms) while catching roughly 76% of actual fraud cases.

## Tools

Python, pandas, XGBoost, scikit-learn, Jupyter

## Project Structure

```
fraud-detection/
├── data/              # (not included, see Dataset section)
├── notebooks/
│   └── 01_exploration.ipynb
└── README.md
```