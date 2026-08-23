# ICS1512 — Machine Learning Algorithms Laboratory

**Student:** Preethi D
**Register Number:** 3122247001046
**Faculty:** Dr. Poreddy Ajay Kumar Reddy
**Academic Year:** 2026–2027 (Odd/Even)

## Overview
This repository contains all lab experiments for ICS1512 — Machine Learning Algorithms Laboratory, covering exploratory data analysis, classical ML algorithms, regularization, kernel methods, and ensemble learning.

## Setup
```bash
pip install -r requirements.txt
```

## Experiments

| # | Title | Dataset | Best Model | Key Result |
|---|---|---|---|---|
| 1 | [EDA & Preprocessing Pipeline](./Experiment_1_EDA/README.md) | Iris, Loan, Diabetes, Spam, MNIST | — | Task-type auto-detection across 5 datasets |
| 2 | [Naive Bayes vs KNN](./Experiment_2_NaiveBayes_KNN/README.md) | Spambase | KNN | 89.14% accuracy |
| 3 | [Linear/Ridge/Lasso/ElasticNet Regression](./Experiment_3_Regression/README.md) | Loan Amount | ElasticNet | R² = 0.587 |
| 4 | [Logistic Regression vs SVM](./Experiment_4_LogReg_SVM/README.md) | Spambase | Linear/RBF SVM | 92.94% accuracy |
| 5 | [Decision Tree vs Random Forest](./Experiment_5_DT_RF/README.md) | Breast Cancer (WDBC) | Random Forest | 95.61% accuracy, AUC 0.994 |
| 6 | [Bagging, Boosting & Stacking](./Experiment_6_Ensembles/README.md) | Breast Cancer (WDBC) | Stacked Ensemble | 97.37% accuracy |

## Repository Structure
```
ICS1512-ML-Lab/
├── README.md                          # this file
├── requirements.txt
├── Experiment_1_EDA/
│   ├── exp1.ipynb
│   └── README.md
├── Experiment_2_NaiveBayes_KNN/
│   ├── exp2.ipynb
│   └── README.md
├── Experiment_3_Regression/
│   ├── exp3.ipynb
│   └── README.md
├── Experiment_4_LogReg_SVM/
│   ├── exp4.ipynb
│   └── README.md
├── Experiment_5_DT_RF/
│   ├── exp5.ipynb
│   └── README.md
└── Experiment_6_Ensembles/
    ├── exp6.ipynb
    └── README.md
```

## Libraries Used
- **pandas** — data loading and manipulation
- **numpy** — numerical computing
- **matplotlib / seaborn** — visualization
- **scikit-learn** — models, preprocessing, metrics, hyperparameter tuning
