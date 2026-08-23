# Experiment 5: Decision Tree and Random Forest — A Comparative Classification Study

## Objective
Implement a Decision Tree classifier, extend it into a Random Forest ensemble, tune hyperparameters via 5-fold cross-validation, and compare single-tree vs. ensemble-tree performance.

## Dataset
Wisconsin Diagnostic Breast Cancer Dataset (loaded via `sklearn.datasets.load_breast_cancer`) — 569 samples, 30 numerical features, binary target (malignant/benign).

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Methodology
1. Load dataset, check for missing values (none) and class distribution (212 malignant, 357 benign).
2. 80/20 stratified train-test split.
3. Train baseline Decision Tree and Random Forest.
4. Tune both via `GridSearchCV` (5-fold): Decision Tree over `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`; Random Forest over `n_estimators`, `max_depth`, `max_features`, `bootstrap`.
5. Compare both models via 5-fold cross-validation, confusion matrices, and ROC-AUC.

## Results

### Baseline (untuned)
| Model | Accuracy |
|---|---|
| Decision Tree | 91.23% |
| Random Forest | 95.61% |

### Tuned Hyperparameters (GridSearchCV)
| Model | Best Parameters | Best CV Accuracy |
|---|---|---|
| Decision Tree | `criterion: gini, max_depth: 5, min_samples_leaf: 1, min_samples_split: 5` | 0.9385 |
| Random Forest | `bootstrap: False, max_depth: None, max_features: log2, n_estimators: 100` | **0.9692** |

### 5-Fold Cross-Validation (tuned models)
| Model | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Average |
|---|---|---|---|---|---|---|
| Decision Tree | 94.51% | 92.31% | 91.21% | 92.31% | 89.01% | 91.87% |
| **Random Forest** | 95.60% | 98.90% | 93.41% | 94.51% | 97.80% | **96.04%** |

### Test Set Performance
| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Decision Tree | 92.11% | 95.65% | 91.67% | 93.62% | 0.916 |
| **Random Forest** | **95.61%** | **95.89%** | **97.22%** | **96.55%** | **0.994** |

## How to Run
```bash
jupyter notebook exp5.ipynb
```
No external file needed — dataset loads directly via scikit-learn.

## Conclusion
Random Forest outperformed the single Decision Tree on every metric, with a notably higher and more stable cross-validation accuracy (96.04% vs 91.87%) and near-perfect AUC (0.994 vs 0.916). Bagging and random feature selection across 100 trees reduced the variance that made the single Decision Tree sensitive to specific data splits (e.g., its accuracy dropped to 89% on one fold).
