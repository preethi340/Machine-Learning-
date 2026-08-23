# Experiment 6: Bagging, Boosting, and Stacked Ensemble Models

## Objective
Understand and implement Bagging, Boosting (AdaBoost, Gradient Boosting), and a Stacked Ensemble to compare ensemble strategies in terms of accuracy, stability, and generalization on a breast cancer classification task.

## Dataset
Wisconsin Diagnostic Breast Cancer Dataset (`wdbc.data`) — 569 samples, 30 numerical features, binary target `diagnosis` (M = Malignant, B = Benign).

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Methodology
1. Load and preprocess data (no missing values), 80/20 stratified train-test split, feature scaling.
2. **Bagging** — `BaggingClassifier` with Decision Tree base estimator, tuned over `n_estimators`, `max_samples`, `max_features`.
3. **Boosting** — `AdaBoostClassifier` (stump base estimator) and `GradientBoostingClassifier`, tuned over `n_estimators`, `learning_rate`, `max_depth`.
4. **Stacking** — `StackingClassifier` combining SVM (RBF), Gaussian Naive Bayes, and Decision Tree as base learners, with Logistic Regression as the meta-learner.
5. Evaluate all four ensemble strategies with 5-fold CV and test-set metrics (accuracy, precision, recall, F1, ROC-AUC).

## Results

### Best Hyperparameters (GridSearchCV, 5-fold)
| Model | Best Parameters | CV Accuracy |
|---|---|---|
| Bagging | `n_estimators: 20, max_samples: 0.5, max_features: 0.5` | 0.9582 |
| AdaBoost | `n_estimators: 100, learning_rate: 0.5` | 0.9780 |
| Gradient Boosting | `n_estimators: 100, learning_rate: 0.2, max_depth: 1` | 0.9802 |
| Stacking | SVM + Naive Bayes + Decision Tree → Logistic Regression | 0.9626 |

### Bagging: Accuracy vs. n_estimators (max_samples=0.8)
| n_estimators | 3 | 5 | 10 | 20 |
|---|---|---|---|---|
| CV Accuracy | 93.63% | **95.16%** | 94.95% | 94.51% |

### Gradient Boosting: Accuracy vs. n_estimators (lr=0.1, depth=2)
| n_estimators | 50 | 100 | 150 |
|---|---|---|---|
| CV Accuracy | 95.16% | 96.48% | **96.92%** |

### Final Test Set Comparison
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Bagging | 92.98% | 95.71% | 93.06% | 94.37% | 0.990 |
| AdaBoost | 95.61% | 94.67% | 98.61% | 96.60% | 0.984 |
| Gradient Boosting | 96.49% | 95.95% | 98.61% | 97.26% | 0.995 |
| **Stacked Ensemble** | **97.37%** | **97.26%** | 98.61% | **97.93%** | 0.994 |

## How to Run
```bash
jupyter notebook exp6.ipynb
```
Requires `wdbc.data` in the same directory.

## Conclusion
The Stacked Ensemble achieved the best overall performance (97.37% accuracy, F1 = 0.979), benefiting from combining three structurally different base learners (margin-based, probabilistic, and rule-based) via a Logistic Regression meta-learner. Among individual ensemble strategies, Gradient Boosting (bias-reducing, sequential) outperformed Bagging (variance-reducing, parallel), suggesting the base Decision Tree wasn't highly variant to begin with, leaving more room for bias-correction methods to add value.
