# Experiment 7: Dimensionality Reduction (PCA) and Comparative Model Evaluation with Statistical Significance Testing

## Objective
- Apply Principal Component Analysis (PCA) to reduce feature dimensionality while retaining 95% variance.
- Train and tune nine classifiers (SVM, Naive Bayes, KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost) plus a Stacked Ensemble, both with and without PCA.
- Compare performance, stability, and computational cost across all models in both settings.
- Statistically test whether PCA significantly changes each model's cross-validation F1 score using paired t-tests and Wilcoxon signed-rank tests.

## Dataset
Wisconsin Diagnostic Breast Cancer Dataset (`wdbc.data`) — 569 samples, 30 numerical features, binary target `Diagnosis` (M = Malignant, B = Benign).
- Class distribution: 357 Benign (62.7%), 212 Malignant (37.3%)
- No missing values, no duplicate rows

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
xgboost
```

## Methodology
1. Load data, drop ID column, encode target (M=0, B=1), check class balance/missing values/duplicates.
2. 80/20 stratified train-test split (455 train / 114 test), standardize features with `StandardScaler`.
3. Run full PCA to compute the explained-variance curve; select the minimum number of components needed to retain ≥95% variance.
4. Define 9 base models with hyperparameter grids; tune each via `GridSearchCV` (5-fold `StratifiedKFold`, scoring=F1) — once on the scaled features (No PCA) and once on the PCA-reduced features (With PCA).
5. Build a `StackingClassifier` (SVM + KNN + Random Forest base learners, Logistic Regression meta-learner) for both settings.
6. Evaluate all 10 models (9 tuned + Stacking) on the held-out test set: accuracy, precision, recall, F1.
7. Run 5-fold cross-validation for every model in both settings to get fold-wise F1 scores and standard deviation (stability).
8. Run paired t-tests and Wilcoxon signed-rank tests (per model, across the same 5 CV folds) to test whether the PCA vs No-PCA F1 difference is statistically significant (α = 0.05).
9. Visualize: scree plot, cumulative variance plot, CV comparison bar chart, accuracy/F1 comparison charts, confusion matrices, ROC curves, precision-recall curves, and a significance-annotated F1 comparison chart.

## Results

### PCA Dimensionality Reduction
| Metric | Value |
|---|---|
| Original features | 30 |
| Components needed for 95% variance | **10** |
| Actual variance retained | 95.27% |
| Feature reduction | 66.67% |

### Best Hyperparameters — No PCA (GridSearchCV, 5-fold, scoring=F1)
| Model | Best Parameters | CV F1 | Tuning Time |
|---|---|---|---|
| SVM | C=0.1, gamma=scale, kernel=linear | 0.9810 | 10.94s |
| Naive Bayes | var_smoothing=1e-11 | 0.9478 | 0.09s |
| KNN | metric=manhattan, n_neighbors=3, weights=uniform | 0.9774 | 0.65s |
| Logistic Regression | C=1, solver=liblinear | 0.9842 | 0.12s |
| Decision Tree | criterion=entropy, max_depth=3, min_samples_split=2 | 0.9464 | 0.86s |
| Random Forest | max_depth=5, min_samples_split=5, n_estimators=200 | 0.9702 | 13.91s |
| AdaBoost | learning_rate=1.0, n_estimators=100 | 0.9844 | 10.52s |
| Gradient Boosting | learning_rate=0.2, max_depth=3, n_estimators=200 | 0.9792 | 33.69s |
| XGBoost | learning_rate=0.2, max_depth=3, n_estimators=50 | 0.9772 | 20.41s |

### Best Hyperparameters — With PCA
| Model | Best Parameters | CV F1 | Tuning Time |
|---|---|---|---|
| SVM | C=10, gamma=0.01, kernel=rbf | 0.9845 | 2.98s |
| Naive Bayes | var_smoothing=1e-11 | 0.9347 | 0.09s |
| KNN | metric=euclidean, n_neighbors=5, weights=uniform | 0.9724 | 0.40s |
| Logistic Regression | C=1, solver=liblinear | 0.9861 | 0.12s |
| Decision Tree | criterion=entropy, max_depth=None, min_samples_split=10 | 0.9522 | 0.68s |
| Random Forest | max_depth=None, min_samples_split=2, n_estimators=200 | 0.9687 | 15.34s |
| AdaBoost | learning_rate=1.0, n_estimators=200 | 0.9622 | 13.14s |
| Gradient Boosting | learning_rate=0.2, max_depth=3, n_estimators=100 | 0.9720 | 25.81s |
| XGBoost | learning_rate=0.1, max_depth=7, n_estimators=200 | 0.9739 | 11.57s |

### Test Set Performance
| Model | No-PCA Acc | PCA Acc | No-PCA F1 | PCA F1 |
|---|---|---|---|---|
| SVM | **0.9825** | 0.9561 | **0.9861** | 0.9645 |
| Naive Bayes | 0.9298 | 0.9211 | 0.9444 | 0.9379 |
| KNN | 0.9649 | 0.9561 | 0.9726 | 0.9655 |
| Logistic Regression | **0.9825** | **0.9649** | **0.9861** | **0.9718** |
| Decision Tree | 0.9474 | 0.9298 | 0.9589 | 0.9444 |
| Random Forest | 0.9561 | 0.9386 | 0.9655 | 0.9510 |
| AdaBoost | 0.9561 | 0.9386 | 0.9660 | 0.9510 |
| Gradient Boosting | 0.9561 | 0.9474 | 0.9660 | 0.9589 |
| XGBoost | 0.9474 | 0.9474 | 0.9589 | 0.9583 |
| Stacking | 0.9737 | 0.9561 | 0.9790 | 0.9650 |

**Best No-PCA model: SVM** (F1 = 0.9861, tied with Logistic Regression)
**Best PCA model: Logistic Regression** (F1 = 0.9718)

### 5-Fold Cross-Validation — Average F1
| Model | Avg No-PCA F1 | Avg PCA F1 | No-PCA Std | PCA Std | More Stable Under PCA? |
|---|---|---|---|---|---|
| SVM | 0.9810 | 0.9845 | 0.0082 | 0.0098 | No |
| Naive Bayes | 0.9478 | 0.9347 | 0.0228 | 0.0149 | Yes |
| KNN | 0.9774 | 0.9724 | 0.0104 | 0.0126 | No |
| Logistic Regression | 0.9842 | 0.9861 | 0.0066 | 0.0042 | Yes |
| Decision Tree | 0.9464 | 0.9522 | 0.0177 | 0.0268 | No |
| Random Forest | 0.9702 | 0.9687 | 0.0145 | 0.0141 | Yes |
| AdaBoost | 0.9844 | 0.9622 | 0.0150 | 0.0112 | Yes |
| Gradient Boosting | 0.9792 | 0.9720 | 0.0118 | 0.0171 | No |
| XGBoost | 0.9772 | 0.9739 | 0.0143 | 0.0121 | Yes |
| Stacking | 0.9772 | 0.9774 | 0.0091 | 0.0089 | Yes |

### Statistical Significance: PCA vs No-PCA (paired t-test & Wilcoxon, α=0.05)
| Model | Mean Diff (PCA − No-PCA) | t-test p-value | Significant? | Wilcoxon p-value | Significant? |
|---|---|---|---|---|---|
| SVM | +0.0034 | 0.1778 | No | 0.5000 | No |
| Naive Bayes | −0.0131 | 0.1833 | No | 0.3125 | No |
| KNN | −0.0050 | 0.2074 | No | 0.5000 | No |
| Logistic Regression | +0.0019 | 0.6006 | No | 0.5000 | No |
| Decision Tree | +0.0058 | 0.6513 | No | 0.8125 | No |
| Random Forest | −0.0015 | 0.6503 | No | 0.7500 | No |
| **AdaBoost** | **−0.0222** | **0.0262** | **Yes** | 0.0625 | No |
| Gradient Boosting | −0.0071 | 0.4648 | No | 0.6250 | No |
| XGBoost | −0.0034 | 0.7895 | No | 0.7500 | No |
| Stacking | +0.0002 | 0.9570 | No | — | No |

Only **AdaBoost** showed a statistically significant drop in F1 with PCA under the parametric t-test (p=0.0262), though this was not confirmed by the more conservative non-parametric Wilcoxon test (p=0.0625) — for every other model, the observed PCA vs No-PCA difference is not distinguishable from fold-to-fold noise.

## How to Run
```bash
jupyter notebook exp7_new.ipynb
```
Requires `wdbc.data` in the same directory. Requires the `xgboost` package in addition to standard scikit-learn dependencies.

## Conclusion
PCA reduced dimensionality by 66.7% (30 → 10 features) while retaining 95.3% of the variance, but this did **not** translate into a meaningful, statistically significant performance change for almost every model — No-PCA and PCA versions performed within noise of each other for 9 of 10 models. SVM and Logistic Regression were the strongest performers overall (No-PCA F1 ≈ 0.986), and Logistic Regression was the only model that improved (marginally) under both settings. AdaBoost was the sole model to show a statistically significant degradation with PCA. This suggests that, for this dataset, the original 30 features are not highly redundant in a way that hurts the top-performing linear/margin-based models, and PCA's main practical benefit here is computational efficiency (most models tuned faster with 10 components) rather than accuracy gain.
