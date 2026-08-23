# Experiment 4: Binary Classification using Logistic Regression and SVM

## Objective
Classify emails as spam or ham using Logistic Regression and Support Vector Machines (with multiple kernels), and analyze the effect of hyperparameter tuning on classification performance.

## Dataset
[Spambase Dataset (Kaggle)](https://www.kaggle.com/datasets/somesh24/spambase) — `spambase_csv.csv`, 4601 samples, 57 numerical features, binary target `class`.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Methodology
1. Preprocess (no missing values found), standardize features, 80/20 stratified train-test split.
2. Train and evaluate baseline Logistic Regression.
3. Tune Logistic Regression via `GridSearchCV` over `penalty` (L1/L2), `C`, `solver`.
4. Train SVM with Linear, Polynomial, RBF, and Sigmoid kernels.
5. Tune SVM via `GridSearchCV` over `kernel`, `C`, `gamma`, `degree`.
6. Run 5-fold cross-validation and compare all models.

## Results

### Baseline Logistic Regression
Accuracy: **92.94%** | Precision: 0.92 | Recall: 0.90 | F1: 0.909 | Training time: 0.173s

### Tuned Logistic Regression (GridSearchCV)
Best params: `{C: 100, penalty: 'l1', solver: 'liblinear'}` → CV accuracy **0.9239**, test accuracy 92.51%

### SVM Kernel Comparison (baseline)
| Kernel | Accuracy | F1 | Training Time |
|---|---|---|---|
| Linear | 92.94% | 0.909 | 0.462s |
| Polynomial | 77.96% | 0.622 | 0.460s |
| RBF | 92.73% | 0.906 | 0.286s |
| Sigmoid | 88.49% | 0.853 | 0.293s |

### Tuned SVM (GridSearchCV)
Best params: `{C: 10, degree: 2, gamma: 'scale', kernel: 'rbf'}` → CV accuracy **0.9332**

### 5-Fold Cross-Validation
| Model | Average Accuracy |
|---|---|
| Logistic Regression | 0.9239 |
| SVM | **0.9332** |

### Final Comparison (test set)
| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression (tuned) | 92.07% | 0.898 |
| Linear SVM | **92.94%** | **0.909** |
| Polynomial SVM | 77.96% | 0.622 |
| RBF SVM | 92.73% | 0.906 |
| Sigmoid SVM | 88.49% | 0.853 |

## How to Run
```bash
jupyter notebook exp4.ipynb
```
Requires `spambase_csv.csv` in the same directory.

## Conclusion
Linear SVM and tuned RBF SVM were the top performers, with tuned SVM edging out Logistic Regression on 5-fold CV accuracy (0.9332 vs 0.9239). Polynomial SVM performed noticeably worse across all metrics, likely due to a mismatch between its default kernel shape and the dataset's actual feature structure. SVM generally required more tuning effort but produced the most consistent results across kernels except polynomial.
