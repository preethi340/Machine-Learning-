# Experiment 3: Regression Analysis using Linear and Regularized Models

## Objective
Implement Linear, Ridge, Lasso, and Elastic Net regression to predict loan sanction amount, tune regularization hyperparameters, and analyze overfitting/underfitting and bias-variance trade-offs.

## Dataset
[Loan Amount Prediction (Kaggle)](https://www.kaggle.com/datasets) — `train.csv`, 30,000 rows, 24 columns (numerical and categorical loan-application features). Target: `Loan Sanction Amount (USD)`.

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Methodology
1. Impute missing values (median for numeric, most-frequent for categorical), label-encode categorical columns, standardize features.
2. Split data 80/20 into train/test.
3. Train baseline Linear Regression.
4. Tune Ridge, Lasso, and Elastic Net via `GridSearchCV` (5-fold, scoring=R²) over their respective alpha (and l1_ratio for Elastic Net) grids.
5. Evaluate all models with MAE, MSE, RMSE, R², and training time.
6. Visualize target distribution, predicted-vs-actual, residuals, learning curves, and coefficient comparison across models.

## Results

### Best Hyperparameters (GridSearchCV, 5-fold)
| Model | Best Parameters | Best CV R² |
|---|---|---|
| Ridge | α = 0.01 | 0.604016 |
| Lasso | α = 0.001 | 0.604018 |
| Elastic Net | α = 0.01, l1_ratio = 0.8 | 0.604775 |

### Test Set Performance
| Model | MAE | RMSE | R² | Training Time |
|---|---|---|---|---|
| Linear | 20785.52 | 31009.45 | 0.5820 | 0.14s |
| Ridge | 20785.49 | 31009.22 | 0.5820 | 0.33s |
| Lasso | 20785.51 | 31009.44 | 0.5820 | 12.47s |
| **Elastic Net** | 20790.81 | **30822.31** | **0.5871** | 2.50s |

## How to Run
```bash
jupyter notebook exp3.ipynb
```
Requires `train.csv` in the same directory.

## Conclusion
All four models converged to nearly identical performance (R² ≈ 0.58–0.59) because the optimal regularization strength found for Ridge/Lasso/Elastic Net was very close to zero — indicating minimal overfitting was present to begin with, given the large sample size (30,000 rows) relative to feature count. Elastic Net marginally outperformed the others on both MAE/RMSE and R², while Lasso was by far the slowest to train due to its iterative coordinate-descent solver. Consistent R² between cross-validation and test sets confirms the models generalize well without overfitting.
