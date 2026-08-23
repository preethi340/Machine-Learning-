# Experiment 2: Email Spam/Ham Classification using Naive Bayes and KNN

## Objective
- Build spam classifiers using Naive Bayes (Gaussian, Multinomial, Bernoulli) and KNN.
- Tune KNN with GridSearchCV and RandomizedSearchCV.
- Compare KDTree vs BallTree, and evaluate via 5-fold cross-validation.

## Dataset
[Spambase Dataset (Kaggle)](https://www.kaggle.com/datasets/somesh24/spambase) — 4601 samples, 57 features (word/character frequencies), binary target `class` (1 = spam, 0 = ham).

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Methodology
1. Load and explore data (shape, dtypes, missing values — none found).
2. Train baseline GaussianNB, MultinomialNB, BernoulliNB, and KNN classifiers.
3. Tune KNN via `GridSearchCV` (exhaustive) and `RandomizedSearchCV` (20 sampled combinations) over `n_neighbors`, `metric`, `weights`, `algorithm`.
4. Compare KDTree vs BallTree search algorithms.
5. Run 5-fold cross-validation and measure training/prediction time per model.
6. Plot confusion matrices, ROC curves, precision-recall curves, accuracy-vs-k, and search result heatmaps.

## Results

### Baseline Model Comparison
| Model | Accuracy | Precision | Recall | F1 | Train Time (s) | Predict Time (s) |
|---|---|---|---|---|---|---|
| GaussianNB | 85.78% | 0.79 | 0.87 | 0.83 | 0.011–0.027 | 0.002–0.006 |
| MultinomialNB | 85.78% | 0.84 | 0.79 | 0.81 | 0.007–0.009 | 0.001 |
| BernoulliNB | 84.69% | 0.83 | 0.76 | 0.80 | 0.019–0.024 | 0.002–0.006 |
| **KNN** | **89.14%** | 0.88 | 0.84 | 0.86 | 0.006–0.011 | 0.063–0.070 |

### Hyperparameter Tuning (KNN)
| Method | Best k | Metric | Weights | CV Accuracy | Time |
|---|---|---|---|---|---|
| GridSearchCV | 11 | manhattan | distance | 0.9141 | 105.77s |
| RandomizedSearchCV | 9 | manhattan | distance | 0.9133 | 18.03s |

### KDTree vs BallTree (k=5)
| Algorithm | Accuracy | Training Time | Prediction Time |
|---|---|---|---|
| KDTree | 89.14% | 0.043s | 0.126s |
| BallTree | 89.14% | 0.014s | 0.220s |

### 5-Fold Cross-Validation (KNN, k=5)
Folds: 0.886, 0.890, 0.912, 0.916, 0.807 → **Average: 0.8822**

## How to Run
```bash
jupyter notebook exp2.ipynb
```
Requires `spambase.csv` in the same directory.

## Conclusion
KNN outperformed all Naive Bayes variants on accuracy (89.14%), at the cost of much slower prediction time due to its lazy-learning nature. RandomizedSearchCV found a near-optimal hyperparameter set almost 6x faster than GridSearchCV with negligible accuracy loss (0.9133 vs 0.9141).
