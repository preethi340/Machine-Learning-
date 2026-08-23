# Experiment 1: EDA & Preprocessing Pipeline with NumPy, Pandas, SciPy, Scikit-learn, Matplotlib

## Objective
Build a reusable EDA + preprocessing pipeline and apply it across five public datasets to identify the correct machine learning task type (classification vs. regression) for each.

## Dataset
| Dataset | File | Samples | Features |
|---|---|---|---|
| Iris | `Iris.csv` | 150 | 6 |
| Loan Amount Prediction | `loan.csv` | 614 | 13 |
| Diabetes | `diabetes.csv` | 768 | 9 |
| Email Spam | `emails.csv` | 5172 | 3002 |
| MNIST (2000-row sample) | `mnist.csv` | 2000 | 785 |

## Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Methodology
1. **`eda_process()`** — loads each CSV, prints shape/dtypes/summary statistics, checks missing values and duplicates, and generates histograms, a correlation heatmap, and (for ≤6 numeric columns) a pairplot.
2. **`detect_task_type()`** — automatically classifies the target column as Classification or Regression: object/category dtype → Classification; numeric with ≤20 unique values → Classification; otherwise → Regression.
3. **`preprocessing_process()`** — handles missing values (median imputation for numeric columns, with domain-specific zero→NaN conversion for the Diabetes dataset), clips outliers using the IQR method, and standardizes features with `StandardScaler`.

## Results
| Dataset | Target | Task Detected | Rows Removed | Missing Values Handled |
|---|---|---|---|---|
| Iris | Species | Classification | 0 | None |
| Loan | LoanAmount | **Regression** | 79 | Median imputation |
| Diabetes | Outcome | Classification | 0 | Zero→NaN→median (Glucose, BP, SkinThickness, Insulin, BMI) |
| Email Spam | Prediction | Classification | 0 | None |
| MNIST (sample) | label | Classification | 0 | None |

No dataset contained duplicate rows.

## How to Run
```bash
jupyter notebook exp1.ipynb
```
Ensure all five CSVs are in the same directory as the notebook before running.

## Conclusion
Four of the five datasets are classification problems (categorical or low-cardinality numeric targets); Loan Amount Prediction is the only regression task, since `LoanAmount` is a continuous value with 203 unique entries. The pipeline generalizes across differently structured datasets (from 6 to 3002 columns) using the same preprocessing logic.
