# ICS1512 – Experiment 9: Perceptron vs. MLP on Handwritten Characters

**Notebook:** `Exp9.ipynb`
**Report:** `Experiment9_PLA_vs_MLP.pdf`

## Dataset
- Name: English Handwritten Characters Dataset (Chars74K hand-drawn subset)
- Download: https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset
- After downloading, place `english.csv` and the `Img/` folder in the notebook's working
  directory (the CSV's `image` column paths point into `Img/`).

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy pillow
jupyter notebook Exp9.ipynb
```
Run all cells top to bottom. Training the MLP grid search takes a few minutes on CPU.

## Notes
- `Experiment9_PLA_vs_MLP.pdf` was generated directly from this notebook's actual outputs —
  all tables, metrics, and figures reflect a real run, not placeholders.
- Still to fill in manually in the report PDF (regenerate the `.tex` if you want these baked in):
  Python version, scikit-learn version, hardware specs, and your GitHub repository link
  (currently a placeholder footnote on the title page).
