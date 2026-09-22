# ICS1512 – Experiment 8: Clustering HAR Data (K-Means, DBSCAN, Hierarchical)

**Notebook:** `Exp8.ipynb`
**Report:** `Experiment8_HAR_Clustering.pdf`

## Dataset
- Name: UCI Human Activity Recognition Using Smartphones
- Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
  (mirror: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
- After downloading, unzip so the notebook's working directory contains:
  ```
  UCI HAR Dataset/
    train/X_train.txt, y_train.txt
    test/X_test.txt, y_test.txt
    features.txt
    activity_labels.txt
  ```
- The notebook reads these paths directly via `base='UCI HAR Dataset/'`.

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
jupyter notebook Exp8.ipynb
```
Run all cells top to bottom. Figures are auto-saved to a `Graphs/` folder as well as shown inline.

## Notes
- `Experiment8_HAR_Clustering.pdf` was generated directly from this notebook's actual outputs —
  all tables, metrics, and figures reflect a real run, not placeholders.
- Still to fill in manually in the report PDF (regenerate the `.tex` if you want these baked in):
  Python version, scikit-learn version, hardware specs, and your GitHub repository link
  (currently a placeholder footnote on the title page).
