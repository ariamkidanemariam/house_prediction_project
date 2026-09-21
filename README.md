# House Prices: Data Cleaning & Categorical Encoding Pipeline

A step-by-step pipeline on the Kaggle "House Prices - Advanced Regression Techniques" dataset (`train.csv`, 1460 rows × 81 columns): clean the data, apply four different categorical encoding methods, then train a model on each to see which encoding actually performs best.

## Pipeline

| # | Notebook | What it does | Output |
|---|---|---|---|
| 1 | `01_data_cleaning.ipynb` | Fills "feature doesn't exist" NaNs with `"None"`; imputes genuinely missing values (`LotFrontage` by neighborhood median, `GarageYrBlt`/`MasVnrArea` with 0, `Electrical` with mode) | `train_cleaned.csv` |
| 2 | `02_label_encoding.ipynb` | Maps ordinal columns (e.g. `KitchenQual`) in the correct rank order; applies `LabelEncoder` to all categorical columns | `train_label_encoded.csv` |
| 3 | `03_onehot_encoding.ipynb` | One binary column per category (`pd.get_dummies`, `drop_first=True`); flags the cost for high-cardinality columns like `Neighborhood` | `train_onehot_encoded.csv` |
| 4 | `04_target_encoding.ipynb` | Replaces each category with its smoothed mean `SalePrice`; notes the target-leakage risk of this simple version | `train_target_encoded.csv` |
| 5 | `05_kfold_target_encoding.ipynb` | Leak-free version: encodes each fold using only the *other* folds' statistics | `train_kfold_target_encoded.csv` |
| 6 | `06_model_comparison.ipynb` | Trains a `RandomForestRegressor` on all four encoded datasets (same train/test split) and compares RMSE / R² | `encoding_comparison_results.csv`, `encoding_comparison.png` |

**Note on the ranking:** Label Encoding wins for this tree-based model, since Random Forests split on thresholds and aren't misled by an arbitrary integer ordering the way a linear model would be. Plain Target Encoding also edges out its own leak-free version (K-Fold), a small illustration of the leakage effect flagged in notebook 4, and a result that likely wouldn't hold on truly unseen data (e.g. Kaggle's real test set).

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
nbformat / jupyter (to run the notebooks)
```

## Running

Notebooks are numbered and meant to be run in order; each one loads the CSV produced by the previous step:

```bash
jupyter nbconvert --to notebook --execute --inplace 01_data_cleaning.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_label_encoding.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_onehot_encoding.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_target_encoding.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_kfold_target_encoding.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_model_comparison.ipynb
```

## Data source

[Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
