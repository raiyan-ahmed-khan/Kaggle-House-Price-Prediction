# House Prices: Advanced Regression Techniques

Predicting Ames, Iowa residential sale prices using the [Kaggle House Prices](https://www.kaggle.com/competitions/home-data-for-ml-course) dataset. Final leaderboard score: **0.12979** (RMSE on log(SalePrice)).

## Overview

This project applies a full regression ML workflow — cleaning, feature engineering, encoding, and model tuning — to a ~80-column tabular dataset, with a focus on handling many missing-value patterns correctly and avoiding data leakage between train and test.

## Dataset

- `train.csv` / `test.csv` — Ames housing data (1460 train / 1459 test rows, 79 explanatory features)
- `data_description.txt` — full feature dictionary
- `sample_submission.csv` — expected submission format
- Target: `SalePrice`, scored via RMSE on `log(SalePrice)`

## Approach

1. **Target transform** — `log1p(SalePrice)` used as the training target, matching the competition's RMSE-on-log metric and correcting for right-skew.
2. **Missing values** — distinguished "NA means the feature doesn't exist" (e.g. no basement, no garage) from genuinely missing values, using the data dictionary rather than blanket imputation. Caught several one-off rows where a structural-NA column hid a real missing value (e.g. a house with a garage but one unknown garage field).
3. **Feature engineering**
   - `Total_sf` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF` (stronger single predictor than any of its parts)
   - Dropped `GarageArea`/`GarageYrBlt` after confirming high collinearity with `GarageCars`/`YearBuilt`
   - Rare categories in high-cardinality columns (`Neighborhood` and others) bucketed into `"Other"`, with thresholds chosen to avoid collapsing categories with distinct price signal
4. **Encoding**
   - Ordinal columns (quality/condition scales, basement exposure, etc.) mapped to explicit worst→best integer orders
   - Nominal columns one-hot encoded
   - All steps fit on train only, applied to test, via a `ColumnTransformer`/`Pipeline`
5. **Modeling** — compared Linear Regression, Ridge, Lasso, Random Forest, and XGBoost, tuned via `GridSearchCV` with 5-fold CV. **XGBoost** (`learning_rate=0.1, max_depth=3, n_estimators=300`) performed best.

## Results

| Model | CV RMSE (log scale) |
|---|---|
| XGBoost | 0.1268 |
| Lasso | 0.1407 |
| Random Forest | 0.1411 |
| Ridge | 0.1427 |
| Linear Regression | 0.1319 (baseline, untuned) |

Final Kaggle leaderboard score: **0.12979**

## Repo contents

- `house_prices_notebook.ipynb` — full workflow: EDA, cleaning, feature engineering, encoding pipeline, model tuning, submission generation
- `data_description.txt` — feature dictionary
- `submission.csv` — final predictions

## Notes

This project follows the same workflow structure as an earlier Titanic classification project, adapted for regression: a bigger feature set, RMSE instead of accuracy, and much more involved missing-value handling.
