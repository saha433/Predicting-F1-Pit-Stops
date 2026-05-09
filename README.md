# Predicting F1 Pit Stops

Starter solution for Kaggle Playground Series S6E5. It trains upgraded
LightGBM/XGBoost models and writes a blended submission file with probabilities
for `PitNextLap`.

## Files Used

- `/Users/saha/Downloads/train.csv`
- `/Users/saha/Downloads/test.csv`
- `/Users/saha/Downloads/sample_submission.csv`

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_lightgbm.py
.venv/bin/python train_lightgbm_variant.py
.venv/bin/python train_xgboost.py
.venv/bin/python blend_predictions.py
```

The script writes:

```text
/Users/saha/Downloads/predicting-f1-pitstops/submission.csv
```

Upload that `submission.csv` file to Kaggle.

## Current Local Result

The best checked blend produced an out-of-fold ROC-AUC of about `0.9520`.

The improvement over the first baseline comes from:

- race/tyre/stint interaction features
- frequency encodings for useful categorical combinations
- fold-safe target encodings, so validation rows do not see their own labels
- 5-fold LightGBM instead of 3-fold LightGBM
- a second LightGBM shape and an XGBoost diversity model
- an OOF-optimized blend: about `0.30` LightGBM, `0.48` LightGBM variant, and
  `0.22` XGBoost
