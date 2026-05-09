# Predicting F1 Pit Stops

Starter solution for Kaggle Playground Series S6E5. It trains a LightGBM binary
classifier and writes a submission file with probabilities for `PitNextLap`.

## Files Used

- `/Users/saha/Downloads/train.csv`
- `/Users/saha/Downloads/test.csv`
- `/Users/saha/Downloads/sample_submission.csv`

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_lightgbm.py
```

The script writes:

```text
/Users/saha/Downloads/predicting-f1-pitstops/submission.csv
```

Upload that `submission.csv` file to Kaggle.

## Current Local Result

The checked run produced a 3-fold out-of-fold ROC-AUC of about `0.9499`.
