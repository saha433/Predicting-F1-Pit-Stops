from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = Path("/Users/saha/Downloads/train.csv")
TEST_PATH = Path("/Users/saha/Downloads/test.csv")
SUBMISSION_PATH = ROOT / "submission.csv"

TARGET = "PitNextLap"
ID_COL = "id"
CATEGORICAL_COLS = ["Driver", "Compound", "Race"]
RANDOM_STATE = 42
N_SPLITS = 3


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    race_progress = df["RaceProgress"].clip(lower=1e-6)
    est_race_laps = df["LapNumber"] / race_progress
    df["EstimatedRaceLaps"] = est_race_laps
    df["LapsRemaining"] = est_race_laps - df["LapNumber"]
    df["TyreLifePctRace"] = df["TyreLife"] / est_race_laps
    df["TyreLifePctRemaining"] = df["TyreLife"] / (df["LapsRemaining"].clip(lower=1.0))
    df["StintProgress"] = df["TyreLife"] / (df["LapNumber"].clip(lower=1.0))
    df["PitStopsPerStint"] = df["PitStop"] / (df["Stint"].clip(lower=1))
    df["DegradationPerTyreLap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1.0)
    df["LapDeltaAbs"] = df["LapTime_Delta"].abs()
    df["PositionChangeAbs"] = df["Position_Change"].abs()
    df["IsFirstStint"] = (df["Stint"] == 1).astype("int8")
    df["LateRace"] = (df["RaceProgress"] > 0.75).astype("int8")

    for col in CATEGORICAL_COLS:
        df[col] = pd.factorize(df[col], sort=True)[0].astype("int16")

    return df


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    y = train[TARGET].astype(int)
    train_ids = train[ID_COL]
    test_ids = test[ID_COL]

    combined = pd.concat(
        [train.drop(columns=[TARGET]), test],
        axis=0,
        ignore_index=True,
    )
    combined = add_features(combined)

    X = combined.iloc[: len(train)].drop(columns=[ID_COL])
    X_test = combined.iloc[len(train) :].drop(columns=[ID_COL])

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 48,
        "max_depth": -1,
        "min_child_samples": 120,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "n_estimators": 1500,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), start=1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[trn_idx],
            y.iloc[trn_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / folds.n_splits
        fold_auc = roc_auc_score(y.iloc[val_idx], oof[val_idx])
        print(f"Fold {fold}: AUC={fold_auc:.6f}, best_iteration={model.best_iteration_}")

    overall_auc = roc_auc_score(y, oof)
    print(f"OOF AUC: {overall_auc:.6f}")
    print(f"Train ids loaded: {train_ids.min()}..{train_ids.max()}")

    submission = pd.DataFrame(
        {
            ID_COL: test_ids,
            TARGET: np.clip(test_pred, 1e-6, 1 - 1e-6),
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Wrote {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
