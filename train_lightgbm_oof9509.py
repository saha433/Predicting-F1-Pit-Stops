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
TARGET_ENCODING_COLS = [
    "Driver",
    "Compound",
    "Race",
    "Year",
    "Stint",
    "Driver_Compound",
    "Race_Compound",
    "Race_Year",
    "Compound_Stint",
    "Race_Year_Compound",
]
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
    df["EarlyRace"] = (df["RaceProgress"] < 0.25).astype("int8")
    df["OldTyre"] = (df["TyreLife"] >= 25).astype("int8")
    df["VeryOldTyre"] = (df["TyreLife"] >= 40).astype("int8")
    df["TopTen"] = (df["Position"] <= 10).astype("int8")
    df["LapTimePerRaceLap"] = df["LapTime (s)"] / est_race_laps
    df["LapTimeXProgress"] = df["LapTime (s)"] * df["RaceProgress"]
    df["TyreLifeXStint"] = df["TyreLife"] * df["Stint"]
    df["TyreLifeXCompoundHard"] = df["TyreLife"] * (df["Compound"] == "HARD")
    df["TyreLifeXCompoundMedium"] = df["TyreLife"] * (df["Compound"] == "MEDIUM")

    df["Driver_Compound"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
    df["Race_Compound"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["Compound_Stint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)
    df["Race_Year_Compound"] = df["Race_Year"] + "_" + df["Compound"].astype(str)

    for col in TARGET_ENCODING_COLS:
        counts = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(counts).astype("float32") / len(df)

    group_cols = ["Compound", "Race", "Race_Compound"]
    stat_cols = ["TyreLife", "LapNumber", "RaceProgress"]
    for group_col in group_cols:
        grouped = df.groupby(group_col, observed=True)
        for stat_col in stat_cols:
            mean_col = f"{group_col}_{stat_col}_mean"
            df[mean_col] = grouped[stat_col].transform("mean").astype("float32")
            df[f"{stat_col}_minus_{group_col}_mean"] = (df[stat_col] - df[mean_col]).astype("float32")

    for col in TARGET_ENCODING_COLS:
        df[col] = pd.factorize(df[col], sort=True)[0].astype("int16")

    return df


def add_target_encoding(
    X_base: pd.DataFrame,
    X_test_base: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    y: pd.Series,
    trn_idx: np.ndarray,
    val_idx: np.ndarray,
    cols: list[str],
    inner_splits: int = 3,
    smoothing: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_trn = X_base.iloc[trn_idx].copy()
    X_val = X_base.iloc[val_idx].copy()
    X_test = X_test_base.copy()
    global_mean = float(y.iloc[trn_idx].mean())
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)

    raw_trn = train_raw.iloc[trn_idx].reset_index(drop=True)
    y_trn = y.iloc[trn_idx].reset_index(drop=True)
    raw_val = train_raw.iloc[val_idx]

    for col in cols:
        encoded_trn = np.full(len(raw_trn), global_mean, dtype="float32")
        for enc_idx, hold_idx in inner.split(raw_trn, y_trn):
            stats = (
                pd.DataFrame({col: raw_trn.iloc[enc_idx][col].values, TARGET: y_trn.iloc[enc_idx].values})
                .groupby(col)[TARGET]
                .agg(["mean", "count"])
            )
            smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
            encoded_trn[hold_idx] = raw_trn.iloc[hold_idx][col].map(smooth).fillna(global_mean).astype("float32")

        stats = (
            pd.DataFrame({col: raw_trn[col].values, TARGET: y_trn.values})
            .groupby(col)[TARGET]
            .agg(["mean", "count"])
        )
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)

        X_trn[f"{col}_te"] = encoded_trn
        X_val[f"{col}_te"] = raw_val[col].map(smooth).fillna(global_mean).astype("float32").values
        X_test[f"{col}_te"] = test_raw[col].map(smooth).fillna(global_mean).astype("float32").values

    return X_trn, X_val, X_test


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
    train_te_source = combined.iloc[: len(train)]
    test_te_source = combined.iloc[len(train) :]

    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.045,
        "num_leaves": 56,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 1.25,
        "n_estimators": 1800,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), start=1):
        X_trn, X_val, X_test_fold = add_target_encoding(
            X,
            X_test,
            train_te_source,
            test_te_source,
            y,
            trn_idx,
            val_idx,
            TARGET_ENCODING_COLS,
        )

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_trn,
            y.iloc[trn_idx],
            eval_set=[(X_val, y.iloc[val_idx])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(120, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_test_fold)[:, 1] / folds.n_splits
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
