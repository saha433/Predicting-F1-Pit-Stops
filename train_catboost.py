from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = Path("/Users/saha/Downloads/train.csv")
TEST_PATH = Path("/Users/saha/Downloads/test.csv")
SUBMISSION_PATH = ROOT / "submission_catboost.csv"

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = [
    "Driver",
    "Compound",
    "Race",
    "Year",
    "PitStop",
    "Stint",
    "Driver_Compound",
    "Race_Compound",
    "Race_Year",
    "Compound_Stint",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    race_progress = df["RaceProgress"].clip(lower=1e-6)
    est_race_laps = df["LapNumber"] / race_progress
    df["EstimatedRaceLaps"] = est_race_laps
    df["LapsRemaining"] = est_race_laps - df["LapNumber"]
    df["TyreLifePctRace"] = df["TyreLife"] / est_race_laps
    df["TyreLifePctRemaining"] = df["TyreLife"] / df["LapsRemaining"].clip(lower=1.0)
    df["StintProgress"] = df["TyreLife"] / df["LapNumber"].clip(lower=1.0)
    df["PitStopsPerStint"] = df["PitStop"] / df["Stint"].clip(lower=1)
    df["DegradationPerTyreLap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1.0)
    df["LapDeltaAbs"] = df["LapTime_Delta"].abs()
    df["PositionChangeAbs"] = df["Position_Change"].abs()
    df["LateRace"] = (df["RaceProgress"] > 0.75).astype("int8")
    df["EarlyRace"] = (df["RaceProgress"] < 0.25).astype("int8")
    df["OldTyre"] = (df["TyreLife"] >= 25).astype("int8")
    df["TopTen"] = (df["Position"] <= 10).astype("int8")
    df["Driver_Compound"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
    df["Race_Compound"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["Compound_Stint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)
    for col in CAT_COLS:
        df[col] = df[col].astype(str)
    return df


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = train[TARGET].astype(int)
    test_ids = test[ID_COL]

    combined = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    combined = add_features(combined)
    X = combined.iloc[: len(train)].drop(columns=[ID_COL])
    X_test = combined.iloc[len(train) :].drop(columns=[ID_COL])
    cat_features = [X.columns.get_loc(c) for c in CAT_COLS]

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y), start=1):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=2500,
            learning_rate=0.035,
            depth=8,
            l2_leaf_reg=5.0,
            random_seed=2026 + fold,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            od_type="Iter",
            od_wait=120,
            allow_writing_files=False,
            verbose=200,
            thread_count=-1,
        )
        train_pool = Pool(X.iloc[trn_idx], y.iloc[trn_idx], cat_features=cat_features)
        val_pool = Pool(X.iloc[val_idx], y.iloc[val_idx], cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        oof[val_idx] = model.predict_proba(val_pool)[:, 1]
        test_pred += model.predict_proba(test_pool)[:, 1] / folds.n_splits
        print(f"Fold {fold}: AUC={roc_auc_score(y.iloc[val_idx], oof[val_idx]):.6f}")

    print(f"OOF AUC: {roc_auc_score(y, oof):.6f}")
    pd.DataFrame({ID_COL: test_ids, TARGET: np.clip(test_pred, 1e-6, 1 - 1e-6)}).to_csv(
        SUBMISSION_PATH,
        index=False,
    )
    np.save(ROOT / "catboost_oof.npy", oof)
    np.save(ROOT / "catboost_test.npy", test_pred)
    print(f"Wrote {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
