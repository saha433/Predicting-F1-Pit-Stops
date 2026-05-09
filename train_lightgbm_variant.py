from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_lightgbm import (
    ID_COL,
    TARGET,
    TARGET_ENCODING_COLS,
    TEST_PATH,
    TRAIN_PATH,
    add_features,
    add_target_encoding,
)


ROOT = Path(__file__).resolve().parent
SUBMISSION_PATH = ROOT / "submission_lightgbm_variant.csv"
RANDOM_STATE = 2027


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = train[TARGET].astype(int)
    test_ids = test[ID_COL]

    combined = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    combined = add_features(combined)
    X = combined.iloc[: len(train)].drop(columns=[ID_COL])
    X_test = combined.iloc[len(train) :].drop(columns=[ID_COL])
    train_te_source = combined.iloc[: len(train)]
    test_te_source = combined.iloc[len(train) :]

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 60,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.03,
        "reg_lambda": 2.0,
        "n_estimators": 2600,
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
            callbacks=[lgb.early_stopping(160, verbose=False), lgb.log_evaluation(100)],
        )
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_test_fold)[:, 1] / folds.n_splits
        print(f"Fold {fold}: AUC={roc_auc_score(y.iloc[val_idx], oof[val_idx]):.6f}")

    print(f"OOF AUC: {roc_auc_score(y, oof):.6f}")
    np.save(ROOT / "lightgbm_variant_oof.npy", oof)
    np.save(ROOT / "lightgbm_variant_test.npy", test_pred)
    pd.DataFrame({ID_COL: test_ids, TARGET: np.clip(test_pred, 1e-6, 1 - 1e-6)}).to_csv(
        SUBMISSION_PATH,
        index=False,
    )
    print(f"Wrote {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
