from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from train_lightgbm import (
    ID_COL,
    RANDOM_STATE,
    TARGET,
    TARGET_ENCODING_COLS,
    TEST_PATH,
    TRAIN_PATH,
    add_features,
    add_target_encoding,
)


ROOT = Path(__file__).resolve().parent
SUBMISSION_PATH = ROOT / "submission_xgboost.csv"


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

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + 77)
    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

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
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            learning_rate=0.035,
            max_depth=7,
            min_child_weight=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            n_estimators=1800,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            early_stopping_rounds=120,
        )
        model.fit(
            X_trn,
            y.iloc[trn_idx],
            eval_set=[(X_val, y.iloc[val_idx])],
            verbose=100,
        )
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_test_fold)[:, 1] / folds.n_splits
        print(f"Fold {fold}: AUC={roc_auc_score(y.iloc[val_idx], oof[val_idx]):.6f}")

    print(f"OOF AUC: {roc_auc_score(y, oof):.6f}")
    np.save(ROOT / "xgboost_oof.npy", oof)
    np.save(ROOT / "xgboost_test.npy", test_pred)
    pd.DataFrame({ID_COL: test_ids, TARGET: np.clip(test_pred, 1e-6, 1 - 1e-6)}).to_csv(
        SUBMISSION_PATH,
        index=False,
    )
    print(f"Wrote {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
