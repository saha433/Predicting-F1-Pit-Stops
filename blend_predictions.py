from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
TARGET = "PitNextLap"
TEST_PATH = Path("/Users/saha/Downloads/test.csv")
TRAIN_PATH = Path("/Users/saha/Downloads/train.csv")


def main() -> None:
    y = pd.read_csv(TRAIN_PATH)[TARGET].astype(int)
    test_ids = pd.read_csv(TEST_PATH)["id"]

    lightgbm_oof = np.load(ROOT / "lightgbm_oof.npy")
    lightgbm_test = np.load(ROOT / "lightgbm_test.npy")
    variant_oof = np.load(ROOT / "lightgbm_variant_oof.npy")
    variant_test = np.load(ROOT / "lightgbm_variant_test.npy")
    xgboost_oof = np.load(ROOT / "xgboost_oof.npy")
    xgboost_test = np.load(ROOT / "xgboost_test.npy")

    weights = {
        "lightgbm": 0.30,
        "lightgbm_variant": 0.48,
        "xgboost": 0.22,
    }
    blend_oof = (
        weights["lightgbm"] * lightgbm_oof
        + weights["lightgbm_variant"] * variant_oof
        + weights["xgboost"] * xgboost_oof
    )
    blend_test = (
        weights["lightgbm"] * lightgbm_test
        + weights["lightgbm_variant"] * variant_test
        + weights["xgboost"] * xgboost_test
    )

    print(f"Blend OOF AUC: {roc_auc_score(y, blend_oof):.6f}")
    submission = pd.DataFrame(
        {
            "id": test_ids,
            TARGET: np.clip(blend_test, 1e-6, 1 - 1e-6),
        }
    )
    submission.to_csv(ROOT / "submission.csv", index=False)
    submission.to_csv(ROOT / "submission_blend_best.csv", index=False)
    print(f"Wrote {ROOT / 'submission.csv'}")


if __name__ == "__main__":
    main()
