import pandas as pd

from src.ml.family_dataset import curve_feature_columns
from src.ml.family_final import (
    build_locked_estimator,
    load_locked_model_profile,
    predict_locked_estimator,
)


def test_model_profile_locks_exact_radius_allowlist():
    profile = load_locked_model_profile()

    assert profile["lock_state"] == "PRE_TEST_LOCKED"
    assert profile["model"]["feature_columns"] == curve_feature_columns("MR")
    assert profile["model"]["C"] == 0.1
    assert profile["final_test"]["use_once"]


def test_locked_prediction_uses_threshold_without_metadata_features():
    profile = load_locked_model_profile()
    estimator = build_locked_estimator(profile)
    features = profile["model"]["feature_columns"]
    training = pd.DataFrame(
        [
            {**dict.fromkeys(features, -1.0), "Label": 0},
            {**dict.fromkeys(features, 1.0), "Label": 1},
        ]
    )
    estimator.fit(training[features], training["Label"])
    frame = pd.DataFrame(
        [
            {
                **dict.fromkeys(features, -1.0),
                "Sample_ID": "H",
                "EoS_ID": "H_TEST",
                "Group_ID": "H_TEST",
                "Perturb_A": 0.0,
                "Label": 0,
            },
            {
                **dict.fromkeys(features, 1.0),
                "Sample_ID": "Q",
                "EoS_ID": "Q_TEST",
                "Group_ID": "Q_TEST",
                "Perturb_A": 0.0,
                "Label": 1,
            },
        ]
    )

    result = predict_locked_estimator(estimator, frame, profile)

    assert result["Correct"].tolist() == [1, 1]
