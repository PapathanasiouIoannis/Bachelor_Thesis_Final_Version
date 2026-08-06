import numpy as np
import pandas as pd
import pytest

from src.ml.family_model_selection import (
    Candidate,
    build_family_pair_folds,
    candidate_grid,
    classification_metrics,
    inverse_family_class_weights,
    run_development_selection,
    validate_model_policy,
)


def _training_metadata() -> pd.DataFrame:
    records = []
    for label, count in ((0, 6), (1, 7)):
        for group_index in range(count):
            group_id = f"{'H' if label == 0 else 'Q'}_{group_index}"
            sample_count = 12 if group_id == "H_0" else 6
            for sample_index in range(sample_count):
                records.append(
                    {
                        "Sample_ID": f"{group_id}:{sample_index}",
                        "Group_ID": group_id,
                        "Label": label,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_family_pair_folds_are_disjoint_and_exhaustive():
    metadata = _training_metadata()
    folds = build_family_pair_folds(metadata)
    scored = []
    for fold in folds:
        assert set(fold["fit_groups"]).isdisjoint(fold["score_groups"])
        score = metadata[metadata["Group_ID"].isin(fold["score_groups"])]
        assert set(score["Label"]) == {0, 1}
        scored.extend(fold["score_groups"])

    assert len(folds) == 6
    assert sorted(scored) == sorted(metadata["Group_ID"].unique())


def test_family_weights_equalize_groups_and_classes():
    metadata = _training_metadata()
    metadata["Weight"] = inverse_family_class_weights(metadata)
    group_totals = metadata.groupby("Group_ID")["Weight"].sum()
    class_totals = metadata.groupby("Label")["Weight"].sum()

    assert np.allclose(
        group_totals[group_totals.index.str.startswith("H_")], group_totals["H_0"]
    )
    assert np.allclose(
        group_totals[group_totals.index.str.startswith("Q_")], group_totals["Q_0"]
    )
    assert class_totals[0] == pytest.approx(class_totals[1])


def test_candidate_grid_is_small_and_predeclared():
    candidates = candidate_grid()

    assert len(candidates) == 23
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)
    assert {candidate.architecture for candidate in candidates} == {
        "dummy",
        "logistic",
        "forest",
    }
    assert {
        candidate.architecture
        for candidate in candidates
        if candidate.eligible_for_reporting_selection
    } == {"dummy", "logistic"}
    assert all(
        not candidate.eligible_for_reporting_selection
        for candidate in candidates
        if candidate.architecture == "forest"
    )


def test_model_policy_rejects_exploratory_reporting_candidates():
    validate_model_policy(
        ("dummy", "logistic_regression"),
        ("xgboost", "mlp"),
    )

    with pytest.raises(ValueError, match="locked to dummy and logistic"):
        validate_model_policy(
            ("dummy", "logistic_regression", "xgboost"),
            ("mlp",),
        )


def test_metrics_give_each_family_equal_weight():
    predictions = pd.DataFrame(
        [
            *(
                {
                    "EoS_ID": "H-large",
                    "Group_ID": "H-large",
                    "Label": 0,
                    "Probability_Quark": 0.9,
                }
                for _ in range(9)
            ),
            {
                "EoS_ID": "H-small",
                "Group_ID": "H-small",
                "Label": 0,
                "Probability_Quark": 0.1,
            },
            {
                "EoS_ID": "Q-one",
                "Group_ID": "Q-one",
                "Label": 1,
                "Probability_Quark": 0.9,
            },
            {
                "EoS_ID": "Q-two",
                "Group_ID": "Q-two",
                "Label": 1,
                "Probability_Quark": 0.9,
            },
        ]
    )

    metrics = classification_metrics(predictions)

    assert metrics["curve_balanced_accuracy"] == pytest.approx(0.55)
    assert metrics["family_balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["family_weighted_accuracy"] == pytest.approx(0.75)
    assert metrics["family_weighted_brier"] != pytest.approx(metrics["curve_brier"])


def test_exploratory_forest_cannot_win_reporting_selection(monkeypatch):
    import src.ml.family_model_selection as selection

    candidates = [
        Candidate("dummy", "dummy", "MR", {}, 0, "reporting"),
        Candidate("logistic", "logistic", "MR", {"C": 0.1}, 10, "reporting"),
        Candidate(
            "forest",
            "forest",
            "MR",
            {"max_depth": 2, "min_samples_leaf": 3},
            20,
            "exploratory",
        ),
    ]
    monkeypatch.setattr(selection, "candidate_grid", lambda: candidates)

    def probabilities(fit, score, candidate):
        del fit
        if candidate.architecture == "dummy":
            return np.full(len(score), 0.5)
        # Forest is deliberately perfect, but is evaluated only in inner CV.
        return np.where(score["Label"].to_numpy(dtype=int) == 1, 0.9, 0.1)

    monkeypatch.setattr(selection, "_fit_predict", probabilities)

    records = []
    for label, prefix in ((0, "H"), (1, "Q")):
        for group_index in range(3):
            group_id = f"{prefix}_{group_index}"
            records.append(
                {
                    "Sample_ID": group_id,
                    "EoS_ID": group_id,
                    "Group_ID": group_id,
                    "Perturb_A": 0.0,
                    "Label": label,
                }
            )
    training = pd.DataFrame.from_records(records)
    validation = pd.DataFrame.from_records(
        [
            {
                "Sample_ID": "H-val",
                "EoS_ID": "H-val",
                "Group_ID": "H-val",
                "Perturb_A": 0.0,
                "Label": 0,
            },
            {
                "Sample_ID": "Q-val",
                "EoS_ID": "Q-val",
                "Group_ID": "Q-val",
                "Perturb_A": 0.0,
                "Label": 1,
            },
        ]
    )

    report, predictions = run_development_selection(training, validation)

    assert report["selected_candidate"]["architecture"] == "logistic"
    assert (
        report["model_policy"]["exploratory_models_can_win_reporting_selection"]
        is False
    )
    assert "validation_metrics" in report["dummy_baseline"]
    assert "validation_metrics" not in report["exploratory_cv_finalists"][0]
    validation_candidates = set(
        predictions.loc[predictions["Stage"] == "outer_validation", "Candidate_ID"]
    )
    assert validation_candidates == {"dummy", "logistic"}
