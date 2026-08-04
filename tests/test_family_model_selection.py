import numpy as np
import pandas as pd
import pytest

from src.ml.family_model_selection import (
    build_family_pair_folds,
    candidate_grid,
    inverse_family_class_weights,
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

    assert np.allclose(group_totals[group_totals.index.str.startswith("H_")], group_totals["H_0"])
    assert np.allclose(group_totals[group_totals.index.str.startswith("Q_")], group_totals["Q_0"])
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
