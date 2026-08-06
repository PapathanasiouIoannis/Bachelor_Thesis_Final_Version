from pathlib import Path

import pandas as pd
import pytest

from src.ml import family_shortcuts
from src.ml.family_shortcuts import (
    _amplitude_balance,
    _probe,
    audit_family_shortcuts,
    load_development_shortcut_inputs,
)


def test_amplitude_balance_requires_equal_class_counts_per_split():
    frame = pd.DataFrame(
        {
            "Sample_ID": ["a", "b", "c", "d"],
            "Split": ["train", "train", "val", "val"],
            "Perturb_A": [0.0, 0.0, 0.01, 0.01],
            "Label": [0, 1, 0, 1],
        }
    )

    assert _amplitude_balance(frame)["balanced_within_split_and_amplitude"]
    assert not _amplitude_balance(frame.iloc[:-1])[
        "balanced_within_split_and_amplitude"
    ]


def test_shortcut_probe_detects_direct_label_proxy():
    train = pd.DataFrame(
        {
            "proxy": [0, 0, 0, 1, 1, 1],
            "Label": [0, 0, 0, 1, 1, 1],
            "Group_ID": ["H1", "H1", "H1", "Q1", "Q1", "Q1"],
        }
    )
    validation = pd.DataFrame(
        {
            "proxy": [0, 0, 1, 1],
            "Label": [0, 0, 1, 1],
            "Group_ID": ["H2", "H2", "Q2", "Q2"],
        }
    )

    result = _probe(train, validation, ["proxy"])

    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["family_balanced_accuracy"] == pytest.approx(1.0)
    assert result["family_weighted_roc_auc"] == pytest.approx(1.0)
    assert result["validation_families"] == 2


def test_development_loader_filters_locked_rows_at_each_parquet_scan(
    tmp_path, monkeypatch
):
    split_manifest_path = tmp_path / "split_manifest.parquet"
    sample_audit_path = tmp_path / "sample_audit.parquet"
    samples_path = tmp_path / "curve_samples.parquet"
    physics_path = tmp_path / "physics.parquet"
    pd.DataFrame(
        {
            "Group_ID": ["H_TRAIN", "Q_VAL", "Q_LOCKED"],
            "Split": ["train", "val", "test"],
        }
    ).to_parquet(split_manifest_path, index=False)
    pd.DataFrame(
        {
            "Sample_ID": ["sample-train", "sample-val", "sample-locked"],
            "Curve_ID": ["curve-train", "curve-val", "curve-locked"],
            "Group_ID": ["H_TRAIN", "Q_VAL", "Q_LOCKED"],
            "Family_Group_ID": ["H_TRAIN", "Q_VAL", "Q_LOCKED"],
            "Split": ["train", "val", "test"],
        }
    ).to_parquet(sample_audit_path, index=False)
    pd.DataFrame(
        {
            "Sample_ID": ["sample-train", "sample-val", "sample-locked"],
            "Curve_ID": ["curve-train", "curve-val", "curve-locked"],
            "Family_Group_ID": ["H_TRAIN", "Q_VAL", "Q_LOCKED"],
            "observable": [1.0, 2.0, 999.0],
        }
    ).to_parquet(samples_path, index=False)
    pd.DataFrame(
        {
            "Curve_ID": ["curve-train", "curve-val", "curve-locked"],
            "Family_Group_ID": ["H_TRAIN", "Q_VAL", "Q_LOCKED"],
            "Mass": [1.0, 1.0, 999.0],
        }
    ).to_parquet(physics_path, index=False)

    original_read_parquet = pd.read_parquet
    calls = []

    def recording_read_parquet(path, *args, **kwargs):
        calls.append((Path(path).name, kwargs.get("filters")))
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", recording_read_parquet)
    loaded = load_development_shortcut_inputs(
        samples_path=samples_path,
        sample_audit_path=sample_audit_path,
        physics_path=physics_path,
        split_manifest_path=split_manifest_path,
    )

    assert set(loaded.sample_audit["Sample_ID"]) == {"sample-train", "sample-val"}
    assert set(loaded.samples["Sample_ID"]) == {"sample-train", "sample-val"}
    assert set(loaded.physics["Curve_ID"]) == {"curve-train", "curve-val"}
    assert loaded.locked_test_family_ids == frozenset({"Q_LOCKED"})
    filters_by_file = dict(calls)
    assert filters_by_file["sample_audit.parquet"] == [
        ("Split", "in", ["train", "val"]),
        ("Group_ID", "in", ["H_TRAIN", "Q_VAL"]),
    ]
    assert filters_by_file["curve_samples.parquet"] == [
        ("Sample_ID", "in", ["sample-train", "sample-val"])
    ]
    assert filters_by_file["physics.parquet"] == [
        ("Curve_ID", "in", ["curve-train", "curve-val"])
    ]


@pytest.mark.parametrize("contaminated_table", ["samples", "physics", "sample_audit"])
def test_locked_test_identity_is_rejected_before_shortcut_processing(
    monkeypatch, contaminated_table
):
    sample_audit = pd.DataFrame(
        {
            "Sample_ID": ["train", "validation"],
            "Curve_ID": ["curve-train", "curve-validation"],
            "Group_ID": ["H_TRAIN", "Q_VALIDATION"],
            "Family_Group_ID": ["H_TRAIN", "Q_VALIDATION"],
            "Split": ["train", "val"],
        }
    )
    samples = pd.DataFrame(
        {
            "Sample_ID": ["train", "validation"],
            "Curve_ID": ["curve-train", "curve-validation"],
            "Family_Group_ID": ["H_TRAIN", "Q_VALIDATION"],
        }
    )
    physics = pd.DataFrame(
        {
            "Curve_ID": ["curve-train", "curve-validation"],
            "Family_Group_ID": ["H_TRAIN", "Q_VALIDATION"],
        }
    )
    if contaminated_table == "samples":
        samples.loc[len(samples)] = ["locked", "curve-locked", "Q_LOCKED"]
    elif contaminated_table == "physics":
        physics.loc[len(physics)] = ["curve-locked", "Q_LOCKED"]
    else:
        sample_audit.loc[len(sample_audit)] = [
            "locked",
            "curve-locked",
            "Q_LOCKED",
            "Q_LOCKED",
            "test",
        ]

    def should_not_process(_physics):
        raise AssertionError("raw shortcut summaries ran before the lock guard")

    monkeypatch.setattr(family_shortcuts, "_raw_curve_summaries", should_not_process)
    with pytest.raises(ValueError, match="locked development ID scope"):
        audit_family_shortcuts(
            samples=samples,
            sample_audit=sample_audit,
            physics=physics,
            feature_manifest={},
            locked_test_family_ids={"Q_LOCKED"},
        )
