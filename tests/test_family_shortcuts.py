import pandas as pd
import pytest

from src.ml.family_shortcuts import _amplitude_balance, _probe


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
        {"proxy": [0, 0, 0, 1, 1, 1], "Label": [0, 0, 0, 1, 1, 1]}
    )
    validation = pd.DataFrame(
        {"proxy": [0, 0, 1, 1], "Label": [0, 0, 1, 1]}
    )

    result = _probe(train, validation, ["proxy"])

    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
