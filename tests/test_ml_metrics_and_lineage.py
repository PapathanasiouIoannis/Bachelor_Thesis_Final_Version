import json

import numpy as np
import pytest

from src.ml.metrics import binary_metrics, select_macro_f1_threshold
from src.runtime import require_artifact_lineage, write_artifact_lineage


def _write_tensor_lineage_fixture(tensor_dir):
    tensor_dir.mkdir()
    preprocessing = {
        "transformer": "sklearn.preprocessing.StandardScaler",
        "outer_fit_scope": "train_only",
        "scaler_filename": "scaler.joblib",
        "hpo_training_filename": "train_unscaled.parquet",
        "hpo_fold_transform": "fit_on_inner_fit_only",
        "split_transform": {"name": "none"},
    }
    with open(tensor_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_sha256": "dataset-hash",
                "tensor_input_sha256": "tensor-input-hash",
                "split_manifest_sha256": "split-hash",
                "approved_features": ["Mass", "Radius", "log10_Lambda"],
                "preprocessing": preprocessing,
            },
            handle,
        )
    for filename in (
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "row_audit.parquet",
        "split_audit.parquet",
        "scaler.joblib",
        "train_unscaled.parquet",
    ):
        (tensor_dir / filename).write_bytes(filename.encode("utf-8"))


def test_threshold_is_selected_from_validation_probabilities():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.45, 0.9])
    threshold = select_macro_f1_threshold(labels, probabilities)
    metrics = binary_metrics(labels, probabilities, threshold)

    assert threshold == pytest.approx(0.45)
    assert metrics["F1-Macro"] == pytest.approx(1.0)
    assert metrics["F1-Quark"] == pytest.approx(1.0)
    assert "PR-AUC-Trapezoidal" in metrics


def test_hpo_artifact_lineage_fails_closed_after_edit(tmp_path):
    tensor_dir = tmp_path / "tensors"
    _write_tensor_lineage_fixture(tensor_dir)
    artifact = tmp_path / "best_params.json"
    artifact.write_text('{"depth": 3}', encoding="utf-8")

    write_artifact_lineage(
        artifact, tensor_dir, "test_hpo", ["Mass", "Radius"]
    )
    require_artifact_lineage(
        artifact,
        tensor_dir,
        "test",
        component="test_hpo",
        selected_features=["Mass", "Radius"],
    )

    artifact.write_text('{"depth": 4}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="lineage mismatch"):
        require_artifact_lineage(
            artifact,
            tensor_dir,
            "test",
            component="test_hpo",
            selected_features=["Mass", "Radius"],
        )


def test_hpo_artifact_lineage_detects_tensor_and_feature_changes(tmp_path):
    tensor_dir = tmp_path / "tensors"
    _write_tensor_lineage_fixture(tensor_dir)
    artifact = tmp_path / "best_params.json"
    artifact.write_text('{"depth": 3}', encoding="utf-8")
    write_artifact_lineage(
        artifact, tensor_dir, "test_hpo", ["Mass", "Radius"]
    )

    with pytest.raises(RuntimeError, match="lineage mismatch"):
        require_artifact_lineage(
            artifact,
            tensor_dir,
            "test",
            selected_features=["Mass", "Radius", "log10_Lambda"],
        )

    (tensor_dir / "train.parquet").write_bytes(b"mutated tensor")
    with pytest.raises(RuntimeError, match="lineage mismatch"):
        require_artifact_lineage(
            artifact,
            tensor_dir,
            "test",
            selected_features=["Mass", "Radius"],
        )
