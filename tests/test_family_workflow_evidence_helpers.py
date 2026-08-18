from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src import family_workflow


def _paths(tmp_path: Path) -> family_workflow.FamilyWorkflowPaths:
    return family_workflow.FamilyWorkflowPaths(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_profile_path=tmp_path / "profiles" / "generation.json",
        split_profile_path=tmp_path / "profiles" / "split.json",
        model_profile_path=tmp_path / "profiles" / "model.json",
        report_dir=tmp_path / "reports",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_artifact_inventory_preserves_category_and_path_order(tmp_path):
    paths = _paths(tmp_path)
    expected_paths = {
        "generation": [
            paths.data_root / "physics_test_dataset.parquet",
            paths.data_root / "family_catalog.parquet",
            paths.data_root / "run_manifest.json",
        ],
        "curve_preparation": [
            paths.family_ml_dir / "curve_samples.parquet",
            paths.family_ml_dir / "sample_audit.parquet",
            paths.family_ml_dir / "split_manifest.parquet",
            paths.family_ml_dir / "feature_manifest.json",
            paths.family_ml_dir / "train.parquet",
            paths.family_ml_dir / "val.parquet",
            paths.family_ml_dir / "test.parquet",
        ],
        "shortcut_audit": [
            paths.report_dir / "family_shortcut_audit.json",
            paths.report_dir / "FAMILY_SHORTCUT_AUDIT.md",
        ],
        "model_selection": [
            paths.report_dir / "family_model_selection.json",
            paths.report_dir / "FAMILY_MODEL_SELECTION.md",
            paths.family_ml_dir / "development_predictions.parquet",
        ],
        "robustness": [
            paths.report_dir / "family_development_robustness.json",
            paths.report_dir / "FAMILY_DEVELOPMENT_ROBUSTNESS.md",
        ],
    }
    missing_paths = {
        expected_paths["curve_preparation"][3],
        *expected_paths["shortcut_audit"],
        expected_paths["model_selection"][1],
    }
    for artifact_paths in expected_paths.values():
        for artifact_path in artifact_paths:
            if artifact_path in missing_paths:
                continue
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("synthetic artifact\n", encoding="utf-8")

    inventory = family_workflow._development_artifacts(paths)

    assert tuple(inventory) == (
        "generation",
        "curve_preparation",
        "shortcut_audit",
        "model_selection",
        "robustness",
    )
    for category, artifact_paths in expected_paths.items():
        expected_missing = [
            str(artifact_path)
            for artifact_path in artifact_paths
            if artifact_path in missing_paths
        ]
        assert tuple(inventory[category]) == ("state", "paths", "missing")
        assert inventory[category] == {
            "state": "missing" if expected_missing else "ready",
            "paths": [str(artifact_path) for artifact_path in artifact_paths],
            "missing": expected_missing,
        }


def test_development_evidence_projection_and_read_error_order(tmp_path):
    paths = _paths(tmp_path)
    shortcut_path = paths.report_dir / "family_shortcut_audit.json"
    selection_path = paths.report_dir / "family_model_selection.json"
    robustness_path = paths.report_dir / "family_development_robustness.json"
    _write_json(shortcut_path, {"passed": False, "test_rows_used": 0})
    _write_json(
        selection_path,
        {"selected_candidate": {"candidate_id": "logistic_synthetic"}},
    )
    _write_json(
        robustness_path,
        {
            "family_label_permutation_null": {"empirical_p_value": 0.125},
            "test_rows_used": 0,
        },
    )

    summary = family_workflow._development_evidence_summary(paths)

    assert tuple(summary) == (
        "shortcut_audit_passed",
        "selected_reporting_model",
        "selected_model_family",
        "family_permutation_empirical_p_value",
        "test_rows_used_by_development",
        "read_errors",
    )
    assert summary == {
        "shortcut_audit_passed": False,
        "selected_reporting_model": "logistic_synthetic",
        "selected_model_family": "logistic regression",
        "family_permutation_empirical_p_value": 0.125,
        "test_rows_used_by_development": {
            "shortcut_audit": 0,
            "robustness": 0,
        },
        "read_errors": [],
    }

    _write_json(shortcut_path, [])
    selection_path.write_text("{not-json\n", encoding="utf-8")
    robustness_path.unlink()

    unreadable = family_workflow._development_evidence_summary(paths)

    assert unreadable == {
        "shortcut_audit_passed": None,
        "selected_reporting_model": None,
        "selected_model_family": None,
        "family_permutation_empirical_p_value": None,
        "test_rows_used_by_development": {
            "shortcut_audit": None,
            "robustness": None,
        },
        "read_errors": unreadable["read_errors"],
    }
    assert len(unreadable["read_errors"]) == 2
    assert unreadable["read_errors"][0] == (
        "family_shortcut_audit.json must contain a JSON object."
    )
    assert unreadable["read_errors"][1].startswith(
        "Could not read family_model_selection.json:"
    )


def test_completed_result_projection_preserves_public_schema_and_claim_boundary(
    tmp_path,
):
    paths = _paths(tmp_path)
    _write_json(paths.model_profile_path, {"profile_id": "synthetic-model"})
    profile_hash = _sha256(paths.model_profile_path)
    result = {
        "locked_git_commit": "synthetic-commit",
        "model_profile_sha256": profile_hash,
        "test_open_count": 1,
        "test_metrics": {
            "balanced_accuracy": 0.75,
            "family_balanced_accuracy": 0.8,
            "roc_auc": 0.9,
            "samples": 18,
            "families": 3,
        },
        "per_eos": [
            {
                "EoS_ID": "H-SYNTHETIC",
                "Label": 0,
                "curves": 6,
                "accuracy": 0.5,
                "mean_probability_quark": 0.2,
                "probability_range_across_A": 0.03,
                "minimum_mmax_msun": 2.1,
                "predictions": [0, 1, 0],
            },
            "non-mapping records are omitted",
            {
                "EoS_ID": "Q-SYNTHETIC",
                "Label": 1,
                "curves": 6,
                "accuracy": 1.0,
                "mean_probability_quark": 0.95,
                "probability_range_across_A": 0.01,
                "minimum_mmax_msun": 2.2,
            },
            {
                "EoS_ID": "UNKNOWN-SYNTHETIC",
                "Label": 7,
                "curves": 6,
                "accuracy": 0.0,
                "mean_probability_quark": 0.5,
                "probability_range_across_A": 0.4,
                "minimum_mmax_msun": 1.9,
            },
        ],
        "independent_test_family_units": 3,
        "strict_2p08_test_applicable": True,
        "claim_boundary": "Synthetic archived claim boundary.",
        "predictions": ["raw predictions must not be exposed"],
    }
    _write_json(paths.final_test_result_path, result)
    marker = {
        "status": "COMPLETED",
        "opened_utc": "2026-08-04T11:30:41+00:00",
        "locked_git_commit": "synthetic-commit",
        "model_profile_sha256": profile_hash,
        "result_sha256": _sha256(paths.final_test_result_path),
    }
    _write_json(paths.final_test_marker_path, marker)

    final = family_workflow._final_test_status(paths)

    assert tuple(final) == (
        "state",
        "open_count",
        "marker_status",
        "opened_utc",
        "marker_path",
        "result",
        "integrity",
        "integrity_errors",
        "rerun_permitted",
    )
    assert final["state"] == "LOCKED_TEST_OPENED"
    assert final["open_count"] == 1
    assert final["marker_status"] == "COMPLETED"
    assert final["opened_utc"] == "2026-08-04T11:30:41+00:00"
    assert final["marker_path"] == str(paths.final_test_marker_path)
    assert final["integrity"] == "valid"
    assert final["integrity_errors"] == []
    assert final["rerun_permitted"] is False

    projection = final["result"]
    assert tuple(projection) == (
        "path",
        "balanced_accuracy",
        "family_balanced_accuracy",
        "roc_auc",
        "samples",
        "families",
        "independent_family_units",
        "strict_2p08_test_applicable",
        "per_family",
        "claim_boundary",
        "interpretation",
    )
    assert projection == {
        "path": str(paths.final_test_result_path),
        "balanced_accuracy": 0.75,
        "family_balanced_accuracy": 0.8,
        "roc_auc": 0.9,
        "samples": 18,
        "families": 3,
        "independent_family_units": 3,
        "strict_2p08_test_applicable": True,
        "per_family": [
            {
                "eos_id": "H-SYNTHETIC",
                "matter_type": "hadronic",
                "curves": 6,
                "accuracy": 0.5,
                "mean_quark_class_model_score": 0.2,
                "model_score_range_across_amplitude": 0.03,
                "minimum_maximum_mass_msun": 2.1,
            },
            {
                "eos_id": "Q-SYNTHETIC",
                "matter_type": "quark",
                "curves": 6,
                "accuracy": 1.0,
                "mean_quark_class_model_score": 0.95,
                "model_score_range_across_amplitude": 0.01,
                "minimum_maximum_mass_msun": 2.2,
            },
            {
                "eos_id": "UNKNOWN-SYNTHETIC",
                "matter_type": None,
                "curves": 6,
                "accuracy": 0.0,
                "mean_quark_class_model_score": 0.5,
                "model_score_range_across_amplitude": 0.4,
                "minimum_maximum_mass_msun": 1.9,
            },
        ],
        "claim_boundary": "Synthetic archived claim boundary.",
        "interpretation": family_workflow.MODEL_SET_CLAIM,
    }
    assert "predictions" not in projection
    assert all("predictions" not in record for record in projection["per_family"])

    with pytest.raises(family_workflow.FinalTestAlreadyOpenedError) as caught:
        family_workflow.assert_final_evaluation_not_opened(paths)
    expected_message = (
        "The locked family test was already opened at "
        "2026-08-04T11:30:41+00:00 (marker status: COMPLETED). It cannot be "
        "opened or scored again. Review the recorded result at "
        f"{paths.final_test_result_path}."
    )
    assert caught.value.args == (expected_message,)

    result.pop("claim_boundary")
    _write_json(paths.final_test_result_path, result)
    marker["result_sha256"] = _sha256(paths.final_test_result_path)
    _write_json(paths.final_test_marker_path, marker)

    fallback = family_workflow._final_test_status(paths)["result"]
    assert fallback["claim_boundary"] == family_workflow.MODEL_SET_CLAIM
    assert fallback["interpretation"] == family_workflow.MODEL_SET_CLAIM
