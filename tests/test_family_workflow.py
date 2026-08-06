from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.family_workflow import (
    FamilyDevelopmentStageError,
    FamilyWorkflowError,
    FinalTestAlreadyOpenedError,
    assert_final_evaluation_not_opened,
    family_workflow_status,
    refuse_final_evaluation_request,
    resolve_family_workflow_paths,
    run_family_development,
)


def _paths(tmp_path: Path):
    return resolve_family_workflow_paths(
        data_root=tmp_path / "data",
        report_dir=tmp_path / "reports",
    )


def _write_completed_final(paths) -> None:
    paths.family_ml_dir.mkdir(parents=True)
    paths.report_dir.mkdir(parents=True)
    profile_hash = hashlib.sha256(paths.model_profile_path.read_bytes()).hexdigest()
    result = {
        "locked_git_commit": "abc123",
        "model_profile_sha256": profile_hash,
        "test_open_count": 1,
        "test_metrics": {"balanced_accuracy": 0.75, "roc_auc": 0.8},
        "independent_test_family_units": 2,
        "strict_2p08_test_applicable": False,
    }
    paths.final_test_result_path.write_text(json.dumps(result) + "\n", encoding="utf-8")
    result_hash = hashlib.sha256(paths.final_test_result_path.read_bytes()).hexdigest()
    marker = {
        "status": "COMPLETED",
        "opened_utc": "2026-08-04T11:30:41+00:00",
        "locked_git_commit": "abc123",
        "model_profile_sha256": profile_hash,
        "result_sha256": result_hash,
    }
    paths.final_test_marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")


def test_status_uses_professional_family_and_model_set_wording(tmp_path):
    status = family_workflow_status(_paths(tmp_path))

    assert status["generation_profile"]["hadronic_eos_baselines"] == 9
    assert status["generation_profile"]["quark_eos_baselines"] == 9
    assert status["generation_profile"]["family_groups"] == 17
    assert status["generation_profile"]["expected_curves"] == 108
    assert status["split_profile"]["splits"]["train"]["family_groups"] == 13
    assert status["split_profile"]["splits"]["val"]["family_groups"] == 2
    assert status["split_profile"]["splits"]["test"]["family_groups"] == 2
    assert status["reporting_model_policy"]["supported"] == [
        "dummy baseline",
        "logistic regression",
    ]
    assert status["reporting_model_policy"]["exploratory_not_run_by_workflow"] == [
        "XGBoost",
        "multilayer perceptron",
    ]
    assert "model-set discrimination" in status["scientific_scope"]
    assert status["final_test"]["state"] == "LOCKED_NOT_EVALUATED"


def test_status_reports_completed_final_once_without_loading_test_tensor(tmp_path):
    paths = _paths(tmp_path)
    _write_completed_final(paths)

    status = family_workflow_status(paths)
    final = status["final_test"]

    assert final["state"] == "LOCKED_TEST_OPENED"
    assert final["open_count"] == 1
    assert final["marker_status"] == "COMPLETED"
    assert final["integrity"] == "valid"
    assert final["rerun_permitted"] is False
    assert final["result"]["balanced_accuracy"] == 0.75
    assert final["result"]["independent_family_units"] == 2
    assert "predictions" not in final["result"]


def test_open_marker_refuses_final_evaluation_and_force_regeneration(tmp_path):
    paths = _paths(tmp_path)
    _write_completed_final(paths)

    with pytest.raises(FinalTestAlreadyOpenedError, match="already opened"):
        assert_final_evaluation_not_opened(paths)
    with pytest.raises(FinalTestAlreadyOpenedError, match="outputs are frozen"):
        run_family_development(paths, force_regenerate=True)
    with pytest.raises(FinalTestAlreadyOpenedError, match="outputs are frozen"):
        run_family_development(paths)


def test_development_runs_only_the_five_safe_stages(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="{}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_family_development(paths, jobs=3, permutations=17)

    assert [record["stage"] for record in result["completed_stages"]] == [
        "generation",
        "curve_preparation",
        "shortcut_audit",
        "model_selection",
        "robustness",
    ]
    assert len(calls) == 5
    assert all(command[0] == sys.executable for command, _ in calls)
    assert not any(
        "family_final_test.py" in part for command, _ in calls for part in command
    )
    assert "--n-jobs" in calls[0][0] and "3" in calls[0][0]
    assert "--permutations" in calls[-1][0] and "17" in calls[-1][0]
    assert result["final_test_accessed"] is False


def test_failed_development_stage_has_actionable_context(tmp_path, monkeypatch):
    paths = _paths(tmp_path)

    def fake_run(command, **kwargs):
        del kwargs
        if "family_shortcut_audit.py" in command[1]:
            return SimpleNamespace(
                returncode=2,
                stdout="",
                stderr="amplitude balance check failed",
            )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(FamilyDevelopmentStageError) as caught:
        run_family_development(paths)

    assert caught.value.stage == "shortcut_audit"
    assert "amplitude balance check failed" in str(caught.value)


def test_final_named_requests_are_never_dispatched(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(FamilyWorkflowError, match="development stages only"):
        run_family_development(_paths(tmp_path), requested_stage="final_evaluation")
    with pytest.raises(FamilyWorkflowError, match="development-only"):
        refuse_final_evaluation_request(_paths(tmp_path))

    assert calls == []
