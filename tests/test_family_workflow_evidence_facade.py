from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src import family_workflow
from src.family_runner import evidence


ROOT = Path(__file__).resolve().parents[1]


def test_family_evidence_facade_preserves_exact_helper_signatures():
    expected = {
        "_read_json_status": "(path: 'Path') -> 'tuple[dict[str, Any] | None, str | None]'",
        "_file_matches_sha256": "(path: 'Path', expected_hash: 'str') -> 'bool'",
        "_artifact_status": "(paths: 'list[Path]') -> 'dict[str, Any]'",
        "_development_artifacts": (
            "(paths: 'FamilyWorkflowPaths') -> 'dict[str, dict[str, Any]]'"
        ),
        "_development_evidence_summary": (
            "(paths: 'FamilyWorkflowPaths') -> 'dict[str, Any]'"
        ),
        "_final_test_status": (
            "(paths: 'FamilyWorkflowPaths', model_profile: 'dict[str, Any] | None' = "
            "None) -> 'dict[str, Any]'"
        ),
    }

    assert {
        name: str(inspect.signature(getattr(family_workflow, name)))
        for name in expected
    } == expected
    expected_leaf_exports = {
        "artifact_status",
        "development_artifacts",
        "development_evidence_summary",
        "file_matches_sha256",
        "final_test_status",
        "read_json_status",
    }
    assert set(evidence.__all__) == expected_leaf_exports
    assert len(evidence.__all__) == len(expected_leaf_exports)
    assert family_workflow._read_json_status is not evidence.read_json_status
    assert family_workflow._file_matches_sha256 is not evidence.file_matches_sha256
    assert family_workflow._artifact_status is not evidence.artifact_status

    required_dependencies = {
        evidence.development_artifacts: ("artifact_status",),
        evidence.development_evidence_summary: ("read_json_status",),
        evidence.final_test_status: (
            "read_json_status",
            "file_matches_sha256",
            "model_set_claim",
        ),
    }
    for function, dependency_names in required_dependencies.items():
        parameters = inspect.signature(function).parameters
        for name in dependency_names:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is inspect.Parameter.empty


def test_family_runner_package_import_is_inert():
    script = textwrap.dedent(
        """
        import sys

        import src.family_runner

        assert "src.family_runner.evidence" not in sys.modules
        assert "src.family_workflow" not in sys.modules
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "leaf_first", (True, False), ids=("leaf-first", "facade-first")
)
def test_family_evidence_leaf_and_facade_import_in_both_orders(leaf_first):
    imports = (
        """
        from src.family_runner import evidence
        assert "src.family_workflow" not in sys.modules
        from src import family_workflow
        """
        if leaf_first
        else """
        from src import family_workflow
        from src.family_runner import evidence
        """
    )
    script = textwrap.dedent(
        f"""
        import sys

        {imports}

        assert callable(evidence.read_json_status)
        assert callable(evidence.file_matches_sha256)
        assert callable(evidence.artifact_status)
        assert callable(evidence.development_artifacts)
        assert callable(evidence.development_evidence_summary)
        assert callable(evidence.final_test_status)
        assert callable(family_workflow._read_json_status)
        assert callable(family_workflow._file_matches_sha256)
        assert callable(family_workflow._artifact_status)
        assert callable(family_workflow._development_artifacts)
        assert callable(family_workflow._development_evidence_summary)
        assert callable(family_workflow._final_test_status)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_family_evidence_composites_use_live_facade_dependencies(
    tmp_path,
    monkeypatch,
):
    paths = family_workflow.FamilyWorkflowPaths(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_profile_path=tmp_path / "generation.json",
        split_profile_path=tmp_path / "split.json",
        model_profile_path=tmp_path / "model.json",
        report_dir=tmp_path / "reports",
    )
    artifact_calls = []

    def artifact_status(path_list):
        artifact_calls.append(path_list)
        return {"sentinel_index": len(artifact_calls)}

    monkeypatch.setattr(family_workflow, "_artifact_status", artifact_status)

    artifacts = family_workflow._development_artifacts(paths)

    assert tuple(artifacts) == (
        "generation",
        "curve_preparation",
        "shortcut_audit",
        "model_selection",
        "robustness",
    )
    assert list(artifacts.values()) == [
        {"sentinel_index": 1},
        {"sentinel_index": 2},
        {"sentinel_index": 3},
        {"sentinel_index": 4},
        {"sentinel_index": 5},
    ]
    assert len(artifact_calls) == 5

    evidence_reads = []
    evidence_payloads = {
        "family_shortcut_audit.json": {"passed": True, "test_rows_used": 0},
        "family_model_selection.json": {
            "selected_candidate": {"candidate_id": "logistic_live"}
        },
        "family_development_robustness.json": {
            "family_label_permutation_null": {"empirical_p_value": 0.25},
            "test_rows_used": 0,
        },
    }

    def read_development(path):
        evidence_reads.append(path.name)
        return evidence_payloads[path.name], None

    monkeypatch.setattr(family_workflow, "_read_json_status", read_development)

    summary = family_workflow._development_evidence_summary(paths)

    assert evidence_reads == [
        "family_shortcut_audit.json",
        "family_model_selection.json",
        "family_development_robustness.json",
    ]
    assert summary["selected_reporting_model"] == "logistic_live"
    assert summary["family_permutation_empirical_p_value"] == 0.25

    marker = {
        "status": "COMPLETED",
        "opened_utc": "synthetic-open-time",
        "locked_git_commit": "synthetic-commit",
        "model_profile_sha256": "profile-hash",
        "result_sha256": "result-hash",
    }
    result = {
        "locked_git_commit": "synthetic-commit",
        "model_profile_sha256": "profile-hash",
        "test_open_count": 1,
        "test_metrics": {},
        "per_eos": [],
    }
    final_reads = []
    match_calls = []

    def read_final(path):
        final_reads.append(path)
        if path == paths.final_test_marker_path:
            return marker, None
        if path == paths.final_test_result_path:
            return result, None
        raise AssertionError(f"Unexpected evidence path: {path}")

    def match_hash(path, expected_hash):
        match_calls.append((path, expected_hash))
        return True

    monkeypatch.setattr(family_workflow, "_read_json_status", read_final)
    monkeypatch.setattr(family_workflow, "_file_matches_sha256", match_hash)
    monkeypatch.setattr(family_workflow, "MODEL_SET_CLAIM", "live facade claim")

    final = family_workflow._final_test_status(paths)

    assert final_reads == [
        paths.final_test_marker_path,
        paths.final_test_result_path,
    ]
    assert match_calls == [
        (paths.final_test_result_path, "result-hash"),
        (paths.model_profile_path, "profile-hash"),
    ]
    assert final["integrity"] == "valid"
    assert final["result"]["claim_boundary"] == "live facade claim"
    assert final["result"]["interpretation"] == "live facade claim"


def test_final_guard_retains_the_live_facade_reader_seam(tmp_path, monkeypatch):
    paths = family_workflow.FamilyWorkflowPaths(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        generation_profile_path=tmp_path / "generation.json",
        split_profile_path=tmp_path / "split.json",
        model_profile_path=tmp_path / "model.json",
        report_dir=tmp_path / "reports",
    )
    marker = {
        "status": "OPENED",
        "opened_utc": "synthetic-open-time",
    }
    calls = []

    def read(path):
        calls.append(path)
        return marker, None

    monkeypatch.setattr(family_workflow, "_read_json_status", read)

    with pytest.raises(family_workflow.FinalTestAlreadyOpenedError) as caught:
        family_workflow.assert_final_evaluation_not_opened(paths)

    assert calls == [paths.final_test_marker_path]
    assert caught.value.args == (
        "The locked family test was already opened at synthetic-open-time "
        "(marker status: OPENED). It cannot be opened or scored again. Review the "
        f"recorded result at {paths.final_test_result_path}.",
    )


def test_family_evidence_facade_uses_reloaded_leaf_and_live_named_helpers():
    script = textwrap.dedent(
        """
        import importlib

        from src import family_workflow
        from src.family_runner import evidence

        importlib.reload(evidence)
        path = object()
        paths = object()
        path_list = [object()]
        model_profile = object()
        read_result = object()
        match_result = object()
        artifact_result = object()
        development_result = object()
        summary_result = object()
        final_result = object()
        calls = []

        def read(path_argument):
            assert path_argument is path
            calls.append("read")
            return read_result

        def match(path_argument, expected_hash):
            assert path_argument is path
            assert expected_hash == "expected-hash"
            calls.append("match")
            return match_result

        def artifact(path_arguments):
            assert path_arguments is path_list
            calls.append("artifact")
            return artifact_result

        def development(paths_argument, *, artifact_status):
            assert paths_argument is paths
            assert artifact_status is facade_artifact_status
            calls.append("development")
            return development_result

        def summary(paths_argument, *, read_json_status):
            assert paths_argument is paths
            assert read_json_status is facade_read_json_status
            calls.append("summary")
            return summary_result

        def final(
            paths_argument,
            model_profile_argument=None,
            *,
            read_json_status,
            file_matches_sha256,
            model_set_claim,
        ):
            assert paths_argument is paths
            assert model_profile_argument is model_profile
            assert read_json_status is facade_read_json_status
            assert file_matches_sha256 is facade_file_matcher
            assert model_set_claim == "facade claim"
            calls.append("final")
            return final_result

        def facade_artifact_status(_paths):
            raise AssertionError("the dispatch spy must not call this helper")

        def facade_read_json_status(_path):
            raise AssertionError("the dispatch spy must not call this helper")

        def facade_file_matcher(_path, _expected):
            raise AssertionError("the dispatch spy must not call this helper")

        evidence.read_json_status = read
        evidence.file_matches_sha256 = match
        evidence.artifact_status = artifact
        evidence.development_artifacts = development
        evidence.development_evidence_summary = summary
        evidence.final_test_status = final

        assert family_workflow._read_json_status(path) is read_result
        assert family_workflow._file_matches_sha256(path, "expected-hash") is match_result
        assert family_workflow._artifact_status(path_list) is artifact_result

        family_workflow._artifact_status = facade_artifact_status
        assert family_workflow._development_artifacts(paths) is development_result

        family_workflow._read_json_status = facade_read_json_status
        assert family_workflow._development_evidence_summary(paths) is summary_result

        family_workflow._file_matches_sha256 = facade_file_matcher
        family_workflow.MODEL_SET_CLAIM = "facade claim"
        assert family_workflow._final_test_status(paths, model_profile) is final_result
        assert calls == ["read", "match", "artifact", "development", "summary", "final"]
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
