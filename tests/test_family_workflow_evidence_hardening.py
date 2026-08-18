from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src import family_workflow


ROOT = Path(__file__).resolve().parents[1]
LOCKED_CONFIGURATION_FILES = (
    "framework/family_pilot_profile.json",
    "framework/family_split_profile.json",
    "framework/family_model_profile.json",
    "docs/family_shortcut_audit.json",
    "docs/family_model_selection.json",
    "docs/family_development_robustness.json",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolated_paths(tmp_path: Path) -> family_workflow.FamilyWorkflowPaths:
    for relative_name in LOCKED_CONFIGURATION_FILES:
        source = ROOT / relative_name
        destination = tmp_path / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    return family_workflow.resolve_family_workflow_paths(
        project_root=tmp_path,
        data_root="data/family_pilot_v1",
        generation_profile_path="framework/family_pilot_profile.json",
        split_profile_path="framework/family_split_profile.json",
        model_profile_path="framework/family_model_profile.json",
        report_dir="docs",
    )


def _evidence_path(
    paths: family_workflow.FamilyWorkflowPaths,
    evidence_kind: str,
) -> Path:
    if evidence_kind == "marker":
        return paths.final_test_marker_path
    if evidence_kind == "result":
        return paths.final_test_result_path
    raise AssertionError(f"Unknown evidence kind: {evidence_kind}")


def _create_non_regular_entry(path: Path, entry_kind: str) -> None:
    if entry_kind == "directory":
        path.mkdir(parents=True)
        return
    if entry_kind == "broken_symlink":
        path.parent.mkdir(parents=True, exist_ok=True)
        missing_target = path.with_name(f"{path.name}.missing-target")
        try:
            path.symlink_to(missing_target.name)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"Broken symlinks are not supported in this environment: {exc}")
        if not path.is_symlink():
            pytest.skip("The platform did not create the requested symbolic link.")
        assert not missing_target.exists()
        assert not path.exists()
        return
    raise AssertionError(f"Unknown entry kind: {entry_kind}")


NON_REGULAR_CASES = tuple(
    pytest.param(evidence_kind, entry_kind, id=f"{evidence_kind}-{entry_kind}")
    for evidence_kind in ("marker", "result")
    for entry_kind in ("directory", "broken_symlink")
)


@pytest.mark.parametrize(("evidence_kind", "entry_kind"), NON_REGULAR_CASES)
def test_non_regular_final_evidence_is_reported_as_structurally_invalid(
    tmp_path: Path,
    evidence_kind: str,
    entry_kind: str,
) -> None:
    paths = _isolated_paths(tmp_path)
    evidence_path = _evidence_path(paths, evidence_kind)
    _create_non_regular_entry(evidence_path, entry_kind)

    final = family_workflow.family_workflow_status(paths)["final_test"]
    errors = final["integrity_errors"]

    assert (
        final["integrity"],
        final["rerun_permitted"],
        isinstance(errors, list) and bool(errors),
        all(isinstance(error, str) for error in errors),
        any(evidence_path.name in error for error in errors),
    ) == ("invalid", False, True, True, True)


@pytest.mark.parametrize(("evidence_kind", "entry_kind"), NON_REGULAR_CASES)
def test_non_regular_final_evidence_refuses_final_evaluation(
    tmp_path: Path,
    evidence_kind: str,
    entry_kind: str,
) -> None:
    paths = _isolated_paths(tmp_path)
    evidence_path = _evidence_path(paths, evidence_kind)
    _create_non_regular_entry(evidence_path, entry_kind)

    with pytest.raises(family_workflow.FinalTestAlreadyOpenedError):
        family_workflow.assert_final_evaluation_not_opened(paths)


HASH_CASES = (
    pytest.param(
        "result",
        7,
        ["The one-shot marker records a final-result hash that is not a string."],
        id="result-integer",
    ),
    pytest.param(
        "model",
        ["not", "a", "hash"],
        ["The one-shot marker records a model-profile hash that is not a string."],
        id="model-list",
    ),
    pytest.param(
        "development_evidence",
        {"not": "a hash"},
        [
            "Locked development evidence hash is not a string: "
            "docs/family_shortcut_audit.json."
        ],
        id="development-evidence-mapping",
    ),
)


def _write_completed_chain_with_hash_value(
    paths: family_workflow.FamilyWorkflowPaths,
    hash_case: str,
    malformed_hash: Any,
) -> None:
    if hash_case == "development_evidence":
        profile = _read_json(paths.model_profile_path)
        profile["development_evidence"]["shortcut_audit_sha256"] = malformed_hash
        _write_json(paths.model_profile_path, profile)

    actual_profile_hash = _sha256(paths.model_profile_path)
    reported_profile_hash = (
        malformed_hash if hash_case == "model" else actual_profile_hash
    )
    result = {
        "locked_git_commit": "synthetic-locked-commit",
        "model_profile_sha256": reported_profile_hash,
        "test_open_count": 1,
        "test_metrics": {},
        "per_eos": [],
    }
    _write_json(paths.final_test_result_path, result)
    reported_result_hash = (
        malformed_hash
        if hash_case == "result"
        else _sha256(paths.final_test_result_path)
    )
    marker = {
        "status": "COMPLETED",
        "opened_utc": "2026-08-04T11:30:41+00:00",
        "locked_git_commit": "synthetic-locked-commit",
        "model_profile_sha256": reported_profile_hash,
        "result_sha256": reported_result_hash,
    }
    _write_json(paths.final_test_marker_path, marker)


@pytest.mark.parametrize(
    ("hash_case", "malformed_hash", "expected_errors"),
    HASH_CASES,
)
def test_truthy_non_string_hash_is_a_structured_integrity_failure(
    tmp_path: Path,
    hash_case: str,
    malformed_hash: Any,
    expected_errors: list[str],
) -> None:
    paths = _isolated_paths(tmp_path)
    _write_completed_chain_with_hash_value(paths, hash_case, malformed_hash)

    final = family_workflow.family_workflow_status(paths)["final_test"]
    errors = final["integrity_errors"]

    assert final["state"] == "LOCKED_TEST_OPENED"
    assert final["open_count"] == 1
    assert final["marker_status"] == "COMPLETED"
    assert final["integrity"] == "invalid"
    assert final["rerun_permitted"] is False
    assert errors == expected_errors
