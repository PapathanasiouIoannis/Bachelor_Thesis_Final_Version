from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.family_workflow import (
    FamilyWorkflowPaths,
    FinalTestAlreadyOpenedError,
    _file_matches_sha256,
    assert_final_evaluation_not_opened,
    family_workflow_status,
    resolve_family_workflow_paths,
)


ROOT = Path(__file__).resolve().parents[1]
TRACKED_MARKER = (
    ROOT / "data" / "family_pilot_v1" / "family_ml" / "LOCKED_TEST_OPENED.json"
)
PORTABLE_MARKER_SHA256 = (
    "b8aae073cae7bc55699e66471e45b2ce06689244d2f2493337222d48d9c88d1c"
)
EXPECTED_TRACKED_MARKER = {
    "status": "COMPLETED",
    "opened_utc": "2026-08-04T11:30:41.002262+00:00",
    "locked_git_commit": "85e3a26059ed26a7af0b7be38ae02bfbf703ca88",
    "model_profile_sha256": (
        "5ecf468528e5086994497e9f828de9a8bd574df2e5e5ca782bc06c784027c1c2"
    ),
    "result_sha256": (
        "8e8abc7693ed67e03ab81c399995a5b794a3df35a83c4d7307cef77d4e68988b"
    ),
}

LOCKED_COMMIT = "synthetic-locked-commit"
OPENED_UTC = "2026-08-04T11:30:41+00:00"

LOCKED_CONFIGURATION_FILES = (
    "framework/family_pilot_profile.json",
    "framework/family_split_profile.json",
    "framework/family_model_profile.json",
    "docs/family_shortcut_audit.json",
    "docs/family_model_selection.json",
    "docs/family_development_robustness.json",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolated_paths(tmp_path: Path) -> FamilyWorkflowPaths:
    for relative_name in LOCKED_CONFIGURATION_FILES:
        source = ROOT / relative_name
        destination = tmp_path / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    return resolve_family_workflow_paths(
        project_root=tmp_path,
        data_root="data/family_pilot_v1",
        generation_profile_path="framework/family_pilot_profile.json",
        split_profile_path="framework/family_split_profile.json",
        model_profile_path="framework/family_model_profile.json",
        report_dir="docs",
    )


def _base_result(paths: FamilyWorkflowPaths) -> dict[str, Any]:
    return {
        "locked_git_commit": LOCKED_COMMIT,
        "model_profile_sha256": _sha256(paths.model_profile_path),
        "test_open_count": 1,
        "test_metrics": {
            "balanced_accuracy": 0.75,
            "family_balanced_accuracy": 0.75,
            "roc_auc": 0.8,
            "samples": 12,
            "families": 2,
        },
        "per_eos": [],
        "independent_test_family_units": 2,
        "strict_2p08_test_applicable": False,
    }


def _base_marker(paths: FamilyWorkflowPaths, *, status: str) -> dict[str, Any]:
    return {
        "status": status,
        "opened_utc": OPENED_UTC,
        "locked_git_commit": LOCKED_COMMIT,
        "model_profile_sha256": _sha256(paths.model_profile_path),
    }


def _write_completed_chain(
    paths: FamilyWorkflowPaths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _base_result(paths)
    _write_json(paths.final_test_result_path, result)
    marker = _base_marker(paths, status="COMPLETED")
    marker["result_sha256"] = _sha256(paths.final_test_result_path)
    _write_json(paths.final_test_marker_path, marker)
    return result, marker


def _rewrite_result_and_rebind_marker(
    paths: FamilyWorkflowPaths,
    result: dict[str, Any],
    marker: dict[str, Any],
) -> None:
    _write_json(paths.final_test_result_path, result)
    marker["result_sha256"] = _sha256(paths.final_test_result_path)
    _write_json(paths.final_test_marker_path, marker)


def _arrange_evidence(paths: FamilyWorkflowPaths, evidence_state: str) -> None:
    if evidence_state == "absent":
        return
    if evidence_state == "opened_marker_without_result":
        _write_json(paths.final_test_marker_path, _base_marker(paths, status="OPENED"))
        return
    if evidence_state == "completed_marker_without_result":
        _write_json(
            paths.final_test_marker_path,
            _base_marker(paths, status="COMPLETED"),
        )
        return
    if evidence_state == "result_without_marker":
        _write_json(paths.final_test_result_path, _base_result(paths))
        return
    if evidence_state == "malformed_marker":
        paths.final_test_marker_path.parent.mkdir(parents=True, exist_ok=True)
        paths.final_test_marker_path.write_text("{not-json\n", encoding="utf-8")
        return
    if evidence_state == "non_object_marker":
        paths.final_test_marker_path.parent.mkdir(parents=True, exist_ok=True)
        paths.final_test_marker_path.write_text("[]\n", encoding="utf-8")
        return
    if evidence_state == "malformed_result":
        paths.final_test_result_path.parent.mkdir(parents=True, exist_ok=True)
        paths.final_test_result_path.write_text("{not-json\n", encoding="utf-8")
        return
    if evidence_state == "marker_with_malformed_result":
        paths.final_test_result_path.parent.mkdir(parents=True, exist_ok=True)
        paths.final_test_result_path.write_text("{not-json\n", encoding="utf-8")
        marker = _base_marker(paths, status="COMPLETED")
        marker["result_sha256"] = "0" * 64
        _write_json(paths.final_test_marker_path, marker)
        return
    if evidence_state == "completed_chain":
        _write_completed_chain(paths)
        return
    raise AssertionError(f"Unknown evidence state: {evidence_state}")


EVIDENCE_CASES = {
    "absent": {
        "state": "LOCKED_NOT_EVALUATED",
        "integrity": "valid",
        "rerun_permitted": None,
        "open_count": None,
        "marker_status": None,
        "result_present": False,
        "errors": (),
        "guard_error": None,
    },
    "opened_marker_without_result": {
        "state": "LOCKED_TEST_OPENED",
        "integrity": "invalid",
        "rerun_permitted": False,
        "open_count": 1,
        "marker_status": "OPENED",
        "result_present": False,
        "errors": (
            "not in COMPLETED state",
            "has no archived final result",
        ),
        "guard_error": "already opened",
    },
    "completed_marker_without_result": {
        "state": "LOCKED_TEST_OPENED",
        "integrity": "invalid",
        "rerun_permitted": False,
        "open_count": 1,
        "marker_status": "COMPLETED",
        "result_present": False,
        "errors": ("has no archived final result",),
        "guard_error": "already opened",
    },
    "result_without_marker": {
        "state": "INCONSISTENT_RESULT_WITHOUT_MARKER",
        "integrity": "invalid",
        "rerun_permitted": False,
        "open_count": 1,
        "marker_status": None,
        "result_present": True,
        "errors": ("final result exists without the one-shot marker",),
        "guard_error": "result already exists without its marker",
    },
    "malformed_marker": {
        "state": "LOCKED_NOT_EVALUATED",
        "integrity": "invalid",
        "rerun_permitted": False,
        "open_count": None,
        "marker_status": None,
        "result_present": False,
        "errors": ("Could not read LOCKED_TEST_OPENED.json",),
        "guard_error": "marker exists but is unreadable",
    },
    "non_object_marker": {
        "state": "LOCKED_NOT_EVALUATED",
        "integrity": "invalid",
        "rerun_permitted": False,
        "open_count": None,
        "marker_status": None,
        "result_present": False,
        "errors": ("LOCKED_TEST_OPENED.json must contain a JSON object",),
        "guard_error": "marker exists but is unreadable",
    },
    "malformed_result": {
        "state": "LOCKED_NOT_EVALUATED",
        "integrity": "invalid",
        "rerun_permitted": False,
        "open_count": None,
        "marker_status": None,
        "result_present": False,
        "errors": ("Could not read family_final_test.json",),
        "guard_error": "result already exists without its marker",
    },
    "marker_with_malformed_result": {
        "state": "LOCKED_TEST_OPENED",
        "integrity": "invalid",
        "rerun_permitted": False,
        "open_count": 1,
        "marker_status": "COMPLETED",
        "result_present": False,
        "errors": (
            "Could not read family_final_test.json",
            "has no archived final result",
        ),
        "guard_error": "already opened",
    },
    "completed_chain": {
        "state": "LOCKED_TEST_OPENED",
        "integrity": "valid",
        "rerun_permitted": False,
        "open_count": 1,
        "marker_status": "COMPLETED",
        "result_present": True,
        "errors": (),
        "guard_error": "already opened",
    },
}


@pytest.mark.parametrize("evidence_state", EVIDENCE_CASES)
def test_final_evidence_state_matrix(tmp_path: Path, evidence_state: str) -> None:
    paths = _isolated_paths(tmp_path)
    _arrange_evidence(paths, evidence_state)
    expected = EVIDENCE_CASES[evidence_state]

    final = family_workflow_status(paths)["final_test"]

    assert final["state"] == expected["state"]
    assert final["integrity"] == expected["integrity"]
    assert final["rerun_permitted"] is expected["rerun_permitted"]
    assert final["open_count"] == expected["open_count"]
    assert final["marker_status"] == expected["marker_status"]
    assert (final["result"] is not None) is expected["result_present"]
    assert len(final["integrity_errors"]) == len(expected["errors"])
    for error, expected_fragment in zip(
        final["integrity_errors"], expected["errors"], strict=True
    ):
        assert expected_fragment in error

    if expected["result_present"]:
        assert Path(final["result"]["path"]) == paths.final_test_result_path

    guard_error = expected["guard_error"]
    if guard_error is None:
        assert assert_final_evaluation_not_opened(paths) is None
    else:
        with pytest.raises(FinalTestAlreadyOpenedError, match=guard_error):
            assert_final_evaluation_not_opened(paths)


INTEGRITY_CASES = {
    "missing_result_hash": (
        "The one-shot marker does not record a final-result hash.",
    ),
    "mismatched_result_hash": (
        "The final result hash does not match the one-shot marker.",
    ),
    "mismatched_commit": ("The final result and marker differ in locked_git_commit.",),
    "mismatched_reported_profile": (
        "The final result and marker differ in model_profile_sha256.",
    ),
    "wrong_open_count": ("The final result does not record exactly one test opening.",),
    "changed_locked_profile": (
        "The locked model profile hash does not match the marker.",
    ),
    "missing_marker_profile_hash": (
        "The final result and marker differ in model_profile_sha256.",
        "The one-shot marker does not record a model-profile hash.",
    ),
    "unsafe_development_evidence": (
        "Locked development evidence path is unsafe: ../outside.json.",
    ),
    "missing_development_evidence": (
        "Locked development evidence is missing: docs/missing_shortcut_audit.json.",
    ),
    "hashless_development_evidence": (
        "Locked development evidence has no hash: docs/family_shortcut_audit.json.",
    ),
    "changed_development_evidence": (
        "Locked development evidence hash changed: docs/family_shortcut_audit.json.",
    ),
}


def _prepare_profile_mutation(paths: FamilyWorkflowPaths, mutation: str) -> None:
    if mutation not in {
        "unsafe_development_evidence",
        "missing_development_evidence",
        "hashless_development_evidence",
    }:
        return

    profile = _read_json(paths.model_profile_path)
    evidence = profile["development_evidence"]
    if mutation == "unsafe_development_evidence":
        evidence["shortcut_audit_path"] = "../outside.json"
    elif mutation == "missing_development_evidence":
        evidence["shortcut_audit_path"] = "docs/missing_shortcut_audit.json"
    else:
        evidence.pop("shortcut_audit_sha256")
    _write_json(paths.model_profile_path, profile)


def _apply_integrity_mutation(
    paths: FamilyWorkflowPaths,
    mutation: str,
    result: dict[str, Any],
    marker: dict[str, Any],
) -> None:
    if mutation == "missing_result_hash":
        marker.pop("result_sha256")
        _write_json(paths.final_test_marker_path, marker)
    elif mutation == "mismatched_result_hash":
        marker["result_sha256"] = "0" * 64
        _write_json(paths.final_test_marker_path, marker)
    elif mutation == "mismatched_commit":
        result["locked_git_commit"] = "different-commit"
        _rewrite_result_and_rebind_marker(paths, result, marker)
    elif mutation == "mismatched_reported_profile":
        result["model_profile_sha256"] = "1" * 64
        _rewrite_result_and_rebind_marker(paths, result, marker)
    elif mutation == "wrong_open_count":
        result["test_open_count"] = 2
        _rewrite_result_and_rebind_marker(paths, result, marker)
    elif mutation == "changed_locked_profile":
        profile = _read_json(paths.model_profile_path)
        profile["claim_boundary"] += " Deliberate test mutation."
        _write_json(paths.model_profile_path, profile)
    elif mutation == "missing_marker_profile_hash":
        marker.pop("model_profile_sha256")
        _write_json(paths.final_test_marker_path, marker)
    elif mutation == "changed_development_evidence":
        evidence_path = paths.report_dir / "family_shortcut_audit.json"
        evidence = _read_json(evidence_path)
        evidence["test_only_mutation"] = True
        _write_json(evidence_path, evidence)


@pytest.mark.parametrize("mutation", INTEGRITY_CASES)
def test_completed_evidence_integrity_matrix(tmp_path: Path, mutation: str) -> None:
    paths = _isolated_paths(tmp_path)
    _prepare_profile_mutation(paths, mutation)
    result, marker = _write_completed_chain(paths)
    _apply_integrity_mutation(paths, mutation, result, marker)

    final = family_workflow_status(paths)["final_test"]

    assert final["state"] == "LOCKED_TEST_OPENED"
    assert final["open_count"] == (2 if mutation == "wrong_open_count" else 1)
    assert final["marker_status"] == "COMPLETED"
    assert final["integrity"] == "invalid"
    assert final["integrity_errors"] == list(INTEGRITY_CASES[mutation])
    assert final["rerun_permitted"] is False
    with pytest.raises(FinalTestAlreadyOpenedError, match="already opened"):
        assert_final_evaluation_not_opened(paths)


def test_checked_in_final_marker_is_the_pinned_semantic_trust_anchor() -> None:
    raw = TRACKED_MARKER.read_bytes()
    newline_normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

    assert json.loads(raw) == EXPECTED_TRACKED_MARKER
    assert hashlib.sha256(newline_normalized).hexdigest() == PORTABLE_MARKER_SHA256
    assert _file_matches_sha256(TRACKED_MARKER, PORTABLE_MARKER_SHA256)
