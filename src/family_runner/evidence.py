"""Read-only evidence and artifact summaries for the family workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol


JsonStatusReader = Callable[
    [Path],
    tuple[dict[str, Any] | None, str | None],
]
HashMatcher = Callable[[Path, str], bool]
ArtifactStatusBuilder = Callable[[list[Path]], dict[str, Any]]


class _EvidencePaths(Protocol):
    @property
    def project_root(self) -> Path: ...

    @property
    def data_root(self) -> Path: ...

    @property
    def model_profile_path(self) -> Path: ...

    @property
    def report_dir(self) -> Path: ...

    @property
    def family_ml_dir(self) -> Path: ...

    @property
    def final_test_marker_path(self) -> Path: ...

    @property
    def final_test_result_path(self) -> Path: ...


def read_json_status(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        path.lstat()
        is_file = path.is_file()
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"Could not inspect {path.name}: {exc}"
    if not is_file:
        return None, f"Could not read {path.name}: expected a regular file."
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"Could not read {path.name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path.name} must contain a JSON object."
    return payload, None


def file_matches_sha256(path: Path, expected_hash: str) -> bool:
    """Match a JSON artifact hash without treating Git newlines as mutations.

    The frozen family evidence was recorded on Windows. Git may materialize the
    same tracked JSON with LF newlines on Linux, so compare the raw digest and
    the two conventional newline representations. No JSON values, spacing, or
    key order are canonicalized; any scientific-content change still fails.
    """

    if not isinstance(expected_hash, str):
        return False
    try:
        raw = path.read_bytes()
    except OSError:
        return False
    candidates = {hashlib.sha256(raw).hexdigest()}
    if path.suffix.casefold() == ".json":
        lf_bytes = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
        candidates.add(hashlib.sha256(lf_bytes).hexdigest())
        candidates.add(hashlib.sha256(crlf_bytes).hexdigest())
    return expected_hash.casefold() in candidates


def artifact_status(paths: list[Path]) -> dict[str, Any]:
    missing = [str(path) for path in paths if not path.is_file()]
    return {
        "state": "ready" if not missing else "missing",
        "paths": [str(path) for path in paths],
        "missing": missing,
    }


def development_artifacts(
    paths: _EvidencePaths,
    *,
    artifact_status: ArtifactStatusBuilder,
) -> dict[str, dict[str, Any]]:
    ml_dir = paths.family_ml_dir
    return {
        "generation": artifact_status(
            [
                paths.data_root / "physics_test_dataset.parquet",
                paths.data_root / "family_catalog.parquet",
                paths.data_root / "run_manifest.json",
            ]
        ),
        "curve_preparation": artifact_status(
            [
                ml_dir / "curve_samples.parquet",
                ml_dir / "sample_audit.parquet",
                ml_dir / "split_manifest.parquet",
                ml_dir / "feature_manifest.json",
                ml_dir / "train.parquet",
                ml_dir / "val.parquet",
                ml_dir / "test.parquet",
            ]
        ),
        "shortcut_audit": artifact_status(
            [
                paths.report_dir / "family_shortcut_audit.json",
                paths.report_dir / "FAMILY_SHORTCUT_AUDIT.md",
            ]
        ),
        "model_selection": artifact_status(
            [
                paths.report_dir / "family_model_selection.json",
                paths.report_dir / "FAMILY_MODEL_SELECTION.md",
                ml_dir / "development_predictions.parquet",
            ]
        ),
        "robustness": artifact_status(
            [
                paths.report_dir / "family_development_robustness.json",
                paths.report_dir / "FAMILY_DEVELOPMENT_ROBUSTNESS.md",
            ]
        ),
    }


def development_evidence_summary(
    paths: _EvidencePaths,
    *,
    read_json_status: JsonStatusReader,
) -> dict[str, Any]:
    shortcut, shortcut_error = read_json_status(
        paths.report_dir / "family_shortcut_audit.json"
    )
    selection, selection_error = read_json_status(
        paths.report_dir / "family_model_selection.json"
    )
    robustness, robustness_error = read_json_status(
        paths.report_dir / "family_development_robustness.json"
    )
    selected = selection.get("selected_candidate", {}) if selection else {}
    permutation = (
        robustness.get("family_label_permutation_null", {}) if robustness else {}
    )
    errors = [
        error
        for error in (shortcut_error, selection_error, robustness_error)
        if error is not None
    ]
    return {
        "shortcut_audit_passed": shortcut.get("passed") if shortcut else None,
        "selected_reporting_model": selected.get("candidate_id"),
        "selected_model_family": "logistic regression"
        if str(selected.get("candidate_id", "")).startswith("logistic_")
        else None,
        "family_permutation_empirical_p_value": permutation.get("empirical_p_value"),
        "test_rows_used_by_development": {
            "shortcut_audit": shortcut.get("test_rows_used") if shortcut else None,
            "robustness": robustness.get("test_rows_used") if robustness else None,
        },
        "read_errors": errors,
    }


def final_test_status(
    paths: _EvidencePaths,
    model_profile: dict[str, Any] | None = None,
    *,
    read_json_status: JsonStatusReader,
    file_matches_sha256: HashMatcher,
    model_set_claim: str,
) -> dict[str, Any]:
    marker, marker_error = read_json_status(paths.final_test_marker_path)
    result, result_error = read_json_status(paths.final_test_result_path)
    final_evidence_present = any(
        value is not None for value in (marker, marker_error, result, result_error)
    )
    errors = [error for error in (marker_error, result_error) if error is not None]

    if marker is not None:
        state = "LOCKED_TEST_OPENED"
        if marker.get("status") != "COMPLETED":
            errors.append("The one-shot marker is not in COMPLETED state.")
    elif result is not None:
        state = "INCONSISTENT_RESULT_WITHOUT_MARKER"
        errors.append("A final result exists without the one-shot marker.")
    else:
        state = "LOCKED_NOT_EVALUATED"

    if marker is not None and result is not None:
        expected_hash = marker.get("result_sha256")
        if not expected_hash:
            errors.append("The one-shot marker does not record a final-result hash.")
        elif not isinstance(expected_hash, str):
            errors.append(
                "The one-shot marker records a final-result hash that is not a string."
            )
        elif not file_matches_sha256(paths.final_test_result_path, expected_hash):
            errors.append("The final result hash does not match the one-shot marker.")
        for key in ("locked_git_commit", "model_profile_sha256"):
            if marker.get(key) != result.get(key):
                errors.append(f"The final result and marker differ in {key}.")
        if result.get("test_open_count") != 1:
            errors.append("The final result does not record exactly one test opening.")
    elif marker is not None:
        errors.append("The completed one-shot marker has no archived final result.")

    if marker is not None:
        expected_profile_hash = marker.get("model_profile_sha256")
        if not expected_profile_hash:
            errors.append("The one-shot marker does not record a model-profile hash.")
        elif not isinstance(expected_profile_hash, str):
            errors.append(
                "The one-shot marker records a model-profile hash that is not a string."
            )
        elif not file_matches_sha256(paths.model_profile_path, expected_profile_hash):
            errors.append("The locked model profile hash does not match the marker.")

    if model_profile is not None:
        for evidence_name, record in (
            model_profile.get("development_evidence", {}) or {}
        ).items():
            if not evidence_name.endswith("_path"):
                continue
            prefix = evidence_name.removesuffix("_path")
            expected_hash = model_profile["development_evidence"].get(
                f"{prefix}_sha256"
            )
            evidence_path = (paths.project_root / str(record)).resolve()
            if not evidence_path.is_relative_to(paths.project_root):
                errors.append(f"Locked development evidence path is unsafe: {record}.")
            elif not evidence_path.is_file():
                errors.append(f"Locked development evidence is missing: {record}.")
            elif not expected_hash:
                errors.append(f"Locked development evidence has no hash: {record}.")
            elif not isinstance(expected_hash, str):
                errors.append(
                    f"Locked development evidence hash is not a string: {record}."
                )
            elif not file_matches_sha256(evidence_path, expected_hash):
                errors.append(f"Locked development evidence hash changed: {record}.")

    result_summary = None
    if result is not None:
        metrics = result.get("test_metrics", {})
        per_family = []
        for record in result.get("per_eos", []):
            if not isinstance(record, dict):
                continue
            label = record.get("Label")
            matter_type = "hadronic" if label == 0 else "quark" if label == 1 else None
            per_family.append(
                {
                    "eos_id": record.get("EoS_ID"),
                    "matter_type": matter_type,
                    "curves": record.get("curves"),
                    "accuracy": record.get("accuracy"),
                    "mean_quark_class_model_score": record.get(
                        "mean_probability_quark"
                    ),
                    "model_score_range_across_amplitude": record.get(
                        "probability_range_across_A"
                    ),
                    "minimum_maximum_mass_msun": record.get("minimum_mmax_msun"),
                }
            )
        result_summary = {
            "path": str(paths.final_test_result_path),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "family_balanced_accuracy": metrics.get("family_balanced_accuracy"),
            "roc_auc": metrics.get("roc_auc"),
            "samples": metrics.get("samples"),
            "families": metrics.get("families"),
            "independent_family_units": result.get("independent_test_family_units"),
            "strict_2p08_test_applicable": result.get("strict_2p08_test_applicable"),
            "per_family": per_family,
            "claim_boundary": result.get("claim_boundary", model_set_claim),
            "interpretation": model_set_claim,
        }

    open_count = result.get("test_open_count") if result is not None else None
    if marker is not None and open_count is None:
        # Creation of this exclusive marker is the opening event. A missing result
        # means that the one allowed attempt may have stopped before completion.
        open_count = 1
    return {
        "state": state,
        "open_count": open_count,
        "marker_status": marker.get("status") if marker else None,
        "opened_utc": marker.get("opened_utc") if marker else None,
        "marker_path": str(paths.final_test_marker_path),
        "result": result_summary,
        "integrity": "valid" if not errors else "invalid",
        "integrity_errors": errors,
        "rerun_permitted": False if final_evidence_present else None,
    }


__all__ = [
    "artifact_status",
    "development_artifacts",
    "development_evidence_summary",
    "file_matches_sha256",
    "final_test_status",
    "read_json_status",
]
