"""Safe orchestration and status reporting for the family-classification pilot.

This module deliberately exposes development-only execution.  The one-time final
test remains owned by ``family_final_test.py`` and is never launched from here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from framework.family_pilot import (
    DEFAULT_PROFILE_PATH,
    load_family_pilot_profile,
    profile_entries,
)
from src.ml.family_final import DEFAULT_MODEL_PROFILE, load_locked_model_profile
from src.ml.family_splitting import (
    DEFAULT_FAMILY_SPLIT_PROFILE,
    load_family_split_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILY_DATA_ROOT = PROJECT_ROOT / "data" / "family_pilot_v1"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "docs"

MODEL_SET_CLAIM = (
    "Repository hadronic-surrogate versus analytic CFL MIT-bag model-set "
    "discrimination on unseen EoS families; not universal matter-phase "
    "identification and not observational deployment."
)
SUPPORTED_REPORTING_MODELS = ("dummy baseline", "logistic regression")
EXPLORATORY_MODELS = ("XGBoost", "multilayer perceptron")


class FamilyWorkflowError(RuntimeError):
    """Base error for safe family-workflow orchestration."""


class FamilyDevelopmentStageError(FamilyWorkflowError):
    """A development stage exited unsuccessfully."""

    def __init__(
        self,
        *,
        stage: str,
        command: tuple[str, ...],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.stage = stage
        self.command = command
        self.returncode = int(returncode)
        self.stdout = stdout
        self.stderr = stderr
        detail = (
            stderr.strip() or stdout.strip() or "No diagnostic output was produced."
        )
        super().__init__(
            f"Family development stage '{stage}' failed with exit code "
            f"{returncode}: {detail}"
        )


class FinalTestAlreadyOpenedError(FamilyWorkflowError):
    """A request would reopen the immutable family test set."""


@dataclass(frozen=True)
class FamilyWorkflowPaths:
    """Resolved paths used by the family-development wrapper."""

    project_root: Path
    data_root: Path
    generation_profile_path: Path
    split_profile_path: Path
    model_profile_path: Path
    report_dir: Path

    @property
    def family_ml_dir(self) -> Path:
        return self.data_root / "family_ml"

    @property
    def final_test_marker_path(self) -> Path:
        return self.family_ml_dir / "LOCKED_TEST_OPENED.json"

    @property
    def final_test_result_path(self) -> Path:
        return self.report_dir / "family_final_test.json"


def resolve_family_workflow_paths(
    *,
    data_root: str | Path = DEFAULT_FAMILY_DATA_ROOT,
    generation_profile_path: str | Path = DEFAULT_PROFILE_PATH,
    split_profile_path: str | Path = DEFAULT_FAMILY_SPLIT_PROFILE,
    model_profile_path: str | Path = DEFAULT_MODEL_PROFILE,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    project_root: str | Path = PROJECT_ROOT,
) -> FamilyWorkflowPaths:
    """Return absolute, normalized paths without changing the filesystem."""

    root = Path(project_root).expanduser().resolve()

    def resolved(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    return FamilyWorkflowPaths(
        project_root=root,
        data_root=resolved(data_root),
        generation_profile_path=resolved(generation_profile_path),
        split_profile_path=resolved(split_profile_path),
        model_profile_path=resolved(model_profile_path),
        report_dir=resolved(report_dir),
    )


def _load_profiles(paths: FamilyWorkflowPaths) -> tuple[dict, dict, dict]:
    generation = load_family_pilot_profile(paths.generation_profile_path)
    split = load_family_split_profile(paths.split_profile_path)
    model = load_locked_model_profile(paths.model_profile_path)

    if split["expected_generation_profile"] != generation["profile_id"]:
        raise ValueError(
            "The family split profile expects generation profile "
            f"'{split['expected_generation_profile']}', but "
            f"'{generation['profile_id']}' was selected."
        )
    identity = model["data_identity"]
    if identity["generation_profile_id"] != generation["profile_id"]:
        raise ValueError(
            "The locked model profile refers to another generation profile."
        )
    if identity["split_profile_id"] != split["profile_id"]:
        raise ValueError(
            "The locked model profile refers to another family split profile."
        )
    return generation, split, model


def _read_json_status(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"Could not read {path.name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path.name} must contain a JSON object."
    return payload, None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_status(paths: list[Path]) -> dict[str, Any]:
    missing = [str(path) for path in paths if not path.is_file()]
    return {
        "state": "ready" if not missing else "missing",
        "paths": [str(path) for path in paths],
        "missing": missing,
    }


def _development_artifacts(paths: FamilyWorkflowPaths) -> dict[str, dict[str, Any]]:
    ml_dir = paths.family_ml_dir
    return {
        "generation": _artifact_status(
            [
                paths.data_root / "physics_test_dataset.parquet",
                paths.data_root / "family_catalog.parquet",
                paths.data_root / "run_manifest.json",
            ]
        ),
        "curve_preparation": _artifact_status(
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
        "shortcut_audit": _artifact_status(
            [
                paths.report_dir / "family_shortcut_audit.json",
                paths.report_dir / "FAMILY_SHORTCUT_AUDIT.md",
            ]
        ),
        "model_selection": _artifact_status(
            [
                paths.report_dir / "family_model_selection.json",
                paths.report_dir / "FAMILY_MODEL_SELECTION.md",
                ml_dir / "development_predictions.parquet",
            ]
        ),
        "robustness": _artifact_status(
            [
                paths.report_dir / "family_development_robustness.json",
                paths.report_dir / "FAMILY_DEVELOPMENT_ROBUSTNESS.md",
            ]
        ),
    }


def _family_split_summary(generation: dict, split: dict) -> dict[str, Any]:
    amplitudes = generation["deformation"]["amplitudes"]
    entries = profile_entries(generation)
    entries_by_group: dict[str, list[Any]] = {}
    for entry in entries:
        entries_by_group.setdefault(str(entry.family_group_id), []).append(entry)

    summary: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        group_ids = [str(value) for value in split["splits"][split_name]]
        hadronic_groups = [value for value in group_ids if value.startswith("H_")]
        quark_groups = [value for value in group_ids if value.startswith("Q_")]
        eos_count = sum(
            len(entries_by_group.get(group_id, [])) for group_id in group_ids
        )
        summary[split_name] = {
            "family_groups": len(group_ids),
            "hadronic_family_groups": len(hadronic_groups),
            "quark_family_groups": len(quark_groups),
            "eos_baselines": eos_count,
            "expected_curves": eos_count * len(amplitudes),
            "group_ids": group_ids,
        }
    return summary


def _development_evidence_summary(paths: FamilyWorkflowPaths) -> dict[str, Any]:
    shortcut, shortcut_error = _read_json_status(
        paths.report_dir / "family_shortcut_audit.json"
    )
    selection, selection_error = _read_json_status(
        paths.report_dir / "family_model_selection.json"
    )
    robustness, robustness_error = _read_json_status(
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


def _final_test_status(
    paths: FamilyWorkflowPaths, model_profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    marker, marker_error = _read_json_status(paths.final_test_marker_path)
    result, result_error = _read_json_status(paths.final_test_result_path)
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
        elif expected_hash != _file_sha256(paths.final_test_result_path):
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
        elif expected_profile_hash != _file_sha256(paths.model_profile_path):
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
            elif _file_sha256(evidence_path) != expected_hash:
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
            "claim_boundary": result.get("claim_boundary", MODEL_SET_CLAIM),
            "interpretation": MODEL_SET_CLAIM,
        }

    open_count = result.get("test_open_count") if result is not None else None
    if marker is not None and open_count is None:
        # Creation of this exclusive marker is the opening event.  A missing result
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
        "rerun_permitted": False if marker is not None else None,
    }


def family_workflow_status(
    paths: FamilyWorkflowPaths | None = None,
    **path_options: str | Path,
) -> dict[str, Any]:
    """Summarize the family pilot without loading or scoring final-test tensors."""

    if paths is not None and path_options:
        raise TypeError("Pass either 'paths' or individual path options, not both.")
    resolved = paths or resolve_family_workflow_paths(**path_options)
    generation, split, model = _load_profiles(resolved)
    amplitudes = [float(value) for value in generation["deformation"]["amplitudes"]]
    split_summary = _family_split_summary(generation, split)
    return {
        "workflow": "family_classification",
        "scientific_scope": MODEL_SET_CLAIM,
        "generation_profile": {
            "profile_id": generation["profile_id"],
            "path": str(resolved.generation_profile_path),
            "hadronic_eos_baselines": len(generation["hadronic_eos_ids"]),
            "quark_eos_baselines": len(generation["quark_eos_ids"]),
            "family_groups": len(
                {str(entry.family_group_id) for entry in profile_entries(generation)}
            ),
            "deformation": {
                "amplitude_symbol": "A",
                "amplitudes": amplitudes,
                "center_energy_density_symbol": "epsilon_0",
                "center_energy_density_mev_fm3": float(
                    generation["deformation"]["epsilon0_mev_fm3"]
                ),
                "width_symbol": "sigma",
                "width_mev_fm3": float(generation["deformation"]["sigma_mev_fm3"]),
            },
            "expected_curves": (
                len(generation["hadronic_eos_ids"]) + len(generation["quark_eos_ids"])
            )
            * len(amplitudes),
        },
        "split_profile": {
            "profile_id": split["profile_id"],
            "path": str(resolved.split_profile_path),
            "primary_split_unit": "physical EoS family",
            "splits": split_summary,
        },
        "reporting_model_policy": {
            "supported": list(SUPPORTED_REPORTING_MODELS),
            "exploratory_not_run_by_workflow": list(EXPLORATORY_MODELS),
            "locked_model_profile_id": model["profile_id"],
        },
        "development_artifacts": _development_artifacts(resolved),
        "development_evidence": _development_evidence_summary(resolved),
        "final_test": _final_test_status(resolved, model),
    }


def assert_final_evaluation_not_opened(
    paths: FamilyWorkflowPaths | None = None,
    **path_options: str | Path,
) -> None:
    """Refuse a final-evaluation request after the exclusive marker is created."""

    if paths is not None and path_options:
        raise TypeError("Pass either 'paths' or individual path options, not both.")
    resolved = paths or resolve_family_workflow_paths(**path_options)
    marker, marker_error = _read_json_status(resolved.final_test_marker_path)
    if marker_error:
        raise FinalTestAlreadyOpenedError(
            "The final-test marker exists but is unreadable; fail-closed refusal: "
            f"{marker_error}"
        )
    if marker is not None:
        opened = marker.get("opened_utc", "an unrecorded time")
        status = marker.get("status", "UNKNOWN")
        raise FinalTestAlreadyOpenedError(
            "The locked family test was already opened at "
            f"{opened} (marker status: {status}). It cannot be opened or scored again. "
            f"Review the recorded result at {resolved.final_test_result_path}."
        )
    if resolved.final_test_result_path.exists():
        raise FinalTestAlreadyOpenedError(
            "A final family-test result already exists without its marker. "
            "Refusing evaluation until this inconsistent state is investigated."
        )


def _development_commands(
    paths: FamilyWorkflowPaths,
    *,
    jobs: int,
    force_regenerate: bool,
    permutations: int,
) -> list[tuple[str, tuple[str, ...]]]:
    python = sys.executable
    commands: list[tuple[str, tuple[str, ...]]] = [
        (
            "generation",
            (
                python,
                str(paths.project_root / "family_physics_main.py"),
                "--data-root",
                str(paths.data_root),
                "--profile",
                str(paths.generation_profile_path),
                "--n-jobs",
                str(jobs),
            ),
        ),
        (
            "curve_preparation",
            (
                python,
                str(paths.project_root / "family_ml_prepare.py"),
                "--data-root",
                str(paths.data_root),
                "--generation-profile",
                str(paths.generation_profile_path),
                "--split-profile",
                str(paths.split_profile_path),
            ),
        ),
        (
            "shortcut_audit",
            (
                python,
                str(paths.project_root / "family_shortcut_audit.py"),
                "--data-root",
                str(paths.data_root),
                "--output-dir",
                str(paths.report_dir),
            ),
        ),
        (
            "model_selection",
            (
                python,
                str(paths.project_root / "family_model_select.py"),
                "--data-root",
                str(paths.data_root),
                "--output-dir",
                str(paths.report_dir),
            ),
        ),
        (
            "robustness",
            (
                python,
                str(paths.project_root / "family_development_robustness.py"),
                "--data-root",
                str(paths.data_root),
                "--output-dir",
                str(paths.report_dir),
                "--permutations",
                str(permutations),
            ),
        ),
    ]
    if force_regenerate:
        stage, command = commands[0]
        commands[0] = (stage, (*command, "--force-regenerate"))
    return commands


def _refuse_final_named_stage(stage: str) -> None:
    normalized = stage.strip().lower().replace("-", "_")
    if "final" in normalized or normalized in {"test", "evaluation", "score_test"}:
        raise FamilyWorkflowError(
            "The unified family workflow supports development stages only. "
            "It never opens or scores the locked final test."
        )


def run_family_development(
    paths: FamilyWorkflowPaths | None = None,
    *,
    jobs: int = 1,
    force_regenerate: bool = False,
    permutations: int = 0,
    requested_stage: str = "development",
    **path_options: str | Path,
) -> dict[str, Any]:
    """Run generation through robustness, never the locked final evaluation.

    All stages execute in separate Python processes so their existing command-line
    contracts stay authoritative.  Standard output and errors are captured and a
    failed stage stops the sequence immediately.
    """

    if paths is not None and path_options:
        raise TypeError("Pass either 'paths' or individual path options, not both.")
    _refuse_final_named_stage(requested_stage)
    if int(jobs) < 1:
        raise ValueError("parallel jobs must be at least 1.")
    if int(permutations) < 0:
        raise ValueError("permutations must be 0 or a positive integer.")
    resolved = paths or resolve_family_workflow_paths(**path_options)
    generation, split, model = _load_profiles(resolved)
    del generation, split
    final_before = _final_test_status(resolved, model)
    if final_before["state"] == "LOCKED_TEST_OPENED":
        raise FinalTestAlreadyOpenedError(
            "Family development outputs are frozen because the locked final test has "
            "already been opened. This command will not overwrite post-test evidence. "
            "Create a separately versioned future experiment instead."
        )

    commands = _development_commands(
        resolved,
        jobs=int(jobs),
        force_regenerate=bool(force_regenerate),
        permutations=int(permutations),
    )
    completed_stages: list[dict[str, Any]] = []
    for stage, command in commands:
        script = Path(command[1])
        if not script.is_file():
            raise FileNotFoundError(
                f"Family development stage '{stage}' is missing its entrypoint: {script}"
            )
        completed = subprocess.run(
            command,
            cwd=resolved.project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise FamilyDevelopmentStageError(
                stage=stage,
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        completed_stages.append(
            {
                "stage": stage,
                "state": "completed",
                "command": list(command),
                "stdout": completed.stdout.strip(),
            }
        )

    return {
        "workflow": "family_development",
        "scientific_scope": MODEL_SET_CLAIM,
        "completed_stages": completed_stages,
        "final_test_accessed": False,
        "final_test_state_before_run": final_before["state"],
        "post_test_context": final_before["state"] == "LOCKED_TEST_OPENED",
        "status": family_workflow_status(resolved),
    }


def refuse_final_evaluation_request(
    paths: FamilyWorkflowPaths | None = None,
    **path_options: str | Path,
) -> NoReturn:
    """Fail closed: final evaluation is intentionally outside this wrapper."""

    assert_final_evaluation_not_opened(paths, **path_options)
    raise FamilyWorkflowError(
        "The unified family workflow intentionally exposes development-only "
        "execution. The final test must remain a separately governed, one-time action."
    )


__all__ = [
    "EXPLORATORY_MODELS",
    "MODEL_SET_CLAIM",
    "SUPPORTED_REPORTING_MODELS",
    "FamilyDevelopmentStageError",
    "FamilyWorkflowError",
    "FamilyWorkflowPaths",
    "FinalTestAlreadyOpenedError",
    "assert_final_evaluation_not_opened",
    "family_workflow_status",
    "refuse_final_evaluation_request",
    "resolve_family_workflow_paths",
    "run_family_development",
]
