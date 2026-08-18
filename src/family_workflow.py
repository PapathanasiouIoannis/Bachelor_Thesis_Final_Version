"""Safe orchestration and status reporting for the family-classification pilot.

This module deliberately exposes development-only execution.  The one-time final
test remains owned by ``family_final_test.py`` and is never launched from here.
"""

from __future__ import annotations

import hashlib
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
from src.family_runner import evidence as _evidence
from src.family_runner import status as _status


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
    return _status.load_profiles(
        paths,
        load_generation_profile=load_family_pilot_profile,
        load_split_profile=load_family_split_profile,
        load_model_profile=load_locked_model_profile,
        file_matches_sha256=_file_matches_sha256,
    )


def _read_json_status(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    return _evidence.read_json_status(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_matches_sha256(path: Path, expected_hash: str) -> bool:
    """Match a JSON artifact hash without treating Git newlines as mutations.

    The frozen family evidence was recorded on Windows. Git may materialize the
    same tracked JSON with LF newlines on Linux, so compare the raw digest and
    the two conventional newline representations. No JSON values, spacing, or
    key order are canonicalized; any scientific-content change still fails.
    """

    return _evidence.file_matches_sha256(path, expected_hash)


def _artifact_status(paths: list[Path]) -> dict[str, Any]:
    return _evidence.artifact_status(paths)


def _development_artifacts(paths: FamilyWorkflowPaths) -> dict[str, dict[str, Any]]:
    return _evidence.development_artifacts(
        paths,
        artifact_status=_artifact_status,
    )


def _family_split_summary(generation: dict, split: dict) -> dict[str, Any]:
    return _status.family_split_summary(
        generation,
        split,
        profile_entries=profile_entries,
    )


def _development_evidence_summary(paths: FamilyWorkflowPaths) -> dict[str, Any]:
    return _evidence.development_evidence_summary(
        paths,
        read_json_status=_read_json_status,
    )


def _final_test_status(
    paths: FamilyWorkflowPaths, model_profile: dict[str, Any] | None = None
) -> dict[str, Any]:
    return _evidence.final_test_status(
        paths,
        model_profile,
        read_json_status=_read_json_status,
        file_matches_sha256=_file_matches_sha256,
        model_set_claim=MODEL_SET_CLAIM,
    )


def family_workflow_status(
    paths: FamilyWorkflowPaths | None = None,
    **path_options: str | Path,
) -> dict[str, Any]:
    """Summarize the family pilot without loading or scoring final-test tensors."""

    if paths is not None and path_options:
        raise TypeError("Pass either 'paths' or individual path options, not both.")
    resolved = paths or resolve_family_workflow_paths(**path_options)
    return _status.family_workflow_status(
        resolved,
        load_profiles=_load_profiles,
        family_split_summary=_family_split_summary,
        profile_entries=profile_entries,
        development_artifacts=_development_artifacts,
        development_evidence_summary=_development_evidence_summary,
        final_test_status=_final_test_status,
        model_set_claim=MODEL_SET_CLAIM,
        supported_reporting_models=SUPPORTED_REPORTING_MODELS,
        exploratory_models=EXPLORATORY_MODELS,
    )


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
    _, result_error = _read_json_status(resolved.final_test_result_path)
    if result_error:
        raise FinalTestAlreadyOpenedError(
            "The final-test result path exists but is unreadable; fail-closed "
            f"refusal: {result_error}"
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
    if final_before["rerun_permitted"] is False:
        details = "; ".join(final_before["integrity_errors"])
        raise FinalTestAlreadyOpenedError(
            "Family development is blocked because final-test evidence is present but "
            "is incomplete, unreadable, or inconsistent; fail-closed refusal. "
            f"Investigate the recorded evidence before continuing. Details: {details}"
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
