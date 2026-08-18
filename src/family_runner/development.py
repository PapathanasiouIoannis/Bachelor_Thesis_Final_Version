"""Pure development-command planning for the family workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class _DevelopmentPaths(Protocol):
    @property
    def project_root(self) -> Path: ...

    @property
    def data_root(self) -> Path: ...

    @property
    def generation_profile_path(self) -> Path: ...

    @property
    def split_profile_path(self) -> Path: ...

    @property
    def report_dir(self) -> Path: ...


def development_commands(
    paths: _DevelopmentPaths,
    *,
    jobs: int,
    force_regenerate: bool,
    permutations: int,
    python_executable: str,
) -> list[tuple[str, tuple[str, ...]]]:
    python = python_executable
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


__all__ = ["development_commands"]
