"""Run-directory, manifest, status, and compact-export helpers for EoS Lab."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class RunLayout:
    root: Path
    logs: Path
    data: Path
    tables: Path
    plots: Path
    resolved_config: Path
    manifest: Path
    report: Path


def create_run_layout(
    experiment_name: str,
    configuration_hash: str,
    *,
    runs_root: Path | None = None,
    now: datetime | None = None,
) -> RunLayout:
    """Create a unique, non-overwriting directory for one experiment attempt."""

    if not _EXPERIMENT_NAME.fullmatch(experiment_name):
        raise ValueError(
            "experiment_name must contain 3-64 lowercase letters, numbers, hyphens, "
            "or underscores, and must start with a letter or number."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", configuration_hash):
        raise ValueError("configuration_hash must be a lowercase SHA-256 value.")
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    base = (runs_root or (PROJECT_ROOT / "runs")).resolve() / experiment_name
    candidate = base / f"{timestamp}-{configuration_hash[:12]}"
    suffix = 2
    while candidate.exists():
        candidate = base / f"{timestamp}-{configuration_hash[:12]}-{suffix}"
        suffix += 1
    return initialize_run_layout(candidate)


def initialize_run_layout(root: Path) -> RunLayout:
    resolved_root = root.resolve()
    logs = resolved_root / "logs"
    data = resolved_root / "data"
    tables = resolved_root / "tables"
    plots = resolved_root / "plots"
    for directory in (logs, data, tables, plots):
        directory.mkdir(parents=True, exist_ok=False)
    return RunLayout(
        root=resolved_root,
        logs=logs,
        data=data,
        tables=tables,
        plots=plots,
        resolved_config=resolved_root / "resolved_config.toml",
        manifest=resolved_root / "run_manifest.json",
        report=resolved_root / "report.md",
    )


def open_run_layout(root: Path) -> RunLayout:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {resolved_root}")
    return RunLayout(
        root=resolved_root,
        logs=resolved_root / "logs",
        data=resolved_root / "data",
        tables=resolved_root / "tables",
        plots=resolved_root / "plots",
        resolved_config=resolved_root / "resolved_config.toml",
        manifest=resolved_root / "run_manifest.json",
        report=resolved_root / "report.md",
    )


def source_tree_sha256() -> str:
    candidates = list((PROJECT_ROOT / "src").rglob("*.py"))
    candidates.extend((PROJECT_ROOT / "framework").rglob("*.py"))
    candidates.extend(PROJECT_ROOT.glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for distribution in (
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "pyarrow",
        "joblib",
        "scikit-learn",
        "xgboost",
        "torch",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": packages,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def read_manifest(run_directory: Path) -> dict[str, Any]:
    layout = open_run_layout(run_directory)
    if not layout.manifest.is_file():
        raise FileNotFoundError(f"Run manifest is missing: {layout.manifest}")
    payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run manifest is not a JSON object: {layout.manifest}")
    return payload


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_recorded_artifacts(
    layout: RunLayout, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Check every artifact hash recorded by a completed EoS Lab run."""

    recorded = manifest.get("artifacts")
    if not isinstance(recorded, dict) or not recorded:
        return {
            "state": "not_recorded",
            "checked": 0,
            "missing": [],
            "mismatched": [],
            "unsafe_paths": [],
        }

    missing: list[str] = []
    mismatched: list[str] = []
    unsafe_paths: list[str] = []
    checked = 0
    root = layout.root.resolve()
    for relative_name, expected_hash in sorted(recorded.items()):
        if not isinstance(relative_name, str) or not isinstance(expected_hash, str):
            unsafe_paths.append(str(relative_name))
            continue
        candidate = (root / relative_name).resolve()
        if not candidate.is_relative_to(root):
            unsafe_paths.append(relative_name)
            continue
        if not candidate.is_file():
            missing.append(relative_name)
            continue
        checked += 1
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            mismatched.append(relative_name)
            continue
        if file_sha256(candidate) != expected_hash:
            mismatched.append(relative_name)

    state = "valid" if not (missing or mismatched or unsafe_paths) else "invalid"
    return {
        "state": state,
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
        "unsafe_paths": unsafe_paths,
    }


def run_status(run_directory: Path) -> dict[str, Any]:
    layout = open_run_layout(run_directory)
    manifest = read_manifest(layout.root)
    expected = {
        "resolved_configuration": layout.resolved_config,
        "eos_tables": layout.data / "eos_tables.parquet",
        "stellar_curves": layout.data / "stellar_curves.parquet",
        "summary": layout.tables / "eos_summary.csv",
        "rejections": layout.tables / "rejections.csv",
        "convergence": layout.tables / "convergence.csv",
        "report": layout.report,
    }
    recorded_source_hash = manifest.get("source_tree_sha256")
    current_source_hash = source_tree_sha256()
    return {
        "run_directory": str(layout.root),
        "status": manifest.get("status", "unknown"),
        "experiment_name": manifest.get("experiment_name"),
        "configuration_hash": manifest.get("configuration_hash"),
        "created_utc": manifest.get("created_utc"),
        "completed_utc": manifest.get("completed_utc"),
        "artifact_integrity": verify_recorded_artifacts(layout, manifest),
        "recorded_source_tree_sha256": recorded_source_hash,
        "current_source_tree_sha256": current_source_hash,
        "source_tree_matches_run": (
            recorded_source_hash == current_source_hash
            if isinstance(recorded_source_hash, str)
            else None
        ),
        "artifacts": {
            name: {"path": str(path), "present": path.is_file()}
            for name, path in expected.items()
        },
    }


def export_summary(
    run_directory: Path,
    *,
    reports_root: Path | None = None,
) -> Path:
    """Copy only compact, reviewable outputs; raw tables and models stay ignored."""

    layout = open_run_layout(run_directory)
    manifest = read_manifest(layout.root)
    exportable_statuses = {
        "completed",
        "completed_with_rejections",
        "failed_convergence",
    }
    if manifest.get("status") not in exportable_statuses:
        raise RuntimeError(
            "Only an artifact-complete terminal run can be exported. Supported "
            f"statuses are: {', '.join(sorted(exportable_statuses))}."
        )
    integrity = verify_recorded_artifacts(layout, manifest)
    if integrity["state"] != "valid":
        raise RuntimeError(
            "The completed run's recorded artifacts did not pass SHA-256 integrity "
            f"verification (state: {integrity['state']})."
        )
    experiment_name = str(manifest.get("experiment_name", ""))
    if not _EXPERIMENT_NAME.fullmatch(experiment_name):
        raise ValueError("The run manifest contains an invalid experiment_name.")
    destination = (reports_root or (PROJECT_ROOT / "reports")).resolve()
    destination = destination / experiment_name / layout.root.name
    if destination.exists():
        raise FileExistsError(
            f"Summary destination already exists and will not be overwritten: {destination}"
        )
    destination.mkdir(parents=True)
    compact_files = (
        layout.resolved_config,
        layout.manifest,
        layout.report,
        layout.tables / "eos_summary.csv",
        layout.tables / "rejections.csv",
        layout.tables / "convergence.csv",
    )
    for source in compact_files:
        if not source.is_file():
            raise FileNotFoundError(
                f"Completed run is missing compact artifact: {source}"
            )
        shutil.copy2(source, destination / source.name)
    plot_destination = destination / "plots"
    plot_destination.mkdir()
    for source in sorted(layout.plots.glob("*.png")):
        shutil.copy2(source, plot_destination / source.name)
    return destination


__all__ = [
    "PROJECT_ROOT",
    "RunLayout",
    "create_run_layout",
    "environment_metadata",
    "export_summary",
    "file_sha256",
    "git_revision",
    "open_run_layout",
    "read_manifest",
    "run_status",
    "source_tree_sha256",
    "verify_recorded_artifacts",
    "write_json",
]
