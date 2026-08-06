from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.eoslab_runtime import (
    create_run_layout,
    export_summary,
    file_sha256,
    run_status,
    source_tree_sha256,
    write_json,
)


CONFIG_HASH = "a" * 64


def test_create_run_layout_is_unique_and_organized(tmp_path: Path):
    now = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
    first = create_run_layout(
        "apr1_cfl4_gaussian", CONFIG_HASH, runs_root=tmp_path, now=now
    )
    second = create_run_layout(
        "apr1_cfl4_gaussian", CONFIG_HASH, runs_root=tmp_path, now=now
    )
    assert first.root.name == "20260806T120000Z-aaaaaaaaaaaa"
    assert second.root.name == "20260806T120000Z-aaaaaaaaaaaa-2"
    assert first.logs.is_dir()
    assert first.data.is_dir()
    assert first.tables.is_dir()
    assert first.plots.is_dir()


def test_create_run_layout_rejects_unsafe_experiment_name(tmp_path: Path):
    with pytest.raises(ValueError, match="experiment_name"):
        create_run_layout("../escape", CONFIG_HASH, runs_root=tmp_path)


@pytest.mark.parametrize(
    "run_state", ["completed", "completed_with_rejections", "failed_convergence"]
)
def test_status_and_compact_export_exclude_raw_data(tmp_path: Path, run_state: str):
    layout = create_run_layout("apr1_cfl4_gaussian", CONFIG_HASH, runs_root=tmp_path)
    layout.resolved_config.write_text("schema_version = 1\n", encoding="utf-8")
    layout.report.write_text("# Report\n", encoding="utf-8")
    (layout.data / "eos_tables.parquet").write_bytes(b"raw-eos")
    (layout.data / "stellar_curves.parquet").write_bytes(b"raw-stars")
    for name in ("eos_summary.csv", "rejections.csv", "convergence.csv"):
        (layout.tables / name).write_text("column\n", encoding="utf-8")
    (layout.plots / "figure.png").write_bytes(b"image")
    recorded_artifacts = {
        path.relative_to(layout.root).as_posix(): file_sha256(path)
        for path in (
            layout.resolved_config,
            layout.data / "eos_tables.parquet",
            layout.data / "stellar_curves.parquet",
            layout.tables / "eos_summary.csv",
            layout.tables / "rejections.csv",
            layout.tables / "convergence.csv",
            layout.report,
            layout.plots / "figure.png",
        )
    }
    write_json(
        layout.manifest,
        {
            "status": run_state,
            "experiment_name": "apr1_cfl4_gaussian",
            "configuration_hash": CONFIG_HASH,
            "source_tree_sha256": source_tree_sha256(),
            "artifacts": recorded_artifacts,
        },
    )

    status = run_status(layout.root)
    assert status["status"] == run_state
    assert all(record["present"] for record in status["artifacts"].values())
    assert status["artifact_integrity"]["state"] == "valid"
    assert status["source_tree_matches_run"] is True

    destination = export_summary(layout.root, reports_root=tmp_path / "reports")
    assert (destination / "eos_summary.csv").is_file()
    assert (destination / "plots" / "figure.png").is_file()
    assert not (destination / "eos_tables.parquet").exists()
    assert not (destination / "stellar_curves.parquet").exists()
    with pytest.raises(FileExistsError):
        export_summary(layout.root, reports_root=tmp_path / "reports")


def test_export_refuses_a_tampered_recorded_artifact(tmp_path: Path):
    layout = create_run_layout("apr1_cfl4_gaussian", CONFIG_HASH, runs_root=tmp_path)
    layout.resolved_config.write_text("schema_version = 1\n", encoding="utf-8")
    layout.report.write_text("# Report\n", encoding="utf-8")
    for name in ("eos_summary.csv", "rejections.csv", "convergence.csv"):
        (layout.tables / name).write_text("column\n", encoding="utf-8")
    recorded = {
        path.relative_to(layout.root).as_posix(): file_sha256(path)
        for path in (
            layout.resolved_config,
            layout.report,
            layout.tables / "eos_summary.csv",
            layout.tables / "rejections.csv",
            layout.tables / "convergence.csv",
        )
    }
    write_json(
        layout.manifest,
        {
            "status": "completed",
            "experiment_name": "apr1_cfl4_gaussian",
            "artifacts": recorded,
        },
    )
    layout.report.write_text("tampered\n", encoding="utf-8")

    assert run_status(layout.root)["artifact_integrity"]["state"] == "invalid"
    with pytest.raises(RuntimeError, match="SHA-256 integrity"):
        export_summary(layout.root, reports_root=tmp_path / "reports")
