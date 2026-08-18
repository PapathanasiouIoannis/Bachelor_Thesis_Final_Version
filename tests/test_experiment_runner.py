from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed

from framework.eos_catalog import CFL_CATALOG, HADRONIC_CATALOG
from src.physics import experiment_runner
from src.physics.experiment_reporting import EOS_COLUMNS, STELLAR_COLUMNS
from src.physics.experiment_runner import (
    CONVERGENCE_COLUMNS,
    _generate_pair,
    _merge_worker_logs,
    _run_convergence_checks,
    _worker_log_path,
    render_resolved_toml,
    run_pair_experiment,
    validate_pair_experiment,
)
from src.utils.logger import close_run_log, configure_run_log, get_logger


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _write_worker_log(path: str, index: int) -> int:
    configure_run_log(_worker_log_path(Path(path)))
    get_logger("EOSLAB").info("worker-isolation-message-%d", index)
    close_run_log()
    return index


def test_reproduction_preflight_reports_exact_pair_and_intervals():
    preflight = validate_pair_experiment(CONFIGS / "apr1_cfl4_reproduction.toml")
    report = preflight.to_dict()
    assert report["expected_curves"] == 30
    assert report["hadronic_eos"]["baseline"] == "APR-1"
    assert report["quark_eos"]["catalog_identifier"] == "CFL4"
    assert report["classification_enabled"] is False
    assert report["deformation"]["amplitudes"][5] == 0.0
    lower, upper = report["admissible_amplitude_intervals"]["common"]
    assert lower < -0.05 < 0.09 < upper
    assert (
        report["baseline_recovery_maximum_relative_pressure_error"]["hadronic"] < 2e-4
    )
    recovery = report["baseline_recovery_maximum_relative_pressure_error"]
    assert 4.0e-5 < recovery["hadronic"] < 5.0e-5
    assert 1.0e-10 < recovery["quark"] < 1.0e-9
    assert report["provenance"] == {
        "hadronic": next(
            entry.as_row() for entry in HADRONIC_CATALOG if entry.eos_id == "APR-1"
        ),
        "quark": next(
            entry.as_row() for entry in CFL_CATALOG if entry.eos_id == "CFL4"
        ),
    }


def test_preflight_out_of_range_error_names_parameter_and_boundary(tmp_path: Path):
    text = (CONFIGS / "apr1_cfl4_exploration.toml").read_text(encoding="utf-8")
    text = text.replace("amplitude_stop = 0.09", "amplitude_stop = 0.65")
    path = tmp_path / "invalid.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"amplitude_stop = 0.65.*permitted maximum"):
        validate_pair_experiment(path)


def test_resolved_toml_round_trips_through_the_strict_loader(tmp_path: Path):
    preflight = validate_pair_experiment(CONFIGS / "smoke.toml")
    text = render_resolved_toml(preflight.runtime_configuration)
    parsed = tomllib.loads(text)
    assert "resolved" not in parsed
    path = tmp_path / "resolved_config.toml"
    path.write_text(text, encoding="utf-8")
    reloaded = validate_pair_experiment(path)
    assert [point.amplitude for point in reloaded.sweep_points] == [-0.01, 0.0, 0.01]
    assert reloaded.resolved.quark_eos_id == "CFL4"


def _fake_pair(runtime, sweep_index, amplitude, configuration_hash, run_log_path):
    del configuration_hash
    assert run_log_path.endswith("pipeline.log")
    sweep_id = f"A{sweep_index:05d}"
    eos_frames = []
    stellar_frames = []
    for matter_type, baseline, radius_shift in (
        ("hadronic", "APR-1", 1.0),
        ("quark", "CFL4", 0.0),
    ):
        eos_frames.append(
            pd.DataFrame(
                {
                    "matter_type": matter_type,
                    "baseline_name": baseline,
                    "model_identifier": baseline,
                    "sweep_id": sweep_id,
                    "deformation_amplitude": amplitude,
                    "pair_accepted": True,
                    "eos_validation_passed": True,
                    "eos_validation_reason": "passed",
                    "eos_region": "core" if matter_type == "hadronic" else "self_bound",
                    "energy_density_mev_fm3": [100.0, 200.0, 300.0, 400.0],
                    "pressure_mev_fm3": [1.0, 20.0, 50.0, 90.0],
                    "sound_speed_squared": [0.2, 0.3, 0.4, 0.5],
                    "causal_prefix_applied": False,
                    "discarded_suffix_points": 0,
                    "first_discarded_sound_speed_squared": np.nan,
                    "causal_cutoff_pressure_mev_fm3": 90.0,
                    "causal_cutoff_energy_density_mev_fm3": 400.0,
                },
                columns=EOS_COLUMNS,
            )
        )
        masses = np.array([1.0, 1.2, 1.4, 1.6, 2.1])
        stellar_frames.append(
            pd.DataFrame(
                {
                    "matter_type": matter_type,
                    "baseline_name": baseline,
                    "model_identifier": baseline,
                    "sweep_id": sweep_id,
                    "curve_id": f"{matter_type}_{sweep_id}",
                    "deformation_amplitude": amplitude,
                    "mass_msun": masses,
                    "radius_km": 12.0 + radius_shift - 0.2 * masses + amplitude,
                    "tidal_deformability": [900.0, 650.0, 400.0, 220.0, 40.0],
                    "central_pressure_mev_fm3": [10.0, 20.0, 40.0, 80.0, 160.0],
                    "central_energy_density_mev_fm3": [
                        200.0,
                        250.0,
                        320.0,
                        420.0,
                        700.0,
                    ],
                    "central_sound_speed_squared": [0.2, 0.25, 0.3, 0.4, 0.5],
                    "surface_energy_density_mev_fm3": 0.0,
                },
                columns=STELLAR_COLUMNS,
            )
        )
    return {
        "accepted": True,
        "eos_frames": eos_frames,
        "stellar_frames": stellar_frames,
        "rejection": None,
    }


def _recording_pair_to_disk(
    runtime, sweep_index, amplitude, configuration_hash, run_log_path
):
    record_path = Path(run_log_path).parent / f"worker_{sweep_index:05d}.json"
    record_path.write_text(
        json.dumps(
            {
                "amplitude": amplitude,
                "parallel_jobs": runtime["execution"]["parallel_jobs"],
                "amplitudes_per_batch": runtime["execution"]["amplitudes_per_batch"],
                "center": runtime["deformation"]["center_energy_density_mev_fm3"],
                "width": runtime["deformation"]["width_mev_fm3"],
            }
        ),
        encoding="utf-8",
    )
    return _fake_pair(runtime, sweep_index, amplitude, configuration_hash, run_log_path)


def test_smoke_orchestration_writes_complete_isolated_artifacts(tmp_path, monkeypatch):
    worker_settings = []

    def recording_pair(
        runtime, sweep_index, amplitude, configuration_hash, run_log_path
    ):
        worker_settings.append(dict(runtime["execution"]))
        return _fake_pair(
            runtime, sweep_index, amplitude, configuration_hash, run_log_path
        )

    monkeypatch.setattr(experiment_runner, "_generate_pair", recording_pair)
    layout = run_pair_experiment(
        CONFIGS / "smoke.toml", parallel_jobs=1, runs_root=tmp_path / "runs"
    )
    manifest = json.loads(layout.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["accepted_pairs"] == 3
    assert manifest["accepted_curves"] == 6
    assert manifest["classification_enabled"] is False
    assert manifest["execution"]["parallel_jobs"] == 1
    assert manifest["runtime_overrides"]["execution.parallel_jobs"] == {
        "configured": 2,
        "effective": 1,
        "source": "command_line",
    }
    assert manifest["source_tree_sha256"]
    assert manifest["convergence_performed"] is False
    assert manifest["convergence_passed"] is None
    assert all(settings["parallel_jobs"] == 1 for settings in worker_settings)
    assert all(settings["amplitudes_per_batch"] == 3 for settings in worker_settings)
    assert (layout.data / "eos_tables.parquet").is_file()
    assert (layout.data / "stellar_curves.parquet").is_file()
    assert (layout.tables / "eos_summary.csv").is_file()
    assert (layout.tables / "rejections.csv").is_file()
    assert (layout.tables / "convergence.csv").is_file()
    assert layout.report.is_file()
    assert len(list(layout.plots.glob("*.png"))) == 4
    summary = pd.read_csv(layout.tables / "eos_summary.csv")
    assert set(summary["matter_type"]) == {"hadronic", "quark"}
    assert len(summary) == 6
    report = layout.report.read_text(encoding="utf-8")
    assert "Maximum mass [$M_\\odot$]" in report
    assert "not a universal matter-phase classifier" in report


def test_nondefault_parameters_reach_two_worker_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_runner, "_generate_pair", _recording_pair_to_disk)
    layout = run_pair_experiment(
        CONFIGS / "smoke.toml", parallel_jobs=2, runs_root=tmp_path / "runs"
    )
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(layout.logs.glob("worker_*.json"))
    ]
    assert len(records) == 3
    assert {record["amplitude"] for record in records} == {-0.01, 0.0, 0.01}
    assert all(record["parallel_jobs"] == 2 for record in records)
    assert all(record["amplitudes_per_batch"] == 3 for record in records)
    assert all(record["center"] == 220.0 for record in records)
    assert all(record["width"] == 50.0 for record in records)


def test_worker_process_logs_are_isolated_in_the_run_directory(tmp_path):
    run_log = tmp_path / "run" / "logs" / "pipeline.log"
    assert Parallel(n_jobs=2, prefer="processes")(
        delayed(_write_worker_log)(str(run_log), index) for index in range(2)
    ) == [0, 1]
    _merge_worker_logs(run_log)
    contents = run_log.read_text(encoding="utf-8")
    assert "worker-isolation-message-0" in contents
    assert "worker-isolation-message-1" in contents


def test_real_smoke_control_is_rejected_with_both_eos_diagnostics(tmp_path):
    preflight = validate_pair_experiment(CONFIGS / "smoke.toml")
    result = _generate_pair(
        preflight.runtime_configuration,
        1,
        0.0,
        preflight.resolved.config_hash,
        str(tmp_path / "worker.log"),
    )
    assert result["accepted"] is False
    assert result["rejection"]["stage"] == "eos_validation"
    assert result["rejection"]["matter_type"] == "hadronic"
    assert "strictly increasing" in result["rejection"]["reason"]
    assert len(result["eos_frames"]) == 2
    by_type = {frame["matter_type"].iloc[0]: frame for frame in result["eos_frames"]}
    assert bool(by_type["hadronic"]["eos_validation_passed"].iloc[0]) is False
    assert bool(by_type["quark"]["eos_validation_passed"].iloc[0]) is True
    assert set(by_type["hadronic"]["eos_region"]) == {"crust", "core"}
    assert by_type["quark"]["baseline_name"].iloc[0] == "CFL4"
    assert not by_type["hadronic"]["pair_accepted"].astype(bool).any()
    assert not by_type["quark"]["pair_accepted"].astype(bool).any()


def _reference_summary(runtime):
    records = []
    for amplitude in (-0.01, 0.0, 0.01):
        for matter_type, baseline in (("hadronic", "APR-1"), ("quark", "CFL4")):
            records.append(
                {
                    "matter_type": matter_type,
                    "baseline_name": baseline,
                    "deformation_amplitude": amplitude,
                    "maximum_mass_msun": 2.1,
                    "radius_1p4_km": 12.0,
                    "tidal_deformability_1p4": 400.0,
                }
            )
    runtime["resolved"]["amplitudes"] = [-0.01, 0.0, 0.01]
    return pd.DataFrame.from_records(records)


def test_convergence_uses_numerical_deltas_before_physical_screen(monkeypatch):
    runtime = validate_pair_experiment(CONFIGS / "smoke.toml").runtime_configuration
    runtime["numerical_settings"]["convergence_check"] = "endpoints_and_zero"
    summary = _reference_summary(runtime)
    requested_screens = []
    refined = {
        "maximum_mass_msun": 2.095,
        "radius_1p4_km": 12.02,
        "tidal_deformability_1p4": 404.0,
    }

    monkeypatch.setattr(
        experiment_runner,
        "_build_eos",
        lambda runtime, matter_type, amplitude, grid_points=None: SimpleNamespace(
            baseline_name="APR-1"
            if matter_type == "hadronic"
            else "CFL_B60_D100_MS150",
            catalog_identifier="APR-1" if matter_type == "hadronic" else "CFL4",
        ),
    )

    def fake_solve(runtime, eos, matter_type, **kwargs):
        del runtime, eos, matter_type
        requested_screens.append(kwargs["enforce_physical_requirements"])
        return [], {}, 0.0

    monkeypatch.setattr(experiment_runner, "_solve", fake_solve)
    monkeypatch.setattr(
        experiment_runner, "stellar_curve_to_frame", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        experiment_runner, "summarize_stellar_curve", lambda frame: dict(refined)
    )

    passing = _run_convergence_checks(runtime, summary)
    assert len(passing) == 18
    assert passing["passed"].astype(bool).all()
    assert requested_screens == [False] * 18

    refined["maximum_mass_msun"] = 2.079
    boundary_reference = summary.copy()
    boundary_reference["maximum_mass_msun"] = 2.081
    physical_failure = _run_convergence_checks(runtime, boundary_reference)
    assert physical_failure["maximum_mass_passed"].astype(bool).all()
    assert (
        not physical_failure["refined_physical_requirements_passed"].astype(bool).any()
    )
    assert not physical_failure["passed"].astype(bool).any()


def test_failed_convergence_is_terminal_reportable_status(tmp_path, monkeypatch):
    text = (
        (CONFIGS / "smoke.toml")
        .read_text(encoding="utf-8")
        .replace(
            'convergence_check = "none"',
            'convergence_check = "endpoints_and_zero"',
        )
    )
    config_path = tmp_path / "convergence_failure.toml"
    config_path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "_generate_pair", _fake_pair)
    failed_record = {column: False for column in CONVERGENCE_COLUMNS}
    failed_record.update(
        {
            "matter_type": "hadronic",
            "baseline_name": "APR-1",
            "deformation_amplitude": 0.0,
            "check": "double_eos_grid",
            "refined_physical_requirements_reason": "passed",
        }
    )
    monkeypatch.setattr(
        experiment_runner,
        "_run_convergence_checks",
        lambda runtime, summary: pd.DataFrame.from_records(
            [failed_record], columns=CONVERGENCE_COLUMNS
        ),
    )

    with pytest.raises(RuntimeError, match="failed_convergence"):
        run_pair_experiment(config_path, parallel_jobs=1, runs_root=tmp_path / "runs")
    run_directory = next((tmp_path / "runs" / "apr1_cfl4_smoke").iterdir())
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    assert manifest["status"] == "failed_convergence"
    assert "error" not in manifest
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    assert "Terminal run status: `failed_convergence`" in report


def test_completed_with_rejections_is_a_terminal_reportable_status(
    tmp_path, monkeypatch
):
    def mixed_pair(runtime, sweep_index, amplitude, configuration_hash, run_log_path):
        result = _fake_pair(
            runtime, sweep_index, amplitude, configuration_hash, run_log_path
        )
        if sweep_index != 1:
            return result
        for frame in result["eos_frames"]:
            frame.loc[:, "pair_accepted"] = False
        return {
            "accepted": False,
            "eos_frames": result["eos_frames"],
            "stellar_frames": [],
            "rejection": {
                "sweep_id": "A00001",
                "deformation_amplitude": amplitude,
                "matter_type": "hadronic",
                "stage": "stellar_sequence",
                "exception_type": "TurningPointError",
                "reason": "The maximum mass was not bracketed.",
            },
        }

    monkeypatch.setattr(experiment_runner, "_generate_pair", mixed_pair)

    with pytest.raises(RuntimeError, match="completed_with_rejections"):
        run_pair_experiment(
            CONFIGS / "smoke.toml", parallel_jobs=1, runs_root=tmp_path / "runs"
        )

    run_directory = next((tmp_path / "runs" / "apr1_cfl4_smoke").iterdir())
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    assert manifest["status"] == "completed_with_rejections"
    assert manifest["accepted_pairs"] == 2
    assert manifest["accepted_curves"] == 4
    assert manifest["rejected_pairs"] == 1
    assert "error" not in manifest
    assert manifest["artifacts"]
    assert all(len(digest) == 64 for digest in manifest["artifacts"].values())

    rejections = pd.read_csv(run_directory / "tables" / "rejections.csv")
    assert rejections.to_dict(orient="records") == [
        {
            "sweep_id": "A00001",
            "deformation_amplitude": 0.0,
            "matter_type": "hadronic",
            "stage": "stellar_sequence",
            "exception_type": "TurningPointError",
            "reason": "The maximum mass was not bracketed.",
        }
    ]
    report = (run_directory / "report.md").read_text(encoding="utf-8")
    assert "Terminal run status: `completed_with_rejections`" in report


def test_unexpected_runner_exception_records_failed_manifest(tmp_path, monkeypatch):
    def unexpected_pair(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("unexpected worker failure")

    monkeypatch.setattr(experiment_runner, "_generate_pair", unexpected_pair)

    with pytest.raises(RuntimeError, match="unexpected worker failure"):
        run_pair_experiment(
            CONFIGS / "smoke.toml", parallel_jobs=1, runs_root=tmp_path / "runs"
        )

    run_directory = next((tmp_path / "runs" / "apr1_cfl4_smoke").iterdir())
    manifest = json.loads((run_directory / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed_utc"]
    assert manifest["configuration_hash"]
    assert manifest["preflight"]["expected_curves"] == 6
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["error"]["message"] == "unexpected worker failure"
    assert "unexpected_pair" in manifest["error"]["traceback"]
