"""Validated execution of one controlled hadronic/CFL sensitivity experiment."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from framework.eos_catalog import CFL_CATALOG, HADRONIC_CATALOG
from framework.eos_sweep import (
    GaussianDeformation,
    QuarkParameters,
    SweepPoint,
    admissible_amplitude_interval,
    build_hadronic_eos,
    build_quark_eos,
    cfl_baseline_grids,
    hadronic_baseline_grids,
    validate_sweep_within_interval,
)
from src.config import CONFIG
from src.eoslab_runtime import (
    RunLayout,
    create_run_layout,
    environment_metadata,
    file_sha256,
    git_revision,
    source_tree_sha256,
    write_json,
)
from src.experiment_config import ResolvedExperiment, resolve_pair_experiment
from src.physics.controlled_generation import solve_and_validate_sequence
from src.physics.experiment_reporting import (
    EOS_COLUMNS,
    STELLAR_COLUMNS,
    SUMMARY_COLUMNS,
    build_causal_domain_table,
    build_summary_table,
    create_standard_plots,
    serialize_eos_table,
    stellar_curve_to_frame,
    summarize_stellar_curve,
    validate_eos_frame,
    write_markdown_report,
)
from src.physics.runner import artifacts as _artifacts
from src.physics.runner import convergence as _convergence
from src.physics.runner import generation as _generation
from src.physics.runner import preflight as _preflight
from src.physics.runner import settings as _settings
from src.utils.logger import close_run_log, configure_run_log, get_logger


LOGGER = get_logger("EOSLAB")
PAIR_INTERPRETATION = _preflight.PAIR_INTERPRETATION
REJECTION_COLUMNS = (
    "sweep_id",
    "deformation_amplitude",
    "matter_type",
    "stage",
    "exception_type",
    "reason",
)
CONVERGENCE_COLUMNS = _convergence.CONVERGENCE_COLUMNS

NUMERICAL_PRESETS = _settings.numerical_presets_from_configuration(CONFIG)
PairPreflight = _preflight.PairPreflight


def _preflight_report_numerical_settings(
    runtime: dict[str, Any],
) -> dict[str, int | float]:
    return _resolved_numerical_settings(runtime)


def _preflight_report_interpretation() -> str:
    return PAIR_INTERPRETATION


def _sync_preflight_facade() -> None:
    _settings.NUMERICAL_PRESETS = NUMERICAL_PRESETS
    _preflight.REPORT_NUMERICAL_SETTINGS = _preflight_report_numerical_settings
    _preflight.REPORT_PAIR_INTERPRETATION = _preflight_report_interpretation


_sync_preflight_facade()


PairGenerationError = _generation.PairGenerationError


def _pair_generation_dependencies() -> _generation.PairGenerationDependencies:
    return _generation.PairGenerationDependencies(
        configure_run_log=configure_run_log,
        worker_log_path=_worker_log_path,
        path_type=Path,
        sweep_point_type=SweepPoint,
        build_eos=_build_eos,
        serialize_eos_table=serialize_eos_table,
        validate_eos_frame=validate_eos_frame,
        solve=_solve,
        stellar_curve_to_frame=stellar_curve_to_frame,
        pair_generation_error_type=PairGenerationError,
        close_run_log=close_run_log,
    )


def _build_eos_dependencies() -> _generation.BuildEosDependencies:
    return _generation.BuildEosDependencies(
        resolved_numerical_settings=_resolved_numerical_settings,
        gaussian_deformation_type=GaussianDeformation,
        build_hadronic_eos=build_hadronic_eos,
        build_quark_eos=build_quark_eos,
        quark_parameters=_quark_parameters,
        configuration=CONFIG,
    )


def _solve_dependencies() -> _generation.SolveDependencies:
    return _generation.SolveDependencies(
        resolved_numerical_settings=_resolved_numerical_settings,
        solve_and_validate_sequence=solve_and_validate_sequence,
    )


def _convergence_dependencies() -> _convergence.ConvergenceDependencies:
    return _convergence.ConvergenceDependencies(
        dataframe_type=pd.DataFrame,
        convergence_columns=CONVERGENCE_COLUMNS,
        resolved_numerical_settings=_resolved_numerical_settings,
        failed_convergence_record=_failed_convergence_record,
        build_eos=_build_eos,
        solve=_solve,
        stellar_curve_to_frame=stellar_curve_to_frame,
        summarize_stellar_curve=summarize_stellar_curve,
        physical_requirements_status=_physical_requirements_status,
    )


def _failed_convergence_record_dependencies(
) -> _convergence.FailedConvergenceRecordDependencies:
    return _convergence.FailedConvergenceRecordDependencies(nan_value=np.nan)


def _preflight_validation_dependencies() -> _preflight.ValidationDependencies:
    return _preflight.ValidationDependencies(
        resolve_pair_experiment=resolve_pair_experiment,
        resolved_experiment_type=ResolvedExperiment,
        sweep_point_type=SweepPoint,
        pair_preflight_type=PairPreflight,
        quark_parameters=_quark_parameters,
        resolved_numerical_settings=_resolved_numerical_settings,
        hadronic_baseline_grids=hadronic_baseline_grids,
        cfl_baseline_grids=cfl_baseline_grids,
        admissible_amplitude_interval=admissible_amplitude_interval,
        validate_sweep_within_interval=validate_sweep_within_interval,
        baseline_recovery_errors=_baseline_recovery_errors,
        provenance=_provenance,
    )


def validate_pair_experiment(
    configuration: ResolvedExperiment | str | Path,
) -> PairPreflight:
    """Validate a profile and its common causal amplitude support without TOV runs."""

    _sync_preflight_facade()
    return _preflight.validate_pair_experiment(
        configuration,
        _preflight_validation_dependencies(),
    )


def run_pair_experiment(
    configuration: ResolvedExperiment | str | Path,
    *,
    parallel_jobs: int | None = None,
    runs_root: Path | None = None,
) -> RunLayout:
    """Generate, validate, persist, plot, and report one controlled pair sweep."""

    preflight = validate_pair_experiment(configuration)
    runtime = json.loads(json.dumps(preflight.runtime_configuration))
    runtime["resolved_numerical_settings"] = _resolved_numerical_settings(runtime)
    runtime_overrides: dict[str, Any] = {}
    if parallel_jobs is not None:
        if type(parallel_jobs) is not int or parallel_jobs < 1:
            raise ValueError("parallel_jobs must be an integer of at least 1.")
        configured_jobs = int(runtime["execution"]["parallel_jobs"])
        runtime["execution"]["parallel_jobs"] = int(parallel_jobs)
        if int(parallel_jobs) != configured_jobs:
            runtime_overrides["execution.parallel_jobs"] = {
                "configured": configured_jobs,
                "effective": int(parallel_jobs),
                "source": "command_line",
            }
    layout = create_run_layout(
        runtime["experiment_name"], preflight.resolved.config_hash, runs_root=runs_root
    )
    configure_run_log(layout.logs / "pipeline.log")
    layout.resolved_config.write_text(render_resolved_toml(runtime), encoding="utf-8")
    created = datetime.now(timezone.utc).isoformat()
    base_manifest = {
        "schema_version": 1,
        "component": "controlled_eos_pair_sensitivity",
        "status": "running",
        "experiment_name": runtime["experiment_name"],
        "workflow": runtime["workflow"],
        "mode": runtime["mode"],
        "configuration_hash": preflight.resolved.config_hash,
        "source_tree_sha256": source_tree_sha256(),
        "git_revision": git_revision(),
        "created_utc": created,
        "environment": environment_metadata(),
        "preflight": preflight.to_dict(),
        "execution": runtime["execution"],
        "resolved_numerical_settings": runtime["resolved_numerical_settings"],
        "runtime_overrides": runtime_overrides,
        "classification_enabled": False,
        "permitted_scientific_interpretation": PAIR_INTERPRETATION,
    }
    write_json(layout.manifest, base_manifest)
    LOGGER.info("Run directory: %s", layout.root)
    LOGGER.info("Generating %d paired amplitudes.", len(preflight.sweep_points))

    try:
        jobs = int(runtime["execution"]["parallel_jobs"])
        amplitudes_per_batch = int(runtime["execution"]["amplitudes_per_batch"])
        tasks = (
            delayed(_generate_pair)(
                runtime,
                point.index,
                point.amplitude,
                preflight.resolved.config_hash,
                str(layout.logs / "pipeline.log"),
            )
            for point in preflight.sweep_points
        )
        results = Parallel(
            n_jobs=jobs,
            prefer="processes",
            batch_size=amplitudes_per_batch,
        )(tasks)
        _merge_worker_logs(layout.logs / "pipeline.log")
        accepted = [result for result in results if result["accepted"]]
        rejected = [result["rejection"] for result in results if not result["accepted"]]

        eos_tables = _concat_frames(
            [frame for result in results for frame in result["eos_frames"]],
            EOS_COLUMNS,
        )
        stellar_curves = _concat_frames(
            [frame for result in accepted for frame in result["stellar_frames"]],
            STELLAR_COLUMNS,
        )
        summary = (
            build_summary_table(stellar_curves)
            if not stellar_curves.empty
            else pd.DataFrame(columns=SUMMARY_COLUMNS)
        )
        rejection_table = pd.DataFrame.from_records(rejected, columns=REJECTION_COLUMNS)
        convergence = _run_convergence_checks(runtime, summary)

        eos_tables.to_parquet(layout.data / "eos_tables.parquet", index=False)
        stellar_curves.to_parquet(layout.data / "stellar_curves.parquet", index=False)
        summary.to_csv(layout.tables / "eos_summary.csv", index=False)
        rejection_table.to_csv(layout.tables / "rejections.csv", index=False)
        convergence.to_csv(layout.tables / "convergence.csv", index=False)

        convergence_performed = (
            runtime["numerical_settings"]["convergence_check"] != "none"
        )
        convergence_passed = (
            bool(convergence["passed"].astype(bool).all())
            if convergence_performed
            else None
        )
        all_pairs_accepted = len(rejection_table) == 0
        status = (
            "completed"
            if all_pairs_accepted
            and (not convergence_performed or convergence_passed is True)
            else "completed_with_rejections"
            if not all_pairs_accepted
            else "failed_convergence"
        )

        plot_paths: list[Path] = []
        if not eos_tables.empty:
            plot_paths = create_standard_plots(
                eos_tables, stellar_curves, summary, layout.plots
            )
        write_markdown_report(
            eos_tables,
            summary,
            rejection_table,
            convergence,
            runtime,
            layout.report,
            run_status=status,
        )
        causal_domains = build_causal_domain_table(eos_tables)
        artifacts = _artifact_hashes(layout)
        manifest = {
            **base_manifest,
            "status": status,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "accepted_pairs": len(accepted),
            "accepted_curves": len(summary),
            "rejected_pairs": len(rejection_table),
            "convergence_performed": convergence_performed,
            "convergence_passed": convergence_passed,
            "causal_domains": json.loads(causal_domains.to_json(orient="records")),
            "plot_files": [path.name for path in plot_paths],
            "artifacts": artifacts,
        }
        write_json(layout.manifest, manifest)
        if status != "completed":
            raise RuntimeError(
                f"Run finished with status {status}. Review {layout.report} and "
                f"{layout.tables / 'rejections.csv'}."
            )
        LOGGER.info("Completed %d curves with no rejected pairs.", len(summary))
        return layout
    except Exception as error:
        _merge_worker_logs(layout.logs / "pipeline.log")
        current = json.loads(layout.manifest.read_text(encoding="utf-8"))
        if current.get("status") == "running":
            current.update(
                {
                    "status": "failed",
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            write_json(layout.manifest, current)
        raise


def _worker_log_path(run_log_path: Path) -> Path:
    """Return a process-isolated temporary log path for one worker."""

    return run_log_path.with_name(
        f"{run_log_path.stem}.worker-{os.getpid()}{run_log_path.suffix}"
    )


def _merge_worker_logs(run_log_path: Path) -> None:
    """Merge process-isolated worker logs into the run's canonical log."""

    pattern = f"{run_log_path.stem}.worker-*{run_log_path.suffix}"
    worker_logs = sorted(run_log_path.parent.glob(pattern))
    if not worker_logs:
        return
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with run_log_path.open("a", encoding="utf-8") as destination:
        for worker_log in worker_logs:
            destination.write(worker_log.read_text(encoding="utf-8"))
    for worker_log in worker_logs:
        try:
            worker_log.unlink()
        except PermissionError:
            # An unexpectedly failed worker may still own its handler. The log
            # has already been merged and remains inside this isolated run.
            pass


def render_resolved_toml(runtime: dict[str, Any]) -> str:
    """Render the known resolved pair schema without a third-party TOML writer."""

    return _artifacts.render_resolved_toml(
        runtime,
        value_renderer=_toml_value,
    )


def _generate_pair(
    runtime: dict[str, Any],
    sweep_index: int,
    amplitude: float,
    configuration_hash: str,
    run_log_path: str,
) -> dict[str, Any]:
    return _generation.generate_pair(
        runtime,
        sweep_index,
        amplitude,
        configuration_hash,
        run_log_path,
        dependencies=_pair_generation_dependencies(),
    )


def _build_eos(
    runtime: dict[str, Any],
    matter_type: str,
    amplitude: float,
    *,
    grid_points: int | None = None,
):
    return _generation.build_eos(
        runtime,
        matter_type,
        amplitude,
        grid_points=grid_points,
        dependencies=_build_eos_dependencies(),
    )


def _solve(
    runtime: dict[str, Any],
    eos,
    matter_type: str,
    *,
    n_points: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    enforce_physical_requirements: bool = True,
) -> tuple[list, dict, float]:
    return _generation.solve(
        runtime,
        eos,
        matter_type,
        n_points=n_points,
        rtol=rtol,
        atol=atol,
        enforce_physical_requirements=enforce_physical_requirements,
        dependencies=_solve_dependencies(),
    )


def _run_convergence_checks(
    runtime: dict[str, Any], summary: pd.DataFrame
) -> pd.DataFrame:
    return _convergence.run_convergence_checks(
        runtime,
        summary,
        dependencies=_convergence_dependencies(),
    )


def _failed_convergence_record(
    runtime: dict[str, Any],
    matter_type: str,
    amplitude: float,
    check: str,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    return _convergence.failed_convergence_record(
        runtime,
        matter_type,
        amplitude,
        check,
        reason=reason,
        dependencies=_failed_convergence_record_dependencies(),
    )


def _physical_requirements_status(
    runtime: dict[str, Any], observables: dict[str, Any]
) -> tuple[bool, str]:
    return _convergence.physical_requirements_status(runtime, observables)


def _baseline_recovery_errors(runtime: dict[str, Any]) -> dict[str, float]:
    return _preflight.baseline_recovery_errors(
        runtime,
        _preflight.BaselineRecoveryDependencies(
            resolved_numerical_settings=_resolved_numerical_settings,
            hadronic_baseline_grids=hadronic_baseline_grids,
            build_hadronic_eos=build_hadronic_eos,
            quark_parameters=_quark_parameters,
            cfl_baseline_grids=cfl_baseline_grids,
            build_quark_eos=build_quark_eos,
            gaussian_deformation=GaussianDeformation,
            maximum_relative_pressure_error=_maximum_relative_pressure_error,
            configuration=CONFIG,
        ),
    )


def _maximum_relative_pressure_error(reconstructed, baseline) -> float:
    return _preflight.maximum_relative_pressure_error(reconstructed, baseline)


def _quark_parameters(runtime: dict[str, Any]) -> QuarkParameters:
    return _settings.quark_parameters(runtime, QuarkParameters)


def _resolved_numerical_settings(runtime: dict[str, Any]) -> dict[str, int | float]:
    return _settings.resolved_numerical_settings(runtime, NUMERICAL_PRESETS)


def _provenance(runtime: dict[str, Any], quark_eos_id: str) -> dict[str, Any]:
    return _preflight.provenance(
        runtime,
        quark_eos_id,
        HADRONIC_CATALOG,
        CFL_CATALOG,
    )


def _concat_frames(frames: list[pd.DataFrame], columns) -> pd.DataFrame:
    return _artifacts.concat_frames(frames, columns)


def _artifact_hashes(layout: RunLayout) -> dict[str, str]:
    return _artifacts.artifact_hashes(layout, file_hasher=file_sha256)


def _toml_value(value: Any) -> str:
    return _artifacts.toml_value(value)


__all__ = [
    "CONVERGENCE_COLUMNS",
    "PAIR_INTERPRETATION",
    "PairPreflight",
    "render_resolved_toml",
    "run_pair_experiment",
    "validate_pair_experiment",
]
