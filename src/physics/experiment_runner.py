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
CONVERGENCE_COLUMNS = (
    "matter_type",
    "baseline_name",
    "deformation_amplitude",
    "check",
    "delta_maximum_mass_msun",
    "delta_radius_1p4_km",
    "relative_delta_tidal_deformability_1p4",
    "maximum_mass_passed",
    "radius_1p4_passed",
    "tidal_deformability_1p4_passed",
    "refined_physical_requirements_passed",
    "refined_physical_requirements_reason",
    "passed",
)

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


class PairGenerationError(RuntimeError):
    def __init__(self, matter_type: str, stage: str, reason: str):
        self.matter_type = matter_type
        self.stage = stage
        super().__init__(reason)


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

    lines = [
        "# Resolved EoS Lab configuration. Generated automatically.",
        f"schema_version = {int(runtime['schema_version'])}",
        f"experiment_name = {_toml_value(runtime['experiment_name'])}",
        f"workflow = {_toml_value(runtime['workflow'])}",
        f"mode = {_toml_value(runtime['mode'])}",
    ]
    for section in (
        "hadronic_eos",
        "quark_eos",
        "deformation",
        "physical_requirements",
        "numerical_settings",
        "execution",
    ):
        lines.extend(("", f"[{section}]"))
        for key, value in runtime[section].items():
            lines.append(f"{key} = {_toml_value(value)}")
    lines.extend(
        (
            "",
            "# Derived amplitude values, catalog identifiers, provenance, and the",
            "# permitted interpretation are recorded in run_manifest.json.",
        )
    )
    return "\n".join(lines) + "\n"


def _generate_pair(
    runtime: dict[str, Any],
    sweep_index: int,
    amplitude: float,
    configuration_hash: str,
    run_log_path: str,
) -> dict[str, Any]:
    configure_run_log(_worker_log_path(Path(run_log_path)))
    point = SweepPoint(sweep_index, amplitude)
    eos_frames: list[pd.DataFrame] = []
    try:
        eoses = {}
        try:
            eoses["hadronic"] = _build_eos(runtime, "hadronic", amplitude)
        except Exception as error:
            raise PairGenerationError(
                "hadronic", "eos_generation", str(error)
            ) from error
        try:
            eoses["quark"] = _build_eos(runtime, "quark", amplitude)
        except Exception as error:
            raise PairGenerationError("quark", "eos_generation", str(error)) from error

        validation_failures: list[tuple[str, Exception]] = []
        for matter_type in ("hadronic", "quark"):
            try:
                frame = serialize_eos_table(
                    eoses[matter_type], matter_type, point.sweep_id
                )
            except Exception as error:
                raise PairGenerationError(
                    matter_type, "eos_serialization", str(error)
                ) from error
            try:
                validate_eos_frame(frame)
                frame.loc[:, "eos_validation_passed"] = True
                frame.loc[:, "eos_validation_reason"] = "passed"
            except Exception as error:
                frame.loc[:, "eos_validation_passed"] = False
                frame.loc[:, "eos_validation_reason"] = (
                    f"{type(error).__name__}: {error}"
                )
                validation_failures.append((matter_type, error))
            eos_frames.append(frame)
        if validation_failures:
            matter_type, error = validation_failures[0]
            raise PairGenerationError(
                matter_type, "eos_validation", str(error)
            ) from error

        stellar_frames = []
        for matter_type in ("hadronic", "quark"):
            eos = eoses[matter_type]
            try:
                curve, _, _ = _solve(runtime, eos, matter_type)
            except Exception as error:
                raise PairGenerationError(
                    matter_type, "stellar_sequence", str(error)
                ) from error
            curve_id = (
                f"{matter_type}_{eos.baseline_name}_{point.sweep_id}_"
                f"{configuration_hash[:10]}"
            )
            try:
                stellar_frames.append(
                    stellar_curve_to_frame(
                        curve, eos, matter_type, point.sweep_id, curve_id
                    )
                )
            except Exception as error:
                raise PairGenerationError(
                    matter_type, "stellar_serialization", str(error)
                ) from error
        result = {
            "accepted": True,
            "eos_frames": eos_frames,
            "stellar_frames": stellar_frames,
            "rejection": None,
        }
        close_run_log()
        return result
    except PairGenerationError as error:
        for frame in eos_frames:
            frame.loc[:, "pair_accepted"] = False
        result = {
            "accepted": False,
            "eos_frames": eos_frames,
            "stellar_frames": [],
            "rejection": {
                "sweep_id": point.sweep_id,
                "deformation_amplitude": amplitude,
                "matter_type": error.matter_type,
                "stage": error.stage,
                "exception_type": type(error.__cause__).__name__
                if error.__cause__
                else type(error).__name__,
                "reason": str(error),
            },
        }
        close_run_log()
        return result


def _build_eos(
    runtime: dict[str, Any],
    matter_type: str,
    amplitude: float,
    *,
    grid_points: int | None = None,
):
    deformation_config = runtime["deformation"]
    if grid_points is None:
        grid_points = int(_resolved_numerical_settings(runtime)["eos_grid_points"])
    deformation = GaussianDeformation(
        amplitude=float(amplitude),
        epsilon0=float(deformation_config["center_energy_density_mev_fm3"]),
        sigma=float(deformation_config["width_mev_fm3"]),
    )
    if matter_type == "hadronic":
        return build_hadronic_eos(
            runtime["hadronic_eos"]["baseline"],
            deformation,
            grid_points=grid_points,
        )
    if matter_type == "quark":
        return build_quark_eos(
            _quark_parameters(runtime),
            deformation,
            maximum_surface_energy_per_baryon=CONFIG["M_N"],
            grid_points=grid_points,
            catalog_identifier=runtime["resolved"]["quark_eos_id"],
        )
    raise ValueError("matter_type must be 'hadronic' or 'quark'.")


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
    screens = runtime["physical_requirements"]
    numerical = _resolved_numerical_settings(runtime)
    return solve_and_validate_sequence(
        eos,
        is_quark=matter_type == "quark",
        minimum_maximum_mass=float(screens["minimum_maximum_mass_msun"]),
        maximum_maximum_mass=float(screens["maximum_maximum_mass_msun"]),
        radius_14_bounds=(
            float(screens["radius_1p4_min_km"]),
            float(screens["radius_1p4_max_km"]),
        ),
        n_points=(
            int(numerical["central_pressure_points"]) if n_points is None else n_points
        ),
        rtol=(float(numerical["tov_relative_tolerance"]) if rtol is None else rtol),
        atol=(float(numerical["tov_absolute_tolerance"]) if atol is None else atol),
        enforce_physical_requirements=enforce_physical_requirements,
    )


def _run_convergence_checks(
    runtime: dict[str, Any], summary: pd.DataFrame
) -> pd.DataFrame:
    if runtime["numerical_settings"]["convergence_check"] == "none":
        return pd.DataFrame(columns=CONVERGENCE_COLUMNS)
    amplitudes = runtime["resolved"]["amplitudes"]
    selected = sorted({float(amplitudes[0]), 0.0, float(amplitudes[-1])})
    records = []
    base_lookup = {
        (str(row.matter_type), float(row.deformation_amplitude)): row
        for row in summary.itertuples(index=False)
    }
    numerical = _resolved_numerical_settings(runtime)
    variants = (
        ("double_eos_grid", 2 * int(numerical["eos_grid_points"]), None, None, None),
        (
            "double_central_pressure_grid",
            None,
            2 * int(numerical["central_pressure_points"]),
            None,
            None,
        ),
        (
            "tighter_tov_tolerances",
            None,
            None,
            float(numerical["tov_relative_tolerance"]) / 10.0,
            float(numerical["tov_absolute_tolerance"]) / 10.0,
        ),
    )
    for amplitude in selected:
        for matter_type in ("hadronic", "quark"):
            baseline = base_lookup.get((matter_type, amplitude))
            if baseline is None:
                records.append(
                    _failed_convergence_record(
                        runtime, matter_type, amplitude, "production_reference_missing"
                    )
                )
                continue
            for check, grid_points, n_points, rtol, atol in variants:
                try:
                    eos = _build_eos(
                        runtime, matter_type, amplitude, grid_points=grid_points
                    )
                    curve, _, _ = _solve(
                        runtime,
                        eos,
                        matter_type,
                        n_points=n_points,
                        rtol=rtol,
                        atol=atol,
                        enforce_physical_requirements=False,
                    )
                    frame = stellar_curve_to_frame(
                        curve,
                        eos,
                        matter_type,
                        "convergence",
                        f"convergence_{matter_type}_{amplitude}_{check}",
                    )
                    refined = summarize_stellar_curve(frame)
                    delta_mass = abs(
                        float(refined["maximum_mass_msun"])
                        - float(baseline.maximum_mass_msun)
                    )
                    delta_radius = abs(
                        float(refined["radius_1p4_km"]) - float(baseline.radius_1p4_km)
                    )
                    reference_tidal = float(baseline.tidal_deformability_1p4)
                    delta_tidal = abs(
                        float(refined["tidal_deformability_1p4"]) - reference_tidal
                    ) / abs(reference_tidal)
                    mass_passed = delta_mass <= 0.01
                    radius_passed = delta_radius <= 0.05
                    tidal_passed = delta_tidal <= 0.02
                    physical_passed, physical_reason = _physical_requirements_status(
                        runtime, refined
                    )
                    records.append(
                        {
                            "matter_type": matter_type,
                            "baseline_name": (
                                eos.catalog_identifier or eos.baseline_name
                            ),
                            "deformation_amplitude": amplitude,
                            "check": check,
                            "delta_maximum_mass_msun": delta_mass,
                            "delta_radius_1p4_km": delta_radius,
                            "relative_delta_tidal_deformability_1p4": delta_tidal,
                            "maximum_mass_passed": mass_passed,
                            "radius_1p4_passed": radius_passed,
                            "tidal_deformability_1p4_passed": tidal_passed,
                            "refined_physical_requirements_passed": physical_passed,
                            "refined_physical_requirements_reason": physical_reason,
                            "passed": (
                                mass_passed
                                and radius_passed
                                and tidal_passed
                                and physical_passed
                            ),
                        }
                    )
                except Exception as error:
                    records.append(
                        _failed_convergence_record(
                            runtime,
                            matter_type,
                            amplitude,
                            check,
                            reason=f"{type(error).__name__}: {error}",
                        )
                    )
    return pd.DataFrame.from_records(records, columns=CONVERGENCE_COLUMNS)


def _failed_convergence_record(
    runtime: dict[str, Any],
    matter_type: str,
    amplitude: float,
    check: str,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    baseline = (
        runtime["hadronic_eos"]["baseline"]
        if matter_type == "hadronic"
        else runtime["resolved"]["quark_eos_id"]
    )
    return {
        "matter_type": matter_type,
        "baseline_name": baseline,
        "deformation_amplitude": amplitude,
        "check": f"{check}: {reason}" if reason else check,
        "delta_maximum_mass_msun": np.nan,
        "delta_radius_1p4_km": np.nan,
        "relative_delta_tidal_deformability_1p4": np.nan,
        "maximum_mass_passed": False,
        "radius_1p4_passed": False,
        "tidal_deformability_1p4_passed": False,
        "refined_physical_requirements_passed": False,
        "refined_physical_requirements_reason": reason or "refinement unavailable",
        "passed": False,
    }


def _physical_requirements_status(
    runtime: dict[str, Any], observables: dict[str, Any]
) -> tuple[bool, str]:
    requirements = runtime["physical_requirements"]
    maximum_mass = float(observables["maximum_mass_msun"])
    radius_1p4 = float(observables["radius_1p4_km"])
    failures = []
    if not (
        float(requirements["minimum_maximum_mass_msun"])
        <= maximum_mass
        <= float(requirements["maximum_maximum_mass_msun"])
    ):
        failures.append("maximum mass outside configured interval")
    if not (
        float(requirements["radius_1p4_min_km"])
        <= radius_1p4
        <= float(requirements["radius_1p4_max_km"])
    ):
        failures.append("radius at 1.4 solar masses outside configured interval")
    return (not failures, "passed" if not failures else "; ".join(failures))


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
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).loc[:, list(columns)]


def _artifact_hashes(layout: RunLayout) -> dict[str, str]:
    paths = [
        layout.resolved_config,
        layout.data / "eos_tables.parquet",
        layout.data / "stellar_curves.parquet",
        layout.tables / "eos_summary.csv",
        layout.tables / "rejections.csv",
        layout.tables / "convergence.csv",
        layout.report,
        *sorted(layout.plots.glob("*.png")),
    ]
    return {
        str(path.relative_to(layout.root).as_posix()): file_sha256(path)
        for path in paths
        if path.is_file()
    }


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Resolved TOML cannot contain a non-finite number.")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported resolved TOML value: {type(value).__name__}")


__all__ = [
    "CONVERGENCE_COLUMNS",
    "PAIR_INTERPRETATION",
    "PairPreflight",
    "render_resolved_toml",
    "run_pair_experiment",
    "validate_pair_experiment",
]
