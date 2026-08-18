"""Preflight validation and report metadata for pair experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from framework.eos_sweep import QuarkParameters, SweepPoint
from src.experiment_config import ResolvedExperiment
from src.physics.runner.settings import resolved_numerical_settings


PAIR_INTERPRETATION = (
    "Controlled sensitivity comparison between one repository hadronic surrogate "
    "and one analytic CFL MIT-bag equation of state. This is not universal "
    "matter-phase classification."
)


def _report_numerical_settings(runtime: dict[str, Any]) -> dict[str, int | float]:
    return resolved_numerical_settings(runtime)


def _report_pair_interpretation() -> str:
    return PAIR_INTERPRETATION


REPORT_NUMERICAL_SETTINGS = _report_numerical_settings
REPORT_PAIR_INTERPRETATION = _report_pair_interpretation


@dataclass(frozen=True)
class PairPreflight:
    resolved: ResolvedExperiment
    runtime_configuration: dict[str, Any]
    sweep_points: tuple[SweepPoint, ...]
    hadronic_interval: tuple[float, float]
    quark_interval: tuple[float, float]
    common_interval: tuple[float, float]
    provenance: dict[str, Any]
    baseline_recovery: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        configuration = self.runtime_configuration
        return {
            "experiment_name": configuration["experiment_name"],
            "workflow": configuration["workflow"],
            "mode": configuration["mode"],
            "configuration_hash": self.resolved.config_hash,
            "hadronic_eos": configuration["hadronic_eos"],
            "quark_eos": {
                **configuration["quark_eos"],
                "catalog_identifier": self.resolved.quark_eos_id,
            },
            "deformation": {
                **configuration["deformation"],
                "amplitudes": [point.amplitude for point in self.sweep_points],
            },
            "physical_requirements": configuration["physical_requirements"],
            "numerical_settings": configuration["numerical_settings"],
            "resolved_numerical_settings": REPORT_NUMERICAL_SETTINGS(configuration),
            "execution": configuration["execution"],
            "admissible_amplitude_intervals": {
                "hadronic": list(self.hadronic_interval),
                "quark": list(self.quark_interval),
                "common": list(self.common_interval),
                "lower_endpoint_is_open": True,
            },
            "baseline_recovery_maximum_relative_pressure_error": (
                self.baseline_recovery
            ),
            "expected_curves": 2 * len(self.sweep_points),
            "classification_enabled": False,
            "permitted_scientific_interpretation": REPORT_PAIR_INTERPRETATION(),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class ValidationDependencies:
    """Facade-owned operations used while validating one pair profile."""

    resolve_pair_experiment: Callable[[Any], ResolvedExperiment]
    resolved_experiment_type: type
    sweep_point_type: Callable[..., SweepPoint]
    pair_preflight_type: Callable[..., PairPreflight]
    quark_parameters: Callable[[dict[str, Any]], QuarkParameters]
    resolved_numerical_settings: Callable[
        [dict[str, Any]], dict[str, int | float]
    ]
    hadronic_baseline_grids: Callable[..., tuple]
    cfl_baseline_grids: Callable[..., tuple]
    admissible_amplitude_interval: Callable[..., tuple[float, float]]
    validate_sweep_within_interval: Callable[..., None]
    baseline_recovery_errors: Callable[[dict[str, Any]], dict[str, float]]
    provenance: Callable[[dict[str, Any], str], dict[str, Any]]


@dataclass(frozen=True)
class BaselineRecoveryDependencies:
    """Facade-owned operations used by the A=0 reconstruction check."""

    resolved_numerical_settings: Callable[
        [dict[str, Any]], dict[str, int | float]
    ]
    hadronic_baseline_grids: Callable[..., tuple]
    build_hadronic_eos: Callable[..., Any]
    quark_parameters: Callable[[dict[str, Any]], QuarkParameters]
    cfl_baseline_grids: Callable[..., tuple]
    build_quark_eos: Callable[..., Any]
    gaussian_deformation: Callable[..., Any]
    maximum_relative_pressure_error: Callable[[Any, Any], float]
    configuration: Mapping[str, Any]


def validate_pair_experiment(
    configuration: ResolvedExperiment | str | Path,
    dependencies: ValidationDependencies,
) -> PairPreflight:
    """Validate a profile and its common causal amplitude support without TOV runs."""

    resolved = (
        configuration
        if isinstance(configuration, dependencies.resolved_experiment_type)
        else dependencies.resolve_pair_experiment(configuration)
    )
    runtime = resolved.to_runtime_dict()
    hadronic_name = str(runtime["hadronic_eos"]["baseline"])
    quark_parameters = dependencies.quark_parameters(runtime)
    deformation = runtime["deformation"]
    numerical = dependencies.resolved_numerical_settings(runtime)
    center = float(deformation["center_energy_density_mev_fm3"])
    width = float(deformation["width_mev_fm3"])
    points = tuple(
        dependencies.sweep_point_type(index=index, amplitude=float(amplitude))
        for index, amplitude in enumerate(runtime["resolved"]["amplitudes"])
    )

    hadronic_energy, hadronic_sound_speed, _, _, _ = (
        dependencies.hadronic_baseline_grids(
            hadronic_name,
            numerical["eos_grid_points"],
        )
    )
    hadronic_interval = dependencies.admissible_amplitude_interval(
        hadronic_energy,
        hadronic_sound_speed,
        center,
        width,
    )
    _, quark_energy, quark_sound_speed, _ = dependencies.cfl_baseline_grids(
        quark_parameters,
        numerical["eos_grid_points"],
    )
    quark_interval = dependencies.admissible_amplitude_interval(
        quark_energy,
        quark_sound_speed,
        center,
        width,
    )
    common_interval = (
        max(hadronic_interval[0], quark_interval[0]),
        min(hadronic_interval[1], quark_interval[1]),
    )
    try:
        dependencies.validate_sweep_within_interval(
            points,
            common_interval,
            "Common pair",
        )
    except ValueError as error:
        start = points[0].amplitude
        stop = points[-1].amplitude
        lower, upper = common_interval
        if start <= lower:
            advice = (
                f"amplitude_start = {start:g} is not above the common permitted "
                f"lower boundary {lower:.6g}. Choose a larger value."
            )
        elif stop > upper:
            advice = (
                f"amplitude_stop = {stop:g} exceeds the common permitted maximum "
                f"of {upper:.6g}. Choose a smaller value."
            )
        else:
            advice = str(error)
        raise ValueError(advice) from error

    recovery = dependencies.baseline_recovery_errors(runtime)
    for matter_type, error in recovery.items():
        if error > 2.0e-4:
            raise ValueError(
                f"A = 0 {matter_type} maximum relative pressure-recovery error "
                f"is {error:.6g}, "
                "above the permitted maximum relative tolerance 0.0002."
            )
    return dependencies.pair_preflight_type(
        resolved=resolved,
        runtime_configuration=runtime,
        sweep_points=points,
        hadronic_interval=hadronic_interval,
        quark_interval=quark_interval,
        common_interval=common_interval,
        provenance=dependencies.provenance(runtime, resolved.quark_eos_id),
        baseline_recovery=recovery,
    )


def baseline_recovery_errors(
    runtime: dict[str, Any],
    dependencies: BaselineRecoveryDependencies,
) -> dict[str, float]:
    """Measure A=0 reconstruction error for both baseline matter models."""

    deformation = runtime["deformation"]
    grid_points = int(
        dependencies.resolved_numerical_settings(runtime)["eos_grid_points"]
    )
    zero = dependencies.gaussian_deformation(
        amplitude=0.0,
        epsilon0=float(deformation["center_energy_density_mev_fm3"]),
        sigma=float(deformation["width_mev_fm3"]),
    )
    name = runtime["hadronic_eos"]["baseline"]
    _, _, _, transition, _ = dependencies.hadronic_baseline_grids(name, grid_points)
    hadronic = dependencies.build_hadronic_eos(name, zero, grid_points=grid_points)
    baseline_hadronic_pressure = np.linspace(
        transition,
        dependencies.configuration["P_GRID_MAX"],
        grid_points,
    )[: len(hadronic.pressure)]
    hadronic_error = dependencies.maximum_relative_pressure_error(
        hadronic.pressure,
        baseline_hadronic_pressure,
    )

    quark_parameters_value = dependencies.quark_parameters(runtime)
    baseline_quark_pressure, _, _, _ = dependencies.cfl_baseline_grids(
        quark_parameters_value,
        grid_points,
    )
    quark = dependencies.build_quark_eos(
        quark_parameters_value,
        zero,
        maximum_surface_energy_per_baryon=dependencies.configuration["M_N"],
        grid_points=grid_points,
    )
    quark_error = dependencies.maximum_relative_pressure_error(
        quark.pressure,
        baseline_quark_pressure[: len(quark.pressure)],
    )
    return {"hadronic": hadronic_error, "quark": quark_error}


def maximum_relative_pressure_error(reconstructed, baseline) -> float:
    """Return the maximum pointwise relative error for aligned pressure grids."""

    reconstructed = np.asarray(reconstructed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if reconstructed.shape != baseline.shape or len(baseline) < 5:
        raise ValueError("A = 0 pressure recovery grids are not aligned.")
    denominator = np.maximum(np.abs(baseline), 1.0e-12)
    relative_error = np.abs(reconstructed - baseline) / denominator
    return float(np.max(relative_error))


def provenance(
    runtime: dict[str, Any],
    quark_eos_id: str,
    hadronic_catalog: Sequence[Any],
    cfl_catalog: Sequence[Any],
) -> dict[str, Any]:
    """Build the catalog-aware provenance payload for one resolved pair."""

    hadronic_name = runtime["hadronic_eos"]["baseline"]
    hadronic = next(
        entry for entry in hadronic_catalog if entry.eos_id == hadronic_name
    )
    quark_match = next(
        (entry for entry in cfl_catalog if entry.eos_id == quark_eos_id),
        None,
    )
    return {
        "hadronic": hadronic.as_row(),
        "quark": (
            quark_match.as_row()
            if quark_match is not None
            else {
                "eos_id": quark_eos_id,
                "model_family": "analytic CFL MIT-bag",
                "provenance_note": (
                    "Exploratory custom parameter tuple; the tuple itself is not a "
                    "named benchmark in the repository literature catalog."
                ),
            }
        ),
    }
