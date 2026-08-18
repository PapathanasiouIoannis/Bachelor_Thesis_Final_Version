"""Shared numerical and analytic-CFL settings for pair experiments."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from framework.eos_sweep import QuarkParameters
from src.config import CONFIG


def numerical_presets_from_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, dict[str, int | float]]:
    """Snapshot named runner presets from the repository configuration."""

    return {
        "production": {
            "eos_grid_points": int(configuration["P_GRID_POINTS"]),
            "central_pressure_points": int(configuration["SOLVER_N_POINTS"]),
            "tov_relative_tolerance": float(configuration["TOV_RTOL"]),
            "tov_absolute_tolerance": float(configuration["TOV_ATOL"]),
        },
        "smoke": {
            "eos_grid_points": 5000,
            "central_pressure_points": 80,
            "tov_relative_tolerance": 1.0e-7,
            "tov_absolute_tolerance": 1.0e-9,
        },
    }


NUMERICAL_PRESETS = numerical_presets_from_configuration(CONFIG)


def quark_parameters(
    runtime: dict[str, Any],
    parameter_type: Callable[..., QuarkParameters] = QuarkParameters,
) -> QuarkParameters:
    """Build the analytic CFL parameter tuple from a resolved runtime mapping."""

    quark = runtime["quark_eos"]
    return parameter_type(
        bag_b=float(quark["bag_constant_mev_fm3"]),
        gap_delta=float(quark["pairing_gap_mev"]),
        strange_mass=float(quark["strange_quark_mass_mev"]),
    )


def resolved_numerical_settings(
    runtime: dict[str, Any],
    numerical_presets: Mapping[str, Mapping[str, int | float]] | None = None,
) -> dict[str, int | float]:
    """Resolve one named numerical preset to a fresh settings mapping."""

    preset = str(runtime["numerical_settings"]["preset"])
    presets = NUMERICAL_PRESETS if numerical_presets is None else numerical_presets
    try:
        return dict(presets[preset])
    except KeyError as error:
        raise ValueError(f"Unknown numerical preset: {preset!r}.") from error
