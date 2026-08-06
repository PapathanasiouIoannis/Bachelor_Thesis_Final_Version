"""Generate the controlled published-CFL4 Gaussian-amplitude sweep."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from framework.eos_sweep import (
    GaussianDeformation,
    QuarkParameters,
    SweepPoint,
    admissible_amplitude_interval,
    build_quark_eos,
    cfl_baseline_grids,
    validate_sweep_within_interval,
)
from src.config import CONFIG
from src.physics.controlled_generation import curve_to_dataframe, solve_and_validate_sequence


def controlled_quark_parameters() -> QuarkParameters:
    """Return the fixed CFL4 tuple in the repository's documented units."""

    return QuarkParameters(
        bag_b=CONFIG["CONTROLLED_QUARK_B"],
        gap_delta=CONFIG["CONTROLLED_QUARK_DELTA"],
        strange_mass=CONFIG["CONTROLLED_QUARK_MS"],
    )


def worker_quark_gen(
    sweep_points: Sequence[SweepPoint],
    batch_idx: int = 0,
) -> pd.DataFrame:
    """Generate exactly one fixed-CFL4 curve for every requested amplitude."""

    del batch_idx
    sweep_points = list(sweep_points)
    if not sweep_points:
        return pd.DataFrame(columns=CONFIG["COLUMN_SCHEMA"])

    parameters = controlled_quark_parameters()
    _, energy_density, baseline_cs2, _ = cfl_baseline_grids(parameters)
    interval = admissible_amplitude_interval(
        energy_density,
        baseline_cs2,
        CONFIG["CONTROLLED_PERTURB_EPS0"],
        CONFIG["CONTROLLED_PERTURB_SIGMA"],
    )
    validate_sweep_within_interval(sweep_points, interval, "CFL4")

    frames = []
    for sweep_point in sweep_points:
        deformation = GaussianDeformation(
            amplitude=sweep_point.amplitude,
            epsilon0=CONFIG["CONTROLLED_PERTURB_EPS0"],
            sigma=CONFIG["CONTROLLED_PERTURB_SIGMA"],
        )
        framework_eos = build_quark_eos(
            parameters,
            deformation,
            maximum_surface_energy_per_baryon=CONFIG["M_N"],
        )
        try:
            curve, features, maximum_mass = solve_and_validate_sequence(
                framework_eos, is_quark=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"CFL4 sweep point {sweep_point.sweep_id} "
                f"(A={sweep_point.amplitude:.8g}) failed: {exc}"
            ) from exc
        frames.append(
            curve_to_dataframe(
                curve,
                features,
                maximum_mass,
                framework_eos,
                sweep_point,
                label=1,
            )
        )
    return pd.concat(frames, ignore_index=True)


__all__ = ["controlled_quark_parameters", "worker_quark_gen"]
