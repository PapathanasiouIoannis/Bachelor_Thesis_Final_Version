"""Generate the controlled APR-1 Gaussian-amplitude sweep."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from framework.eos_sweep import (
    GaussianDeformation,
    SweepPoint,
    admissible_amplitude_interval,
    build_hadronic_eos,
    hadronic_baseline_grids,
    resolve_density_shifted_transition,
    validate_sweep_within_interval,
)
from src.config import CONFIG
from src.physics.controlled_generation import curve_to_dataframe, solve_and_validate_sequence


def worker_hadronic_gen(
    sweep_points: Sequence[SweepPoint],
    batch_idx: int = 0,
) -> pd.DataFrame:
    """Generate exactly one APR-1 curve for every requested amplitude."""

    del batch_idx  # IDs derive from physical parameters, not execution order.
    sweep_points = list(sweep_points)
    if not sweep_points:
        return pd.DataFrame(columns=CONFIG["COLUMN_SCHEMA"])

    baseline_name = CONFIG["CONTROLLED_HADRONIC_BASELINE"]
    if baseline_name != "APR-1":
        raise ValueError(
            "The controlled experiment is intentionally restricted to the APR-1 baseline."
        )
    energy_density, baseline_cs2, _, _, _ = hadronic_baseline_grids(baseline_name)
    interval = admissible_amplitude_interval(
        energy_density,
        baseline_cs2,
        CONFIG["CONTROLLED_PERTURB_EPS0"],
        CONFIG["CONTROLLED_PERTURB_SIGMA"],
    )
    validate_sweep_within_interval(sweep_points, interval, "APR-1")

    frames = []
    for sweep_point in sweep_points:
        deformation = GaussianDeformation(
            amplitude=sweep_point.amplitude,
            epsilon0=CONFIG["CONTROLLED_PERTURB_EPS0"],
            sigma=CONFIG["CONTROLLED_PERTURB_SIGMA"],
        )
        framework_eos = build_hadronic_eos(baseline_name, deformation)
        try:
            curve, features, maximum_mass = solve_and_validate_sequence(
                framework_eos, is_quark=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"APR-1 sweep point {sweep_point.sweep_id} "
                f"(A={sweep_point.amplitude:.8g}) failed: {exc}"
            ) from exc
        frames.append(
            curve_to_dataframe(
                curve,
                features,
                maximum_mass,
                framework_eos,
                sweep_point,
                label=0,
            )
        )
    return pd.concat(frames, ignore_index=True)


__all__ = ["resolve_density_shifted_transition", "worker_hadronic_gen"]
