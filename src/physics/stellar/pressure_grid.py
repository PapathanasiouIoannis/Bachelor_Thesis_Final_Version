"""Central-pressure sampling grids for stellar sequences."""

from typing import Any, Mapping

import numpy as np


def build_central_pressure_grid(
    *,
    is_quark: bool,
    p_max_causal: float | None,
    n_points: int | None,
    configuration: Mapping[str, Any],
) -> np.ndarray:
    """Build the legacy stratified central-pressure grid."""
    if n_points is None:
        solver_n_points = int(configuration["SOLVER_N_POINTS"])
    elif isinstance(n_points, (bool, np.bool_)) or not isinstance(
        n_points, (int, np.integer)
    ):
        raise ValueError("n_points must be an integer greater than or equal to 4.")
    else:
        solver_n_points = int(n_points)
    if solver_n_points < 4:
        raise ValueError("n_points must be an integer greater than or equal to 4.")

    n_low = int(solver_n_points * configuration["SOLVER_N_LOW_RATIO"])
    n_high = solver_n_points - n_low

    p_max = (
        p_max_causal
        if p_max_causal is not None
        else configuration["ABSOLUTE_P_MAX_FALLBACK"]
    )
    p_max_log = np.log10(p_max) if p_max > 10**2.0 else 2.1

    if is_quark:
        p_low = np.logspace(-1.0, 2.0, n_low, endpoint=False)
        p_high = np.logspace(2.0, p_max_log, n_high)
        return np.concatenate((p_low, p_high))

    return np.geomspace(
        configuration["GRID_P_MIN_LOG"],
        p_max if p_max_causal is not None else 1000.0,
        solver_n_points,
    )
