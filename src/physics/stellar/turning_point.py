"""Turning-point extraction for pressure-ordered stellar sequences."""

from collections.abc import Sequence
from typing import Any

import numpy as np


_MASS_NOISE_RTOL = 1.0e-6
_MASS_NOISE_ATOL = 1.0e-10


class TurningPointError(ValueError):
    """Raised when a stellar sequence does not bracket a usable first mass peak."""


def _extract_first_turning_point(
    curve_data: Sequence[Sequence[float]],
    dense_profiles: Sequence[Any],
) -> tuple[list[list[float]], list[Any], float]:
    """Return the branch through the first resolved mass turning point.

    The input must be ordered by strictly increasing central pressure (column
    index 3).  A peak is considered resolved only after the sequence has shown
    a mass increase and then a decrease larger than a small numerical-noise
    tolerance.  This is a turning-point stability estimate; it is not a proof
    based on a radial-mode calculation.
    """

    try:
        curve_array = np.asarray(curve_data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TurningPointError(
            "Stellar sequence data must form a rectangular numeric table."
        ) from exc

    if curve_array.ndim != 2 or curve_array.shape[1] < 4:
        raise TurningPointError(
            "Stellar sequence data must be a two-dimensional table with mass "
            "and central-pressure columns."
        )
    point_count = curve_array.shape[0]
    if point_count < 3:
        raise TurningPointError(
            "At least three stellar models are required to bracket a mass turning point."
        )
    if len(dense_profiles) != point_count:
        raise TurningPointError(
            "The dense-profile count must match the stellar-sequence point count."
        )
    if not np.all(np.isfinite(curve_array)):
        raise TurningPointError("Stellar sequence data contain non-finite values.")

    masses = curve_array[:, 0]
    central_pressures = curve_array[:, 3]
    if np.any(masses <= 0.0):
        raise TurningPointError("Stellar masses must be finite and strictly positive.")
    if np.any(np.diff(central_pressures) <= 0.0):
        raise TurningPointError(
            "Stellar models must be ordered by strictly increasing central pressure."
        )

    mass_scale = max(1.0, float(np.max(np.abs(masses))))
    mass_tolerance = max(_MASS_NOISE_ATOL, _MASS_NOISE_RTOL * mass_scale)
    mass_changes = np.diff(masses)

    increasing_branch_seen = False
    first_decrease_index: int | None = None
    for index, change in enumerate(mass_changes):
        if change > mass_tolerance:
            increasing_branch_seen = True
        elif change < -mass_tolerance:
            if not increasing_branch_seen:
                raise TurningPointError(
                    "The sequence decreases before a resolved increasing-mass branch; "
                    "its first mass maximum is at the lower-pressure boundary."
                )
            first_decrease_index = index
            break

    if first_decrease_index is None:
        if increasing_branch_seen:
            raise TurningPointError(
                "Maximum mass is not bracketed: no resolved post-peak decrease was "
                "sampled, so the maximum may lie beyond the final central pressure."
            )
        raise TurningPointError(
            "The sequence contains no resolved increasing-mass branch or mass maximum."
        )

    # Tiny changes around a flat peak are treated as numerical noise.  Retain
    # the actual largest sampled mass preceding the first resolved decrease.
    peak_index = int(np.argmax(masses[: first_decrease_index + 1]))
    if peak_index == 0:
        raise TurningPointError(
            "The first resolved mass maximum occurs at the lower-pressure boundary."
        )
    if peak_index >= point_count - 1:
        raise TurningPointError(
            "The first resolved mass maximum occurs at the final sampled pressure."
        )

    curve_stable = [list(point) for point in curve_data[: peak_index + 1]]
    profiles_stable = list(dense_profiles[: peak_index + 1])
    return curve_stable, profiles_stable, float(masses[peak_index])
