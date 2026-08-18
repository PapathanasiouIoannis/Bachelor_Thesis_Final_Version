"""Surface-density and tidal-deformability calculations for stellar models."""

import numpy as np


def apply_surface_density_correction(
    yR: float,
    R: float,
    M: float,
    eps_surf: float,
    *,
    gravitational_conversion: float,
) -> float:
    """Apply the self-bound surface-density jump correction to y(R)."""
    if eps_surf <= 0.0:
        return yR
    delta_yR = gravitational_conversion * (R**3) * eps_surf / M
    return yR - delta_yR


def _tidal_lambda_from_y(C: float, yR: float) -> float | None:
    num = (
        (8.0 / 5.0)
        * (1.0 - 2.0 * C) ** 2
        * C**5
        * (2.0 * C * (yR - 1.0) - yR + 2.0)
    )

    den_term1 = 2.0 * C * (6.0 - 3.0 * yR + 3.0 * C * (5.0 * yR - 8.0))
    den_term2 = (
        4.0
        * (C**3)
        * (13.0 - 11.0 * yR + C * (3.0 * yR - 2.0) + 2.0 * (C**2) * (1.0 + yR))
    )
    den_term3 = (
        3.0
        * (1.0 - 2.0 * C) ** 2
        * (2.0 - yR + 2.0 * C * (yR - 1.0))
        * np.log(1.0 - 2.0 * C)
    )

    den = den_term1 + den_term2 + den_term3
    if abs(den) < 1e-25:
        return None

    k2 = num / den
    return (2.0 / 3.0) * k2 * (C**-5)
