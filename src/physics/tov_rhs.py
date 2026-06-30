# src/physics/tov_rhs.py

"""
  Computes the derivatives for the Tolman-Oppenheimer-Volkoff (TOV) equations
  coupled with the Tidal Deformability Riccati equation.

Refactored:
  - PURE GRAVITY ENGINE: Decoupled entirely from specific EoS physics.
    All Hadronic mixing, Generalized CFL root-finding, and crust transitions
    have been moved out. It now purely evaluates General Relativity.
  - SMOOTH INTEGRATION: By receiving a pre-clamped, C^1 continuous `eos_callable`
    from the workers, we eliminate the C^0 continuous shockwaves (if cs2 > 1: clamp)
    that previously caused Runge-Kutta step-size collapse and stiffness.
"""

import numba
from typing import Callable
from typing import List

from src.config import CONFIG

_P_MIN_SAFE = CONFIG["TOV_P_MIN_SAFE"]
_G_CONV = CONFIG["G_CONV"]
_A_CONV = CONFIG["A_CONV"]
_TOV_SINGULARITY_LIMIT = CONFIG["TOV_SINGULARITY_LIMIT"]


@numba.njit
def taylor_expansion(r: float, P_safe: float, epsilon: float, cs2_local: float, G_CONV: float, A_CONV: float) -> list:
    dm_dr = (r**2) * epsilon * G_CONV
    dP_dr = -A_CONV * G_CONV * (epsilon + P_safe) * (epsilon / 3.0 + P_safe) * r
    dy_dr = (
        -(2.0 / 7.0)
        * A_CONV
        * G_CONV
        * r
        * (11.0 * P_safe + epsilon / 3.0 + (epsilon + P_safe) / cs2_local)
    )
    return [dm_dr, dP_dr, dy_dr]


@numba.njit
def tov_equations(r: float, m: float, P_safe: float, y_tidal: float, epsilon: float, cs2_local: float, G_CONV: float, A_CONV: float) -> list:
    term_1 = epsilon + P_safe
    term_2 = m + (r**3 * P_safe * G_CONV)
    term_3 = r * (r - 2.0 * m * A_CONV)

    # singularity / Horizon protection
    if abs(term_3) < _TOV_SINGULARITY_LIMIT:
        return [0.0, 0.0, 0.0]

    dP_dr = -A_CONV * (term_1 * term_2) / term_3
    dm_dr = (r**2) * epsilon * G_CONV

    exp_lambda = 1.0 / (1.0 - 2.0 * A_CONV * m / r)
    Q = (
        A_CONV
        * G_CONV
        * (5.0 * epsilon + 9.0 * P_safe + (epsilon + P_safe) / cs2_local)
        * (r**2)
    )
    Q -= 6.0 * exp_lambda

    F = (1.0 - A_CONV * G_CONV * (r**2) * (epsilon - P_safe)) * exp_lambda
    dy_dr = -(y_tidal**2 + y_tidal * F + Q) / r

    return [dm_dr, dP_dr, dy_dr]


def tov_rhs(r: float, y_state: list, eos_callable: Callable) -> list:
    """
    Computes the derivatives for the TOV and Tidal Deformability equations.

    Parameters:
    - r: Current radius (integration variable) [km]
    - y_state: Array containing[Mass (M_sun), Pressure (MeV/fm^3), y_tidal (dimensionless)]
    - eos_callable: A function/closure that takes Pressure and returns
                    (Energy_Density [MeV/fm^3], Sound_Speed_Squared).

    Returns:
    - [dm_dr, dP_dr, dy_dr]
    """
    # 1. Prevent 0/0 Division: Safe limit handler at the center of the star
    r = max(r, 1e-10)

    m, P, y_tidal = y_state

    # 2. Surface Cutoff: Defensive guard clause for non-physical negative pressure
    if P < CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"]:
        return [0.0, 0.0, 0.0]

    # ensure P is never strictly zero to avoid numerical instability
    P_safe = max(P, _P_MIN_SAFE)

    # ==========================================
    # 1. MICROPHYSICS (Thermodynamics)
    # ==========================================
    # evaluate the agnostic EoS callable (Spline for Hadronic, Algebraic for Quark)
    epsilon, cs2_local = eos_callable(P_safe)

    # terminate integration safely if density becomes unphysical
    if epsilon <= 0:
        return [0.0, 0.0, 0.0]

    # protect against divisions by zero in the Riccati equation
    if cs2_local < 1e-10:
        cs2_local = 1e-10

    # center boundary condition (r -> 0) using a regularized core expansion
    # to prevent 0/0 NaN division spikes when the integrator initializes.
    if r < CONFIG["TOV_CENTER_R_LIMIT"]:
        return taylor_expansion(r, P_safe, epsilon, cs2_local, _G_CONV, _A_CONV)

    # ==========================================
    # 2. MACROPHYSICS (General Relativity)
    # ==========================================
    return tov_equations(r, m, P_safe, y_tidal, epsilon, cs2_local, _G_CONV, _A_CONV)
