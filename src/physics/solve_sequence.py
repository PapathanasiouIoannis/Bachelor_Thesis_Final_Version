# src/physics/solve_sequence.py

r"""
  Integrates the TOV equations over a range of central pressures to generate
  a full Mass-Radius-Lambda sequence (an EoS curve).

Refactored:
  - MORPHOLOGY FIX: Expanded the central pressure search grid according to
    CONFIG["GRID_P_MIN_LOG"] and CONFIG["SOLVER_N_POINTS"]. This allows the solver 
    to find the extremely low-mass stars with highly expanded crusts, completely
    eliminating the "floating cut-off" visual artifact.
  - SENTINEL CHECKS: Safely skips unphysical branches returned by the EoS callable.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable

from src.config import CONFIG
from src.physics.tov_rhs import tov_rhs
from src.utils.exceptions import TovConvergenceError
from src.utils.logger import get_logger

logger = get_logger("SOLVE_SEQ")

_R_MIN = CONFIG["TOV_R_MIN"]
_R_MAX = CONFIG["TOV_R_MAX"]
_G_CONV = CONFIG["G_CONV"]
_A_CONV = CONFIG["A_CONV"]
_BUCHDAHL_LIMIT = CONFIG["BUCHDAHL_LIMIT"]


# event to detect surface
def _surface_event(t, y, *args):
    return y[1] - CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"]


_surface_event.terminal = True
_surface_event.direction = -1


def solve_sequence(eos_callable: Callable, is_quark: bool = False, p_max_causal: float = None, rtol: float = None, atol: float = None) -> tuple:
    """
    Integrates TOV for a sequence of central pressures to form a full star curve.

    Parameters:
    - eos_callable: A function/closure that takes Pressure [MeV/fm^3] and
                    returns (Energy_Density [MeV/fm^3], Sound_Speed_Squared).
    - is_quark: Boolean flag used strictly to optimize the pressure search grid.

    Returns:
    - curve_data: List of[Mass, Radius, Lambda, Pc, Eps_c, CS2_c, Eps_surf]
    - max_m: The maximum mass found in this sequence.
    """
    r_min = _R_MIN

    # ---------------------------------------------------------
    # PRESSURE GRID (The Morphology Fix)
    # ---------------------------------------------------------
    n_points = CONFIG["SOLVER_N_POINTS"]
    n_low = int(n_points * CONFIG["SOLVER_N_LOW_RATIO"])
    n_high = n_points - n_low

    p_max = p_max_causal if p_max_causal is not None else CONFIG["ABSOLUTE_P_MAX_FALLBACK"]
    p_max_log = np.log10(p_max) if p_max > 10**2.0 else 2.1 # fallback safeguard

    if is_quark:
        # quark stars are self-bound and have higher central pressures even at low masses
        # STRATIFIED SAMPLING: Boost high-mass core generation frequency
        p_low = np.logspace(-1.0, 2.0, n_low, endpoint=False)
        p_high = np.logspace(2.0, p_max_log, n_high)
        pressures = np.concatenate((p_low, p_high))
    else:
        pressures = np.geomspace(CONFIG["GRID_P_MIN_LOG"], p_max if p_max_causal is not None else 1000.0, CONFIG["SOLVER_N_POINTS"])

    curve_data = []
    dense_profiles = []

    # extract surface density (Default to 0.0 for Hadronic models)
    eps_surf = getattr(eos_callable, "eps_surf", 0.0)

    for pc in pressures:
        # ==============================================================
        # 1. INITIALIZATION (Get Core Microphysics)
        # ==============================================================
        eps_init, cs2_init = eos_callable(pc)

        # sentinel value check for unphysical roots/branches
        if np.isnan(eps_init) or eps_init < 0:
            continue

        # ==============================================================
        # 2. INTEGRATION (TOV Solver)
        # ==============================================================
        # initial Mass (Approximation for small r_min)
        m_init = (r_min**3) * eps_init * (_G_CONV / 3.0)

        # state Vector:[Mass, Pressure, y_tidal]
        y0 = [m_init, pc, 2.0]

        try:
            # integrate from r_min out to the boundary where P = 0
            # note: TOV_R_MAX was increased to 50.0 in const.py so thick crusts aren't truncated
            sol = solve_ivp(
                fun=tov_rhs,
                t_span=(r_min, _R_MAX),
                y0=y0,
                args=(eos_callable,),
                events=_surface_event,
                method="RK45",
                dense_output=True,
                rtol=rtol if rtol is not None else CONFIG["ODE_RTOL"],  # tightened for mathematical rigor
                atol=atol if atol is not None else CONFIG["ODE_ATOL"],
            )

            # check if the integration successfully hit the surface
            if sol.status == 1 and len(sol.t_events[0]) > 0:
                R = sol.t_events[0][0]
                M = sol.y_events[0][0][0]
                yR = sol.y_events[0][0][2]

                # boundary Invariant Assertions
                assert not np.isnan(M) and not np.isnan(R), "NaN detected in TOV Mass or Radius!"
                assert not np.isinf(M) and not np.isinf(R), "Inf detected in TOV Mass or Radius!"
                assert M > 0.0, f"Unphysical mass detected! M={M} is not strictly positive."
                assert R > 0.0, f"Unphysical radius detected! R={R} is not strictly positive."

                # filter unphysical results
                if R < CONFIG["MIN_RADIUS_CUTOFF"] or M < CONFIG["MIN_MASS_CUTOFF"]:
                    continue

                # generate dense evaluation grid for the interior profile
                r_dense = np.linspace(r_min, R, CONFIG["DENSE_PROFILES_POINTS"])
                y_dense = sol.sol(r_dense)
                m_dense = y_dense[0]

                # ==============================================================
                # 3. MACROPHYSICS (Tidal Deformability)
                # ==============================================================
                # calculate Compactness
                C = (M * _A_CONV) / R

                # STRICT BUCHDAHL LIMIT (C < 4/9)
                if C >= _BUCHDAHL_LIMIT:
                    continue

                # complex Tidal Love Number (k2) formula (Hinderer et al. 2008)
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
                    * (
                        13.0
                        - 11.0 * yR
                        + C * (3.0 * yR - 2.0)
                        + 2.0 * (C**2) * (1.0 + yR)
                    )
                )
                den_term3 = (
                    3.0
                    * (1.0 - 2.0 * C) ** 2
                    * (2.0 - yR + 2.0 * C * (yR - 1.0))
                    * np.log(1.0 - 2.0 * C)
                )

                den = den_term1 + den_term2 + den_term3

                if abs(den) < 1e-25:
                    continue

                k2 = num / den

                # dimensionless Tidal Deformability
                Lam = (2.0 / 3.0) * k2 * (C**-5)

                if M <= 0.0:
                    break

                # record point
                curve_data.append([M, R, Lam, pc, eps_init, cs2_init, eps_surf])
                dense_profiles.append((r_dense, m_dense))

        except (ValueError, RuntimeError, ArithmeticError) as e:
            # trap specific ODE solver integration faults and re-raise them as our domain error
            try:
                raise TovConvergenceError(pc=pc, reason=str(e)) from e
            except TovConvergenceError:
                logger.exception("ODE Solver failed due to domain error")
                continue

    if not curve_data:
        return [], [], 0.0

    # convert to arrays for brute-force slicing
    curve_arr = np.array(curve_data)
    mass_array = curve_arr[:, 0]
    radius_array = curve_arr[:, 1]
    lambda_array = curve_arr[:, 2]
    pressure_array = curve_arr[:, 3]
    density_array = curve_arr[:, 4]
    cs2_array = curve_arr[:, 5]
    eps_surf_array = curve_arr[:, 6]

    # implement a hard, brute-force slice using np.argmax
    max_mass_idx = int(np.argmax(mass_array))

    mass_stable = mass_array[: max_mass_idx + 1]
    radius_stable = radius_array[: max_mass_idx + 1]
    lambda_stable = lambda_array[: max_mass_idx + 1]
    pressure_stable = pressure_array[: max_mass_idx + 1]
    density_stable = density_array[: max_mass_idx + 1]
    cs2_stable = cs2_array[: max_mass_idx + 1]
    eps_surf_stable = eps_surf_array[: max_mass_idx + 1]

    # reconstruct the expected list of lists and truncate profiles
    curve_stable = [
        [m, r, lam, p, d, cs2, e_surf]
        for m, r, lam, p, d, cs2, e_surf in zip(
            mass_stable,
            radius_stable,
            lambda_stable,
            pressure_stable,
            density_stable,
            cs2_stable,
            eps_surf_stable,
        )
    ]

    dense_profiles_stable = dense_profiles[: max_mass_idx + 1]
    max_m_stable = float(mass_stable[max_mass_idx])

    return curve_stable, dense_profiles_stable, max_m_stable
