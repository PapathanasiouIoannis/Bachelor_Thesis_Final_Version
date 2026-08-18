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
from src.physics.stellar import integration as _integration
from src.physics.stellar import pressure_grid as _pressure_grid
from src.physics.stellar import tidal as _tidal
from src.physics.stellar.turning_point import (
    TurningPointError as TurningPointError,
    _MASS_NOISE_ATOL as _MASS_NOISE_ATOL,
    _MASS_NOISE_RTOL as _MASS_NOISE_RTOL,
    _extract_first_turning_point as _extract_first_turning_point,
)
from src.physics.tov_rhs import tov_rhs
from src.utils.exceptions import TovConvergenceError
from src.utils.logger import get_logger

logger = get_logger("SOLVE_SEQ")

_R_MIN = CONFIG["TOV_R_MIN"]
_R_MAX = CONFIG["TOV_R_MAX"]
_G_CONV = CONFIG["G_CONV"]
_A_CONV = CONFIG["A_CONV"]
_BUCHDAHL_LIMIT = CONFIG["BUCHDAHL_LIMIT"]

_tidal_lambda_from_y = _tidal._tidal_lambda_from_y


def _apply_surface_density_correction(
    yR: float, R: float, M: float, eps_surf: float
) -> float:
    """Apply the self-bound surface-density jump correction to y(R)."""
    return _tidal.apply_surface_density_correction(
        yR,
        R,
        M,
        eps_surf,
        gravitational_conversion=_G_CONV,
    )


# event to detect surface
def _surface_event(t, y, *args):
    return y[1] - CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"]


_surface_event.terminal = True
_surface_event.direction = -1


def solve_sequence(
    eos_callable: Callable,
    is_quark: bool = False,
    p_max_causal: float = None,
    rtol: float = None,
    atol: float = None,
    *,
    n_points: int | None = None,
) -> tuple:
    """
    Integrates TOV for a sequence of central pressures to form a full star curve.

    Parameters:
    - eos_callable: A function/closure that takes Pressure [MeV/fm^3] and
                    returns (Energy_Density [MeV/fm^3], Sound_Speed_Squared).
    - is_quark: Boolean flag used strictly to optimize the pressure search grid.
    - n_points: Optional central-pressure sample count. This supports numerical
                convergence checks without changing the global configuration.

    Returns:
    - curve_data: List of[Mass, Radius, Lambda, Pc, Eps_c, CS2_c, Eps_surf]
    - max_m: The maximum mass found in this sequence.
    """
    r_min = _R_MIN

    pressures = _pressure_grid.build_central_pressure_grid(
        is_quark=is_quark,
        p_max_causal=p_max_causal,
        n_points=n_points,
        configuration=CONFIG,
    )

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

        initial_mass = (r_min**3) * eps_init * (_G_CONV / 3.0)
        initial_state = [initial_mass, pc, 2.0]
        integration_context = _integration.IntegrationContext(
            integrator=solve_ivp,
            rhs=tov_rhs,
            surface_event=_surface_event,
            surface_density_correction=_apply_surface_density_correction,
            tidal_lambda=_tidal_lambda_from_y,
            r_min=r_min,
            r_max=_R_MAX,
            compactness_conversion=_A_CONV,
            buchdahl_limit=_BUCHDAHL_LIMIT,
        )

        try:
            integration_result = _integration.integrate_stellar_model(
                eos_callable=eos_callable,
                central_pressure=pc,
                central_energy_density=eps_init,
                central_sound_speed_squared=cs2_init,
                surface_energy_density=eps_surf,
                initial_state=initial_state,
                rtol=rtol,
                atol=atol,
                configuration=CONFIG,
                context=integration_context,
            )
            if integration_result is _integration.IntegrationDecision.STOP:
                break
            if integration_result is _integration.IntegrationDecision.SKIP:
                continue

            assert isinstance(
                integration_result,
                _integration.IntegratedStellarModel,
            )
            curve_data.append(integration_result.curve_row)
            dense_profiles.append(integration_result.dense_profile)

        except (ValueError, RuntimeError, ArithmeticError) as e:
            # trap specific ODE solver integration faults and re-raise them as our domain error
            try:
                raise TovConvergenceError(pc=pc, reason=str(e)) from e
            except TovConvergenceError:
                logger.exception("ODE Solver failed due to domain error")
                continue

    if not curve_data:
        return [], [], 0.0

    return _extract_first_turning_point(curve_data, dense_profiles)
