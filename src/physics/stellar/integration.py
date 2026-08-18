"""Single-pressure TOV integration and stellar-model assembly."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

import numpy as np


class IntegrationDecision(Enum):
    """Non-model outcomes from one central-pressure integration."""

    SKIP = "skip"
    STOP = "stop"


@dataclass(frozen=True)
class IntegrationContext:
    """Runtime dependencies and import-time constants for one solver call."""

    integrator: Callable[..., Any]
    rhs: Callable[..., Any]
    surface_event: Callable[..., Any]
    surface_density_correction: Callable[[float, float, float, float], float]
    tidal_lambda: Callable[[float, float], float | None]
    r_min: float
    r_max: float
    compactness_conversion: float
    buchdahl_limit: float


@dataclass(frozen=True)
class IntegratedStellarModel:
    """One accepted stellar row and its aligned dense mass profile."""

    curve_row: list[float]
    dense_profile: tuple[np.ndarray, np.ndarray]


def integrate_stellar_model(
    *,
    eos_callable: Callable,
    central_pressure: float,
    central_energy_density: float,
    central_sound_speed_squared: float,
    surface_energy_density: float,
    initial_state: list[float],
    rtol: float | None,
    atol: float | None,
    configuration: Mapping[str, Any],
    context: IntegrationContext,
) -> IntegratedStellarModel | IntegrationDecision:
    """Integrate and validate one stellar model at a central pressure."""
    solution = context.integrator(
        fun=context.rhs,
        t_span=(context.r_min, context.r_max),
        y0=initial_state,
        args=(eos_callable,),
        events=context.surface_event,
        method="RK45",
        dense_output=True,
        rtol=rtol if rtol is not None else configuration["ODE_RTOL"],
        atol=atol if atol is not None else configuration["ODE_ATOL"],
    )

    if not (solution.status == 1 and len(solution.t_events[0]) > 0):
        return IntegrationDecision.SKIP

    radius = solution.t_events[0][0]
    mass = solution.y_events[0][0][0]
    y_surface = solution.y_events[0][0][2]

    assert not np.isnan(mass) and not np.isnan(radius), (
        "NaN detected in TOV Mass or Radius!"
    )
    assert not np.isinf(mass) and not np.isinf(radius), (
        "Inf detected in TOV Mass or Radius!"
    )
    assert mass > 0.0, (
        f"Unphysical mass detected! M={mass} is not strictly positive."
    )
    assert radius > 0.0, (
        f"Unphysical radius detected! R={radius} is not strictly positive."
    )

    if (
        radius < configuration["MIN_RADIUS_CUTOFF"]
        or mass < configuration["MIN_MASS_CUTOFF"]
    ):
        return IntegrationDecision.SKIP

    dense_radii = np.linspace(
        context.r_min,
        radius,
        configuration["DENSE_PROFILES_POINTS"],
    )
    dense_solution = solution.sol(dense_radii)
    dense_masses = dense_solution[0]

    compactness = (mass * context.compactness_conversion) / radius
    if compactness >= context.buchdahl_limit:
        return IntegrationDecision.SKIP

    y_surface = context.surface_density_correction(
        y_surface,
        radius,
        mass,
        surface_energy_density,
    )
    tidal_deformability = context.tidal_lambda(compactness, y_surface)
    if tidal_deformability is None:
        return IntegrationDecision.SKIP

    if mass <= 0.0:
        return IntegrationDecision.STOP

    return IntegratedStellarModel(
        curve_row=[
            mass,
            radius,
            tidal_deformability,
            central_pressure,
            central_energy_density,
            central_sound_speed_squared,
            surface_energy_density,
        ],
        dense_profile=(dense_radii, dense_masses),
    )
