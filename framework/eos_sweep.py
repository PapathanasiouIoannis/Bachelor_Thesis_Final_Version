"""Deterministic Gaussian sound-speed sweeps for compact-star equations of state.

The production workers call this module so the controlled hadronic and quark
experiments share one deformation, causality, and pressure-reconstruction path.
All dimensional quantities use the repository convention of MeV and fm.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import PchipInterpolator

from src.config import CONFIG
from src.physics.get_eos_library import get_eos_library
from src.utils.exceptions import CrustStitchingError, ThermodynamicInstabilityError


EosCallable = Callable[[float], tuple[float, float]]


@dataclass(frozen=True)
class GaussianDeformation:
    r"""A Gaussian additive deformation of :math:`c_s^2(\epsilon)`."""

    amplitude: float
    epsilon0: float
    sigma: float

    def __post_init__(self) -> None:
        values = (self.amplitude, self.epsilon0, self.sigma)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Gaussian deformation parameters must be finite.")
        if self.sigma <= 0.0:
            raise ValueError("Gaussian sigma must be strictly positive.")


@dataclass(frozen=True)
class QuarkParameters:
    """Parameters of the lowest-order analytic CFL MIT-bag approximation."""

    bag_b: float  # MeV/fm^3
    gap_delta: float  # MeV
    strange_mass: float  # MeV

    def __post_init__(self) -> None:
        values = (self.bag_b, self.gap_delta, self.strange_mass)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("CFL parameters B, Delta, and m_s must be finite.")
        if self.bag_b <= 0.0 or self.gap_delta <= 0.0 or self.strange_mass < 0.0:
            raise ValueError(
                "CFL parameters B and Delta must be positive; m_s must be non-negative."
            )

    @property
    def baseline_name(self) -> str:
        def token(value: float) -> str:
            return f"{value:g}".replace("-", "m").replace(".", "p")

        return (
            f"CFL_B{token(self.bag_b)}"
            f"_D{token(self.gap_delta)}"
            f"_MS{token(self.strange_mass)}"
        )


@dataclass(frozen=True)
class SweepPoint:
    """One deterministic and class-paired amplitude value."""

    index: int
    amplitude: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Sweep-point indices must be non-negative.")
        if not np.isfinite(self.amplitude):
            raise ValueError("Sweep amplitudes must be finite.")

    @property
    def sweep_id(self) -> str:
        return f"A{self.index:05d}"


@dataclass(frozen=True)
class FrameworkEos:
    """A framework-generated EoS and its auditable construction metadata."""

    eos_callable: EosCallable
    pressure: np.ndarray
    energy_density: np.ndarray
    sound_speed_squared: np.ndarray
    p_max_causal: float
    eps_surface: float
    baseline_name: str
    deformation: GaussianDeformation
    catalog_identifier: str | None = None
    quark_parameters: QuarkParameters | None = None
    transition_pressure: float | None = None
    energy_shift: float = 0.0
    energy_per_baryon_surface: float | None = None
    input_grid_points: int = 0
    discarded_suffix_points: int = 0
    first_discarded_sound_speed_squared: float | None = None


def tabulate_complete_eos(
    framework_eos: FrameworkEos,
    *,
    crust_points: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the full pressure domain used by the stellar solver.

    Self-bound quark tables already start at zero pressure. Hadronic framework
    arrays contain the causal core, so this helper prepends a logarithmic crust
    table down to the same surface-pressure cutoff used by the TOV event. The
    final array identifies every row as ``crust``, ``core``, or ``self_bound``.
    """

    if framework_eos.transition_pressure is None:
        count = len(framework_eos.pressure)
        return (
            np.asarray(framework_eos.pressure, dtype=float).copy(),
            np.asarray(framework_eos.energy_density, dtype=float).copy(),
            np.asarray(framework_eos.sound_speed_squared, dtype=float).copy(),
            np.full(count, "self_bound", dtype=object),
        )
    if isinstance(crust_points, (bool, np.bool_)) or not isinstance(
        crust_points, (int, np.integer)
    ):
        raise ValueError("crust_points must be an integer of at least 16.")
    if int(crust_points) < 16:
        raise ValueError("crust_points must be an integer of at least 16.")

    transition = float(framework_eos.transition_pressure)
    surface_pressure = float(CONFIG["SURFACE_PRESSURE_EVENT_CUTOFF"])
    crust_pressure = np.geomspace(
        surface_pressure,
        transition,
        int(crust_points),
        endpoint=False,
    )
    crust_values = np.asarray(
        [framework_eos.eos_callable(float(pressure)) for pressure in crust_pressure],
        dtype=float,
    )
    pressure = np.concatenate((crust_pressure, framework_eos.pressure))
    energy_density = np.concatenate((crust_values[:, 0], framework_eos.energy_density))
    sound_speed_squared = np.concatenate(
        (crust_values[:, 1], framework_eos.sound_speed_squared)
    )
    regions = np.concatenate(
        (
            np.full(len(crust_pressure), "crust", dtype=object),
            np.full(len(framework_eos.pressure), "core", dtype=object),
        )
    )
    return pressure, energy_density, sound_speed_squared, regions


def amplitude_grid(a_min: float, a_max: float, points: int) -> list[SweepPoint]:
    """Return a common, ordered A grid used once for each matter class."""

    if points < 2:
        raise ValueError("An A sweep requires at least two points.")
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_min >= a_max:
        raise ValueError("A sweep requires finite bounds with a_min < a_max.")
    amplitudes = np.linspace(a_min, a_max, points)
    # Keep the undeformed control in any sweep whose support straddles zero.
    # Replacement is deterministic and preserves the requested point count.
    if a_min < 0.0 < a_max and not np.any(np.isclose(amplitudes, 0.0, atol=1e-14)):
        amplitudes[int(np.argmin(np.abs(amplitudes)))] = 0.0
    return [
        SweepPoint(index=index, amplitude=float(amplitude))
        for index, amplitude in enumerate(amplitudes)
    ]


def gaussian_sound_speed(
    energy_density: np.ndarray,
    baseline_cs2: np.ndarray,
    deformation: GaussianDeformation,
) -> np.ndarray:
    """Apply the shared Gaussian deformation without hiding invalid values."""

    energy_density = np.asarray(energy_density, dtype=float)
    baseline_cs2 = np.asarray(baseline_cs2, dtype=float)
    if energy_density.shape != baseline_cs2.shape:
        raise ValueError(
            "Energy-density and sound-speed grids must have identical shapes."
        )
    bump = deformation.amplitude * np.exp(
        -0.5 * ((energy_density - deformation.epsilon0) / deformation.sigma) ** 2
    )
    return baseline_cs2 + bump


def admissible_amplitude_interval(
    energy_density: np.ndarray,
    baseline_cs2: np.ndarray,
    epsilon0: float,
    sigma: float,
    *,
    gaussian_floor: float = 1e-12,
) -> tuple[float, float]:
    """Derive the open A interval satisfying ``0 < c_s^2 <= 1``.

    The calculation stops immediately before the baseline's first invalid point,
    matching the framework's causal-prefix rule. The returned lower endpoint is
    mathematically open because exactly zero sound speed is not admissible.
    """

    energy_density = np.asarray(energy_density, dtype=float)
    baseline_cs2 = np.asarray(baseline_cs2, dtype=float)
    _validate_energy_grid(energy_density)
    if energy_density.shape != baseline_cs2.shape:
        raise ValueError(
            "Energy-density and sound-speed grids must have identical shapes."
        )

    invalid = np.flatnonzero(
        ~np.isfinite(baseline_cs2) | (baseline_cs2 <= 0.0) | (baseline_cs2 > 1.0)
    )
    stop = int(invalid[0]) if invalid.size else len(baseline_cs2)
    if stop < 4:
        raise ValueError("Baseline has fewer than four causal, stable grid points.")

    deformation = GaussianDeformation(0.0, epsilon0, sigma)
    gaussian = np.exp(
        -0.5 * ((energy_density[:stop] - deformation.epsilon0) / deformation.sigma) ** 2
    )
    informative = gaussian > gaussian_floor
    if not np.any(informative):
        raise ValueError("Gaussian support does not overlap the baseline energy grid.")

    gaussian = gaussian[informative]
    cs2 = baseline_cs2[:stop][informative]
    lower = float(np.max(-cs2 / gaussian))
    upper = float(np.min((1.0 - cs2) / gaussian))
    if lower >= upper:
        raise ValueError(
            "No causal and thermodynamically stable amplitude interval exists."
        )
    return lower, upper


def _validate_energy_grid(energy_density: np.ndarray) -> None:
    if energy_density.ndim != 1 or len(energy_density) < 4:
        raise ValueError(
            "Energy-density grid must be one-dimensional with at least four points."
        )
    if np.any(~np.isfinite(energy_density)) or np.any(energy_density <= 0.0):
        raise ValueError(
            "Energy-density grid contains non-finite or non-positive values."
        )
    if np.any(np.diff(energy_density) <= 0.0):
        raise ThermodynamicInstabilityError(
            deps=float(np.min(np.diff(energy_density))),
            dp=1.0,
            message="Energy-density grid is not strictly increasing.",
        )


def _causal_prefix(
    energy_density: np.ndarray,
    sound_speed_squared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_energy_grid(energy_density)
    sound_speed_squared = np.asarray(sound_speed_squared, dtype=float)
    if energy_density.shape != sound_speed_squared.shape:
        raise ValueError(
            "Energy-density and sound-speed grids must have identical shapes."
        )

    invalid = np.flatnonzero(
        ~np.isfinite(sound_speed_squared)
        | (sound_speed_squared <= 0.0)
        | (sound_speed_squared > 1.0)
    )
    stop = int(invalid[0]) if invalid.size else len(sound_speed_squared)
    if stop < 4:
        first = int(invalid[0]) if invalid.size else 0
        value = float(sound_speed_squared[first])
        raise ValueError(
            "Deformation leaves fewer than four causal, stable points "
            f"(first invalid c_s^2={value:.6g})."
        )
    return energy_density[:stop], sound_speed_squared[:stop]


def _deform_and_reconstruct_pressure(
    energy_density: np.ndarray,
    baseline_cs2: np.ndarray,
    start_pressure: float,
    deformation: GaussianDeformation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | None]]:
    deformed_cs2 = gaussian_sound_speed(energy_density, baseline_cs2, deformation)
    input_grid_points = len(deformed_cs2)
    invalid = np.flatnonzero(
        ~np.isfinite(deformed_cs2) | (deformed_cs2 <= 0.0) | (deformed_cs2 > 1.0)
    )
    first_discarded = float(deformed_cs2[invalid[0]]) if invalid.size else None
    energy_density, deformed_cs2 = _causal_prefix(energy_density, deformed_cs2)
    pressure = float(start_pressure) + cumulative_simpson(
        deformed_cs2, x=energy_density, initial=0.0
    )
    if np.any(~np.isfinite(pressure)) or np.any(np.diff(pressure) <= 0.0):
        raise ValueError(
            "Reconstructed pressure grid is not finite and strictly increasing."
        )
    return (
        pressure,
        energy_density,
        deformed_cs2,
        {
            "input_grid_points": input_grid_points,
            "discarded_suffix_points": input_grid_points - len(deformed_cs2),
            "first_discarded_sound_speed_squared": first_discarded,
        },
    )


def _eval_crust(crusts: dict, pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.asarray(pressure, dtype=float)
    conditions = [
        pressure > CONFIG["P_C1"],
        (pressure <= CONFIG["P_C1"]) & (pressure > CONFIG["P_C2"]),
        (pressure <= CONFIG["P_C2"]) & (pressure > CONFIG["P_C3"]),
        pressure <= CONFIG["P_C3"],
    ]
    eps_functions = [crusts[key][0] for key in ("c1", "c2", "c3", "c4")]
    deps_functions = [crusts[key][1] for key in ("c1", "c2", "c3", "c4")]
    return (
        np.piecewise(pressure, conditions, eps_functions),
        np.piecewise(pressure, conditions, deps_functions),
    )


def _crust_eos(crusts: dict, pressure: float) -> tuple[float, float]:
    eps, deps_dp = _eval_crust(crusts, np.array([pressure], dtype=float))
    derivative = float(deps_dp[0])
    if derivative <= 0.0 or not np.isfinite(derivative):
        return -1.0, -1.0
    return float(eps[0]), float(1.0 / derivative)


def resolve_density_shifted_transition(
    crusts: dict,
    core_anchor: tuple,
    transition_pressure: float,
) -> tuple[float, float]:
    """Return the fixed transition pressure and additive core-density shift."""

    eps_function, deps_function = core_anchor
    transition_pressure = float(transition_pressure)
    try:
        raw_core_eps = float(eps_function(transition_pressure))
        core_deps_dp = float(deps_function(transition_pressure))
        crust_eps, _ = _eval_crust(crusts, np.array([transition_pressure], dtype=float))
        crust_eps = float(crust_eps[0])
    except Exception as exc:
        raise CrustStitchingError(
            p_trans=transition_pressure,
            message=f"Invalid anchor at transition pressure: {exc}",
        ) from exc

    values = (raw_core_eps, core_deps_dp, crust_eps)
    if not all(np.isfinite(value) for value in values):
        raise CrustStitchingError(
            p_trans=transition_pressure,
            message="Anchor or crust produced non-finite transition values.",
        )
    if raw_core_eps <= 0.0 or crust_eps <= 0.0 or core_deps_dp <= 0.0:
        raise CrustStitchingError(
            p_trans=transition_pressure,
            message="Anchor or crust is non-physical at the transition pressure.",
        )

    energy_shift = raw_core_eps - crust_eps
    shifted_core_eps = raw_core_eps - energy_shift
    tolerance = max(
        CONFIG["CRUST_CORE_EPS_ABS_TOL"],
        CONFIG["CRUST_CORE_EPS_REL_TOL"]
        * max(abs(shifted_core_eps), abs(crust_eps), 1.0),
    )
    if abs(shifted_core_eps - crust_eps) > tolerance:
        raise CrustStitchingError(
            p_trans=transition_pressure,
            message="Density-shifted crust/core continuity check failed.",
        )
    return transition_pressure, energy_shift


@lru_cache(maxsize=None)
def hadronic_baseline_grids(
    baseline_name: str,
    grid_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, float, float]:
    """Return the shifted hadronic baseline grids used by the sweep."""

    resolved_grid_points = (
        CONFIG["P_GRID_POINTS"] if grid_points is None else int(grid_points)
    )
    if resolved_grid_points < 4:
        raise ValueError("The EoS pressure grid requires at least four points.")

    core_library, crusts = get_eos_library()
    if baseline_name not in core_library:
        raise KeyError(f"Unknown hadronic baseline: {baseline_name}")
    anchor = core_library[baseline_name]
    # The source of the repository's analytic fits specifies a distinct
    # crust/core pressure for PS and a common value for the other 20 models.
    # Keeping this decision here makes all framework consumers use the same
    # auditable matching convention.
    transition_pressure_requested = (
        0.696 if baseline_name == "PS" else CONFIG["P_TRANS_DEFAULT"]
    )
    transition_pressure, energy_shift = resolve_density_shifted_transition(
        crusts, anchor, transition_pressure_requested
    )
    pressure = np.linspace(
        transition_pressure, CONFIG["P_GRID_MAX"], resolved_grid_points
    )
    energy_density = np.asarray(anchor[0](pressure), dtype=float) - energy_shift
    deps_dp = np.asarray(anchor[1](pressure), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_cs2 = 1.0 / deps_dp
    return energy_density, baseline_cs2, crusts, transition_pressure, energy_shift


def build_hadronic_eos(
    baseline_name: str,
    deformation: GaussianDeformation,
    *,
    grid_points: int | None = None,
) -> FrameworkEos:
    """Build one framework-controlled, crust-matched hadronic EoS."""

    (
        energy_density,
        baseline_cs2,
        crusts,
        transition_pressure,
        energy_shift,
    ) = hadronic_baseline_grids(baseline_name, grid_points)
    pressure, energy_density, deformed_cs2, causal_metadata = (
        _deform_and_reconstruct_pressure(
            energy_density, baseline_cs2, transition_pressure, deformation
        )
    )
    eps_interpolator = PchipInterpolator(pressure, energy_density, extrapolate=False)
    cs2_interpolator = PchipInterpolator(pressure, deformed_cs2, extrapolate=False)
    p_max_causal = float(pressure[-1])

    def eos_callable(p: float) -> tuple[float, float]:
        p = float(p)
        if p <= transition_pressure:
            return _crust_eos(crusts, p)
        if p > p_max_causal:
            return -1.0, -1.0
        return float(eps_interpolator(p)), float(cs2_interpolator(p))

    eos_callable.eps_surf = 0.0
    return FrameworkEos(
        eos_callable=eos_callable,
        pressure=pressure,
        energy_density=energy_density,
        sound_speed_squared=deformed_cs2,
        p_max_causal=p_max_causal,
        eps_surface=0.0,
        baseline_name=baseline_name,
        deformation=deformation,
        catalog_identifier=baseline_name,
        transition_pressure=transition_pressure,
        energy_shift=energy_shift,
        **causal_metadata,
    )


@lru_cache(maxsize=None)
def cfl_baseline_grids(
    parameters: QuarkParameters,
    grid_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate the analytic lowest-order CFL baseline on the pressure grid."""

    resolved_grid_points = (
        CONFIG["P_GRID_POINTS"] if grid_points is None else int(grid_points)
    )
    if resolved_grid_points < 4:
        raise ValueError("The EoS pressure grid requires at least four points.")

    hc = float(CONFIG["HC"])
    pressure = np.linspace(0.0, CONFIG["P_GRID_MAX"], resolved_grid_points)
    pressure_geom = pressure / hc
    bag_geom = parameters.bag_b / hc
    delta_geom = parameters.gap_delta / hc
    strange_mass_geom = parameters.strange_mass / hc
    effective_gap_squared = delta_geom**2 - strange_mass_geom**2 / 4.0

    coeff_a = 3.0 / (4.0 * np.pi**2)
    coeff_b = 3.0 * effective_gap_squared / np.pi**2
    determinant = coeff_b**2 + 4.0 * coeff_a * (pressure_geom + bag_geom)
    if np.any(determinant < 0.0):
        raise ValueError("CFL baseline has no real chemical-potential root.")
    mu_squared = (-coeff_b + np.sqrt(determinant)) / (2.0 * coeff_a)
    if np.any(mu_squared <= 0.0) or np.any(~np.isfinite(mu_squared)):
        raise ValueError("CFL baseline produced a non-positive chemical potential.")

    energy_density_geom = (
        3.0 * coeff_a * mu_squared**2 + coeff_b * mu_squared + bag_geom
    )
    energy_density = energy_density_geom * hc
    shift = 2.0 * effective_gap_squared
    denominator = 3.0 * mu_squared + shift
    if np.any(denominator <= 0.0):
        raise ValueError("CFL sound-speed denominator is non-positive.")
    baseline_cs2 = (mu_squared + shift) / denominator
    surface_chemical_potential = float(np.sqrt(mu_squared[0]) * hc)
    if parameters.strange_mass**2 >= (
        2.0 * surface_chemical_potential * parameters.gap_delta
    ):
        raise ValueError(
            "CFL pairing condition m_s^2 < 2 mu Delta fails at the stellar surface."
        )
    energy_per_baryon_surface = 3.0 * surface_chemical_potential
    return pressure, energy_density, baseline_cs2, energy_per_baryon_surface


def build_quark_eos(
    parameters: QuarkParameters,
    deformation: GaussianDeformation,
    *,
    maximum_surface_energy_per_baryon: float | None = None,
    grid_points: int | None = None,
    catalog_identifier: str | None = None,
) -> FrameworkEos:
    """Build one self-bound analytic CFL EoS with the shared deformation."""

    (
        _,
        energy_density,
        baseline_cs2,
        energy_per_baryon_surface,
    ) = cfl_baseline_grids(parameters, grid_points)
    if (
        maximum_surface_energy_per_baryon is not None
        and energy_per_baryon_surface > maximum_surface_energy_per_baryon
    ):
        raise ValueError(
            "CFL surface energy per baryon exceeds the configured stability limit: "
            f"{energy_per_baryon_surface:.3f} > "
            f"{maximum_surface_energy_per_baryon:.3f} MeV."
        )

    pressure, energy_density, deformed_cs2, causal_metadata = (
        _deform_and_reconstruct_pressure(energy_density, baseline_cs2, 0.0, deformation)
    )
    eps_interpolator = PchipInterpolator(pressure, energy_density, extrapolate=False)
    cs2_interpolator = PchipInterpolator(pressure, deformed_cs2, extrapolate=False)
    p_max_causal = float(pressure[-1])
    eps_surface = float(energy_density[0])

    def eos_callable(p: float) -> tuple[float, float]:
        p = max(float(p), 0.0)
        if p > p_max_causal:
            return -1.0, -1.0
        return float(eps_interpolator(p)), float(cs2_interpolator(p))

    eos_callable.eps_surf = eps_surface
    return FrameworkEos(
        eos_callable=eos_callable,
        pressure=pressure,
        energy_density=energy_density,
        sound_speed_squared=deformed_cs2,
        p_max_causal=p_max_causal,
        eps_surface=eps_surface,
        baseline_name=parameters.baseline_name,
        deformation=deformation,
        catalog_identifier=catalog_identifier or parameters.baseline_name,
        quark_parameters=parameters,
        energy_per_baryon_surface=energy_per_baryon_surface,
        **causal_metadata,
    )


def validate_sweep_within_interval(
    sweep_points: Sequence[SweepPoint],
    interval: tuple[float, float],
    label: str,
) -> None:
    """Fail before generation when a requested A lies outside model bounds."""

    lower, upper = interval
    invalid = [
        point.amplitude
        for point in sweep_points
        if not (point.amplitude > lower and point.amplitude <= upper)
    ]
    if invalid:
        raise ValueError(
            f"{label} sweep amplitudes fall outside ({lower:.6g}, {upper:.6g}]: "
            f"{invalid[:5]}"
        )
