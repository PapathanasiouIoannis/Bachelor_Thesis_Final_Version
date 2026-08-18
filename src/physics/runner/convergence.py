"""Convergence refinements and physical-screen reporting for pair experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


CONVERGENCE_COLUMNS = (
    "matter_type",
    "baseline_name",
    "deformation_amplitude",
    "check",
    "delta_maximum_mass_msun",
    "delta_radius_1p4_km",
    "relative_delta_tidal_deformability_1p4",
    "maximum_mass_passed",
    "radius_1p4_passed",
    "tidal_deformability_1p4_passed",
    "refined_physical_requirements_passed",
    "refined_physical_requirements_reason",
    "passed",
)


@dataclass(frozen=True)
class ConvergenceDependencies:
    """Facade-owned operations used by the convergence refinement loop."""

    dataframe_type: Any
    convergence_columns: Any
    resolved_numerical_settings: Callable[..., Any]
    failed_convergence_record: Callable[..., Any]
    build_eos: Callable[..., Any]
    solve: Callable[..., Any]
    stellar_curve_to_frame: Callable[..., Any]
    summarize_stellar_curve: Callable[..., Any]
    physical_requirements_status: Callable[..., Any]


@dataclass(frozen=True)
class FailedConvergenceRecordDependencies:
    """Facade-owned values used to construct an unavailable-refinement row."""

    nan_value: Any


def run_convergence_checks(
    runtime: dict[str, Any],
    summary: Any,
    *,
    dependencies: ConvergenceDependencies,
) -> Any:
    """Run configured numerical refinements and return their audit table."""

    if runtime["numerical_settings"]["convergence_check"] == "none":
        return dependencies.dataframe_type(columns=dependencies.convergence_columns)
    amplitudes = runtime["resolved"]["amplitudes"]
    selected = sorted({float(amplitudes[0]), 0.0, float(amplitudes[-1])})
    records = []
    base_lookup = {
        (str(row.matter_type), float(row.deformation_amplitude)): row
        for row in summary.itertuples(index=False)
    }
    numerical = dependencies.resolved_numerical_settings(runtime)
    variants = (
        ("double_eos_grid", 2 * int(numerical["eos_grid_points"]), None, None, None),
        (
            "double_central_pressure_grid",
            None,
            2 * int(numerical["central_pressure_points"]),
            None,
            None,
        ),
        (
            "tighter_tov_tolerances",
            None,
            None,
            float(numerical["tov_relative_tolerance"]) / 10.0,
            float(numerical["tov_absolute_tolerance"]) / 10.0,
        ),
    )
    for amplitude in selected:
        for matter_type in ("hadronic", "quark"):
            baseline = base_lookup.get((matter_type, amplitude))
            if baseline is None:
                records.append(
                    dependencies.failed_convergence_record(
                        runtime, matter_type, amplitude, "production_reference_missing"
                    )
                )
                continue
            for check, grid_points, n_points, rtol, atol in variants:
                try:
                    eos = dependencies.build_eos(
                        runtime, matter_type, amplitude, grid_points=grid_points
                    )
                    curve, _, _ = dependencies.solve(
                        runtime,
                        eos,
                        matter_type,
                        n_points=n_points,
                        rtol=rtol,
                        atol=atol,
                        enforce_physical_requirements=False,
                    )
                    frame = dependencies.stellar_curve_to_frame(
                        curve,
                        eos,
                        matter_type,
                        "convergence",
                        f"convergence_{matter_type}_{amplitude}_{check}",
                    )
                    refined = dependencies.summarize_stellar_curve(frame)
                    delta_mass = abs(
                        float(refined["maximum_mass_msun"])
                        - float(baseline.maximum_mass_msun)
                    )
                    delta_radius = abs(
                        float(refined["radius_1p4_km"])
                        - float(baseline.radius_1p4_km)
                    )
                    reference_tidal = float(baseline.tidal_deformability_1p4)
                    delta_tidal = abs(
                        float(refined["tidal_deformability_1p4"]) - reference_tidal
                    ) / abs(reference_tidal)
                    mass_passed = delta_mass <= 0.01
                    radius_passed = delta_radius <= 0.05
                    tidal_passed = delta_tidal <= 0.02
                    physical_passed, physical_reason = (
                        dependencies.physical_requirements_status(runtime, refined)
                    )
                    records.append(
                        {
                            "matter_type": matter_type,
                            "baseline_name": (
                                eos.catalog_identifier or eos.baseline_name
                            ),
                            "deformation_amplitude": amplitude,
                            "check": check,
                            "delta_maximum_mass_msun": delta_mass,
                            "delta_radius_1p4_km": delta_radius,
                            "relative_delta_tidal_deformability_1p4": delta_tidal,
                            "maximum_mass_passed": mass_passed,
                            "radius_1p4_passed": radius_passed,
                            "tidal_deformability_1p4_passed": tidal_passed,
                            "refined_physical_requirements_passed": physical_passed,
                            "refined_physical_requirements_reason": physical_reason,
                            "passed": (
                                mass_passed
                                and radius_passed
                                and tidal_passed
                                and physical_passed
                            ),
                        }
                    )
                except Exception as error:
                    records.append(
                        dependencies.failed_convergence_record(
                            runtime,
                            matter_type,
                            amplitude,
                            check,
                            reason=f"{type(error).__name__}: {error}",
                        )
                    )
    return dependencies.dataframe_type.from_records(
        records,
        columns=dependencies.convergence_columns,
    )


def failed_convergence_record(
    runtime: dict[str, Any],
    matter_type: str,
    amplitude: float,
    check: str,
    *,
    reason: str | None = None,
    dependencies: FailedConvergenceRecordDependencies,
) -> dict[str, Any]:
    """Build one failed or unavailable convergence-refinement record."""

    baseline = (
        runtime["hadronic_eos"]["baseline"]
        if matter_type == "hadronic"
        else runtime["resolved"]["quark_eos_id"]
    )
    return {
        "matter_type": matter_type,
        "baseline_name": baseline,
        "deformation_amplitude": amplitude,
        "check": f"{check}: {reason}" if reason else check,
        "delta_maximum_mass_msun": dependencies.nan_value,
        "delta_radius_1p4_km": dependencies.nan_value,
        "relative_delta_tidal_deformability_1p4": dependencies.nan_value,
        "maximum_mass_passed": False,
        "radius_1p4_passed": False,
        "tidal_deformability_1p4_passed": False,
        "refined_physical_requirements_passed": False,
        "refined_physical_requirements_reason": reason or "refinement unavailable",
        "passed": False,
    }


def physical_requirements_status(
    runtime: dict[str, Any],
    observables: dict[str, Any],
) -> tuple[bool, str]:
    """Evaluate refined mass and radius against configured inclusive bounds."""

    requirements = runtime["physical_requirements"]
    maximum_mass = float(observables["maximum_mass_msun"])
    radius_1p4 = float(observables["radius_1p4_km"])
    failures = []
    if not (
        float(requirements["minimum_maximum_mass_msun"])
        <= maximum_mass
        <= float(requirements["maximum_maximum_mass_msun"])
    ):
        failures.append("maximum mass outside configured interval")
    if not (
        float(requirements["radius_1p4_min_km"])
        <= radius_1p4
        <= float(requirements["radius_1p4_max_km"])
    ):
        failures.append("radius at 1.4 solar masses outside configured interval")
    return (not failures, "passed" if not failures else "; ".join(failures))
