"""Worker generation operations for controlled pair experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


class PairGenerationError(RuntimeError):
    def __init__(self, matter_type: str, stage: str, reason: str):
        self.matter_type = matter_type
        self.stage = stage
        super().__init__(reason)

    def __reduce__(self):
        return (
            type(self),
            (self.matter_type, self.stage, self.args[0]),
            self.__dict__,
        )


@dataclass(frozen=True)
class PairGenerationDependencies:
    """Facade-owned operations used to generate one paired sweep point."""

    configure_run_log: Callable[..., Any]
    worker_log_path: Callable[..., Any]
    path_type: Callable[..., Any]
    sweep_point_type: Callable[..., Any]
    build_eos: Callable[..., Any]
    serialize_eos_table: Callable[..., Any]
    validate_eos_frame: Callable[..., Any]
    solve: Callable[..., Any]
    stellar_curve_to_frame: Callable[..., Any]
    pair_generation_error_type: type[PairGenerationError]
    close_run_log: Callable[..., Any]


@dataclass(frozen=True)
class BuildEosDependencies:
    """Facade-owned operations used to construct one deformed EoS."""

    resolved_numerical_settings: Callable[..., Any]
    gaussian_deformation_type: Callable[..., Any]
    build_hadronic_eos: Callable[..., Any]
    build_quark_eos: Callable[..., Any]
    quark_parameters: Callable[..., Any]
    configuration: Mapping[str, Any]


@dataclass(frozen=True)
class SolveDependencies:
    """Facade-owned operations used to solve one stellar sequence."""

    resolved_numerical_settings: Callable[..., Any]
    solve_and_validate_sequence: Callable[..., Any]


def generate_pair(
    runtime: dict[str, Any],
    sweep_index: int,
    amplitude: float,
    configuration_hash: str,
    run_log_path: str,
    *,
    dependencies: PairGenerationDependencies,
) -> dict[str, Any]:
    """Generate and serialize one paired hadronic/quark sweep point."""

    dependencies.configure_run_log(
        dependencies.worker_log_path(dependencies.path_type(run_log_path))
    )
    point = dependencies.sweep_point_type(sweep_index, amplitude)
    pair_generation_error_type = dependencies.pair_generation_error_type
    eos_frames: list[Any] = []
    try:
        eoses = {}
        try:
            eoses["hadronic"] = dependencies.build_eos(
                runtime, "hadronic", amplitude
            )
        except Exception as error:
            raise pair_generation_error_type(
                "hadronic", "eos_generation", str(error)
            ) from error
        try:
            eoses["quark"] = dependencies.build_eos(runtime, "quark", amplitude)
        except Exception as error:
            raise pair_generation_error_type(
                "quark", "eos_generation", str(error)
            ) from error

        validation_failures: list[tuple[str, Exception]] = []
        for matter_type in ("hadronic", "quark"):
            try:
                frame = dependencies.serialize_eos_table(
                    eoses[matter_type], matter_type, point.sweep_id
                )
            except Exception as error:
                raise pair_generation_error_type(
                    matter_type, "eos_serialization", str(error)
                ) from error
            try:
                dependencies.validate_eos_frame(frame)
                frame.loc[:, "eos_validation_passed"] = True
                frame.loc[:, "eos_validation_reason"] = "passed"
            except Exception as error:
                frame.loc[:, "eos_validation_passed"] = False
                frame.loc[:, "eos_validation_reason"] = (
                    f"{type(error).__name__}: {error}"
                )
                validation_failures.append((matter_type, error))
            eos_frames.append(frame)
        if validation_failures:
            matter_type, error = validation_failures[0]
            raise pair_generation_error_type(
                matter_type, "eos_validation", str(error)
            ) from error

        stellar_frames = []
        for matter_type in ("hadronic", "quark"):
            eos = eoses[matter_type]
            try:
                curve, _, _ = dependencies.solve(runtime, eos, matter_type)
            except Exception as error:
                raise pair_generation_error_type(
                    matter_type, "stellar_sequence", str(error)
                ) from error
            curve_id = (
                f"{matter_type}_{eos.baseline_name}_{point.sweep_id}_"
                f"{configuration_hash[:10]}"
            )
            try:
                stellar_frames.append(
                    dependencies.stellar_curve_to_frame(
                        curve, eos, matter_type, point.sweep_id, curve_id
                    )
                )
            except Exception as error:
                raise pair_generation_error_type(
                    matter_type, "stellar_serialization", str(error)
                ) from error
        result = {
            "accepted": True,
            "eos_frames": eos_frames,
            "stellar_frames": stellar_frames,
            "rejection": None,
        }
        dependencies.close_run_log()
        return result
    except pair_generation_error_type as error:
        for frame in eos_frames:
            frame.loc[:, "pair_accepted"] = False
        result = {
            "accepted": False,
            "eos_frames": eos_frames,
            "stellar_frames": [],
            "rejection": {
                "sweep_id": point.sweep_id,
                "deformation_amplitude": amplitude,
                "matter_type": error.matter_type,
                "stage": error.stage,
                "exception_type": type(error.__cause__).__name__
                if error.__cause__
                else type(error).__name__,
                "reason": str(error),
            },
        }
        dependencies.close_run_log()
        return result


def build_eos(
    runtime: dict[str, Any],
    matter_type: str,
    amplitude: float,
    *,
    grid_points: int | None = None,
    dependencies: BuildEosDependencies,
):
    """Build one deformed EoS using facade-provided scientific operations."""

    deformation_config = runtime["deformation"]
    if grid_points is None:
        grid_points = int(
            dependencies.resolved_numerical_settings(runtime)["eos_grid_points"]
        )
    deformation = dependencies.gaussian_deformation_type(
        amplitude=float(amplitude),
        epsilon0=float(deformation_config["center_energy_density_mev_fm3"]),
        sigma=float(deformation_config["width_mev_fm3"]),
    )
    if matter_type == "hadronic":
        return dependencies.build_hadronic_eos(
            runtime["hadronic_eos"]["baseline"],
            deformation,
            grid_points=grid_points,
        )
    if matter_type == "quark":
        return dependencies.build_quark_eos(
            dependencies.quark_parameters(runtime),
            deformation,
            maximum_surface_energy_per_baryon=dependencies.configuration["M_N"],
            grid_points=grid_points,
            catalog_identifier=runtime["resolved"]["quark_eos_id"],
        )
    raise ValueError("matter_type must be 'hadronic' or 'quark'.")


def solve(
    runtime: dict[str, Any],
    eos,
    matter_type: str,
    *,
    n_points: int | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    enforce_physical_requirements: bool = True,
    dependencies: SolveDependencies,
) -> tuple[list, dict, float]:
    """Solve one sequence using facade-provided numerical settings and solver."""

    screens = runtime["physical_requirements"]
    numerical = dependencies.resolved_numerical_settings(runtime)
    return dependencies.solve_and_validate_sequence(
        eos,
        is_quark=matter_type == "quark",
        minimum_maximum_mass=float(screens["minimum_maximum_mass_msun"]),
        maximum_maximum_mass=float(screens["maximum_maximum_mass_msun"]),
        radius_14_bounds=(
            float(screens["radius_1p4_min_km"]),
            float(screens["radius_1p4_max_km"]),
        ),
        n_points=(
            int(numerical["central_pressure_points"])
            if n_points is None
            else n_points
        ),
        rtol=(
            float(numerical["tov_relative_tolerance"]) if rtol is None else rtol
        ),
        atol=(
            float(numerical["tov_absolute_tolerance"]) if atol is None else atol
        ),
        enforce_physical_requirements=enforce_physical_requirements,
    )
