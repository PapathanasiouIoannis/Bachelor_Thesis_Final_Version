"""Parsing and reproduction-policy validation for paired EoS experiments."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Generic, Mapping, TypeVar

from src.configuration.common import (
    ConfigurationError,
    _decimal,
    _decimal_text,
    _integer,
    _require_exact_keys,
    _require_table,
    _string,
    _validate_common_header,
)


PairSpecificationT = TypeVar("PairSpecificationT")


@dataclass(frozen=True, slots=True)
class PairParserDependencies(Generic[PairSpecificationT]):
    """Model constructors and constants supplied by the compatibility facade."""

    hadronic_spec_type: type[Any]
    quark_spec_type: type[Any]
    deformation_spec_type: type[Any]
    physical_requirements_spec_type: type[Any]
    numerical_settings_spec_type: type[Any]
    execution_spec_type: type[Any]
    pair_spec_type: Callable[..., PairSpecificationT]
    supported_hadronic_baselines: frozenset[str]
    pair_workflow: str
    reproduction_mode: str
    exploration_mode: str


def parse_pair(
    root: Mapping[str, Any],
    dependencies: PairParserDependencies[PairSpecificationT],
    *,
    reproduction_validator: Callable[[PairSpecificationT], None] | None = None,
) -> PairSpecificationT:
    """Parse a validated pair-experiment mapping into the facade's model types."""

    expected_root = {
        "schema_version",
        "experiment_name",
        "workflow",
        "mode",
        "hadronic_eos",
        "quark_eos",
        "deformation",
        "physical_requirements",
        "numerical_settings",
        "execution",
    }
    _require_exact_keys(root, expected_root, "pair experiment")
    schema_version, name, workflow, mode = _validate_common_header(root)
    if workflow != dependencies.pair_workflow:
        raise ConfigurationError(
            f"Pair experiment workflow must be {dependencies.pair_workflow!r}."
        )
    if mode not in {dependencies.reproduction_mode, dependencies.exploration_mode}:
        raise ConfigurationError(
            "Pair experiment mode must be 'reproduction' or 'exploration'."
        )

    hadronic = _require_table(root["hadronic_eos"], "hadronic_eos")
    _require_exact_keys(hadronic, {"baseline"}, "hadronic_eos")
    hadronic_spec = dependencies.hadronic_spec_type(
        baseline=_string(hadronic["baseline"], "hadronic_eos.baseline")
    )
    if hadronic_spec.baseline not in dependencies.supported_hadronic_baselines:
        choices = ", ".join(sorted(dependencies.supported_hadronic_baselines))
        raise ConfigurationError(
            f"hadronic_eos.baseline = {hadronic_spec.baseline!r} is not in the "
            f"repository catalog. Choose one of: {choices}."
        )

    quark = _require_table(root["quark_eos"], "quark_eos")
    _require_exact_keys(
        quark,
        {
            "model",
            "bag_constant_mev_fm3",
            "pairing_gap_mev",
            "strange_quark_mass_mev",
        },
        "quark_eos",
    )
    quark_spec = dependencies.quark_spec_type(
        model=_string(quark["model"], "quark_eos.model"),
        bag_constant_mev_fm3=_decimal(
            quark["bag_constant_mev_fm3"],
            "quark_eos.bag_constant_mev_fm3",
        ),
        pairing_gap_mev=_decimal(
            quark["pairing_gap_mev"], "quark_eos.pairing_gap_mev"
        ),
        strange_quark_mass_mev=_decimal(
            quark["strange_quark_mass_mev"],
            "quark_eos.strange_quark_mass_mev",
        ),
    )
    if quark_spec.model != "cfl_mit_bag":
        raise ConfigurationError(
            "quark_eos.model must be 'cfl_mit_bag'; other quark models are not "
            "implemented by the controlled framework."
        )
    if quark_spec.bag_constant_mev_fm3 <= 0:
        raise ConfigurationError(
            "quark_eos.bag_constant_mev_fm3 must be strictly positive."
        )
    if quark_spec.pairing_gap_mev <= 0:
        raise ConfigurationError("quark_eos.pairing_gap_mev must be strictly positive.")
    if quark_spec.strange_quark_mass_mev < 0:
        raise ConfigurationError(
            "quark_eos.strange_quark_mass_mev must be non-negative."
        )

    deformation = _require_table(root["deformation"], "deformation")
    _require_exact_keys(
        deformation,
        {
            "method",
            "center_energy_density_mev_fm3",
            "width_mev_fm3",
            "amplitude_start",
            "amplitude_stop",
            "amplitude_step",
        },
        "deformation",
    )
    deformation_spec = dependencies.deformation_spec_type(
        method=_string(deformation["method"], "deformation.method"),
        center_energy_density_mev_fm3=_decimal(
            deformation["center_energy_density_mev_fm3"],
            "deformation.center_energy_density_mev_fm3",
        ),
        width_mev_fm3=_decimal(
            deformation["width_mev_fm3"], "deformation.width_mev_fm3"
        ),
        amplitude_start=_decimal(
            deformation["amplitude_start"], "deformation.amplitude_start"
        ),
        amplitude_stop=_decimal(
            deformation["amplitude_stop"], "deformation.amplitude_stop"
        ),
        amplitude_step=_decimal(
            deformation["amplitude_step"], "deformation.amplitude_step"
        ),
    )
    if deformation_spec.method != "additive_gaussian_sound_speed":
        raise ConfigurationError(
            "deformation.method must be 'additive_gaussian_sound_speed'."
        )
    if deformation_spec.center_energy_density_mev_fm3 <= 0:
        raise ConfigurationError(
            "deformation.center_energy_density_mev_fm3 must be strictly positive."
        )
    if deformation_spec.width_mev_fm3 <= 0:
        raise ConfigurationError("deformation.width_mev_fm3 must be strictly positive.")
    # Evaluate now so invalid grids fail during loading, not during execution.
    deformation_spec.amplitudes

    screens = _require_table(root["physical_requirements"], "physical_requirements")
    _require_exact_keys(
        screens,
        {
            "minimum_maximum_mass_msun",
            "maximum_maximum_mass_msun",
            "radius_1p4_min_km",
            "radius_1p4_max_km",
        },
        "physical_requirements",
    )
    screen_spec = dependencies.physical_requirements_spec_type(
        minimum_maximum_mass_msun=_decimal(
            screens["minimum_maximum_mass_msun"],
            "physical_requirements.minimum_maximum_mass_msun",
        ),
        maximum_maximum_mass_msun=_decimal(
            screens["maximum_maximum_mass_msun"],
            "physical_requirements.maximum_maximum_mass_msun",
        ),
        radius_1p4_min_km=_decimal(
            screens["radius_1p4_min_km"],
            "physical_requirements.radius_1p4_min_km",
        ),
        radius_1p4_max_km=_decimal(
            screens["radius_1p4_max_km"],
            "physical_requirements.radius_1p4_max_km",
        ),
    )
    if screen_spec.minimum_maximum_mass_msun <= 0:
        raise ConfigurationError(
            "physical_requirements.minimum_maximum_mass_msun must be positive."
        )
    if screen_spec.maximum_maximum_mass_msun <= screen_spec.minimum_maximum_mass_msun:
        raise ConfigurationError(
            "physical_requirements.maximum_maximum_mass_msun must exceed the "
            "minimum maximum mass."
        )
    if screen_spec.radius_1p4_min_km <= 0:
        raise ConfigurationError(
            "physical_requirements.radius_1p4_min_km must be positive."
        )
    if screen_spec.radius_1p4_max_km <= screen_spec.radius_1p4_min_km:
        raise ConfigurationError(
            "physical_requirements.radius_1p4_max_km must exceed radius_1p4_min_km."
        )

    numerical = _require_table(root["numerical_settings"], "numerical_settings")
    _require_exact_keys(
        numerical, {"preset", "convergence_check"}, "numerical_settings"
    )
    numerical_spec = dependencies.numerical_settings_spec_type(
        preset=_string(numerical["preset"], "numerical_settings.preset"),
        convergence_check=_string(
            numerical["convergence_check"],
            "numerical_settings.convergence_check",
        ),
    )
    if numerical_spec.preset not in {"production", "smoke"}:
        raise ConfigurationError(
            "numerical_settings.preset must be 'production' or 'smoke'."
        )
    if numerical_spec.convergence_check not in {"endpoints_and_zero", "none"}:
        raise ConfigurationError(
            "numerical_settings.convergence_check must be 'endpoints_and_zero' "
            "or 'none'."
        )
    if (
        mode == dependencies.reproduction_mode
        and numerical_spec.preset != "production"
    ):
        raise ConfigurationError(
            "Reproduction mode requires numerical_settings.preset = 'production'."
        )

    execution = _require_table(root["execution"], "execution")
    _require_exact_keys(
        execution,
        {"random_seed", "parallel_jobs", "amplitudes_per_batch"},
        "execution",
    )
    execution_spec = dependencies.execution_spec_type(
        random_seed=_integer(execution["random_seed"], "execution.random_seed"),
        parallel_jobs=_integer(execution["parallel_jobs"], "execution.parallel_jobs"),
        amplitudes_per_batch=_integer(
            execution["amplitudes_per_batch"], "execution.amplitudes_per_batch"
        ),
    )
    if execution_spec.random_seed < 0:
        raise ConfigurationError("execution.random_seed must be non-negative.")
    if execution_spec.parallel_jobs < 1:
        raise ConfigurationError("execution.parallel_jobs must be at least 1.")
    if execution_spec.amplitudes_per_batch < 1:
        raise ConfigurationError("execution.amplitudes_per_batch must be at least 1.")

    specification = dependencies.pair_spec_type(
        schema_version=schema_version,
        experiment_name=name,
        workflow=workflow,
        mode=mode,
        hadronic_eos=hadronic_spec,
        quark_eos=quark_spec,
        deformation=deformation_spec,
        physical_requirements=screen_spec,
        numerical_settings=numerical_spec,
        execution=execution_spec,
    )
    if mode == dependencies.reproduction_mode:
        validator = reproduction_validator or validate_reproduction_lock
        validator(specification)
    return specification


def validate_reproduction_lock(specification: Any) -> None:
    """Enforce the documented scientific values for reproduction mode."""

    expected = {
        "hadronic_eos.baseline": "APR-1",
        "quark_eos.model": "cfl_mit_bag",
        "quark_eos.bag_constant_mev_fm3": Decimal("60"),
        "quark_eos.pairing_gap_mev": Decimal("100"),
        "quark_eos.strange_quark_mass_mev": Decimal("150"),
        "deformation.method": "additive_gaussian_sound_speed",
        "deformation.center_energy_density_mev_fm3": Decimal("220"),
        "deformation.width_mev_fm3": Decimal("50"),
        "deformation.amplitude_start": Decimal("-0.05"),
        "deformation.amplitude_stop": Decimal("0.09"),
        "deformation.amplitude_step": Decimal("0.01"),
        "physical_requirements.minimum_maximum_mass_msun": Decimal("2.08"),
        "physical_requirements.maximum_maximum_mass_msun": Decimal("3.0"),
        "physical_requirements.radius_1p4_min_km": Decimal("9.5"),
        "physical_requirements.radius_1p4_max_km": Decimal("14.5"),
        "numerical_settings.preset": "production",
        "numerical_settings.convergence_check": "endpoints_and_zero",
        "execution.random_seed": 20260804,
    }
    actual = {
        "hadronic_eos.baseline": specification.hadronic_eos.baseline,
        "quark_eos.model": specification.quark_eos.model,
        "quark_eos.bag_constant_mev_fm3": (
            specification.quark_eos.bag_constant_mev_fm3
        ),
        "quark_eos.pairing_gap_mev": specification.quark_eos.pairing_gap_mev,
        "quark_eos.strange_quark_mass_mev": (
            specification.quark_eos.strange_quark_mass_mev
        ),
        "deformation.method": specification.deformation.method,
        "deformation.center_energy_density_mev_fm3": (
            specification.deformation.center_energy_density_mev_fm3
        ),
        "deformation.width_mev_fm3": specification.deformation.width_mev_fm3,
        "deformation.amplitude_start": specification.deformation.amplitude_start,
        "deformation.amplitude_stop": specification.deformation.amplitude_stop,
        "deformation.amplitude_step": specification.deformation.amplitude_step,
        "physical_requirements.minimum_maximum_mass_msun": (
            specification.physical_requirements.minimum_maximum_mass_msun
        ),
        "physical_requirements.maximum_maximum_mass_msun": (
            specification.physical_requirements.maximum_maximum_mass_msun
        ),
        "physical_requirements.radius_1p4_min_km": (
            specification.physical_requirements.radius_1p4_min_km
        ),
        "physical_requirements.radius_1p4_max_km": (
            specification.physical_requirements.radius_1p4_max_km
        ),
        "numerical_settings.preset": specification.numerical_settings.preset,
        "numerical_settings.convergence_check": (
            specification.numerical_settings.convergence_check
        ),
        "execution.random_seed": specification.execution.random_seed,
    }
    for field, required in expected.items():
        if actual[field] != required:
            display = (
                _decimal_text(required) if isinstance(required, Decimal) else required
            )
            raise ConfigurationError(
                f"{field} = {actual[field]!s} is not permitted in reproduction "
                f"mode; the documented value is {display}. Use mode = "
                "'exploration' for a different sensitivity experiment."
            )


__all__ = [
    "PairParserDependencies",
    "parse_pair",
    "validate_reproduction_lock",
]
