"""Strict, immutable configuration for reproducible EoS experiments.

The public loader deliberately accepts a small schema.  Scientific parameters
are represented by :class:`~decimal.Decimal` so an amplitude sweep is never
silently changed by binary floating-point rounding.  Conversion to ordinary
``float`` values happens only through ``to_runtime_dict`` at the boundary to
the numerical framework.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, TypeAlias

from framework.eos_catalog import CFL_CATALOG, HADRONIC_CATALOG
from src.configuration.common import (
    ConfigurationError,
    _boolean,
    _canonicalize,
    _decimal,
    _decimal_text,
    _integer,
    _require_exact_keys,
    _require_table,
    _runtimeize,
    _string,
    _string_tuple,
    _validate_common_header,
    canonical_sha256,
    decimal_amplitude_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIR_WORKFLOW = "pair_sensitivity"
FAMILY_WORKFLOW = "family_classification"
REPRODUCTION_MODE = "reproduction"
EXPLORATION_MODE = "exploration"

PAIR_INTERPRETATION = (
    "Controlled sensitivity comparison between one repository hadronic "
    "surrogate and one analytic CFL MIT-bag equation of state; not universal "
    "matter-phase classification."
)
FAMILY_INTERPRETATION = (
    "Repository hadronic-surrogate versus analytic CFL MIT-bag curve "
    "discrimination on held-out EoS families; not universal phase "
    "identification or an astrophysical posterior probability."
)

_SUPPORTED_HADRONIC_BASELINES = frozenset(entry.eos_id for entry in HADRONIC_CATALOG)


@dataclass(frozen=True, slots=True)
class HadronicEosSpec:
    """One repository hadronic EoS baseline."""

    baseline: str


@dataclass(frozen=True, slots=True)
class QuarkEosSpec:
    """One analytic CFL MIT-bag parameter tuple."""

    model: str
    bag_constant_mev_fm3: Decimal
    pairing_gap_mev: Decimal
    strange_quark_mass_mev: Decimal


@dataclass(frozen=True, slots=True)
class DeformationSpec:
    """Additive Gaussian sound-speed deformation parameters."""

    method: str
    center_energy_density_mev_fm3: Decimal
    width_mev_fm3: Decimal
    amplitude_start: Decimal
    amplitude_stop: Decimal
    amplitude_step: Decimal

    @property
    def amplitudes(self) -> tuple[Decimal, ...]:
        return decimal_amplitude_grid(
            self.amplitude_start,
            self.amplitude_stop,
            self.amplitude_step,
        )


@dataclass(frozen=True, slots=True)
class PhysicalRequirementsSpec:
    """Macroscopic acceptance requirements with explicit units."""

    minimum_maximum_mass_msun: Decimal
    maximum_maximum_mass_msun: Decimal
    radius_1p4_min_km: Decimal
    radius_1p4_max_km: Decimal


@dataclass(frozen=True, slots=True)
class NumericalSettingsSpec:
    """Named numerical-quality settings interpreted by the runner."""

    preset: str
    convergence_check: str


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Non-scientific execution controls."""

    random_seed: int
    parallel_jobs: int
    amplitudes_per_batch: int


@dataclass(frozen=True, slots=True)
class PairExperimentSpec:
    """Immutable specification of a paired hadronic/CFL sensitivity sweep."""

    schema_version: int
    experiment_name: str
    workflow: str
    mode: str
    hadronic_eos: HadronicEosSpec
    quark_eos: QuarkEosSpec
    deformation: DeformationSpec
    physical_requirements: PhysicalRequirementsSpec
    numerical_settings: NumericalSettingsSpec
    execution: ExecutionSpec

    @property
    def amplitudes(self) -> tuple[Decimal, ...]:
        return self.deformation.amplitudes

    @property
    def config_hash(self) -> str:
        return canonical_sha256(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-safe configuration dictionary."""

        return _canonicalize(self)

    def to_runtime_dict(self) -> dict[str, Any]:
        """Return numerical values as floats for the existing framework."""

        return _runtimeize(asdict(self))

    def resolve(self) -> "ResolvedExperiment":
        return resolve_pair_experiment(self)


@dataclass(frozen=True, slots=True)
class FamilyProfilesSpec:
    generation_profile: str
    split_profile: str
    model_profile: str


@dataclass(frozen=True, slots=True)
class ObservableGridSpec:
    minimum_mass_msun: Decimal
    maximum_mass_msun: Decimal
    mass_points: int


@dataclass(frozen=True, slots=True)
class FamilyModelsSpec:
    primary: tuple[str, ...]
    exploratory: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FinalTestSpec:
    policy: str
    allow_evaluation: bool


@dataclass(frozen=True, slots=True)
class FamilyClassificationSpec:
    """Read-only wrapper around the audited family-classification profiles."""

    schema_version: int
    experiment_name: str
    workflow: str
    mode: str
    profiles: FamilyProfilesSpec
    observable_grid: ObservableGridSpec
    models: FamilyModelsSpec
    final_test: FinalTestSpec

    @property
    def config_hash(self) -> str:
        return canonical_sha256(self)

    def to_dict(self) -> dict[str, Any]:
        return _canonicalize(self)


ExperimentSpec: TypeAlias = PairExperimentSpec | FamilyClassificationSpec


@dataclass(frozen=True, slots=True)
class ResolvedExperiment:
    """A pair experiment plus derived, deterministic identifiers and grid."""

    specification: PairExperimentSpec
    amplitudes: tuple[Decimal, ...]
    quark_eos_id: str
    permitted_scientific_interpretation: str = PAIR_INTERPRETATION

    @property
    def config_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification": self.specification.to_dict(),
            "resolved": {
                "amplitudes": [_decimal_text(value) for value in self.amplitudes],
                "quark_eos_id": self.quark_eos_id,
                "permitted_scientific_interpretation": (
                    self.permitted_scientific_interpretation
                ),
            },
        }

    def to_runtime_dict(self) -> dict[str, Any]:
        output = self.specification.to_runtime_dict()
        output["resolved"] = {
            "amplitudes": [float(value) for value in self.amplitudes],
            "quark_eos_id": self.quark_eos_id,
            "permitted_scientific_interpretation": (
                self.permitted_scientific_interpretation
            ),
        }
        return output


def _parse_pair(root: Mapping[str, Any]) -> PairExperimentSpec:
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
    if workflow != PAIR_WORKFLOW:
        raise ConfigurationError(f"Pair experiment workflow must be {PAIR_WORKFLOW!r}.")
    if mode not in {REPRODUCTION_MODE, EXPLORATION_MODE}:
        raise ConfigurationError(
            "Pair experiment mode must be 'reproduction' or 'exploration'."
        )

    hadronic = _require_table(root["hadronic_eos"], "hadronic_eos")
    _require_exact_keys(hadronic, {"baseline"}, "hadronic_eos")
    hadronic_spec = HadronicEosSpec(
        baseline=_string(hadronic["baseline"], "hadronic_eos.baseline")
    )
    if hadronic_spec.baseline not in _SUPPORTED_HADRONIC_BASELINES:
        choices = ", ".join(sorted(_SUPPORTED_HADRONIC_BASELINES))
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
    quark_spec = QuarkEosSpec(
        model=_string(quark["model"], "quark_eos.model"),
        bag_constant_mev_fm3=_decimal(
            quark["bag_constant_mev_fm3"],
            "quark_eos.bag_constant_mev_fm3",
        ),
        pairing_gap_mev=_decimal(quark["pairing_gap_mev"], "quark_eos.pairing_gap_mev"),
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
    deformation_spec = DeformationSpec(
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
    screen_spec = PhysicalRequirementsSpec(
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
    numerical_spec = NumericalSettingsSpec(
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
    if mode == REPRODUCTION_MODE and numerical_spec.preset != "production":
        raise ConfigurationError(
            "Reproduction mode requires numerical_settings.preset = 'production'."
        )

    execution = _require_table(root["execution"], "execution")
    _require_exact_keys(
        execution,
        {"random_seed", "parallel_jobs", "amplitudes_per_batch"},
        "execution",
    )
    execution_spec = ExecutionSpec(
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

    specification = PairExperimentSpec(
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
    if mode == REPRODUCTION_MODE:
        _validate_reproduction_lock(specification)
    return specification


def _validate_reproduction_lock(specification: PairExperimentSpec) -> None:
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


def _parse_family(root: Mapping[str, Any]) -> FamilyClassificationSpec:
    expected_root = {
        "schema_version",
        "experiment_name",
        "workflow",
        "mode",
        "profiles",
        "observable_grid",
        "models",
        "final_test",
    }
    _require_exact_keys(root, expected_root, "family classification")
    schema_version, name, workflow, mode = _validate_common_header(root)
    if workflow != FAMILY_WORKFLOW:
        raise ConfigurationError(
            f"Family configuration workflow must be {FAMILY_WORKFLOW!r}."
        )
    if mode != "development":
        raise ConfigurationError(
            "Family classification mode must be 'development'; the final test "
            "has already been opened and is read-only."
        )

    profiles = _require_table(root["profiles"], "profiles")
    _require_exact_keys(
        profiles,
        {"generation_profile", "split_profile", "model_profile"},
        "profiles",
    )
    profile_spec = FamilyProfilesSpec(
        generation_profile=_string(
            profiles["generation_profile"], "profiles.generation_profile"
        ),
        split_profile=_string(profiles["split_profile"], "profiles.split_profile"),
        model_profile=_string(profiles["model_profile"], "profiles.model_profile"),
    )
    required_profiles = {
        "generation_profile": "framework/family_pilot_profile.json",
        "split_profile": "framework/family_split_profile.json",
        "model_profile": "framework/family_model_profile.json",
    }
    for field, required in required_profiles.items():
        configured = getattr(profile_spec, field)
        if Path(configured).as_posix() != required:
            raise ConfigurationError(
                f"profiles.{field} must reference the audited profile {required!r}."
            )
        if not (PROJECT_ROOT / required).is_file():
            raise ConfigurationError(
                f"Audited profile {required!r} is missing from the repository."
            )

    grid = _require_table(root["observable_grid"], "observable_grid")
    _require_exact_keys(
        grid,
        {"minimum_mass_msun", "maximum_mass_msun", "mass_points"},
        "observable_grid",
    )
    grid_spec = ObservableGridSpec(
        minimum_mass_msun=_decimal(
            grid["minimum_mass_msun"], "observable_grid.minimum_mass_msun"
        ),
        maximum_mass_msun=_decimal(
            grid["maximum_mass_msun"], "observable_grid.maximum_mass_msun"
        ),
        mass_points=_integer(grid["mass_points"], "observable_grid.mass_points"),
    )
    if grid_spec != ObservableGridSpec(Decimal("1.0"), Decimal("2.0"), 21):
        raise ConfigurationError(
            "The audited family workflow requires 21 mass points from 1.0 to "
            "2.0 solar masses."
        )

    models = _require_table(root["models"], "models")
    _require_exact_keys(models, {"primary", "exploratory"}, "models")
    model_spec = FamilyModelsSpec(
        primary=_string_tuple(models["primary"], "models.primary"),
        exploratory=_string_tuple(models["exploratory"], "models.exploratory"),
    )
    if model_spec.primary != ("dummy", "logistic_regression"):
        raise ConfigurationError(
            "models.primary must be ['dummy', 'logistic_regression'] in that order."
        )
    if model_spec.exploratory != ("xgboost", "mlp"):
        raise ConfigurationError(
            "models.exploratory must be ['xgboost', 'mlp'] in that order."
        )

    final_test = _require_table(root["final_test"], "final_test")
    _require_exact_keys(final_test, {"policy", "allow_evaluation"}, "final_test")
    final_test_spec = FinalTestSpec(
        policy=_string(final_test["policy"], "final_test.policy"),
        allow_evaluation=_boolean(
            final_test["allow_evaluation"], "final_test.allow_evaluation"
        ),
    )
    if final_test_spec != FinalTestSpec("already_opened_read_only", False):
        raise ConfigurationError(
            "The family final test has already been opened. Set policy = "
            "'already_opened_read_only' and allow_evaluation = false."
        )

    return FamilyClassificationSpec(
        schema_version=schema_version,
        experiment_name=name,
        workflow=workflow,
        mode=mode,
        profiles=profile_spec,
        observable_grid=grid_spec,
        models=model_spec,
        final_test=final_test_spec,
    )


def load_experiment_config(path: str | Path) -> ExperimentSpec:
    """Load and strictly validate a TOML experiment profile."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            root = tomllib.load(handle, parse_float=Decimal)
    except FileNotFoundError:
        raise ConfigurationError(
            f"Configuration file does not exist: {config_path}"
        ) from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(
            f"Configuration file is not valid TOML: {config_path}: {error}"
        ) from None

    root = _require_table(root, "configuration")
    if "workflow" not in root:
        raise ConfigurationError("configuration: missing field 'workflow'.")
    workflow = _string(root["workflow"], "workflow")
    if workflow == PAIR_WORKFLOW:
        return _parse_pair(root)
    if workflow == FAMILY_WORKFLOW:
        return _parse_family(root)
    raise ConfigurationError(
        f"workflow = {workflow!r} is unsupported; choose {PAIR_WORKFLOW!r} or "
        f"{FAMILY_WORKFLOW!r}."
    )


def load_pair_experiment(path: str | Path) -> PairExperimentSpec:
    """Load a pair profile and reject a family-classification profile."""

    specification = load_experiment_config(path)
    if not isinstance(specification, PairExperimentSpec):
        raise ConfigurationError(
            f"{path} describes the family-classification workflow, not a paired "
            "EoS sensitivity experiment."
        )
    return specification


def _quark_eos_id(specification: QuarkEosSpec) -> str:
    for entry in CFL_CATALOG:
        values = (
            Decimal(str(entry.bag_b_mev_fm3)),
            Decimal(str(entry.gap_delta_mev)),
            Decimal(str(entry.strange_mass_mev)),
        )
        configured = (
            specification.bag_constant_mev_fm3,
            specification.pairing_gap_mev,
            specification.strange_quark_mass_mev,
        )
        if configured == values:
            return entry.eos_id

    def token(value: Decimal) -> str:
        return _decimal_text(value).replace("-", "m").replace(".", "p")

    return (
        f"CFL_B{token(specification.bag_constant_mev_fm3)}"
        f"_D{token(specification.pairing_gap_mev)}"
        f"_MS{token(specification.strange_quark_mass_mev)}"
    )


def resolve_pair_experiment(
    specification_or_path: PairExperimentSpec | str | Path,
) -> ResolvedExperiment:
    """Resolve a pair profile to its exact grid and catalog-aware identifiers."""

    if isinstance(specification_or_path, PairExperimentSpec):
        specification = specification_or_path
    else:
        specification = load_pair_experiment(specification_or_path)
    return ResolvedExperiment(
        specification=specification,
        amplitudes=specification.amplitudes,
        quark_eos_id=_quark_eos_id(specification.quark_eos),
    )


__all__ = [
    "ConfigurationError",
    "DeformationSpec",
    "ExecutionSpec",
    "ExperimentSpec",
    "FamilyClassificationSpec",
    "FamilyModelsSpec",
    "FamilyProfilesSpec",
    "FinalTestSpec",
    "HadronicEosSpec",
    "NumericalSettingsSpec",
    "ObservableGridSpec",
    "PAIR_INTERPRETATION",
    "PairExperimentSpec",
    "PhysicalRequirementsSpec",
    "QuarkEosSpec",
    "ResolvedExperiment",
    "canonical_sha256",
    "decimal_amplitude_grid",
    "load_experiment_config",
    "load_pair_experiment",
    "resolve_pair_experiment",
]
