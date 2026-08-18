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
    _canonicalize,
    _decimal_text,
    _require_table,
    _runtimeize,
    _string,
    canonical_sha256,
    decimal_amplitude_grid,
)
from src.configuration.family import (
    FamilyParserDependencies as _FamilyParserDependencies,
    parse_family as _parse_family_impl,
)
from src.configuration.pair import (
    PairParserDependencies as _PairParserDependencies,
    parse_pair as _parse_pair_impl,
    validate_reproduction_lock as _validate_reproduction_lock_impl,
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


def _pair_parser_dependencies() -> _PairParserDependencies[PairExperimentSpec]:
    return _PairParserDependencies(
        hadronic_spec_type=HadronicEosSpec,
        quark_spec_type=QuarkEosSpec,
        deformation_spec_type=DeformationSpec,
        physical_requirements_spec_type=PhysicalRequirementsSpec,
        numerical_settings_spec_type=NumericalSettingsSpec,
        execution_spec_type=ExecutionSpec,
        pair_spec_type=PairExperimentSpec,
        supported_hadronic_baselines=_SUPPORTED_HADRONIC_BASELINES,
        pair_workflow=PAIR_WORKFLOW,
        reproduction_mode=REPRODUCTION_MODE,
        exploration_mode=EXPLORATION_MODE,
    )


def _parse_pair(root: Mapping[str, Any]) -> PairExperimentSpec:
    return _parse_pair_impl(
        root,
        _pair_parser_dependencies(),
        reproduction_validator=_validate_reproduction_lock,
    )


def _validate_reproduction_lock(specification: PairExperimentSpec) -> None:
    _validate_reproduction_lock_impl(specification)


def _family_parser_dependencies() -> _FamilyParserDependencies[FamilyClassificationSpec]:
    return _FamilyParserDependencies(
        profiles_spec_type=FamilyProfilesSpec,
        observable_grid_spec_type=ObservableGridSpec,
        models_spec_type=FamilyModelsSpec,
        final_test_spec_type=FinalTestSpec,
        family_spec_type=FamilyClassificationSpec,
        project_root=PROJECT_ROOT,
        family_workflow=FAMILY_WORKFLOW,
    )


def _parse_family(root: Mapping[str, Any]) -> FamilyClassificationSpec:
    return _parse_family_impl(root, _family_parser_dependencies())


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
