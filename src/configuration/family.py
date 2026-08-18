"""Parsing and scientific-policy validation for family classification."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, TypeVar

from src.configuration.common import (
    ConfigurationError,
    _boolean,
    _decimal,
    _integer,
    _require_exact_keys,
    _require_table,
    _string,
    _string_tuple,
    _validate_common_header,
)


FamilySpecificationT = TypeVar("FamilySpecificationT")


@dataclass(frozen=True, slots=True)
class FamilyParserDependencies(Generic[FamilySpecificationT]):
    """Model constructors and constants supplied by the compatibility facade."""

    profiles_spec_type: type[Any]
    observable_grid_spec_type: type[Any]
    models_spec_type: type[Any]
    final_test_spec_type: type[Any]
    family_spec_type: Callable[..., FamilySpecificationT]
    project_root: Path
    family_workflow: str


def parse_family(
    root: Mapping[str, Any],
    dependencies: FamilyParserDependencies[FamilySpecificationT],
) -> FamilySpecificationT:
    """Parse a validated family-classification mapping into facade model types."""

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
    if workflow != dependencies.family_workflow:
        raise ConfigurationError(
            f"Family configuration workflow must be {dependencies.family_workflow!r}."
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
    profile_spec = dependencies.profiles_spec_type(
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
        if not (dependencies.project_root / required).is_file():
            raise ConfigurationError(
                f"Audited profile {required!r} is missing from the repository."
            )

    grid = _require_table(root["observable_grid"], "observable_grid")
    _require_exact_keys(
        grid,
        {"minimum_mass_msun", "maximum_mass_msun", "mass_points"},
        "observable_grid",
    )
    grid_spec = dependencies.observable_grid_spec_type(
        minimum_mass_msun=_decimal(
            grid["minimum_mass_msun"], "observable_grid.minimum_mass_msun"
        ),
        maximum_mass_msun=_decimal(
            grid["maximum_mass_msun"], "observable_grid.maximum_mass_msun"
        ),
        mass_points=_integer(grid["mass_points"], "observable_grid.mass_points"),
    )
    required_grid = dependencies.observable_grid_spec_type(
        Decimal("1.0"), Decimal("2.0"), 21
    )
    if grid_spec != required_grid:
        raise ConfigurationError(
            "The audited family workflow requires 21 mass points from 1.0 to "
            "2.0 solar masses."
        )

    models = _require_table(root["models"], "models")
    _require_exact_keys(models, {"primary", "exploratory"}, "models")
    model_spec = dependencies.models_spec_type(
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
    final_test_spec = dependencies.final_test_spec_type(
        policy=_string(final_test["policy"], "final_test.policy"),
        allow_evaluation=_boolean(
            final_test["allow_evaluation"], "final_test.allow_evaluation"
        ),
    )
    required_final_test = dependencies.final_test_spec_type(
        "already_opened_read_only", False
    )
    if final_test_spec != required_final_test:
        raise ConfigurationError(
            "The family final test has already been opened. Set policy = "
            "'already_opened_read_only' and allow_evaluation = false."
        )

    return dependencies.family_spec_type(
        schema_version=schema_version,
        experiment_name=name,
        workflow=workflow,
        mode=mode,
        profiles=profile_spec,
        observable_grid=grid_spec,
        models=model_spec,
        final_test=final_test_spec,
    )


__all__ = ["FamilyParserDependencies", "parse_family"]
