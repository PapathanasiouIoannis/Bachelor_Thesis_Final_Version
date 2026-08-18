from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

import pytest

from src.experiment_config import (
    ConfigurationError,
    FamilyClassificationSpec,
    PairExperimentSpec,
    canonical_sha256,
    decimal_amplitude_grid,
    load_experiment_config,
    load_pair_experiment,
    resolve_pair_experiment,
)


CONFIGS = Path(__file__).resolve().parents[1] / "configs"

PAIR_ROOT_KEYS = (
    "deformation",
    "execution",
    "experiment_name",
    "hadronic_eos",
    "mode",
    "numerical_settings",
    "physical_requirements",
    "quark_eos",
    "schema_version",
    "workflow",
)


def _replace_profile(tmp_path: Path, source_name: str, old: str, new: str) -> Path:
    text = (CONFIGS / source_name).read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / source_name
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_reproduction_profile_loads_as_immutable_typed_specification():
    specification = load_pair_experiment(CONFIGS / "apr1_cfl4_reproduction.toml")

    assert isinstance(specification, PairExperimentSpec)
    assert specification.hadronic_eos.baseline == "APR-1"
    assert specification.quark_eos.bag_constant_mev_fm3 == Decimal("60.0")
    assert specification.deformation.center_energy_density_mev_fm3 == Decimal("220.0")
    assert len(specification.amplitudes) == 15
    assert specification.amplitudes[0] == Decimal("-0.05")
    assert specification.amplitudes[-1] == Decimal("0.09")
    assert Decimal("0") in specification.amplitudes
    with pytest.raises(FrozenInstanceError):
        specification.mode = "exploration"  # type: ignore[misc]


def test_exact_decimal_grid_does_not_replace_or_shift_values():
    values = decimal_amplitude_grid(Decimal("-0.03"), Decimal("0.03"), Decimal("0.01"))
    assert values == tuple(
        Decimal(value)
        for value in ("-0.03", "-0.02", "-0.01", "0.00", "0.01", "0.02", "0.03")
    )


@pytest.mark.parametrize(
    ("start", "stop", "step", "message"),
    [
        ("0.01", "0.03", "0.01", "A = 0"),
        ("-0.03", "0.032", "0.01", "not exactly aligned"),
        ("0", "0.03", "0", "strictly positive"),
        ("0.03", "-0.03", "0.01", "smaller"),
    ],
)
def test_invalid_decimal_grids_are_rejected(start, stop, step, message):
    with pytest.raises(ConfigurationError, match=message):
        decimal_amplitude_grid(Decimal(start), Decimal(stop), Decimal(step))


def test_unknown_and_missing_fields_are_rejected(tmp_path):
    unknown = _replace_profile(
        tmp_path,
        "apr1_cfl4_exploration.toml",
        "width_mev_fm3 = 50.0",
        "width_mev_fm3 = 50.0\nwidht_mev_fm3 = 50.0",
    )
    with pytest.raises(ConfigurationError, match="unknown fields.*widht_mev_fm3"):
        load_experiment_config(unknown)

    missing = _replace_profile(
        tmp_path,
        "apr1_cfl4_reproduction.toml",
        "parallel_jobs = 4\n",
        "",
    )
    with pytest.raises(ConfigurationError, match="missing fields.*parallel_jobs"):
        load_experiment_config(missing)


@pytest.mark.parametrize("name", ["ab", "a" * 65, "Upper_case"])
def test_experiment_name_matches_run_directory_contract(tmp_path, name):
    path = _replace_profile(
        tmp_path,
        "apr1_cfl4_exploration.toml",
        'experiment_name = "apr1_cfl4_exploration"',
        f'experiment_name = "{name}"',
    )
    with pytest.raises(ConfigurationError, match="3-64"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ('baseline = "APR-1"', 'baseline = "BGP"', "hadronic_eos.baseline"),
        ("bag_constant_mev_fm3 = 60.0", "bag_constant_mev_fm3 = 61.0", "bag_constant"),
        ("pairing_gap_mev = 100.0", "pairing_gap_mev = 90.0", "pairing_gap"),
        (
            "strange_quark_mass_mev = 150.0",
            "strange_quark_mass_mev = 140.0",
            "strange_quark",
        ),
        (
            "center_energy_density_mev_fm3 = 220.0",
            "center_energy_density_mev_fm3 = 221.0",
            "center_energy",
        ),
        ("width_mev_fm3 = 50.0", "width_mev_fm3 = 51.0", "width"),
        ("amplitude_stop = 0.09", "amplitude_stop = 0.08", "amplitude_stop"),
        (
            "minimum_maximum_mass_msun = 2.08",
            "minimum_maximum_mass_msun = 2.0",
            "minimum_maximum",
        ),
        (
            'convergence_check = "endpoints_and_zero"',
            'convergence_check = "none"',
            "convergence_check",
        ),
        ("random_seed = 20260804", "random_seed = 7", "random_seed"),
    ],
)
def test_reproduction_scientific_values_are_locked(tmp_path, old, new, field):
    path = _replace_profile(tmp_path, "apr1_cfl4_reproduction.toml", old, new)
    with pytest.raises(ConfigurationError, match=field):
        load_experiment_config(path)


def test_exploration_allows_one_catalog_baseline_and_custom_positive_cfl_tuple(
    tmp_path,
):
    text = (CONFIGS / "apr1_cfl4_exploration.toml").read_text(encoding="utf-8")
    text = text.replace('baseline = "APR-1"', 'baseline = "BGP"')
    text = text.replace("bag_constant_mev_fm3 = 60.0", "bag_constant_mev_fm3 = 72.5")
    text = text.replace("pairing_gap_mev = 100.0", "pairing_gap_mev = 80.0")
    text = text.replace(
        "strange_quark_mass_mev = 150.0", "strange_quark_mass_mev = 95.0"
    )
    path = tmp_path / "custom.toml"
    path.write_text(text, encoding="utf-8")

    resolved = resolve_pair_experiment(path)
    assert resolved.specification.hadronic_eos.baseline == "BGP"
    assert resolved.quark_eos_id == "CFL_B72p5_D80_MS95"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "bag_constant_mev_fm3 = 60.0",
            "bag_constant_mev_fm3 = 0.0",
            "strictly positive",
        ),
        ("pairing_gap_mev = 100.0", "pairing_gap_mev = -1.0", "strictly positive"),
        (
            "strange_quark_mass_mev = 150.0",
            "strange_quark_mass_mev = -1.0",
            "non-negative",
        ),
        ("width_mev_fm3 = 50.0", "width_mev_fm3 = 0.0", "strictly positive"),
    ],
)
def test_exploration_rejects_nonphysical_parameter_values(tmp_path, old, new, message):
    path = _replace_profile(tmp_path, "apr1_cfl4_exploration.toml", old, new)
    with pytest.raises(ConfigurationError, match=message):
        load_experiment_config(path)


def test_exploration_rejects_unknown_hadronic_baseline(tmp_path):
    path = _replace_profile(
        tmp_path,
        "apr1_cfl4_exploration.toml",
        'baseline = "APR-1"',
        'baseline = "NOT-AN-EOS"',
    )
    with pytest.raises(ConfigurationError, match="repository catalog"):
        load_experiment_config(path)


def test_resolved_reproduction_identifies_cfl4_and_has_stable_hash():
    path = CONFIGS / "apr1_cfl4_reproduction.toml"
    first = resolve_pair_experiment(path)
    second = load_pair_experiment(path).resolve()

    assert first.quark_eos_id == "CFL4"
    assert first.to_dict() == second.to_dict()
    assert first.config_hash == second.config_hash
    assert len(first.config_hash) == 64
    assert first.to_dict()["resolved"]["amplitudes"][5] == "0"
    assert first.to_runtime_dict()["resolved"]["amplitudes"][5] == 0.0


@pytest.mark.parametrize(
    ("profile_name", "specification_hash", "resolved_hash"),
    [
        (
            "apr1_cfl4_reproduction.toml",
            "4573d40cd820a76a6919595b96933d34f41cd9aabcf04c824229c7af55a3cf26",
            "d1c447f5d5bc73834ea74cfa964329d4caeba3856bae1abfc9a435236a37d942",
        ),
        (
            "apr1_cfl4_exploration.toml",
            "061cd262ad8f1e06f6bbccf909c0bdff52312ecf52def6a566c44fa1176df680",
            "491e134d61eba6d28b7266cd4f268e7ec66845dbcfeb4615e86c761c001d23eb",
        ),
        (
            "smoke.toml",
            "7f2da3358eeb309ed5ea421424a4cb1b43efa1756fc67f4cbdbbe6620ccbb119",
            "2d65745f1dadb695468ce453e802494701e9d47dd9ab2e9c935cfd6edb92c6cc",
        ),
    ],
)
def test_shipped_pair_configuration_contracts_have_literal_hashes(
    profile_name, specification_hash, resolved_hash
):
    specification = load_pair_experiment(CONFIGS / profile_name)
    resolved = resolve_pair_experiment(CONFIGS / profile_name)

    assert specification.config_hash == specification_hash
    assert resolved.config_hash == resolved_hash
    assert tuple(specification.to_dict()) == PAIR_ROOT_KEYS
    assert tuple(resolved.to_dict()) == ("specification", "resolved")
    assert tuple(resolved.to_dict()["resolved"]) == (
        "amplitudes",
        "quark_eos_id",
        "permitted_scientific_interpretation",
    )


def test_shipped_pair_serialization_preserves_section_shapes():
    payload = load_pair_experiment(
        CONFIGS / "apr1_cfl4_reproduction.toml"
    ).to_dict()

    assert {
        key: tuple(value)
        for key, value in payload.items()
        if isinstance(value, dict)
    } == {
        "deformation": (
            "amplitude_start",
            "amplitude_step",
            "amplitude_stop",
            "center_energy_density_mev_fm3",
            "method",
            "width_mev_fm3",
        ),
        "execution": ("amplitudes_per_batch", "parallel_jobs", "random_seed"),
        "hadronic_eos": ("baseline",),
        "numerical_settings": ("convergence_check", "preset"),
        "physical_requirements": (
            "maximum_maximum_mass_msun",
            "minimum_maximum_mass_msun",
            "radius_1p4_max_km",
            "radius_1p4_min_km",
        ),
        "quark_eos": (
            "bag_constant_mev_fm3",
            "model",
            "pairing_gap_mev",
            "strange_quark_mass_mev",
        ),
    }


def test_canonical_hash_normalizes_equivalent_decimal_notation():
    assert canonical_sha256({"value": Decimal("60.0")}) == canonical_sha256(
        {"value": Decimal("6E+1")}
    )


def test_smoke_profile_has_three_amplitudes_and_six_expected_curves():
    resolved = resolve_pair_experiment(CONFIGS / "smoke.toml")
    assert resolved.amplitudes == (
        Decimal("-0.01"),
        Decimal("0.00"),
        Decimal("0.01"),
    )
    assert 2 * len(resolved.amplitudes) == 6
    assert resolved.specification.numerical_settings.preset == "smoke"


def test_family_profile_loads_with_read_only_final_test_policy():
    specification = load_experiment_config(CONFIGS / "family_classification.toml")
    assert isinstance(specification, FamilyClassificationSpec)
    assert specification.observable_grid.mass_points == 21
    assert specification.models.primary == ("dummy", "logistic_regression")
    assert specification.final_test.policy == "already_opened_read_only"
    assert specification.final_test.allow_evaluation is False

    with pytest.raises(ConfigurationError, match="not a paired"):
        load_pair_experiment(CONFIGS / "family_classification.toml")


def test_shipped_family_configuration_contract_has_literal_hash_and_shape():
    specification = load_experiment_config(CONFIGS / "family_classification.toml")
    payload = specification.to_dict()

    assert (
        specification.config_hash
        == "bb1ef4c1f6536768b5e951ae1032038439e2704c512490c922a50736c9e02ca2"
    )
    assert tuple(payload) == (
        "experiment_name",
        "final_test",
        "mode",
        "models",
        "observable_grid",
        "profiles",
        "schema_version",
        "workflow",
    )
    assert {
        key: tuple(value)
        for key, value in payload.items()
        if isinstance(value, dict)
    } == {
        "final_test": ("allow_evaluation", "policy"),
        "models": ("exploratory", "primary"),
        "observable_grid": (
            "mass_points",
            "maximum_mass_msun",
            "minimum_mass_msun",
        ),
        "profiles": ("generation_profile", "model_profile", "split_profile"),
    }


def test_family_profile_refuses_final_test_re_evaluation(tmp_path):
    path = _replace_profile(
        tmp_path,
        "family_classification.toml",
        "allow_evaluation = false",
        "allow_evaluation = true",
    )
    with pytest.raises(ConfigurationError, match="already been opened"):
        load_experiment_config(path)


def test_loader_rejects_unsupported_workflow_and_malformed_toml(tmp_path):
    unsupported = _replace_profile(
        tmp_path,
        "smoke.toml",
        'workflow = "pair_sensitivity"',
        'workflow = "universal_classifier"',
    )
    with pytest.raises(ConfigurationError, match="unsupported"):
        load_experiment_config(unsupported)

    malformed = tmp_path / "malformed.toml"
    malformed.write_text("not = [valid", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid TOML"):
        load_experiment_config(malformed)
