import subprocess
import sys
import textwrap
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from src.experiment_config import load_pair_experiment, resolve_pair_experiment
from src.physics import experiment_runner


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RECOVERY_LIMIT = 2.0e-4


@pytest.mark.parametrize("matter_type", ["hadronic", "quark"])
def test_preflight_baseline_recovery_limit_is_strict_and_matter_specific(
    monkeypatch,
    matter_type,
):
    at_limit = {"hadronic": 0.0, "quark": 0.0}
    at_limit[matter_type] = RECOVERY_LIMIT
    monkeypatch.setattr(
        experiment_runner,
        "_baseline_recovery_errors",
        lambda runtime: at_limit,
    )

    accepted = experiment_runner.validate_pair_experiment(CONFIGS / "smoke.toml")

    assert accepted.baseline_recovery == at_limit

    above_limit = np.nextafter(RECOVERY_LIMIT, np.inf)
    rejected = {"hadronic": 0.0, "quark": 0.0}
    rejected[matter_type] = above_limit
    monkeypatch.setattr(
        experiment_runner,
        "_baseline_recovery_errors",
        lambda runtime: rejected,
    )

    with pytest.raises(ValueError) as raised:
        experiment_runner.validate_pair_experiment(CONFIGS / "smoke.toml")

    assert str(raised.value) == (
        f"A = 0 {matter_type} maximum relative pressure-recovery error is "
        f"{above_limit:.6g}, above the permitted maximum relative tolerance "
        "0.0002."
    )


def test_preflight_common_lower_endpoint_is_open(monkeypatch):
    interval_calls = []

    def interval_with_start_as_lower_endpoint(*args):
        interval_calls.append(args)
        return -0.01, 0.5

    monkeypatch.setattr(
        experiment_runner,
        "admissible_amplitude_interval",
        interval_with_start_as_lower_endpoint,
    )

    with pytest.raises(ValueError) as raised:
        experiment_runner.validate_pair_experiment(CONFIGS / "smoke.toml")

    assert str(raised.value) == (
        "amplitude_start = -0.01 is not above the common permitted lower "
        "boundary -0.01. Choose a larger value."
    )
    assert len(interval_calls) == 2


def test_preflight_preserves_an_already_resolved_experiment():
    resolved = resolve_pair_experiment(CONFIGS / "smoke.toml")

    preflight = experiment_runner.validate_pair_experiment(resolved)

    assert preflight.resolved is resolved


def test_preflight_report_preserves_the_manifest_schema():
    preflight = experiment_runner.validate_pair_experiment(CONFIGS / "smoke.toml")

    report = preflight.to_dict()

    assert tuple(report) == (
        "experiment_name",
        "workflow",
        "mode",
        "configuration_hash",
        "hadronic_eos",
        "quark_eos",
        "deformation",
        "physical_requirements",
        "numerical_settings",
        "resolved_numerical_settings",
        "execution",
        "admissible_amplitude_intervals",
        "baseline_recovery_maximum_relative_pressure_error",
        "expected_curves",
        "classification_enabled",
        "permitted_scientific_interpretation",
        "provenance",
    )
    assert report["configuration_hash"] == preflight.resolved.config_hash
    assert report["deformation"]["amplitudes"] == [
        point.amplitude for point in preflight.sweep_points
    ]
    assert report["resolved_numerical_settings"] == (
        experiment_runner.NUMERICAL_PRESETS["smoke"]
    )
    assert report["admissible_amplitude_intervals"] == {
        "hadronic": list(preflight.hadronic_interval),
        "quark": list(preflight.quark_interval),
        "common": list(preflight.common_interval),
        "lower_endpoint_is_open": True,
    }
    assert report["baseline_recovery_maximum_relative_pressure_error"] == (
        preflight.baseline_recovery
    )
    assert report["provenance"] == preflight.provenance
    assert report["expected_curves"] == 2 * len(preflight.sweep_points)
    assert report["classification_enabled"] is False
    assert (
        report["permitted_scientific_interpretation"]
        == experiment_runner.PAIR_INTERPRETATION
    )


def test_preflight_reports_fallback_provenance_for_custom_cfl_parameters():
    specification = load_pair_experiment(CONFIGS / "apr1_cfl4_exploration.toml")
    custom_quark = replace(
        specification.quark_eos,
        bag_constant_mev_fm3=Decimal("72.5"),
        pairing_gap_mev=Decimal("80"),
        strange_quark_mass_mev=Decimal("95"),
    )
    resolved = resolve_pair_experiment(
        replace(specification, quark_eos=custom_quark)
    )

    preflight = experiment_runner.validate_pair_experiment(resolved)

    assert preflight.provenance["quark"] == {
        "eos_id": "CFL_B72p5_D80_MS95",
        "model_family": "analytic CFL MIT-bag",
        "provenance_note": (
            "Exploratory custom parameter tuple; the tuple itself is not a named "
            "benchmark in the repository literature catalog."
        ),
    }
    assert (
        preflight.to_dict()["quark_eos"]["catalog_identifier"]
        == "CFL_B72p5_D80_MS95"
    )


def test_resolved_numerical_settings_are_fresh_and_fail_with_an_exact_cause():
    runtime = experiment_runner.validate_pair_experiment(
        CONFIGS / "smoke.toml"
    ).runtime_configuration

    first = experiment_runner._resolved_numerical_settings(runtime)
    second = experiment_runner._resolved_numerical_settings(runtime)

    assert first == second == experiment_runner.NUMERICAL_PRESETS["smoke"]
    assert first is not second

    unknown = {
        **runtime,
        "numerical_settings": {
            **runtime["numerical_settings"],
            "preset": "unknown",
        },
    }
    with pytest.raises(ValueError) as raised:
        experiment_runner._resolved_numerical_settings(unknown)

    assert str(raised.value) == "Unknown numerical preset: 'unknown'."
    assert isinstance(raised.value.__cause__, KeyError)


def test_quark_parameters_use_the_facade_constructor(monkeypatch):
    sentinel = object()
    constructor_calls = []

    def recording_constructor(**arguments):
        constructor_calls.append(arguments)
        return sentinel

    monkeypatch.setattr(
        experiment_runner,
        "QuarkParameters",
        recording_constructor,
    )

    result = experiment_runner._quark_parameters(
        {
            "quark_eos": {
                "bag_constant_mev_fm3": "72.5",
                "pairing_gap_mev": "80",
                "strange_quark_mass_mev": "95",
            }
        }
    )

    assert result is sentinel
    assert constructor_calls == [
        {
            "bag_b": 72.5,
            "gap_delta": 80.0,
            "strange_mass": 95.0,
        }
    ]


def test_preflight_report_resolves_facade_settings_and_interpretation_late(
    monkeypatch,
):
    preflight = experiment_runner.validate_pair_experiment(CONFIGS / "smoke.toml")
    sentinel_settings = {"sentinel": 17}
    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        lambda runtime: sentinel_settings,
    )
    monkeypatch.setattr(
        experiment_runner,
        "PAIR_INTERPRETATION",
        "patched interpretation",
    )

    report = preflight.to_dict()

    assert report["resolved_numerical_settings"] is sentinel_settings
    assert report["permitted_scientific_interpretation"] == "patched interpretation"


def test_facade_import_and_reload_resnapshot_numerical_configuration():
    script = textwrap.dedent(
        """
        import importlib

        from src.config import CONFIG
        from src.physics.runner import settings

        original = int(CONFIG["P_GRID_POINTS"])
        CONFIG["P_GRID_POINTS"] = original + 17

        from src.physics import experiment_runner

        assert (
            experiment_runner.NUMERICAL_PRESETS["production"]["eos_grid_points"]
            == original + 17
        )
        assert experiment_runner.NUMERICAL_PRESETS is settings.NUMERICAL_PRESETS

        CONFIG["P_GRID_POINTS"] = original + 23
        assert (
            experiment_runner.NUMERICAL_PRESETS["production"]["eos_grid_points"]
            == original + 17
        )

        importlib.reload(experiment_runner)
        assert (
            experiment_runner.NUMERICAL_PRESETS["production"]["eos_grid_points"]
            == original + 23
        )
        assert experiment_runner.NUMERICAL_PRESETS is settings.NUMERICAL_PRESETS
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=CONFIGS.parent,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
