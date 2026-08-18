from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.physics import experiment_runner


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


EXPECTED_SMOKE_TOML = """\
# Resolved EoS Lab configuration. Generated automatically.
schema_version = 1
experiment_name = "apr1_cfl4_smoke"
workflow = "pair_sensitivity"
mode = "exploration"

[hadronic_eos]
baseline = "APR-1"

[quark_eos]
model = "cfl_mit_bag"
bag_constant_mev_fm3 = 60.0
pairing_gap_mev = 100.0
strange_quark_mass_mev = 150.0

[deformation]
method = "additive_gaussian_sound_speed"
center_energy_density_mev_fm3 = 220.0
width_mev_fm3 = 50.0
amplitude_start = -0.01
amplitude_stop = 0.01
amplitude_step = 0.01

[physical_requirements]
minimum_maximum_mass_msun = 2.08
maximum_maximum_mass_msun = 3.0
radius_1p4_min_km = 9.5
radius_1p4_max_km = 14.5

[numerical_settings]
preset = "smoke"
convergence_check = "none"

[execution]
random_seed = 20260804
parallel_jobs = 2
amplitudes_per_batch = 3

# Derived amplitude values, catalog identifiers, provenance, and the
# permitted interpretation are recorded in run_manifest.json.
"""


def test_resolved_toml_facade_preserves_exact_signatures():
    assert str(inspect.signature(experiment_runner.render_resolved_toml)) == (
        "(runtime: 'dict[str, Any]') -> 'str'"
    )
    assert str(inspect.signature(experiment_runner._toml_value)) == (
        "(value: 'Any') -> 'str'"
    )


def test_smoke_resolved_toml_is_byte_stable_and_excludes_derived_values():
    runtime = experiment_runner.validate_pair_experiment(
        CONFIGS / "smoke.toml"
    ).runtime_configuration
    runtime["resolved"]["unpersisted_probe"] = "derived-only"
    runtime["resolved_numerical_settings"] = {"unpersisted": 123}

    rendered = experiment_runner.render_resolved_toml(runtime)

    assert rendered == EXPECTED_SMOKE_TOML
    assert "derived-only" not in rendered
    assert "resolved_numerical_settings" not in rendered
    assert rendered.endswith("\n")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(True, "true", id="true-before-int"),
        pytest.param(False, "false", id="false-before-int"),
        pytest.param('delta Δ "quoted"\nline', '"delta Δ \\"quoted\\"\\nline"', id="unicode-string"),
        pytest.param(0, "0", id="integer"),
        pytest.param(-0.0, "-0.0", id="negative-zero"),
        pytest.param(1.25, "1.25", id="finite-float"),
        pytest.param(
            [True, "Δ", (-0.0, 2)],
            '[true, "Δ", [-0.0, 2]]',
            id="recursive-list-and-tuple",
        ),
    ],
)
def test_toml_value_preserves_supported_scalar_and_sequence_rendering(
    value, expected
):
    assert experiment_runner._toml_value(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_toml_value_rejects_every_nonfinite_float(value):
    with pytest.raises(
        ValueError,
        match=r"^Resolved TOML cannot contain a non-finite number\.$",
    ):
        experiment_runner._toml_value(value)


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        pytest.param(None, "NoneType", id="none"),
        pytest.param({"key": "value"}, "dict", id="mapping"),
        pytest.param(object(), "object", id="object"),
    ],
)
def test_toml_value_rejects_unsupported_types_with_the_exact_type(value, type_name):
    with pytest.raises(
        TypeError,
        match=rf"^Unsupported resolved TOML value: {type_name}$",
    ):
        experiment_runner._toml_value(value)


def test_render_resolved_toml_uses_the_live_facade_value_renderer(monkeypatch):
    runtime = experiment_runner.validate_pair_experiment(
        CONFIGS / "smoke.toml"
    ).runtime_configuration
    expected_values = [
        runtime["experiment_name"],
        runtime["workflow"],
        runtime["mode"],
        *runtime["hadronic_eos"].values(),
        *runtime["quark_eos"].values(),
        *runtime["deformation"].values(),
        *runtime["physical_requirements"].values(),
        *runtime["numerical_settings"].values(),
        *runtime["execution"].values(),
    ]
    calls = []

    def render_value(value):
        calls.append(value)
        return f'"value-{len(calls)}"'

    monkeypatch.setattr(experiment_runner, "_toml_value", render_value)

    rendered = experiment_runner.render_resolved_toml(runtime)

    assert calls == expected_values
    assert rendered.splitlines()[1:5] == [
        "schema_version = 1",
        'experiment_name = "value-1"',
        'workflow = "value-2"',
        'mode = "value-3"',
    ]
