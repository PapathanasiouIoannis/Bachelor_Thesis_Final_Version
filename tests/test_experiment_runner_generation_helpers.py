from __future__ import annotations

from pathlib import Path

import pytest
from joblib import Parallel, delayed

from src.physics import experiment_runner


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_build_eos_forwards_defaults_overrides_and_live_dependencies(monkeypatch):
    runtime = {
        "deformation": {
            "center_energy_density_mev_fm3": "220.5",
            "width_mev_fm3": "50.25",
        },
        "hadronic_eos": {"baseline": "APR-1"},
        "resolved": {"quark_eos_id": "CFL_CUSTOM"},
    }
    settings_calls = []
    deformation_calls = []
    hadronic_calls = []
    quark_calls = []
    quark_parameter_calls = []
    deformations = (object(), object(), object())
    deformation_sentinels = iter(deformations)
    quark_parameters = object()
    hadronic_result = object()
    quark_result = object()

    def resolve_settings(runtime_argument):
        settings_calls.append(runtime_argument)
        return {"eos_grid_points": 321}

    def build_deformation(**arguments):
        deformation_calls.append(arguments)
        return next(deformation_sentinels)

    def build_hadronic(*args, **kwargs):
        hadronic_calls.append((args, kwargs))
        return hadronic_result

    def resolve_quark_parameters(runtime_argument):
        quark_parameter_calls.append(runtime_argument)
        return quark_parameters

    def build_quark(*args, **kwargs):
        quark_calls.append((args, kwargs))
        return quark_result

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        resolve_settings,
    )
    monkeypatch.setattr(experiment_runner, "GaussianDeformation", build_deformation)
    monkeypatch.setattr(experiment_runner, "build_hadronic_eos", build_hadronic)
    monkeypatch.setattr(
        experiment_runner,
        "_quark_parameters",
        resolve_quark_parameters,
    )
    monkeypatch.setattr(experiment_runner, "build_quark_eos", build_quark)
    monkeypatch.setitem(experiment_runner.CONFIG, "M_N", 938.75)

    hadronic = experiment_runner._build_eos(runtime, "hadronic", "0.125")
    quark = experiment_runner._build_eos(
        runtime,
        "quark",
        "-0.25",
        grid_points=654,
    )

    assert hadronic is hadronic_result
    assert quark is quark_result
    assert settings_calls == [runtime]
    assert deformation_calls[:2] == [
        {"amplitude": 0.125, "epsilon0": 220.5, "sigma": 50.25},
        {"amplitude": -0.25, "epsilon0": 220.5, "sigma": 50.25},
    ]
    assert hadronic_calls == [
        (("APR-1", deformations[0]), {"grid_points": 321})
    ]
    assert quark_parameter_calls == [runtime]
    assert quark_calls == [
        (
            (quark_parameters, deformations[1]),
            {
                "maximum_surface_energy_per_baryon": 938.75,
                "grid_points": 654,
                "catalog_identifier": "CFL_CUSTOM",
            },
        )
    ]

    with pytest.raises(ValueError) as raised:
        experiment_runner._build_eos(
            runtime,
            "hybrid",
            "0.5",
            grid_points=777,
        )

    assert str(raised.value) == "matter_type must be 'hadronic' or 'quark'."
    assert settings_calls == [runtime]
    assert deformation_calls == [
        {"amplitude": 0.125, "epsilon0": 220.5, "sigma": 50.25},
        {"amplitude": -0.25, "epsilon0": 220.5, "sigma": 50.25},
        {"amplitude": 0.5, "epsilon0": 220.5, "sigma": 50.25},
    ]
    assert hadronic_calls == [
        (("APR-1", deformations[0]), {"grid_points": 321})
    ]
    assert quark_parameter_calls == [runtime]
    assert quark_calls == [
        (
            (quark_parameters, deformations[1]),
            {
                "maximum_surface_energy_per_baryon": 938.75,
                "grid_points": 654,
                "catalog_identifier": "CFL_CUSTOM",
            },
        )
    ]


def test_solve_forwards_resolved_defaults_and_explicit_overrides(monkeypatch):
    runtime = {
        "physical_requirements": {
            "minimum_maximum_mass_msun": "2.08",
            "maximum_maximum_mass_msun": "3.0",
            "radius_1p4_min_km": "9.5",
            "radius_1p4_max_km": "14.5",
        }
    }
    settings_calls = []
    solver_calls = []
    results = [(object(), object(), object()), (object(), object(), object())]
    eos = object()

    def resolve_settings(runtime_argument):
        settings_calls.append(runtime_argument)
        return {
            "central_pressure_points": 80,
            "tov_relative_tolerance": 1.0e-7,
            "tov_absolute_tolerance": 1.0e-9,
        }

    def solve_and_validate(*args, **kwargs):
        solver_calls.append((args, kwargs))
        return results[len(solver_calls) - 1]

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        resolve_settings,
    )
    monkeypatch.setattr(
        experiment_runner,
        "solve_and_validate_sequence",
        solve_and_validate,
    )

    default_result = experiment_runner._solve(runtime, eos, "quark")
    override_result = experiment_runner._solve(
        runtime,
        eos,
        "hadronic",
        n_points=17,
        rtol=2.0e-8,
        atol=3.0e-10,
        enforce_physical_requirements=False,
    )

    assert default_result is results[0]
    assert override_result is results[1]
    assert settings_calls == [runtime, runtime]
    assert solver_calls == [
        (
            (eos,),
            {
                "is_quark": True,
                "minimum_maximum_mass": 2.08,
                "maximum_maximum_mass": 3.0,
                "radius_14_bounds": (9.5, 14.5),
                "n_points": 80,
                "rtol": 1.0e-7,
                "atol": 1.0e-9,
                "enforce_physical_requirements": True,
            },
        ),
        (
            (eos,),
            {
                "is_quark": False,
                "minimum_maximum_mass": 2.08,
                "maximum_maximum_mass": 3.0,
                "radius_14_bounds": (9.5, 14.5),
                "n_points": 17,
                "rtol": 2.0e-8,
                "atol": 3.0e-10,
                "enforce_physical_requirements": False,
            },
        ),
    ]


def test_generate_pair_facade_runs_in_worker_processes(tmp_path):
    preflight = experiment_runner.validate_pair_experiment(
        CONFIGS / "smoke.toml"
    )
    run_log = tmp_path / "logs" / "pipeline.log"

    results = Parallel(n_jobs=2, prefer="processes")(
        delayed(experiment_runner._generate_pair)(
            preflight.runtime_configuration,
            index,
            amplitude,
            preflight.resolved.config_hash,
            str(run_log),
        )
        for index, amplitude in enumerate((-0.01, 0.0))
    )

    assert [result["accepted"] for result in results] == [False, False]
    assert [result["rejection"]["stage"] for result in results] == [
        "eos_validation",
        "eos_validation",
    ]
    assert [len(result["eos_frames"]) for result in results] == [2, 2]
