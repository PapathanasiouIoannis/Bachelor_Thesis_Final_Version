from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.physics import experiment_runner


SELECTED_AMPLITUDES = (-0.1, 0.0, 0.2)
MATTER_TYPES = ("hadronic", "quark")
VARIANT_NAMES = (
    "double_eos_grid",
    "double_central_pressure_grid",
    "tighter_tov_tolerances",
)
EXPECTED_CONVERGENCE_COLUMNS = (
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


def _runtime(*, amplitudes=(-0.1, -0.05, 0.2), mode="endpoints_and_zero"):
    return {
        "numerical_settings": {"convergence_check": mode},
        "resolved": {
            "amplitudes": list(amplitudes),
            "quark_eos_id": "CFL4-production",
        },
        "hadronic_eos": {"baseline": "APR-1-production"},
        "physical_requirements": {
            "minimum_maximum_mass_msun": "2.08",
            "maximum_maximum_mass_msun": "3.0",
            "radius_1p4_min_km": "9.5",
            "radius_1p4_max_km": "14.5",
        },
    }


def _reference_summary(amplitudes=SELECTED_AMPLITUDES):
    records = [
        {
            "matter_type": matter_type,
            "baseline_name": f"production-{matter_type}",
            "deformation_amplitude": amplitude,
            "maximum_mass_msun": 0.0,
            "radius_1p4_km": 0.0,
            "tidal_deformability_1p4": 100.0,
        }
        for amplitude in amplitudes
        for matter_type in MATTER_TYPES
    ]
    return pd.DataFrame.from_records(
        [records[index] for index in (4, 1, 5, 0, 3, 2)]
    )


def test_convergence_success_matrix_preserves_order_forwarding_and_identities(
    monkeypatch,
):
    runtime = _runtime()
    summary = _reference_summary()
    expected_keys = [
        (amplitude, matter_type, check)
        for amplitude in SELECTED_AMPLITUDES
        for matter_type in MATTER_TYPES
        for check in VARIANT_NAMES
    ]
    settings_calls = []
    build_calls = []
    solve_calls = []
    serialization_calls = []
    summary_calls = []
    physical_calls = []
    events = []

    def resolve_settings(runtime_argument):
        settings_calls.append(runtime_argument)
        events.append(("settings",))
        return {
            "eos_grid_points": 100,
            "central_pressure_points": 80,
            "tov_relative_tolerance": 1.0e-6,
            "tov_absolute_tolerance": 1.0e-8,
        }

    def build_eos(*args, **kwargs):
        runtime_argument, matter_type, amplitude = args
        check = VARIANT_NAMES[len(build_calls) % len(VARIANT_NAMES)]
        key = (float(amplitude), matter_type, check)
        eos = SimpleNamespace(
            key=key,
            baseline_name=(
                "APR-1-refined" if matter_type == "hadronic" else "CFL analytic"
            ),
            catalog_identifier=("" if matter_type == "hadronic" else "CFL4-refined"),
        )
        build_calls.append(
            SimpleNamespace(
                runtime=runtime_argument,
                matter_type=matter_type,
                amplitude=float(amplitude),
                kwargs=dict(kwargs),
                eos=eos,
            )
        )
        events.append(("build", key))
        return eos

    def solve(*args, **kwargs):
        runtime_argument, eos, matter_type = args
        curve = object()
        solve_calls.append(
            SimpleNamespace(
                runtime=runtime_argument,
                eos=eos,
                matter_type=matter_type,
                kwargs=dict(kwargs),
                curve=curve,
            )
        )
        events.append(("solve", eos.key))
        return curve, object(), 2.1

    def serialize_stellar(*args):
        curve, eos, matter_type, sweep_id, curve_id = args
        frame = object()
        serialization_calls.append(
            SimpleNamespace(
                curve=curve,
                eos=eos,
                matter_type=matter_type,
                sweep_id=sweep_id,
                curve_id=curve_id,
                frame=frame,
            )
        )
        events.append(("serialize", eos.key))
        return frame

    def summarize(frame):
        serialization = serialization_calls[len(summary_calls)]
        key = serialization.eos.key
        assert frame is serialization.frame
        check = key[2]
        if check == "double_eos_grid":
            values = (0.01, 0.05, 102.0)
        elif check == "double_central_pressure_grid":
            values = (
                np.nextafter(0.01, np.inf),
                0.02,
                101.0,
            )
        elif key[1] == "hadronic":
            values = (0.005, np.nextafter(0.05, np.inf), 101.0)
        else:
            values = (0.005, 0.02, np.nextafter(102.0, np.inf))
        refined = {
            "maximum_mass_msun": values[0],
            "radius_1p4_km": values[1],
            "tidal_deformability_1p4": values[2],
            "test_key": key,
        }
        summary_calls.append((frame, refined))
        events.append(("summarize", key))
        return refined

    def physical_status(runtime_argument, refined):
        key = refined["test_key"]
        result = (
            (False, "injected physical screen failure")
            if key == (0.0, "quark", "double_eos_grid")
            else (True, "passed")
        )
        physical_calls.append((runtime_argument, refined, result))
        events.append(("physical", key))
        return result

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        resolve_settings,
    )
    monkeypatch.setattr(experiment_runner, "_build_eos", build_eos)
    monkeypatch.setattr(experiment_runner, "_solve", solve)
    monkeypatch.setattr(
        experiment_runner,
        "stellar_curve_to_frame",
        serialize_stellar,
    )
    monkeypatch.setattr(
        experiment_runner,
        "summarize_stellar_curve",
        summarize,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_physical_requirements_status",
        physical_status,
    )

    result = experiment_runner._run_convergence_checks(runtime, summary)

    assert settings_calls == [runtime]
    assert settings_calls[0] is runtime
    assert experiment_runner.CONVERGENCE_COLUMNS == EXPECTED_CONVERGENCE_COLUMNS
    assert tuple(result.columns) == EXPECTED_CONVERGENCE_COLUMNS
    assert list(
        result[["deformation_amplitude", "matter_type", "check"]].itertuples(
            index=False, name=None
        )
    ) == expected_keys
    expected_events = [("settings",)]
    for key in expected_keys:
        expected_events.extend(
            (
                ("build", key),
                ("solve", key),
                ("serialize", key),
                ("summarize", key),
                ("physical", key),
            )
        )
    assert events == expected_events

    expected_overrides = {
        "double_eos_grid": (
            {"grid_points": 200},
            {
                "n_points": None,
                "rtol": None,
                "atol": None,
                "enforce_physical_requirements": False,
            },
        ),
        "double_central_pressure_grid": (
            {"grid_points": None},
            {
                "n_points": 160,
                "rtol": None,
                "atol": None,
                "enforce_physical_requirements": False,
            },
        ),
        "tighter_tov_tolerances": (
            {"grid_points": None},
            {
                "n_points": None,
                "rtol": 1.0e-7,
                "atol": 1.0e-9,
                "enforce_physical_requirements": False,
            },
        ),
    }
    for key, build, solved, serialized, summarized, physical in zip(
        expected_keys,
        build_calls,
        solve_calls,
        serialization_calls,
        summary_calls,
        physical_calls,
    ):
        amplitude, matter_type, check = key
        expected_build, expected_solve = expected_overrides[check]
        assert build.runtime is runtime
        assert (build.amplitude, build.matter_type, build.kwargs) == (
            amplitude,
            matter_type,
            expected_build,
        )
        assert solved.runtime is runtime
        assert solved.eos is build.eos
        assert solved.matter_type == matter_type
        assert solved.kwargs == expected_solve
        assert serialized.curve is solved.curve
        assert serialized.eos is build.eos
        assert serialized.matter_type == matter_type
        assert serialized.sweep_id == "convergence"
        assert serialized.curve_id == (
            f"convergence_{matter_type}_{amplitude}_{check}"
        )
        assert summarized[0] is serialized.frame
        assert physical[0] is runtime
        assert physical[1] is summarized[1]

    expected_baselines = [
        "APR-1-refined" if matter_type == "hadronic" else "CFL4-refined"
        for _, matter_type, _ in expected_keys
    ]
    assert result["baseline_name"].tolist() == expected_baselines
    expected_mass = []
    expected_radius = []
    expected_tidal = []
    expected_mass_flags = []
    expected_radius_flags = []
    expected_tidal_flags = []
    expected_physical = []
    expected_reasons = []
    for key in expected_keys:
        check = key[2]
        if check == "double_eos_grid":
            expected_mass.append(0.01)
            expected_radius.append(0.05)
            expected_tidal.append(0.02)
            expected_mass_flags.append(True)
            expected_radius_flags.append(True)
            expected_tidal_flags.append(True)
        elif check == "double_central_pressure_grid":
            expected_mass.append(np.nextafter(0.01, np.inf))
            expected_radius.append(0.02)
            expected_tidal.append(0.01)
            expected_mass_flags.append(False)
            expected_radius_flags.append(True)
            expected_tidal_flags.append(True)
        elif key[1] == "hadronic":
            expected_mass.append(0.005)
            expected_radius.append(np.nextafter(0.05, np.inf))
            expected_tidal.append(0.01)
            expected_mass_flags.append(True)
            expected_radius_flags.append(False)
            expected_tidal_flags.append(True)
        else:
            expected_mass.append(0.005)
            expected_radius.append(0.02)
            expected_tidal.append(
                (np.nextafter(102.0, np.inf) - 100.0) / 100.0
            )
            expected_mass_flags.append(True)
            expected_radius_flags.append(True)
            expected_tidal_flags.append(False)
        physical_passed = key != (0.0, "quark", "double_eos_grid")
        expected_physical.append(physical_passed)
        expected_reasons.append(
            "passed" if physical_passed else "injected physical screen failure"
        )

    assert result["delta_maximum_mass_msun"].tolist() == pytest.approx(
        expected_mass
    )
    assert result["delta_radius_1p4_km"].tolist() == pytest.approx(expected_radius)
    assert result["relative_delta_tidal_deformability_1p4"].tolist() == (
        pytest.approx(expected_tidal)
    )
    assert result["maximum_mass_passed"].tolist() == expected_mass_flags
    assert result["radius_1p4_passed"].tolist() == expected_radius_flags
    assert result["tidal_deformability_1p4_passed"].tolist() == (
        expected_tidal_flags
    )
    assert result["refined_physical_requirements_passed"].tolist() == (
        expected_physical
    )
    assert result["refined_physical_requirements_reason"].tolist() == (
        expected_reasons
    )
    assert result["passed"].tolist() == [
        mass and radius and tidal and physical
        for mass, radius, tidal, physical in zip(
            expected_mass_flags,
            expected_radius_flags,
            expected_tidal_flags,
            expected_physical,
        )
    ]
