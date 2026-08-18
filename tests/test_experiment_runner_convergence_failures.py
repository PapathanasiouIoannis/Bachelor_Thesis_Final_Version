from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.physics import experiment_runner


CHECKS = (
    "double_eos_grid",
    "double_central_pressure_grid",
    "tighter_tov_tolerances",
)
DELTA_COLUMNS = (
    "delta_maximum_mass_msun",
    "delta_radius_1p4_km",
    "relative_delta_tidal_deformability_1p4",
)
PASS_COLUMNS = (
    "maximum_mass_passed",
    "radius_1p4_passed",
    "tidal_deformability_1p4_passed",
    "refined_physical_requirements_passed",
    "passed",
)


class InjectedFailure(Exception):
    pass


def _runtime() -> dict:
    return {
        "numerical_settings": {"convergence_check": "endpoints_and_zero"},
        "resolved": {
            "amplitudes": [0.0],
            "quark_eos_id": "CFL4",
        },
        "hadronic_eos": {"baseline": "APR-1"},
        "physical_requirements": {
            "minimum_maximum_mass_msun": "2.08",
            "maximum_maximum_mass_msun": "3.0",
            "radius_1p4_min_km": "9.5",
            "radius_1p4_max_km": "14.5",
        },
    }


def _summary(*matter_types: str) -> pd.DataFrame:
    records = [
        {
            "matter_type": matter_type,
            "baseline_name": "APR-1" if matter_type == "hadronic" else "CFL4",
            "deformation_amplitude": 0.0,
            "maximum_mass_msun": 2.1,
            "radius_1p4_km": 12.0,
            "tidal_deformability_1p4": 400.0,
        }
        for matter_type in matter_types
    ]
    return pd.DataFrame.from_records(records)


@pytest.mark.parametrize(
    "failure_stage",
    (
        "build_eos",
        "solve",
        "stellar_curve_to_frame",
        "summarize_stellar_curve",
        "physical_requirements_status",
    ),
)
def test_refinement_stage_failures_become_rows_and_continue(
    monkeypatch,
    failure_stage,
):
    runtime = _runtime()
    summary = _summary("hadronic")
    events = []
    settings_calls = []
    build_calls = []
    solve_calls = []
    stellar_calls = []
    summary_calls = []
    physical_calls = []
    failed_record_calls = []
    original_failed_record = experiment_runner._failed_convergence_record

    def fail_if_selected(stage, check):
        events.append((check, stage))
        if failure_stage == stage:
            raise InjectedFailure(f"injected {stage} failure")

    def resolve_settings(runtime_argument):
        settings_calls.append(runtime_argument)
        return {
            "eos_grid_points": 100,
            "central_pressure_points": 80,
            "tov_relative_tolerance": 1.0e-6,
            "tov_absolute_tolerance": 1.0e-8,
        }

    def build_eos(runtime_argument, matter_type, amplitude, *, grid_points=None):
        check = CHECKS[len(build_calls)]
        build_calls.append((runtime_argument, matter_type, amplitude, grid_points))
        fail_if_selected("build_eos", check)
        return SimpleNamespace(
            check=check,
            baseline_name=f"generated-{matter_type}",
            catalog_identifier=f"catalog-{matter_type}",
        )

    def solve(runtime_argument, eos, matter_type, **kwargs):
        solve_calls.append((runtime_argument, eos, matter_type, kwargs))
        fail_if_selected("solve", eos.check)
        return SimpleNamespace(check=eos.check), object(), 2.1

    def stellar_curve_to_frame(curve, eos, matter_type, sweep_id, curve_id):
        stellar_calls.append((curve, eos, matter_type, sweep_id, curve_id))
        fail_if_selected("stellar_curve_to_frame", eos.check)
        return SimpleNamespace(check=eos.check)

    def summarize_stellar_curve(frame):
        summary_calls.append(frame)
        fail_if_selected("summarize_stellar_curve", frame.check)
        return {
            "check": frame.check,
            "maximum_mass_msun": 2.1,
            "radius_1p4_km": 12.0,
            "tidal_deformability_1p4": 400.0,
        }

    def physical_requirements_status(runtime_argument, observables):
        physical_calls.append((runtime_argument, observables))
        fail_if_selected("physical_requirements_status", observables["check"])
        return True, "passed"

    def failed_convergence_record(
        runtime_argument,
        matter_type,
        amplitude,
        check,
        *,
        reason=None,
    ):
        failed_record_calls.append(
            (runtime_argument, matter_type, amplitude, check, reason)
        )
        return original_failed_record(
            runtime_argument,
            matter_type,
            amplitude,
            check,
            reason=reason,
        )

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
        stellar_curve_to_frame,
    )
    monkeypatch.setattr(
        experiment_runner,
        "summarize_stellar_curve",
        summarize_stellar_curve,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_physical_requirements_status",
        physical_requirements_status,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_failed_convergence_record",
        failed_convergence_record,
    )

    result = experiment_runner._run_convergence_checks(runtime, summary)

    stage_order = (
        "build_eos",
        "solve",
        "stellar_curve_to_frame",
        "summarize_stellar_curve",
        "physical_requirements_status",
    )
    expected_stage_prefix = stage_order[: stage_order.index(failure_stage) + 1]
    assert events == [
        (check, stage)
        for check in CHECKS
        for stage in expected_stage_prefix
    ]
    assert settings_calls == [runtime]
    assert settings_calls[0] is runtime
    assert [call[1:] for call in build_calls] == [
        ("hadronic", 0.0, 200),
        ("hadronic", 0.0, None),
        ("hadronic", 0.0, None),
    ]
    assert all(call[0] is runtime for call in build_calls)

    expected_reason = f"InjectedFailure: injected {failure_stage} failure"
    assert all(call[0] is runtime for call in failed_record_calls)
    assert [call[1:] for call in failed_record_calls] == [
        ("hadronic", 0.0, check, expected_reason) for check in CHECKS
    ] + [
        ("quark", 0.0, "production_reference_missing", None),
    ]

    assert tuple(result.columns) == experiment_runner.CONVERGENCE_COLUMNS
    assert list(
        zip(
            result["matter_type"],
            result["deformation_amplitude"],
            result["check"],
        )
    ) == [
        ("hadronic", 0.0, f"{check}: {expected_reason}")
        for check in CHECKS
    ] + [
        ("quark", 0.0, "production_reference_missing"),
    ]
    assert result["baseline_name"].tolist() == ["APR-1"] * 3 + ["CFL4"]
    assert result.loc[:2, "refined_physical_requirements_reason"].tolist() == [
        expected_reason
    ] * 3
    assert result.loc[3, "refined_physical_requirements_reason"] == (
        "refinement unavailable"
    )
    for column in DELTA_COLUMNS:
        assert result[column].isna().all()
    for column in PASS_COLUMNS:
        assert result[column].tolist() == [False] * 4

    reached_stage_index = stage_order.index(failure_stage)
    assert len(solve_calls) == (3 if reached_stage_index >= 1 else 0)
    assert len(stellar_calls) == (3 if reached_stage_index >= 2 else 0)
    assert len(summary_calls) == (3 if reached_stage_index >= 3 else 0)
    assert len(physical_calls) == (3 if reached_stage_index >= 4 else 0)


@pytest.mark.parametrize(
    ("failure_mode", "expected_reason", "expected_stage_order"),
    (
        (
            "solve-unpack",
            "RuntimeError: unpack exploded",
            ("build", "solve", "unpack"),
        ),
        (
            "zero-reference-tidal",
            "ZeroDivisionError: float division by zero",
            ("build", "solve", "serialize", "summarize"),
        ),
    ),
)
def test_post_call_and_arithmetic_failures_become_rows_and_continue(
    monkeypatch,
    failure_mode,
    expected_reason,
    expected_stage_order,
):
    runtime = _runtime()
    summary = _summary("hadronic")
    if failure_mode == "zero-reference-tidal":
        summary.loc[:, "tidal_deformability_1p4"] = 0.0
    events = []
    build_count = 0

    class ExplodingSolution:
        def __init__(self, check):
            self.check = check

        def __iter__(self):
            events.append((self.check, "unpack"))
            raise RuntimeError("unpack exploded")

    def resolve_settings(_runtime_argument):
        return {
            "eos_grid_points": 100,
            "central_pressure_points": 80,
            "tov_relative_tolerance": 1.0e-6,
            "tov_absolute_tolerance": 1.0e-8,
        }

    def build_eos(_runtime_argument, matter_type, _amplitude, **_kwargs):
        nonlocal build_count
        check = CHECKS[build_count]
        build_count += 1
        events.append((check, "build"))
        return SimpleNamespace(
            check=check,
            baseline_name=f"generated-{matter_type}",
            catalog_identifier=f"catalog-{matter_type}",
        )

    def solve(_runtime_argument, eos, _matter_type, **_kwargs):
        events.append((eos.check, "solve"))
        if failure_mode == "solve-unpack":
            return ExplodingSolution(eos.check)
        return SimpleNamespace(check=eos.check), object(), 2.1

    def serialize(curve, eos, *_args):
        assert curve.check == eos.check
        events.append((eos.check, "serialize"))
        return SimpleNamespace(check=eos.check)

    def summarize(frame):
        events.append((frame.check, "summarize"))
        return {
            "maximum_mass_msun": 2.1,
            "radius_1p4_km": 12.0,
            "tidal_deformability_1p4": 1.0,
        }

    def forbidden_physical_screen(*_args, **_kwargs):
        pytest.fail("failure path reached the physical screen")

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        resolve_settings,
    )
    monkeypatch.setattr(experiment_runner, "_build_eos", build_eos)
    monkeypatch.setattr(experiment_runner, "_solve", solve)
    monkeypatch.setattr(experiment_runner, "stellar_curve_to_frame", serialize)
    monkeypatch.setattr(experiment_runner, "summarize_stellar_curve", summarize)
    monkeypatch.setattr(
        experiment_runner,
        "_physical_requirements_status",
        forbidden_physical_screen,
    )

    result = experiment_runner._run_convergence_checks(runtime, summary)

    assert events == [
        (check, stage)
        for check in CHECKS
        for stage in expected_stage_order
    ]
    assert result["check"].tolist() == [
        f"{check}: {expected_reason}" for check in CHECKS
    ] + ["production_reference_missing"]
    assert result["refined_physical_requirements_reason"].tolist() == [
        expected_reason
    ] * 3 + ["refinement unavailable"]
    for column in DELTA_COLUMNS:
        assert result[column].isna().all()
    for column in PASS_COLUMNS:
        assert result[column].tolist() == [False] * 4
