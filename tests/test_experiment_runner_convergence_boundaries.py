from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.physics import experiment_runner


class InjectedFailure(Exception):
    pass


class ResolverFailure(Exception):
    pass


class FatalConvergenceSignal(BaseException):
    pass


class FailureRecordFailure(Exception):
    pass


class FinalFrameFailure(Exception):
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


def _valid_numerical_settings() -> dict[str, int | float]:
    return {
        "eos_grid_points": 100,
        "central_pressure_points": 80,
        "tov_relative_tolerance": 1.0e-6,
        "tov_absolute_tolerance": 1.0e-8,
    }


def test_pre_loop_numerical_settings_failure_propagates(monkeypatch):
    runtime = _runtime()
    summary = _summary("hadronic", "quark")
    resolver_calls = []
    downstream_calls = []

    def fail_settings(runtime_argument):
        resolver_calls.append(runtime_argument)
        raise ResolverFailure("numerical settings unavailable")

    def forbidden(name):
        def call(*_args, **_kwargs):
            downstream_calls.append(name)
            raise AssertionError(f"unexpected downstream call: {name}")

        return call

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        fail_settings,
    )
    monkeypatch.setattr(experiment_runner, "_build_eos", forbidden("build"))
    monkeypatch.setattr(
        experiment_runner,
        "_failed_convergence_record",
        forbidden("failed-record"),
    )

    with pytest.raises(ResolverFailure) as raised:
        experiment_runner._run_convergence_checks(runtime, summary)

    assert str(raised.value) == "numerical settings unavailable"
    assert resolver_calls == [runtime]
    assert resolver_calls[0] is runtime
    assert downstream_calls == []


def test_pre_loop_numerical_conversion_failure_propagates(monkeypatch):
    runtime = _runtime()
    summary = _summary("hadronic", "quark")
    downstream_calls = []

    def invalid_settings(_runtime_argument):
        return {**_valid_numerical_settings(), "eos_grid_points": "not-an-integer"}

    def forbidden(name):
        def call(*_args, **_kwargs):
            downstream_calls.append(name)
            raise AssertionError(f"unexpected downstream call: {name}")

        return call

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        invalid_settings,
    )
    monkeypatch.setattr(experiment_runner, "_build_eos", forbidden("build"))
    monkeypatch.setattr(
        experiment_runner,
        "_failed_convergence_record",
        forbidden("failed-record"),
    )

    with pytest.raises(ValueError, match="invalid literal for int"):
        experiment_runner._run_convergence_checks(runtime, summary)

    assert downstream_calls == []


def test_base_exception_from_refinement_propagates_without_failure_row(monkeypatch):
    runtime = _runtime()
    summary = _summary("hadronic", "quark")
    build_calls = []
    failure_calls = []
    fatal_error = FatalConvergenceSignal("stop convergence now")

    def fatal_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        raise fatal_error

    def record_failure(*args, **kwargs):
        failure_calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        lambda _runtime_argument: _valid_numerical_settings(),
    )
    monkeypatch.setattr(experiment_runner, "_build_eos", fatal_build)
    monkeypatch.setattr(
        experiment_runner,
        "_failed_convergence_record",
        record_failure,
    )

    with pytest.raises(FatalConvergenceSignal) as raised:
        experiment_runner._run_convergence_checks(runtime, summary)

    assert raised.value is fatal_error
    assert len(build_calls) == 1
    assert build_calls[0][0][0] is runtime
    assert failure_calls == []


def test_failure_record_error_propagates_from_caught_refinement(monkeypatch):
    runtime = _runtime()
    summary = _summary("hadronic", "quark")
    build_calls = []
    failure_calls = []
    record_error = FailureRecordFailure("could not record convergence failure")

    def failing_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        raise InjectedFailure("build failed first")

    def failing_record(*args, **kwargs):
        failure_calls.append((args, kwargs))
        raise record_error

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        lambda _runtime_argument: _valid_numerical_settings(),
    )
    monkeypatch.setattr(experiment_runner, "_build_eos", failing_build)
    monkeypatch.setattr(
        experiment_runner,
        "_failed_convergence_record",
        failing_record,
    )

    with pytest.raises(FailureRecordFailure) as raised:
        experiment_runner._run_convergence_checks(runtime, summary)

    assert raised.value is record_error
    assert len(build_calls) == 1
    assert build_calls[0][0][0] is runtime
    assert len(failure_calls) == 1
    assert failure_calls[0][0][0] is runtime
    assert failure_calls[0][0][1:] == (
        "hadronic",
        0.0,
        "double_eos_grid",
    )
    assert failure_calls[0][1] == {
        "reason": "InjectedFailure: build failed first"
    }


def test_final_frame_error_uses_live_facade_schema_and_propagates(monkeypatch):
    runtime = _runtime()
    summary = pd.DataFrame(columns=("matter_type", "deformation_amplitude"))
    frame_calls = []
    sentinel_columns = ("live_facade_column",)
    frame_error = FinalFrameFailure("final frame construction failed")

    class FailingDataFrameType:
        @classmethod
        def from_records(cls, records, *, columns):
            frame_calls.append((records, columns))
            raise frame_error

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        lambda _runtime_argument: _valid_numerical_settings(),
    )
    monkeypatch.setattr(
        experiment_runner,
        "pd",
        SimpleNamespace(DataFrame=FailingDataFrameType),
    )
    monkeypatch.setattr(
        experiment_runner,
        "CONVERGENCE_COLUMNS",
        sentinel_columns,
    )

    with pytest.raises(FinalFrameFailure) as raised:
        experiment_runner._run_convergence_checks(runtime, summary)

    assert raised.value is frame_error
    assert len(frame_calls) == 1
    records, columns = frame_calls[0]
    assert columns is sentinel_columns
    assert [record["matter_type"] for record in records] == [
        "hadronic",
        "quark",
    ]
    assert [record["check"] for record in records] == [
        "production_reference_missing",
        "production_reference_missing",
    ]


@pytest.mark.parametrize(
    ("maximum_mass", "radius_1p4", "expected"),
    (
        (2.08, 9.5, (True, "passed")),
        (3.0, 14.5, (True, "passed")),
        (
            2.079,
            12.0,
            (False, "maximum mass outside configured interval"),
        ),
        (
            2.1,
            14.501,
            (False, "radius at 1.4 solar masses outside configured interval"),
        ),
        (
            3.001,
            9.499,
            (
                False,
                "maximum mass outside configured interval; "
                "radius at 1.4 solar masses outside configured interval",
            ),
        ),
    ),
    ids=("lower-bound", "upper-bound", "mass", "radius", "both"),
)
def test_physical_requirements_status_preserves_inclusive_bounds_and_reasons(
    maximum_mass,
    radius_1p4,
    expected,
):
    runtime = _runtime()
    observables = {
        "maximum_mass_msun": str(maximum_mass),
        "radius_1p4_km": str(radius_1p4),
    }

    assert (
        experiment_runner._physical_requirements_status(runtime, observables)
        == expected
    )
