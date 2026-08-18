from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.physics import experiment_runner


def _runtime(*, amplitudes=(-0.1, -0.05, 0.2), mode="endpoints_and_zero"):
    return {
        "numerical_settings": {"convergence_check": mode},
        "resolved": {
            "amplitudes": list(amplitudes),
            "quark_eos_id": "CFL4-production",
        },
        "hadronic_eos": {"baseline": "APR-1-production"},
    }


def _unexpected_call(*_args, **_kwargs):
    pytest.fail("disabled or missing-reference convergence called a refinement")


def test_convergence_none_short_circuits_with_exact_empty_schema(monkeypatch):
    class SummaryTrap:
        def itertuples(self, **_kwargs):
            pytest.fail("disabled convergence consumed the production summary")

    runtime = _runtime(mode="none")
    real_dataframe_type = pd.DataFrame
    dataframe_calls = []
    sentinel_columns = ("facade_column",)

    def dataframe_type(*args, **kwargs):
        dataframe_calls.append((args, kwargs))
        return real_dataframe_type(*args, **kwargs)

    monkeypatch.setattr(
        experiment_runner,
        "pd",
        SimpleNamespace(DataFrame=dataframe_type),
    )
    monkeypatch.setattr(
        experiment_runner,
        "CONVERGENCE_COLUMNS",
        sentinel_columns,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        _unexpected_call,
    )
    monkeypatch.setattr(experiment_runner, "_build_eos", _unexpected_call)
    monkeypatch.setattr(experiment_runner, "_solve", _unexpected_call)

    result = experiment_runner._run_convergence_checks(runtime, SummaryTrap())

    assert result.empty
    assert result.shape == (0, 1)
    assert tuple(result.columns) == sentinel_columns
    assert dataframe_calls == [((), {"columns": sentinel_columns})]


def test_missing_references_emit_one_failure_row_per_pair_and_skip_refinements(
    monkeypatch,
):
    runtime = _runtime(amplitudes=(0.0, 0.05, 0.1))
    summary = pd.DataFrame(columns=("matter_type", "deformation_amplitude"))
    settings_calls = []
    failed_calls = []
    original_failed_record = experiment_runner._failed_convergence_record

    def resolve_settings(runtime_argument):
        settings_calls.append(runtime_argument)
        return {
            "eos_grid_points": 100,
            "central_pressure_points": 80,
            "tov_relative_tolerance": 1.0e-6,
            "tov_absolute_tolerance": 1.0e-8,
        }

    def record_failure(*args, **kwargs):
        failed_calls.append((args, kwargs))
        return original_failed_record(*args, **kwargs)

    monkeypatch.setattr(
        experiment_runner,
        "_resolved_numerical_settings",
        resolve_settings,
    )
    monkeypatch.setattr(
        experiment_runner,
        "_failed_convergence_record",
        record_failure,
    )
    for dependency in (
        "_build_eos",
        "_solve",
        "stellar_curve_to_frame",
        "summarize_stellar_curve",
        "_physical_requirements_status",
    ):
        monkeypatch.setattr(experiment_runner, dependency, _unexpected_call)

    result = experiment_runner._run_convergence_checks(runtime, summary)

    expected_pairs = [
        (0.0, "hadronic"),
        (0.0, "quark"),
        (0.1, "hadronic"),
        (0.1, "quark"),
    ]
    assert settings_calls == [runtime]
    assert settings_calls[0] is runtime
    assert all(args[0] is runtime for args, _ in failed_calls)
    assert list(
        result[["deformation_amplitude", "matter_type"]].itertuples(
            index=False, name=None
        )
    ) == expected_pairs
    assert [
        (args[2], args[1], args[3], kwargs)
        for args, kwargs in failed_calls
    ] == [
        (amplitude, matter_type, "production_reference_missing", {})
        for amplitude, matter_type in expected_pairs
    ]
    assert tuple(result.columns) == experiment_runner.CONVERGENCE_COLUMNS
    assert result["baseline_name"].tolist() == [
        "APR-1-production",
        "CFL4-production",
        "APR-1-production",
        "CFL4-production",
    ]
    assert result["check"].tolist() == ["production_reference_missing"] * 4
    for column in (
        "delta_maximum_mass_msun",
        "delta_radius_1p4_km",
        "relative_delta_tidal_deformability_1p4",
    ):
        assert np.isnan(result[column].to_numpy()).all()
    for column in (
        "maximum_mass_passed",
        "radius_1p4_passed",
        "tidal_deformability_1p4_passed",
        "refined_physical_requirements_passed",
        "passed",
    ):
        assert result[column].tolist() == [False] * 4
    assert result["refined_physical_requirements_reason"].tolist() == (
        ["refinement unavailable"] * 4
    )
