from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from src.physics import experiment_runner
from src.physics.runner import convergence


ROOT = Path(__file__).resolve().parents[1]


def test_convergence_facade_preserves_schema_identity_and_exact_signatures():
    assert experiment_runner.CONVERGENCE_COLUMNS is convergence.CONVERGENCE_COLUMNS
    assert str(inspect.signature(experiment_runner._run_convergence_checks)) == (
        "(runtime: 'dict[str, Any]', summary: 'pd.DataFrame') -> 'pd.DataFrame'"
    )
    assert str(inspect.signature(experiment_runner._failed_convergence_record)) == (
        "(runtime: 'dict[str, Any]', matter_type: 'str', amplitude: 'float', "
        "check: 'str', *, reason: 'str | None' = None) -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(experiment_runner._physical_requirements_status)) == (
        "(runtime: 'dict[str, Any]', observables: 'dict[str, Any]') -> "
        "'tuple[bool, str]'"
    )


def test_failed_convergence_record_uses_live_facade_nan_value(monkeypatch):
    nan_sentinel = object()
    monkeypatch.setattr(
        experiment_runner,
        "np",
        SimpleNamespace(nan=nan_sentinel),
    )
    runtime = {
        "hadronic_eos": {"baseline": "APR-1"},
        "resolved": {"quark_eos_id": "CFL4"},
    }

    result = experiment_runner._failed_convergence_record(
        runtime,
        "hadronic",
        0.0,
        "injected-check",
    )

    assert result["delta_maximum_mass_msun"] is nan_sentinel
    assert result["delta_radius_1p4_km"] is nan_sentinel
    assert result["relative_delta_tidal_deformability_1p4"] is nan_sentinel


def test_convergence_facade_uses_reloaded_leaf_and_current_dependencies():
    script = textwrap.dedent(
        """
        import importlib

        from src.physics import experiment_runner
        from src.physics.runner import convergence

        importlib.reload(convergence)
        runtime = object()
        summary = object()
        observables = object()
        run_result = object()
        failed_result = object()
        physical_result = object()
        calls = []

        def run(runtime_argument, summary_argument, *, dependencies):
            assert runtime_argument is runtime
            assert summary_argument is summary
            assert type(dependencies) is convergence.ConvergenceDependencies
            assert dependencies.dataframe_type is experiment_runner.pd.DataFrame
            assert (
                dependencies.convergence_columns
                is experiment_runner.CONVERGENCE_COLUMNS
            )
            assert (
                dependencies.resolved_numerical_settings
                is experiment_runner._resolved_numerical_settings
            )
            assert (
                dependencies.failed_convergence_record
                is experiment_runner._failed_convergence_record
            )
            assert dependencies.build_eos is experiment_runner._build_eos
            assert dependencies.solve is experiment_runner._solve
            assert (
                dependencies.stellar_curve_to_frame
                is experiment_runner.stellar_curve_to_frame
            )
            assert (
                dependencies.summarize_stellar_curve
                is experiment_runner.summarize_stellar_curve
            )
            assert (
                dependencies.physical_requirements_status
                is experiment_runner._physical_requirements_status
            )
            calls.append("run")
            return run_result

        def failed(
            runtime_argument,
            matter_type,
            amplitude,
            check,
            *,
            reason,
            dependencies,
        ):
            assert runtime_argument is runtime
            assert (matter_type, amplitude, check, reason) == (
                "hadronic",
                0.25,
                "double_eos_grid",
                "injected reason",
            )
            assert (
                type(dependencies)
                is convergence.FailedConvergenceRecordDependencies
            )
            assert dependencies.nan_value is experiment_runner.np.nan
            calls.append("failed")
            return failed_result

        def physical(runtime_argument, observables_argument):
            assert runtime_argument is runtime
            assert observables_argument is observables
            calls.append("physical")
            return physical_result

        convergence.run_convergence_checks = run
        convergence.failed_convergence_record = failed
        convergence.physical_requirements_status = physical

        assert (
            experiment_runner._run_convergence_checks(runtime, summary)
            is run_result
        )
        assert (
            experiment_runner._failed_convergence_record(
                runtime,
                "hadronic",
                0.25,
                "double_eos_grid",
                reason="injected reason",
            )
            is failed_result
        )
        assert (
            experiment_runner._physical_requirements_status(runtime, observables)
            is physical_result
        )
        assert calls == ["run", "failed", "physical"]
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
