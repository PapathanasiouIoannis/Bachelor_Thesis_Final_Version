from __future__ import annotations

import inspect
from types import MappingProxyType

from src.physics import experiment_runner
from src.physics.runner import manifest


BASE_KEYS = (
    "schema_version",
    "component",
    "status",
    "experiment_name",
    "workflow",
    "mode",
    "configuration_hash",
    "source_tree_sha256",
    "git_revision",
    "created_utc",
    "environment",
    "preflight",
    "execution",
    "resolved_numerical_settings",
    "runtime_overrides",
    "classification_enabled",
    "permitted_scientific_interpretation",
)
TERMINAL_KEYS = BASE_KEYS + (
    "completed_utc",
    "accepted_pairs",
    "accepted_curves",
    "rejected_pairs",
    "convergence_performed",
    "convergence_passed",
    "causal_domains",
    "plot_files",
    "artifacts",
)
FAILED_KEYS = BASE_KEYS + ("completed_utc", "error")


def test_manifest_builders_preserve_exact_schema_order_and_are_pure():
    environment = MappingProxyType({"python": "3.13"})
    preflight = MappingProxyType({"expected_curves": 6})
    execution = MappingProxyType({"parallel_jobs": 2})
    numerical = MappingProxyType({"eos_grid_points": 800})
    overrides = MappingProxyType({"execution.parallel_jobs": {"effective": 2}})
    running = manifest.running_manifest(
        experiment_name="manifest_probe",
        workflow="pair_sensitivity",
        mode="exploration",
        configuration_hash="ab" * 32,
        source_tree_sha256="source-hash",
        git_revision=None,
        created_utc="created",
        environment=environment,
        preflight=preflight,
        execution=execution,
        resolved_numerical_settings=numerical,
        runtime_overrides=overrides,
        permitted_scientific_interpretation="sensitivity only",
    )

    assert tuple(running) == BASE_KEYS
    assert running == {
        "schema_version": 1,
        "component": "controlled_eos_pair_sensitivity",
        "status": "running",
        "experiment_name": "manifest_probe",
        "workflow": "pair_sensitivity",
        "mode": "exploration",
        "configuration_hash": "ab" * 32,
        "source_tree_sha256": "source-hash",
        "git_revision": None,
        "created_utc": "created",
        "environment": environment,
        "preflight": preflight,
        "execution": execution,
        "resolved_numerical_settings": numerical,
        "runtime_overrides": overrides,
        "classification_enabled": False,
        "permitted_scientific_interpretation": "sensitivity only",
    }
    running_snapshot = running.copy()

    terminal = manifest.terminal_manifest(
        MappingProxyType(running),
        status="completed",
        completed_utc="completed",
        accepted_pairs=3,
        accepted_curves=6,
        rejected_pairs=0,
        convergence_performed=True,
        convergence_passed=True,
        causal_domains=[{"matter_type": "hadronic"}],
        plot_files=["mass_radius.png"],
        artifacts=MappingProxyType({"report.md": "digest"}),
    )

    assert tuple(terminal) == TERMINAL_KEYS
    assert terminal == {
        **running,
        "status": "completed",
        "completed_utc": "completed",
        "accepted_pairs": 3,
        "accepted_curves": 6,
        "rejected_pairs": 0,
        "convergence_performed": True,
        "convergence_passed": True,
        "causal_domains": [{"matter_type": "hadronic"}],
        "plot_files": ["mass_radius.png"],
        "artifacts": MappingProxyType({"report.md": "digest"}),
    }
    assert running == running_snapshot

    failed = manifest.failed_manifest(
        MappingProxyType(running),
        completed_utc="failed",
        error_type="RuntimeError",
        error_message="injected failure",
        error_traceback="traceback text",
    )

    assert tuple(failed) == FAILED_KEYS
    assert tuple(failed["error"]) == ("type", "message", "traceback")
    assert failed == {
        **running,
        "status": "failed",
        "completed_utc": "failed",
        "error": {
            "type": "RuntimeError",
            "message": "injected failure",
            "traceback": "traceback text",
        },
    }
    assert running == running_snapshot


def test_terminal_status_supported_matrix():
    cases = (
        (True, False, None, "completed"),
        (True, True, True, "completed"),
        (True, True, False, "failed_convergence"),
        (False, False, None, "completed_with_rejections"),
        (False, True, True, "completed_with_rejections"),
        (False, True, False, "completed_with_rejections"),
    )

    for accepted, performed, passed, expected in cases:
        assert (
            manifest.terminal_status(
                all_pairs_accepted=accepted,
                convergence_performed=performed,
                convergence_passed=passed,
            )
            == expected
        )


def test_run_pair_experiment_preserves_public_signature():
    assert str(inspect.signature(experiment_runner.run_pair_experiment)) == (
        "(configuration: 'ResolvedExperiment | str | Path', *, "
        "parallel_jobs: 'int | None' = None, runs_root: 'Path | None' = None) "
        "-> 'RunLayout'"
    )
