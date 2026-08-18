"""Pure manifest construction for controlled pair experiments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def running_manifest(
    *,
    experiment_name: str,
    workflow: str,
    mode: str,
    configuration_hash: str,
    source_tree_sha256: str,
    git_revision: str | None,
    created_utc: str,
    environment: Mapping[str, Any],
    preflight: Mapping[str, Any],
    execution: Mapping[str, Any],
    resolved_numerical_settings: Mapping[str, int | float],
    runtime_overrides: Mapping[str, Any],
    permitted_scientific_interpretation: str,
) -> dict[str, Any]:
    """Build the durable running checkpoint in its stable schema order."""

    return {
        "schema_version": 1,
        "component": "controlled_eos_pair_sensitivity",
        "status": "running",
        "experiment_name": experiment_name,
        "workflow": workflow,
        "mode": mode,
        "configuration_hash": configuration_hash,
        "source_tree_sha256": source_tree_sha256,
        "git_revision": git_revision,
        "created_utc": created_utc,
        "environment": environment,
        "preflight": preflight,
        "execution": execution,
        "resolved_numerical_settings": resolved_numerical_settings,
        "runtime_overrides": runtime_overrides,
        "classification_enabled": False,
        "permitted_scientific_interpretation": (permitted_scientific_interpretation),
    }


def terminal_status(
    *,
    all_pairs_accepted: bool,
    convergence_performed: bool,
    convergence_passed: bool | None,
) -> str:
    """Select the terminal status, with pair rejection taking precedence."""

    return (
        "completed"
        if all_pairs_accepted
        and (not convergence_performed or convergence_passed is True)
        else "completed_with_rejections"
        if not all_pairs_accepted
        else "failed_convergence"
    )


def terminal_manifest(
    base_manifest: Mapping[str, Any],
    *,
    status: str,
    completed_utc: str,
    accepted_pairs: int,
    accepted_curves: int,
    rejected_pairs: int,
    convergence_performed: bool,
    convergence_passed: bool | None,
    causal_domains: list[dict[str, Any]],
    plot_files: list[str],
    artifacts: Mapping[str, str],
) -> dict[str, Any]:
    """Extend a running checkpoint with terminal run evidence."""

    return {
        **base_manifest,
        "status": status,
        "completed_utc": completed_utc,
        "accepted_pairs": accepted_pairs,
        "accepted_curves": accepted_curves,
        "rejected_pairs": rejected_pairs,
        "convergence_performed": convergence_performed,
        "convergence_passed": convergence_passed,
        "causal_domains": causal_domains,
        "plot_files": plot_files,
        "artifacts": artifacts,
    }


def failed_manifest(
    current: Mapping[str, Any],
    *,
    completed_utc: str,
    error_type: str,
    error_message: str,
    error_traceback: str,
) -> dict[str, Any]:
    """Extend the persisted running checkpoint with failure evidence."""

    return {
        **current,
        "status": "failed",
        "completed_utc": completed_utc,
        "error": {
            "type": error_type,
            "message": error_message,
            "traceback": error_traceback,
        },
    }


__all__ = [
    "failed_manifest",
    "running_manifest",
    "terminal_manifest",
    "terminal_status",
]
