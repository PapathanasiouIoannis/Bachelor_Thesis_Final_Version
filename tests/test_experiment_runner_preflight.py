from pathlib import Path

import numpy as np
import pytest

from src.experiment_config import resolve_pair_experiment
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
