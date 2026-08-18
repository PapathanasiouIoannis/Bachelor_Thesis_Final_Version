from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.physics import experiment_runner


RUN_LOG = Path("run") / "logs" / "pipeline.log"
WORKER_LOG = RUN_LOG.with_name("pipeline.worker-test.log")
CONFIGURATION_HASH = "0123456789abcdef"
AMPLITUDE = 0.125
SWEEP_INDEX = 7


class InjectedFailure(Exception):
    pass


class UnexpectedFailure(Exception):
    pass


@dataclass(frozen=True)
class FailureCase:
    failures: frozenset[tuple[str, str]]
    primary_matter: str
    stage: str
    retained_matter: tuple[str, ...]
    events: tuple[str, ...]


def _event(stage: str, matter_type: str) -> str:
    return f"{stage}:{matter_type}"


def _failure_message(stage: str, matter_type: str) -> str:
    return f"injected {stage} failure for {matter_type}"


def _install_worker_harness(monkeypatch, failures=frozenset()):
    runtime = {"runtime_marker": object()}
    events = []
    worker_path_calls = []
    configure_calls = []
    sweep_point_calls = []
    build_calls = []
    eos_serialization_calls = []
    solve_calls = []
    stellar_serialization_calls = []
    eoses = {
        "hadronic": SimpleNamespace(
            matter_type="hadronic",
            baseline_name="APR-1",
        ),
        "quark": SimpleNamespace(
            matter_type="quark",
            baseline_name="CFL4",
        ),
    }
    eos_frames = {
        matter_type: pd.DataFrame(
            {
                "matter_type": [matter_type],
                "pair_accepted": [True],
            }
        )
        for matter_type in ("hadronic", "quark")
    }
    curves = {"hadronic": object(), "quark": object()}
    stellar_frames = {"hadronic": object(), "quark": object()}

    def record(stage, matter_type):
        events.append(_event(stage, matter_type))
        if (stage, matter_type) in failures:
            raise InjectedFailure(_failure_message(stage, matter_type))

    def worker_log_path(path):
        worker_path_calls.append(path)
        return WORKER_LOG

    def configure(path):
        configure_calls.append(path)
        events.append("configure")

    def close():
        events.append("close")

    def sweep_point(index, amplitude):
        sweep_point_calls.append((index, amplitude))
        events.append("sweep-point")
        return SimpleNamespace(sweep_id=f"A{index:05d}")

    def build(runtime_argument, matter_type, amplitude):
        build_calls.append((runtime_argument, matter_type, amplitude))
        record("eos_generation", matter_type)
        return eoses[matter_type]

    def serialize_eos(eos, matter_type, sweep_id):
        eos_serialization_calls.append((eos, matter_type, sweep_id))
        record("eos_serialization", matter_type)
        return eos_frames[matter_type]

    def validate_eos(frame):
        matter_type = str(frame["matter_type"].iloc[0])
        record("eos_validation", matter_type)

    def solve(runtime_argument, eos, matter_type):
        solve_calls.append((runtime_argument, eos, matter_type))
        record("stellar_sequence", matter_type)
        return curves[matter_type], object(), 2.1

    def serialize_stellar(curve, eos, matter_type, sweep_id, curve_id):
        stellar_serialization_calls.append(
            (curve, eos, matter_type, sweep_id, curve_id)
        )
        record("stellar_serialization", matter_type)
        return stellar_frames[matter_type]

    monkeypatch.setattr(experiment_runner, "_worker_log_path", worker_log_path)
    monkeypatch.setattr(experiment_runner, "configure_run_log", configure)
    monkeypatch.setattr(experiment_runner, "close_run_log", close)
    monkeypatch.setattr(experiment_runner, "SweepPoint", sweep_point)
    monkeypatch.setattr(experiment_runner, "_build_eos", build)
    monkeypatch.setattr(experiment_runner, "serialize_eos_table", serialize_eos)
    monkeypatch.setattr(experiment_runner, "validate_eos_frame", validate_eos)
    monkeypatch.setattr(experiment_runner, "_solve", solve)
    monkeypatch.setattr(
        experiment_runner,
        "stellar_curve_to_frame",
        serialize_stellar,
    )

    return SimpleNamespace(
        runtime=runtime,
        events=events,
        worker_path_calls=worker_path_calls,
        configure_calls=configure_calls,
        sweep_point_calls=sweep_point_calls,
        build_calls=build_calls,
        eos_serialization_calls=eos_serialization_calls,
        solve_calls=solve_calls,
        stellar_serialization_calls=stellar_serialization_calls,
        eoses=eoses,
        eos_frames=eos_frames,
        curves=curves,
        stellar_frames=stellar_frames,
    )


FULL_EOS_PREFIX = (
    "configure",
    "sweep-point",
    _event("eos_generation", "hadronic"),
    _event("eos_generation", "quark"),
    _event("eos_serialization", "hadronic"),
    _event("eos_validation", "hadronic"),
    _event("eos_serialization", "quark"),
    _event("eos_validation", "quark"),
)
FULL_SUCCESS_TRACE = FULL_EOS_PREFIX + (
    _event("stellar_sequence", "hadronic"),
    _event("stellar_serialization", "hadronic"),
    _event("stellar_sequence", "quark"),
    _event("stellar_serialization", "quark"),
    "close",
)


def test_pair_generation_error_preserves_domain_context():
    error = experiment_runner.PairGenerationError(
        "quark",
        "stellar_sequence",
        "synthetic reason",
    )

    assert isinstance(error, RuntimeError)
    assert error.matter_type == "quark"
    assert error.stage == "stellar_sequence"
    assert error.args == ("synthetic reason",)
    assert str(error) == "synthetic reason"


def test_generate_pair_success_preserves_order_ids_objects_and_log_lifecycle(
    monkeypatch,
):
    harness = _install_worker_harness(monkeypatch)

    result = experiment_runner._generate_pair(
        harness.runtime,
        SWEEP_INDEX,
        AMPLITUDE,
        CONFIGURATION_HASH,
        str(RUN_LOG),
    )

    assert tuple(result) == (
        "accepted",
        "eos_frames",
        "stellar_frames",
        "rejection",
    )
    assert result["accepted"] is True
    assert result["rejection"] is None
    assert len(result["eos_frames"]) == len(result["stellar_frames"]) == 2
    assert result["eos_frames"][0] is harness.eos_frames["hadronic"]
    assert result["eos_frames"][1] is harness.eos_frames["quark"]
    assert result["stellar_frames"][0] is harness.stellar_frames["hadronic"]
    assert result["stellar_frames"][1] is harness.stellar_frames["quark"]
    assert harness.events == list(FULL_SUCCESS_TRACE)
    assert harness.worker_path_calls == [RUN_LOG]
    assert harness.configure_calls == [WORKER_LOG]
    assert harness.sweep_point_calls == [(SWEEP_INDEX, AMPLITUDE)]
    assert all(call[0] is harness.runtime for call in harness.build_calls)
    assert harness.build_calls == [
        (harness.runtime, "hadronic", AMPLITUDE),
        (harness.runtime, "quark", AMPLITUDE),
    ]
    assert harness.eos_serialization_calls == [
        (harness.eoses["hadronic"], "hadronic", "A00007"),
        (harness.eoses["quark"], "quark", "A00007"),
    ]
    assert harness.eos_serialization_calls[0][0] is harness.eoses["hadronic"]
    assert harness.eos_serialization_calls[1][0] is harness.eoses["quark"]
    assert all(call[0] is harness.runtime for call in harness.solve_calls)
    assert harness.solve_calls == [
        (harness.runtime, harness.eoses["hadronic"], "hadronic"),
        (harness.runtime, harness.eoses["quark"], "quark"),
    ]
    assert harness.solve_calls[0][1] is harness.eoses["hadronic"]
    assert harness.solve_calls[1][1] is harness.eoses["quark"]
    assert harness.stellar_serialization_calls == [
        (
            harness.curves["hadronic"],
            harness.eoses["hadronic"],
            "hadronic",
            "A00007",
            "hadronic_APR-1_A00007_0123456789",
        ),
        (
            harness.curves["quark"],
            harness.eoses["quark"],
            "quark",
            "A00007",
            "quark_CFL4_A00007_0123456789",
        ),
    ]
    assert harness.stellar_serialization_calls[0][1] is harness.eoses["hadronic"]
    assert harness.stellar_serialization_calls[1][1] is harness.eoses["quark"]
    for frame in result["eos_frames"]:
        assert frame["pair_accepted"].tolist() == [True]
        assert frame["eos_validation_passed"].tolist() == [True]
        assert frame["eos_validation_reason"].tolist() == ["passed"]


def test_generate_pair_propagates_unexpected_post_solve_errors(monkeypatch):
    class EosWithBrokenBaselineName:
        matter_type = "hadronic"

        @property
        def baseline_name(self):
            raise UnexpectedFailure("unexpected baseline-name failure")

    harness = _install_worker_harness(monkeypatch)
    harness.eoses["hadronic"] = EosWithBrokenBaselineName()

    with pytest.raises(
        UnexpectedFailure,
        match="unexpected baseline-name failure",
    ):
        experiment_runner._generate_pair(
            harness.runtime,
            SWEEP_INDEX,
            AMPLITUDE,
            CONFIGURATION_HASH,
            str(RUN_LOG),
        )

    domain_events = [event for event in harness.events if event != "close"]
    assert domain_events == [
        *FULL_EOS_PREFIX,
        _event("stellar_sequence", "hadronic"),
    ]
    assert harness.stellar_serialization_calls == []


FAILURE_CASES = (
    FailureCase(
        frozenset({("eos_generation", "hadronic")}),
        "hadronic",
        "eos_generation",
        (),
        (
            "configure",
            "sweep-point",
            _event("eos_generation", "hadronic"),
        ),
    ),
    FailureCase(
        frozenset({("eos_generation", "quark")}),
        "quark",
        "eos_generation",
        (),
        (
            "configure",
            "sweep-point",
            _event("eos_generation", "hadronic"),
            _event("eos_generation", "quark"),
        ),
    ),
    FailureCase(
        frozenset({("eos_serialization", "hadronic")}),
        "hadronic",
        "eos_serialization",
        (),
        (
            "configure",
            "sweep-point",
            _event("eos_generation", "hadronic"),
            _event("eos_generation", "quark"),
            _event("eos_serialization", "hadronic"),
        ),
    ),
    FailureCase(
        frozenset({("eos_serialization", "quark")}),
        "quark",
        "eos_serialization",
        ("hadronic",),
        FULL_EOS_PREFIX[:7],
    ),
    FailureCase(
        frozenset(
            {
                ("eos_validation", "hadronic"),
                ("eos_validation", "quark"),
            }
        ),
        "hadronic",
        "eos_validation",
        ("hadronic", "quark"),
        FULL_EOS_PREFIX,
    ),
    FailureCase(
        frozenset({("eos_validation", "quark")}),
        "quark",
        "eos_validation",
        ("hadronic", "quark"),
        FULL_EOS_PREFIX,
    ),
    FailureCase(
        frozenset(
            {
                ("eos_validation", "hadronic"),
                ("eos_serialization", "quark"),
            }
        ),
        "quark",
        "eos_serialization",
        ("hadronic",),
        FULL_EOS_PREFIX[:-1],
    ),
    FailureCase(
        frozenset({("stellar_sequence", "hadronic")}),
        "hadronic",
        "stellar_sequence",
        ("hadronic", "quark"),
        FULL_EOS_PREFIX + (_event("stellar_sequence", "hadronic"),),
    ),
    FailureCase(
        frozenset({("stellar_sequence", "quark")}),
        "quark",
        "stellar_sequence",
        ("hadronic", "quark"),
        FULL_SUCCESS_TRACE[:-2],
    ),
    FailureCase(
        frozenset({("stellar_serialization", "hadronic")}),
        "hadronic",
        "stellar_serialization",
        ("hadronic", "quark"),
        FULL_EOS_PREFIX
        + (
            _event("stellar_sequence", "hadronic"),
            _event("stellar_serialization", "hadronic"),
        ),
    ),
    FailureCase(
        frozenset({("stellar_serialization", "quark")}),
        "quark",
        "stellar_serialization",
        ("hadronic", "quark"),
        FULL_SUCCESS_TRACE[:-1],
    ),
)


@pytest.mark.parametrize(
    "case",
    FAILURE_CASES,
    ids=lambda case: "+".join(
        f"{stage}-{matter_type}"
        for stage, matter_type in sorted(case.failures)
    ),
)
def test_generate_pair_handled_failures_preserve_stage_protocol(
    monkeypatch,
    case,
):
    harness = _install_worker_harness(monkeypatch, case.failures)

    result = experiment_runner._generate_pair(
        harness.runtime,
        SWEEP_INDEX,
        AMPLITUDE,
        CONFIGURATION_HASH,
        str(RUN_LOG),
    )

    primary_message = _failure_message(case.stage, case.primary_matter)
    assert tuple(result) == (
        "accepted",
        "eos_frames",
        "stellar_frames",
        "rejection",
    )
    assert result["accepted"] is False
    assert result["stellar_frames"] == []
    assert result["rejection"] == {
        "sweep_id": "A00007",
        "deformation_amplitude": AMPLITUDE,
        "matter_type": case.primary_matter,
        "stage": case.stage,
        "exception_type": "InjectedFailure",
        "reason": primary_message,
    }
    assert harness.events == [*case.events, "close"]
    assert harness.worker_path_calls == [RUN_LOG]
    assert harness.configure_calls == [WORKER_LOG]
    assert harness.sweep_point_calls == [(SWEEP_INDEX, AMPLITUDE)]
    assert len(result["eos_frames"]) == len(case.retained_matter)
    for frame, matter_type in zip(result["eos_frames"], case.retained_matter):
        assert frame is harness.eos_frames[matter_type]
        assert frame["pair_accepted"].tolist() == [False]
        validation_failed = ("eos_validation", matter_type) in case.failures
        assert frame["eos_validation_passed"].tolist() == [not validation_failed]
        expected_reason = (
            f"InjectedFailure: {_failure_message('eos_validation', matter_type)}"
            if validation_failed
            else "passed"
        )
        assert frame["eos_validation_reason"].tolist() == [expected_reason]
