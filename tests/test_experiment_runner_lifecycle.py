from __future__ import annotations

import json

import pytest

from tests.experiment_runner_lifecycle_support import LifecycleHarness


BASE_MANIFEST_KEYS = (
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
TERMINAL_MANIFEST_KEYS = BASE_MANIFEST_KEYS + (
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
FAILED_MANIFEST_KEYS = BASE_MANIFEST_KEYS + ("completed_utc", "error")


class InjectedFailure(RuntimeError):
    pass


class InjectedAbort(BaseException):
    pass


def _event_names(events):
    return [event[0] if isinstance(event, tuple) else event for event in events]


def _written_payloads(harness):
    assert all(path == harness.layout.manifest for path, _ in harness.writes)
    return [payload for _, payload in harness.writes]


def test_running_and_completed_manifest_checkpoints_preserve_schema(
    tmp_path, monkeypatch
):
    harness = LifecycleHarness(
        monkeypatch,
        tmp_path,
        convergence_check="endpoints_and_zero",
        convergence_values=(True, True),
    )

    result = harness.run()

    assert result is harness.layout
    checkpoint, terminal = _written_payloads(harness)
    assert list(checkpoint) == list(BASE_MANIFEST_KEYS)
    assert list(terminal) == list(TERMINAL_MANIFEST_KEYS)
    assert [checkpoint["status"], terminal["status"]] == ["running", "completed"]
    for key in BASE_MANIFEST_KEYS:
        if key != "status":
            assert terminal[key] == checkpoint[key]
    assert checkpoint == {
        "schema_version": 1,
        "component": "controlled_eos_pair_sensitivity",
        "status": "running",
        "experiment_name": "lifecycle_probe",
        "workflow": "pair_sensitivity",
        "mode": "exploration",
        "configuration_hash": "ab" * 32,
        "source_tree_sha256": "source-hash",
        "git_revision": "git-revision",
        "created_utc": "time-1",
        "environment": {"environment": "metadata"},
        "preflight": harness.preflight_report,
        "execution": {"parallel_jobs": 1, "amplitudes_per_batch": 2},
        "resolved_numerical_settings": {"eos_grid_points": 8},
        "runtime_overrides": {},
        "classification_enabled": False,
        "permitted_scientific_interpretation": "lifecycle interpretation",
    }
    assert terminal["completed_utc"] == "time-2"
    assert terminal["accepted_pairs"] == 1
    assert terminal["accepted_curves"] == 1
    assert terminal["rejected_pairs"] == 0
    assert terminal["convergence_performed"] is True
    assert terminal["convergence_passed"] is True
    assert terminal["causal_domains"] == [{"domain": "core"}]
    assert terminal["plot_files"] == ["lifecycle.png"]
    assert terminal["artifacts"] == {"report.md": "artifact-hash"}
    assert "error" not in terminal
    assert harness.report_statuses == ["completed"]
    assert harness.merge_calls == [harness.layout.logs / "pipeline.log"]
    assert harness.merge_exceptions == [None]
    assert _event_names(harness.events)[:13] == [
        "validate",
        "numerical",
        "layout",
        "configure-log",
        "render",
        "clock",
        "source-hash",
        "git-revision",
        "environment",
        "preflight-report",
        "write",
        "log",
        "log",
    ]
    assert _event_names(harness.events)[-8:] == [
        "plots",
        "report",
        "causal-domains",
        "artifact-hashes",
        "clock",
        "causal-json",
        "write",
        "log",
    ]
    assert json.loads(harness.layout.manifest.read_text(encoding="utf-8")) == terminal


def test_rejection_precedes_failed_convergence_and_terminal_manifest_is_preserved(
    tmp_path, monkeypatch
):
    harness = LifecycleHarness(
        monkeypatch,
        tmp_path,
        rejected=True,
        convergence_check="endpoints_and_zero",
        convergence_values=(False,),
    )

    with pytest.raises(RuntimeError, match="completed_with_rejections") as raised:
        harness.run()

    checkpoint, terminal = _written_payloads(harness)
    assert [checkpoint["status"], terminal["status"]] == [
        "running",
        "completed_with_rejections",
    ]
    assert list(terminal) == list(TERMINAL_MANIFEST_KEYS)
    assert terminal["accepted_pairs"] == 0
    assert terminal["accepted_curves"] == 0
    assert terminal["rejected_pairs"] == 1
    assert terminal["convergence_performed"] is True
    assert terminal["convergence_passed"] is False
    assert "error" not in terminal
    assert harness.report_statuses == ["completed_with_rejections"]
    canonical_log = harness.layout.logs / "pipeline.log"
    assert harness.merge_calls == [canonical_log, canonical_log]
    assert harness.merge_exceptions[0] is None
    assert harness.merge_exceptions[1] is raised.value
    assert _event_names(harness.events)[-7:] == [
        "report",
        "causal-domains",
        "artifact-hashes",
        "clock",
        "causal-json",
        "write",
        "merge-logs",
    ]
    assert json.loads(harness.layout.manifest.read_text(encoding="utf-8")) == terminal


@pytest.mark.parametrize(
    "parallel_jobs",
    [
        pytest.param(True, id="bool"),
        pytest.param(0, id="zero"),
        pytest.param(1.5, id="float"),
    ],
)
def test_invalid_parallel_override_is_rejected_before_layout(
    tmp_path, monkeypatch, parallel_jobs
):
    harness = LifecycleHarness(monkeypatch, tmp_path)

    with pytest.raises(ValueError) as raised:
        harness.run(parallel_jobs=parallel_jobs)

    assert str(raised.value) == "parallel_jobs must be an integer of at least 1."
    assert _event_names(harness.events) == ["validate", "numerical"]
    assert harness.create_layout_calls == []
    assert harness.writes == []
    assert harness.merge_calls == []
    assert not harness.layout.resolved_config.exists()
    assert not harness.layout.manifest.exists()


def test_pre_try_provider_failure_bypasses_recovery(tmp_path, monkeypatch):
    sentinel = InjectedFailure("source fingerprint failed")
    harness = LifecycleHarness(
        monkeypatch,
        tmp_path,
        source_error=sentinel,
    )

    with pytest.raises(InjectedFailure) as raised:
        harness.run()

    assert raised.value is sentinel
    assert harness.create_layout_calls == [
        ("lifecycle_probe", "ab" * 32, harness.runs_root)
    ]
    assert harness.layout.resolved_config.read_text(encoding="utf-8") == (
        "resolved lifecycle configuration\n"
    )
    assert not harness.layout.manifest.exists()
    assert harness.writes == []
    assert harness.merge_calls == []
    assert "generate" not in _event_names(harness.events)
    assert harness.report_statuses == []
    assert _event_names(harness.events) == [
        "validate",
        "numerical",
        "layout",
        "configure-log",
        "render",
        "clock",
        "source-hash",
    ]


def test_second_pre_try_log_failure_leaves_durable_running_checkpoint(
    tmp_path, monkeypatch
):
    sentinel = InjectedFailure("second startup log failed")
    harness = LifecycleHarness(
        monkeypatch,
        tmp_path,
        log_error=sentinel,
        log_error_call=2,
    )

    with pytest.raises(InjectedFailure) as raised:
        harness.run()

    assert raised.value is sentinel
    [checkpoint] = _written_payloads(harness)
    assert checkpoint["status"] == "running"
    assert checkpoint["created_utc"] == "time-1"
    assert harness.clock_calls == 1
    assert harness.log_calls == 2
    assert harness.merge_calls == []
    assert "parallel" not in _event_names(harness.events)
    assert "generate" not in _event_names(harness.events)
    assert json.loads(harness.layout.manifest.read_text(encoding="utf-8")) == checkpoint


def test_in_try_exception_writes_failed_manifest_and_reraises_same_object(
    tmp_path, monkeypatch
):
    sentinel = InjectedFailure("injected worker failure")
    harness = LifecycleHarness(
        monkeypatch,
        tmp_path,
        generation_error=sentinel,
    )

    with pytest.raises(InjectedFailure) as raised:
        harness.run()

    assert raised.value is sentinel
    checkpoint, failed = _written_payloads(harness)
    assert [checkpoint["status"], failed["status"]] == ["running", "failed"]
    assert list(failed) == list(FAILED_MANIFEST_KEYS)
    for key in BASE_MANIFEST_KEYS:
        if key != "status":
            assert failed[key] == checkpoint[key]
    assert list(failed["error"]) == ["type", "message", "traceback"]
    assert failed["error"]["type"] == "InjectedFailure"
    assert failed["error"]["message"] == "injected worker failure"
    assert failed["error"]["traceback"].rstrip().endswith(
        "InjectedFailure: injected worker failure"
    )
    assert failed["completed_utc"] == "time-2"
    assert "artifacts" not in failed
    assert harness.merge_calls == [harness.layout.logs / "pipeline.log"]
    assert harness.merge_exceptions == [sentinel]
    assert harness.report_statuses == []
    assert _event_names(harness.events)[-3:] == ["merge-logs", "clock", "write"]
    assert json.loads(harness.layout.manifest.read_text(encoding="utf-8")) == failed


def test_post_merge_exception_runs_recovery_merge_before_failed_checkpoint(
    tmp_path, monkeypatch
):
    sentinel = InjectedFailure("artifact inventory failed")
    harness = LifecycleHarness(
        monkeypatch,
        tmp_path,
        artifact_error=sentinel,
    )

    with pytest.raises(InjectedFailure) as raised:
        harness.run()

    assert raised.value is sentinel
    checkpoint, failed = _written_payloads(harness)
    assert [checkpoint["status"], failed["status"]] == ["running", "failed"]
    assert failed["completed_utc"] == "time-2"
    assert failed["error"]["type"] == "InjectedFailure"
    assert failed["error"]["message"] == "artifact inventory failed"
    canonical_log = harness.layout.logs / "pipeline.log"
    assert harness.merge_calls == [canonical_log, canonical_log]
    assert harness.merge_exceptions == [None, sentinel]
    assert _event_names(harness.events)[-4:] == [
        "artifact-hashes",
        "merge-logs",
        "clock",
        "write",
    ]
    assert json.loads(harness.layout.manifest.read_text(encoding="utf-8")) == failed


def test_base_exception_bypasses_recovery_and_leaves_running_checkpoint(
    tmp_path, monkeypatch
):
    sentinel = InjectedAbort("stop immediately")
    harness = LifecycleHarness(
        monkeypatch,
        tmp_path,
        generation_error=sentinel,
    )

    with pytest.raises(InjectedAbort) as raised:
        harness.run()

    assert raised.value is sentinel
    [checkpoint] = _written_payloads(harness)
    assert list(checkpoint) == list(BASE_MANIFEST_KEYS)
    assert checkpoint["status"] == "running"
    assert checkpoint["created_utc"] == "time-1"
    assert harness.clock_calls == 1
    assert "completed_utc" not in checkpoint
    assert "error" not in checkpoint
    assert harness.merge_calls == []
    assert harness.report_statuses == []
    assert json.loads(harness.layout.manifest.read_text(encoding="utf-8")) == checkpoint
