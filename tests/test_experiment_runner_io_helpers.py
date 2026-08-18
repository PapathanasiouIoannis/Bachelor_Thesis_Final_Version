from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from src.eoslab_runtime import initialize_run_layout
from src.physics import experiment_runner


def test_runner_log_facade_preserves_exact_helper_signatures():
    assert str(inspect.signature(experiment_runner._worker_log_path)) == (
        "(run_log_path: 'Path') -> 'Path'"
    )
    assert str(inspect.signature(experiment_runner._merge_worker_logs)) == (
        "(run_log_path: 'Path') -> 'None'"
    )


@pytest.mark.parametrize(
    ("run_log_path", "expected"),
    [
        pytest.param(
            Path("relative") / "logs" / "pipeline.log",
            Path("relative") / "logs" / "pipeline.worker-4321.log",
            id="production-name",
        ),
        pytest.param(
            Path("relative") / "logs" / "pipeline.audit.log",
            Path("relative") / "logs" / "pipeline.audit.worker-4321.log",
            id="multiple-suffixes",
        ),
        pytest.param(
            Path("relative") / "logs" / "pipeline",
            Path("relative") / "logs" / "pipeline.worker-4321",
            id="no-suffix",
        ),
    ],
)
def test_worker_log_path_uses_the_live_pid_and_preserves_path_shape(
    monkeypatch, run_log_path, expected
):
    monkeypatch.setattr(experiment_runner.os, "getpid", lambda: 4321)

    result = experiment_runner._worker_log_path(run_log_path)

    assert result == expected
    assert result.parent == run_log_path.parent


def test_merge_worker_logs_without_matches_is_a_total_no_op(tmp_path):
    run_log_path = tmp_path / "not-created" / "logs" / "pipeline.log"

    assert experiment_runner._merge_worker_logs(run_log_path) is None

    assert not run_log_path.parent.exists()
    assert not run_log_path.exists()


def test_merge_worker_logs_appends_raw_text_in_lexical_order_and_cleans_up(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    run_log_path = logs / "pipeline.log"
    run_log_path.write_text("canonical|", encoding="utf-8")
    worker_20 = logs / "pipeline.worker-20.log"
    worker_3 = logs / "pipeline.worker-3.log"
    worker_20.write_text("worker-20|", encoding="utf-8")
    worker_3.write_text("worker-3|", encoding="utf-8")
    unrelated = logs / "other.worker-1.log"
    unrelated.write_text("unrelated|", encoding="utf-8")
    wrong_suffix = logs / "pipeline.worker-1.txt"
    wrong_suffix.write_text("wrong-suffix|", encoding="utf-8")

    experiment_runner._merge_worker_logs(run_log_path)

    merged = run_log_path.read_text(encoding="utf-8")
    assert merged == (
        "canonical|worker-20|worker-3|"
    )
    assert not worker_20.exists()
    assert not worker_3.exists()
    assert unrelated.read_text(encoding="utf-8") == "unrelated|"
    assert wrong_suffix.read_text(encoding="utf-8") == "wrong-suffix|"

    experiment_runner._merge_worker_logs(run_log_path)

    assert run_log_path.read_text(encoding="utf-8") == merged


def test_permission_error_during_cleanup_is_suppressed_and_worker_log_remains(
    tmp_path, monkeypatch
):
    logs = tmp_path / "logs"
    logs.mkdir()
    run_log_path = logs / "pipeline.log"
    run_log_path.write_text("canonical|", encoding="utf-8")
    retained = logs / "pipeline.worker-1.log"
    removed = logs / "pipeline.worker-2.log"
    retained.write_text("retained|", encoding="utf-8")
    removed.write_text("removed|", encoding="utf-8")
    original_unlink = Path.unlink

    def controlled_unlink(path, *args, **kwargs):
        if path == retained:
            raise PermissionError("worker still owns the log")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", controlled_unlink)

    experiment_runner._merge_worker_logs(run_log_path)

    assert run_log_path.read_text(encoding="utf-8") == (
        "canonical|retained|removed|"
    )
    assert retained.is_file()
    assert not removed.exists()


def test_non_permission_cleanup_error_propagates_after_every_log_is_appended(
    tmp_path, monkeypatch
):
    logs = tmp_path / "logs"
    logs.mkdir()
    run_log_path = logs / "pipeline.log"
    first = logs / "pipeline.worker-1.log"
    second = logs / "pipeline.worker-2.log"
    first.write_text("first|", encoding="utf-8")
    second.write_text("second|", encoding="utf-8")
    sentinel = OSError("unlink failed")
    unlink_calls = []

    def fail_first_unlink(path, *args, **kwargs):
        del args, kwargs
        unlink_calls.append(path)
        if path == first:
            raise sentinel
        raise AssertionError("cleanup continued after the first unlink failure")

    monkeypatch.setattr(Path, "unlink", fail_first_unlink)

    with pytest.raises(OSError) as raised:
        experiment_runner._merge_worker_logs(run_log_path)

    assert raised.value is sentinel
    assert run_log_path.read_text(encoding="utf-8") == "first|second|"
    assert unlink_calls == [first]
    assert first.is_file()
    assert second.is_file()


def test_concat_frames_empty_path_uses_live_dataframe_and_schema_identity(monkeypatch):
    columns = ("second", "first")
    sentinel = object()
    calls = []

    def dataframe(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(experiment_runner.pd, "DataFrame", dataframe)

    result = experiment_runner._concat_frames([], columns)

    assert result is sentinel
    assert calls == [((), {"columns": columns})]
    assert calls[0][1]["columns"] is columns


def test_concat_frames_preserves_input_order_resets_index_and_projects_columns(
    monkeypatch,
):
    first = pd.DataFrame(
        {"extra": ["a"], "second": [20], "first": [10]},
        index=[7],
    )
    second = pd.DataFrame(
        {"extra": ["b"], "second": [40], "first": [30]},
        index=[9],
    )
    first_before = first.copy(deep=True)
    second_before = second.copy(deep=True)
    frames = [first, second]
    columns = ("first", "second")
    calls = []
    original_concat = pd.concat

    def concat(received_frames, *, ignore_index):
        calls.append((received_frames, ignore_index))
        return original_concat(received_frames, ignore_index=ignore_index)

    monkeypatch.setattr(experiment_runner.pd, "concat", concat)

    result = experiment_runner._concat_frames(frames, columns)

    assert calls == [(frames, True)]
    assert calls[0][0] is frames
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"first": [10, 30], "second": [20, 40]}),
    )
    pd.testing.assert_frame_equal(first, first_before)
    pd.testing.assert_frame_equal(second, second_before)


def test_concat_frames_propagates_missing_projected_columns():
    frame = pd.DataFrame({"present": [1]})

    with pytest.raises(KeyError) as raised:
        experiment_runner._concat_frames([frame], ("present", "missing"))

    assert "missing" in str(raised.value)


def test_artifact_hashes_use_fixed_order_sorted_plots_and_live_hasher(
    tmp_path, monkeypatch
):
    layout = initialize_run_layout(tmp_path / "run")
    present_paths = [
        layout.resolved_config,
        layout.data / "eos_tables.parquet",
        layout.data / "stellar_curves.parquet",
        layout.tables / "eos_summary.csv",
        layout.tables / "rejections.csv",
        layout.tables / "convergence.csv",
        layout.report,
        layout.plots / "z-last.png",
        layout.plots / "a-first.png",
    ]
    for index, path in enumerate(present_paths):
        path.write_text(f"artifact-{index}", encoding="utf-8")
    layout.manifest.write_text("manifest", encoding="utf-8")
    (layout.logs / "pipeline.log").write_text("log", encoding="utf-8")
    (layout.plots / "not-a-plot.svg").write_text("svg", encoding="utf-8")
    hash_calls = []

    def hasher(path):
        hash_calls.append(path)
        return f"hash:{path.relative_to(layout.root).as_posix()}"

    monkeypatch.setattr(experiment_runner, "file_sha256", hasher)

    result = experiment_runner._artifact_hashes(layout)

    expected_paths = [
        layout.resolved_config,
        layout.data / "eos_tables.parquet",
        layout.data / "stellar_curves.parquet",
        layout.tables / "eos_summary.csv",
        layout.tables / "rejections.csv",
        layout.tables / "convergence.csv",
        layout.report,
        layout.plots / "a-first.png",
        layout.plots / "z-last.png",
    ]
    expected_keys = [path.relative_to(layout.root).as_posix() for path in expected_paths]
    assert hash_calls == expected_paths
    assert list(result) == expected_keys
    assert result == {key: f"hash:{key}" for key in expected_keys}
    assert "run_manifest.json" not in result
    assert "logs/pipeline.log" not in result
    assert "plots/not-a-plot.svg" not in result


def test_artifact_hashes_silently_omit_missing_fixed_candidates(tmp_path, monkeypatch):
    layout = initialize_run_layout(tmp_path / "run")
    only_present = layout.tables / "eos_summary.csv"
    only_present.write_text("summary", encoding="utf-8")
    monkeypatch.setattr(experiment_runner, "file_sha256", lambda path: "summary-hash")

    result = experiment_runner._artifact_hashes(layout)

    assert result == {"tables/eos_summary.csv": "summary-hash"}


def test_artifact_hash_failure_propagates_and_stops_later_hashing(
    tmp_path, monkeypatch
):
    layout = initialize_run_layout(tmp_path / "run")
    layout.resolved_config.write_text("config", encoding="utf-8")
    eos_path = layout.data / "eos_tables.parquet"
    eos_path.write_text("eos", encoding="utf-8")
    later_path = layout.report
    later_path.write_text("report", encoding="utf-8")
    calls = []
    sentinel = RuntimeError("hash failed")

    def hasher(path):
        calls.append(path)
        if path == eos_path:
            raise sentinel
        return "first-hash"

    monkeypatch.setattr(experiment_runner, "file_sha256", hasher)

    with pytest.raises(RuntimeError) as raised:
        experiment_runner._artifact_hashes(layout)

    assert raised.value is sentinel
    assert calls == [layout.resolved_config, eos_path]
    assert later_path not in calls
