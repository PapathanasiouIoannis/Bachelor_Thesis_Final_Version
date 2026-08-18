from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import family_workflow as workflow


STAGES = (
    "generation",
    "curve_preparation",
    "shortcut_audit",
    "model_selection",
    "robustness",
)


class LaunchFailure(RuntimeError):
    pass


class LaunchAbort(BaseException):
    pass


class ProfileFailure(RuntimeError):
    pass


class PostStatusFailure(RuntimeError):
    pass


class FinalStatusFailure(RuntimeError):
    pass


def _paths(tmp_path: Path, *, project_root: Path | None = None):
    root = workflow.PROJECT_ROOT if project_root is None else project_root
    return workflow.resolve_family_workflow_paths(
        project_root=root,
        data_root=tmp_path / "data",
        generation_profile_path=tmp_path / "generation-profile.json",
        split_profile_path=tmp_path / "split-profile.json",
        model_profile_path=tmp_path / "model-profile.json",
        report_dir=tmp_path / "reports",
    )


def _expected_commands(
    paths,
    *,
    jobs: int = 1,
    force_regenerate: bool = False,
    permutations: int = 0,
) -> list[tuple[str, tuple[str, ...]]]:
    commands = [
        (
            "generation",
            (
                sys.executable,
                str(paths.project_root / "family_physics_main.py"),
                "--data-root",
                str(paths.data_root),
                "--profile",
                str(paths.generation_profile_path),
                "--n-jobs",
                str(jobs),
            ),
        ),
        (
            "curve_preparation",
            (
                sys.executable,
                str(paths.project_root / "family_ml_prepare.py"),
                "--data-root",
                str(paths.data_root),
                "--generation-profile",
                str(paths.generation_profile_path),
                "--split-profile",
                str(paths.split_profile_path),
            ),
        ),
        (
            "shortcut_audit",
            (
                sys.executable,
                str(paths.project_root / "family_shortcut_audit.py"),
                "--data-root",
                str(paths.data_root),
                "--output-dir",
                str(paths.report_dir),
            ),
        ),
        (
            "model_selection",
            (
                sys.executable,
                str(paths.project_root / "family_model_select.py"),
                "--data-root",
                str(paths.data_root),
                "--output-dir",
                str(paths.report_dir),
            ),
        ),
        (
            "robustness",
            (
                sys.executable,
                str(paths.project_root / "family_development_robustness.py"),
                "--data-root",
                str(paths.data_root),
                "--output-dir",
                str(paths.report_dir),
                "--permutations",
                str(permutations),
            ),
        ),
    ]
    if force_regenerate:
        stage, command = commands[0]
        commands[0] = (stage, (*command, "--force-regenerate"))
    return commands


def _allow_development(monkeypatch, *, post_status=None, events=None):
    model = {"profile_id": "synthetic-model"}
    post_status = (
        {"workflow": "synthetic-family-status"} if post_status is None else post_status
    )

    def load_profiles(paths):
        del paths
        if events is not None:
            events.append("profiles")
        return {"generation": True}, {"split": True}, model

    def final_status(paths, loaded_model):
        del paths
        assert loaded_model is model
        if events is not None:
            events.append("final_gate")
        return {
            "state": "LOCKED_NOT_EVALUATED",
            "rerun_permitted": None,
            "integrity_errors": [],
        }

    def status(paths):
        del paths
        if events is not None:
            events.append("post_status")
        if isinstance(post_status, BaseException):
            raise post_status
        return post_status

    monkeypatch.setattr(workflow, "_load_profiles", load_profiles)
    monkeypatch.setattr(workflow, "_final_test_status", final_status)
    monkeypatch.setattr(workflow, "family_workflow_status", status)
    return post_status


def test_development_forwards_exact_five_commands_and_subprocess_contract(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    post_status = _allow_development(monkeypatch)
    original_planner = workflow._development_commands
    planner_calls = []
    calls = []

    def plan(value, **options):
        planner_calls.append((value, options))
        return original_planner(value, **options)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=f" output from {Path(command[1]).stem} \n",
            stderr="ignored success stderr",
        )

    monkeypatch.setattr(workflow, "_development_commands", plan)
    monkeypatch.setattr(workflow.subprocess, "run", run)

    result = workflow.run_family_development(
        paths,
        jobs=3,
        force_regenerate=True,
        permutations=17,
    )

    expected = _expected_commands(
        paths,
        jobs=3,
        force_regenerate=True,
        permutations=17,
    )
    assert [command for command, _ in calls] == [command for _, command in expected]
    assert planner_calls == [
        (
            paths,
            {
                "jobs": 3,
                "force_regenerate": True,
                "permutations": 17,
            },
        )
    ]
    assert [stage for stage, _ in expected] == list(STAGES)
    assert all(
        kwargs
        == {
            "cwd": paths.project_root,
            "text": True,
            "capture_output": True,
            "check": False,
        }
        for _, kwargs in calls
    )
    assert expected[0][1][-1] == "--force-regenerate"
    assert all("--force-regenerate" not in command for _, command in expected[1:])
    assert not any(
        "family_final_test.py" in part for _, command in expected for part in command
    )
    assert result == {
        "workflow": "family_development",
        "scientific_scope": workflow.MODEL_SET_CLAIM,
        "completed_stages": [
            {
                "stage": stage,
                "state": "completed",
                "command": list(command),
                "stdout": f"output from {Path(command[1]).stem}",
            }
            for stage, command in expected
        ],
        "final_test_accessed": False,
        "final_test_state_before_run": "LOCKED_NOT_EVALUATED",
        "post_test_context": False,
        "status": post_status,
    }


@pytest.mark.parametrize(
    ("failure_index", "returncode", "stdout", "stderr", "detail"),
    (
        (0, 2, "ignored stdout", " generation failed \n", "generation failed"),
        (2, 3, " shortcut output \n", "", "shortcut output"),
        (4, 4, "", "", "No diagnostic output was produced."),
    ),
    ids=("first", "middle", "last"),
)
def test_stage_failure_preserves_error_contract_and_short_circuits_later_stages(
    tmp_path,
    monkeypatch,
    failure_index,
    returncode,
    stdout,
    stderr,
    detail,
):
    paths = _paths(tmp_path)
    _allow_development(
        monkeypatch,
        post_status=AssertionError("post status must not run after a stage failure"),
    )
    expected = _expected_commands(paths)
    calls = []

    def run(command, **kwargs):
        del kwargs
        calls.append(command)
        index = len(calls) - 1
        if index == failure_index:
            return SimpleNamespace(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
        return SimpleNamespace(returncode=0, stdout="completed", stderr="")

    monkeypatch.setattr(workflow.subprocess, "run", run)

    with pytest.raises(workflow.FamilyDevelopmentStageError) as caught:
        workflow.run_family_development(paths)

    stage, command = expected[failure_index]
    message = (
        f"Family development stage '{stage}' failed with exit code "
        f"{returncode}: {detail}"
    )
    assert calls == [entry[1] for entry in expected[: failure_index + 1]]
    assert caught.value.stage == stage
    assert caught.value.command == command
    assert caught.value.returncode == returncode
    assert caught.value.stdout == stdout
    assert caught.value.stderr == stderr
    assert caught.value.args == (message,)
    assert str(caught.value) == message


def test_missing_later_entrypoint_is_detected_after_prior_stage_side_effects(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "synthetic-project"
    project_root.mkdir()
    paths = _paths(tmp_path, project_root=project_root)
    expected = _expected_commands(paths)
    for _, command in expected[:2]:
        Path(command[1]).touch()
    calls = []
    _allow_development(
        monkeypatch,
        post_status=AssertionError("post status must not run for a missing entrypoint"),
    )

    def run(command, **kwargs):
        del kwargs
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="completed", stderr="")

    monkeypatch.setattr(workflow.subprocess, "run", run)

    missing_stage, missing_command = expected[2]
    missing_script = Path(missing_command[1])
    message = (
        f"Family development stage '{missing_stage}' is missing its entrypoint: "
        f"{missing_script}"
    )
    with pytest.raises(FileNotFoundError) as caught:
        workflow.run_family_development(paths)

    assert calls == [entry[1] for entry in expected[:2]]
    assert caught.value.args == (message,)
    assert str(caught.value) == message


@pytest.mark.parametrize("signal_type", (LaunchFailure, LaunchAbort))
def test_raw_subprocess_exception_and_base_exception_propagate_by_identity(
    tmp_path, monkeypatch, signal_type
):
    paths = _paths(tmp_path)
    _allow_development(
        monkeypatch,
        post_status=AssertionError("post status must not run after a launch signal"),
    )
    expected = _expected_commands(paths)
    signal = signal_type("raw launch signal")
    calls = []

    def run(command, **kwargs):
        del kwargs
        calls.append(command)
        if len(calls) == 2:
            raise signal
        return SimpleNamespace(returncode=0, stdout="completed", stderr="")

    monkeypatch.setattr(workflow.subprocess, "run", run)

    with pytest.raises(signal_type) as caught:
        workflow.run_family_development(paths)

    assert caught.value is signal
    assert calls == [entry[1] for entry in expected[:2]]


def test_post_status_failure_propagates_after_all_five_stages(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    events = []
    signal = PostStatusFailure("post-status failed")
    _allow_development(monkeypatch, post_status=signal, events=events)
    expected = _expected_commands(paths)

    def run(command, **kwargs):
        del kwargs
        events.append(Path(command[1]).name)
        return SimpleNamespace(returncode=0, stdout="completed", stderr="")

    monkeypatch.setattr(workflow.subprocess, "run", run)

    with pytest.raises(PostStatusFailure) as caught:
        workflow.run_family_development(paths)

    assert caught.value is signal
    assert events == [
        "profiles",
        "final_gate",
        *(Path(command[1]).name for _, command in expected),
        "post_status",
    ]


def test_paths_options_conflict_precedes_every_other_gate(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    events = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        events.append("unexpected")
        raise AssertionError("a later gate was reached")

    monkeypatch.setattr(workflow, "_refuse_final_named_stage", forbidden)
    monkeypatch.setattr(workflow, "_load_profiles", forbidden)
    monkeypatch.setattr(workflow, "_final_test_status", forbidden)
    monkeypatch.setattr(workflow, "family_workflow_status", forbidden)

    with pytest.raises(
        TypeError,
        match="Pass either 'paths' or individual path options, not both",
    ):
        workflow.run_family_development(
            paths,
            requested_stage="final_evaluation",
            jobs="not-an-integer",
            report_dir=tmp_path / "other-reports",
        )

    assert events == []


@pytest.mark.parametrize(
    "requested_stage",
    (" FINAL-evaluation ", " test ", "evaluation", "score-test"),
)
def test_final_named_request_precedes_numeric_profile_and_process_gates(
    tmp_path, monkeypatch, requested_stage
):
    paths = _paths(tmp_path)
    events = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        events.append("unexpected")
        raise AssertionError("a later gate was reached")

    monkeypatch.setattr(workflow, "_load_profiles", forbidden)
    monkeypatch.setattr(workflow, "_final_test_status", forbidden)
    monkeypatch.setattr(workflow, "family_workflow_status", forbidden)
    monkeypatch.setattr(workflow.subprocess, "run", forbidden)

    with pytest.raises(workflow.FamilyWorkflowError) as caught:
        workflow.run_family_development(
            paths,
            requested_stage=requested_stage,
            jobs="not-an-integer",
            permutations="not-an-integer",
        )

    assert caught.value.args == (
        "The unified family workflow supports development stages only. "
        "It never opens or scores the locked final test.",
    )
    assert events == []


@pytest.mark.parametrize(
    ("jobs", "permutations", "error_type", "message"),
    (
        (0, -1, ValueError, "parallel jobs must be at least 1."),
        (1, -1, ValueError, "permutations must be 0 or a positive integer."),
        ("not-an-integer", 0, ValueError, None),
        (None, 0, TypeError, None),
    ),
    ids=("jobs-range", "permutations-range", "jobs-value", "jobs-type"),
)
def test_numeric_input_failures_precede_path_resolution_and_profile_loading(
    tmp_path, monkeypatch, jobs, permutations, error_type, message
):
    events = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        events.append("unexpected")
        raise AssertionError("path/profile work was reached")

    monkeypatch.setattr(workflow, "resolve_family_workflow_paths", forbidden)
    monkeypatch.setattr(workflow, "_load_profiles", forbidden)

    with pytest.raises(error_type) as caught:
        workflow.run_family_development(
            jobs=jobs,
            permutations=permutations,
            report_dir=tmp_path / "reports",
        )

    if message is not None:
        assert caught.value.args == (message,)
    assert events == []


def test_profile_failure_precedes_final_evidence_commands_and_processes(
    tmp_path, monkeypatch
):
    paths = _paths(tmp_path)
    signal = ProfileFailure("profile validation failed")
    events = []

    def load_profiles(resolved):
        assert resolved is paths
        events.append("profiles")
        raise signal

    def forbidden(*args, **kwargs):
        del args, kwargs
        events.append("unexpected")
        raise AssertionError("a later gate was reached")

    monkeypatch.setattr(workflow, "_load_profiles", load_profiles)
    monkeypatch.setattr(workflow, "_final_test_status", forbidden)
    monkeypatch.setattr(workflow, "_development_commands", forbidden)
    monkeypatch.setattr(workflow, "family_workflow_status", forbidden)
    monkeypatch.setattr(workflow.subprocess, "run", forbidden)

    with pytest.raises(ProfileFailure) as caught:
        workflow.run_family_development(paths)

    assert caught.value is signal
    assert events == ["profiles"]


def test_final_status_failure_propagates_before_commands_and_processes(
    tmp_path,
    monkeypatch,
):
    paths = _paths(tmp_path)
    model = object()
    signal = FinalStatusFailure("final evidence inspection failed")
    events = []

    def load_profiles(resolved):
        assert resolved is paths
        events.append("profiles")
        return {}, {}, model

    def inspect_final(resolved, loaded_model):
        assert resolved is paths
        assert loaded_model is model
        events.append("final_gate")
        raise signal

    def forbidden(*args, **kwargs):
        del args, kwargs
        events.append("unexpected")
        raise AssertionError("command, process, or post-status work was reached")

    monkeypatch.setattr(workflow, "_load_profiles", load_profiles)
    monkeypatch.setattr(workflow, "_final_test_status", inspect_final)
    monkeypatch.setattr(workflow, "_development_commands", forbidden)
    monkeypatch.setattr(workflow, "family_workflow_status", forbidden)
    monkeypatch.setattr(workflow.subprocess, "run", forbidden)

    with pytest.raises(FinalStatusFailure) as caught:
        workflow.run_family_development(paths)

    assert caught.value is signal
    assert events == ["profiles", "final_gate"]


@pytest.mark.parametrize(
    ("final_status", "message"),
    (
        (
            {
                "state": "LOCKED_TEST_OPENED",
                "rerun_permitted": False,
                "integrity_errors": ["ignored by the opened-state branch"],
            },
            "Family development outputs are frozen because the locked final test "
            "has already been opened. This command will not overwrite post-test "
            "evidence. Create a separately versioned future experiment instead.",
        ),
        (
            {
                "state": "INCONSISTENT_RESULT_WITHOUT_MARKER",
                "rerun_permitted": False,
                "integrity_errors": ["orphan result", "hash mismatch"],
            },
            "Family development is blocked because final-test evidence is present "
            "but is incomplete, unreadable, or inconsistent; fail-closed refusal. "
            "Investigate the recorded evidence before continuing. Details: orphan "
            "result; hash mismatch",
        ),
    ),
    ids=("opened", "inconsistent"),
)
def test_final_evidence_gate_follows_profiles_and_precedes_command_planning(
    tmp_path, monkeypatch, final_status, message
):
    paths = _paths(tmp_path)
    model = object()
    events = []

    def load_profiles(resolved):
        assert resolved is paths
        events.append("profiles")
        return {}, {}, model

    def inspect_final(resolved, loaded_model):
        assert resolved is paths
        assert loaded_model is model
        events.append("final_gate")
        return final_status

    def forbidden(*args, **kwargs):
        del args, kwargs
        events.append("unexpected")
        raise AssertionError("command/process work was reached")

    monkeypatch.setattr(workflow, "_load_profiles", load_profiles)
    monkeypatch.setattr(workflow, "_final_test_status", inspect_final)
    monkeypatch.setattr(workflow, "_development_commands", forbidden)
    monkeypatch.setattr(workflow, "family_workflow_status", forbidden)
    monkeypatch.setattr(workflow.subprocess, "run", forbidden)

    with pytest.raises(workflow.FinalTestAlreadyOpenedError) as caught:
        workflow.run_family_development(paths)

    assert caught.value.args == (message,)
    assert events == ["profiles", "final_gate"]


@pytest.mark.parametrize(
    ("jobs", "permutations", "expected_jobs", "expected_permutations"),
    (
        ("3", "4", 3, 4),
        (2.9, 4.9, 2, 4),
        (True, False, 1, 0),
    ),
    ids=("numeric-strings", "truncated-floats", "booleans"),
)
def test_legacy_integer_coercion_is_applied_to_child_commands(
    tmp_path,
    monkeypatch,
    jobs,
    permutations,
    expected_jobs,
    expected_permutations,
):
    paths = _paths(tmp_path)
    _allow_development(monkeypatch)
    calls = []

    def run(command, **kwargs):
        del kwargs
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="completed", stderr="")

    monkeypatch.setattr(workflow.subprocess, "run", run)

    workflow.run_family_development(
        paths,
        jobs=jobs,
        permutations=permutations,
    )

    expected = _expected_commands(
        paths,
        jobs=expected_jobs,
        permutations=expected_permutations,
    )
    assert calls == [command for _, command in expected]
