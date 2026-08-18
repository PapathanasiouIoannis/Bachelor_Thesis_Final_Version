from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

from src import family_workflow
from src.family_runner import development


def test_family_development_facade_preserves_planner_contract():
    assert str(inspect.signature(family_workflow._development_commands)) == (
        "(paths: 'FamilyWorkflowPaths', *, jobs: 'int', "
        "force_regenerate: 'bool', permutations: 'int') -> "
        "'list[tuple[str, tuple[str, ...]]]'"
    )
    assert development.__all__ == ["development_commands"]
    assert family_workflow._development_commands is not development.development_commands

    parameters = inspect.signature(development.development_commands).parameters
    for name in (
        "jobs",
        "force_regenerate",
        "permutations",
        "python_executable",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


def test_family_development_facade_dispatches_to_reloaded_leaf_with_live_python(
    monkeypatch,
):
    importlib.reload(development)
    paths = object()
    planned = object()
    calls = []

    def plan(value, **options):
        calls.append((value, options))
        return planned

    monkeypatch.setattr(development, "development_commands", plan)
    monkeypatch.setattr(
        family_workflow,
        "sys",
        SimpleNamespace(executable="python-from-live-facade"),
    )

    assert (
        family_workflow._development_commands(
            paths,
            jobs=3,
            force_regenerate=True,
            permutations=17,
        )
        is planned
    )
    assert calls == [
        (
            paths,
            {
                "jobs": 3,
                "force_regenerate": True,
                "permutations": 17,
                "python_executable": "python-from-live-facade",
            },
        )
    ]


def test_real_development_leaf_uses_live_facade_python_executable(monkeypatch):
    paths = SimpleNamespace(
        project_root=Path("project"),
        data_root=Path("data"),
        generation_profile_path=Path("generation.json"),
        split_profile_path=Path("split.json"),
        report_dir=Path("reports"),
    )
    monkeypatch.setattr(
        family_workflow,
        "sys",
        SimpleNamespace(executable="python-from-live-facade"),
    )

    commands = family_workflow._development_commands(
        paths,
        jobs=1,
        force_regenerate=False,
        permutations=0,
    )

    assert len(commands) == 5
    assert [command[0] for _, command in commands] == ["python-from-live-facade"] * 5
