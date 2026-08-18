from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

from src import family_workflow


ROOT = Path(__file__).resolve().parents[1]
FAMILY_RUNNER_DIRECTORY = ROOT / "src" / "family_runner"
FAMILY_RUNNER_MODULE_PATHS = tuple(sorted(FAMILY_RUNNER_DIRECTORY.glob("*.py")))
LEAF_MODULES = tuple(
    path.stem for path in FAMILY_RUNNER_MODULE_PATHS if path.stem != "__init__"
)
WRAPPER_MAPPINGS = (
    ("development", "_development_commands", "development_commands"),
    ("evidence", "_read_json_status", "read_json_status"),
    ("evidence", "_file_matches_sha256", "file_matches_sha256"),
    ("evidence", "_artifact_status", "artifact_status"),
    ("evidence", "_development_artifacts", "development_artifacts"),
    (
        "evidence",
        "_development_evidence_summary",
        "development_evidence_summary",
    ),
    ("evidence", "_final_test_status", "final_test_status"),
    ("status", "_load_profiles", "load_profiles"),
    ("status", "_family_split_summary", "family_split_summary"),
    ("status", "family_workflow_status", "family_workflow_status"),
)


def test_family_facade_preserves_supported_surface_and_distinct_wrappers():
    expected_exports = {
        "EXPLORATORY_MODELS",
        "MODEL_SET_CLAIM",
        "SUPPORTED_REPORTING_MODELS",
        "FamilyDevelopmentStageError",
        "FamilyWorkflowError",
        "FamilyWorkflowPaths",
        "FinalTestAlreadyOpenedError",
        "assert_final_evaluation_not_opened",
        "family_workflow_status",
        "refuse_final_evaluation_request",
        "resolve_family_workflow_paths",
        "run_family_development",
    }
    assert set(family_workflow.__all__) == expected_exports
    assert len(family_workflow.__all__) == len(expected_exports)
    for facade_type_name in (
        "FamilyDevelopmentStageError",
        "FamilyWorkflowError",
        "FamilyWorkflowPaths",
        "FinalTestAlreadyOpenedError",
    ):
        assert getattr(family_workflow, facade_type_name).__module__ == (
            "src.family_workflow"
        )

    modules = {
        name: importlib.import_module(f"src.family_runner.{name}")
        for name in LEAF_MODULES
    }
    for leaf_name, facade_name, implementation_name in WRAPPER_MAPPINGS:
        facade_operation = getattr(family_workflow, facade_name)
        leaf_operation = getattr(modules[leaf_name], implementation_name)
        assert callable(facade_operation)
        assert callable(leaf_operation)
        assert facade_operation is not leaf_operation


@pytest.mark.parametrize(
    "module_name",
    ("__package__", *LEAF_MODULES),
    ids=("package-only", *(f"{name}-only" for name in LEAF_MODULES)),
)
def test_family_package_and_each_leaf_have_clean_fresh_imports(module_name):
    target = (
        "src.family_runner"
        if module_name == "__package__"
        else f"src.family_runner.{module_name}"
    )
    expected_loaded = () if module_name == "__package__" else (module_name,)
    script = "\n".join(
        (
            "import importlib",
            "import sys",
            f"leaf_names = {LEAF_MODULES!r}",
            f"importlib.import_module({target!r})",
            'assert "src.family_workflow" not in sys.modules',
            'assert "family_final_test" not in sys.modules',
            "loaded = tuple(",
            "    name for name in leaf_names",
            "    if f'src.family_runner.{name}' in sys.modules",
            ")",
            f"assert loaded == {expected_loaded!r}",
            *(
                ('importlib.import_module("src.family_workflow")',)
                if module_name != "__package__"
                else ()
            ),
            'assert "family_final_test" not in sys.modules',
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "facade_first",
    (False, True),
    ids=("leaves-then-facade", "facade-then-leaves"),
)
def test_family_leaves_and_facade_compose_cleanly_in_both_orders(facade_first):
    script = "\n".join(
        (
            "import importlib",
            "import sys",
            f"leaf_names = {LEAF_MODULES!r}",
            f"wrapper_mappings = {WRAPPER_MAPPINGS!r}",
            f"facade_first = {facade_first!r}",
            "if facade_first:",
            '    facade = importlib.import_module("src.family_workflow")',
            "modules = {",
            "    name: importlib.import_module(f'src.family_runner.{name}')",
            "    for name in leaf_names",
            "}",
            "if not facade_first:",
            '    assert "src.family_workflow" not in sys.modules',
            '    facade = importlib.import_module("src.family_workflow")',
            'assert "family_final_test" not in sys.modules',
            "for leaf_name, facade_name, implementation_name in wrapper_mappings:",
            "    facade_operation = getattr(facade, facade_name)",
            "    leaf_operation = getattr(modules[leaf_name], implementation_name)",
            "    assert callable(facade_operation)",
            "    assert callable(leaf_operation)",
            "    assert facade_operation is not leaf_operation",
            "public_operations = (",
            "    facade.assert_final_evaluation_not_opened,",
            "    facade.family_workflow_status,",
            "    facade.refuse_final_evaluation_request,",
            "    facade.resolve_family_workflow_paths,",
            "    facade.run_family_development,",
            ")",
            "assert all(callable(operation) for operation in public_operations)",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _resolved_import_from(node: ast.ImportFrom, package: str) -> str | None:
    if not node.level:
        return node.module
    return resolve_name("." * node.level + (node.module or ""), package)


def test_family_leaves_have_no_facade_or_sibling_imports():
    violations = []
    leaf_module_names = {f"src.family_runner.{name}" for name in LEAF_MODULES}
    for path in FAMILY_RUNNER_MODULE_PATHS:
        module_name = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                if "src.family_workflow" in imported:
                    violations.append((module_name, node.lineno, "facade"))
                sibling_imports = imported & leaf_module_names
                if sibling_imports:
                    violations.append((module_name, node.lineno, "sibling"))
            elif isinstance(node, ast.ImportFrom):
                base_module = _resolved_import_from(node, "src.family_runner")
                imports_facade = base_module == "src.family_workflow" or (
                    base_module == "src"
                    and any(alias.name == "family_workflow" for alias in node.names)
                )
                imports_sibling = base_module in leaf_module_names or (
                    base_module == "src.family_runner"
                    and any(alias.name in LEAF_MODULES for alias in node.names)
                )
                if imports_facade:
                    violations.append((module_name, node.lineno, "facade"))
                if imports_sibling:
                    violations.append((module_name, node.lineno, "sibling"))

    assert violations == []


def test_family_workflow_and_leaves_never_import_final_test_owner():
    violations = []
    paths = (ROOT / "src" / "family_workflow.py", *FAMILY_RUNNER_MODULE_PATHS)
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name.split(".")[-1] == "family_final_test"
                    for alias in node.names
                ):
                    violations.append((path.name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                package = (
                    "src.family_runner"
                    if path.parent == FAMILY_RUNNER_DIRECTORY
                    else "src"
                )
                base_module = _resolved_import_from(node, package)
                if (
                    base_module is not None
                    and base_module.split(".")[-1] == "family_final_test"
                ):
                    violations.append((path.name, node.lineno))

    assert violations == []
