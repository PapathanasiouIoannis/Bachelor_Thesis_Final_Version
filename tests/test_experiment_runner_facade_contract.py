from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from importlib.util import resolve_name
from pathlib import Path

import pytest

from src.physics import experiment_runner
from src.physics.runner import convergence, generation, preflight


ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIRECTORY = ROOT / "src" / "physics" / "runner"
RUNNER_MODULE_PATHS = tuple(sorted(RUNNER_DIRECTORY.glob("*.py")))
LEAF_MODULES = tuple(
    path.stem for path in RUNNER_MODULE_PATHS if path.stem != "__init__"
)


def test_runner_facade_preserves_supported_surface_and_canonical_identities():
    expected_exports = frozenset(
        {
            "CONVERGENCE_COLUMNS",
            "PAIR_INTERPRETATION",
            "PairPreflight",
            "render_resolved_toml",
            "run_pair_experiment",
            "validate_pair_experiment",
        }
    )
    assert frozenset(experiment_runner.__all__) == expected_exports
    assert len(experiment_runner.__all__) == len(expected_exports)
    assert str(inspect.signature(experiment_runner.validate_pair_experiment)) == (
        "(configuration: 'ResolvedExperiment | str | Path') -> 'PairPreflight'"
    )
    assert experiment_runner.PairPreflight is preflight.PairPreflight
    assert experiment_runner.PairGenerationError is generation.PairGenerationError
    assert experiment_runner.CONVERGENCE_COLUMNS is convergence.CONVERGENCE_COLUMNS


@pytest.mark.parametrize(
    "module_name",
    ("__package__", *LEAF_MODULES),
    ids=("package-only", *(f"{name}-only" for name in LEAF_MODULES)),
)
def test_runner_package_and_each_leaf_have_clean_fresh_imports(module_name):
    target = (
        "src.physics.runner"
        if module_name == "__package__"
        else f"src.physics.runner.{module_name}"
    )
    script = "\n".join(
        (
            "import importlib",
            "import sys",
            f"importlib.import_module({target!r})",
            'assert "src.physics.experiment_runner" not in sys.modules',
            *(
                (
                    "assert not any(",
                    '    name.startswith("src.physics.runner.")',
                    "    for name in sys.modules",
                    ")",
                )
                if module_name == "__package__"
                else ()
            ),
            *(
                ('importlib.import_module("src.physics.experiment_runner")',)
                if module_name != "__package__"
                else ()
            ),
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
def test_runner_leaves_and_facade_compose_cleanly_in_both_orders(facade_first):
    script = "\n".join(
        (
            "import importlib",
            "import sys",
            f"leaf_names = {LEAF_MODULES!r}",
            f"facade_first = {facade_first!r}",
            "if facade_first:",
            "    experiment_runner = importlib.import_module("
            '"src.physics.experiment_runner")',
            "modules = {",
            "    name: importlib.import_module(f'src.physics.runner.{name}')",
            "    for name in leaf_names",
            "}",
            "if not facade_first:",
            '    assert "src.physics.experiment_runner" not in sys.modules',
            "    experiment_runner = importlib.import_module("
            '"src.physics.experiment_runner")',
            "artifacts = modules['artifacts']",
            "convergence = modules['convergence']",
            "generation = modules['generation']",
            "manifest = modules['manifest']",
            "preflight = modules['preflight']",
            "run_logs = modules['run_logs']",
            "settings = modules['settings']",
            "assert experiment_runner.PairPreflight is preflight.PairPreflight",
            "assert experiment_runner.PairGenerationError is "
            "generation.PairGenerationError",
            "assert experiment_runner.CONVERGENCE_COLUMNS is "
            "convergence.CONVERGENCE_COLUMNS",
            "assert experiment_runner.NUMERICAL_PRESETS is settings.NUMERICAL_PRESETS",
            "operations = (",
            "    experiment_runner.run_pair_experiment,",
            "    experiment_runner.validate_pair_experiment,",
            "    experiment_runner.render_resolved_toml,",
            "    experiment_runner._toml_value,",
            "    experiment_runner._concat_frames,",
            "    experiment_runner._artifact_hashes,",
            "    experiment_runner._run_convergence_checks,",
            "    experiment_runner._failed_convergence_record,",
            "    experiment_runner._physical_requirements_status,",
            "    experiment_runner._generate_pair,",
            "    experiment_runner._build_eos,",
            "    experiment_runner._solve,",
            "    experiment_runner._worker_log_path,",
            "    experiment_runner._merge_worker_logs,",
            "    artifacts.render_resolved_toml,",
            "    artifacts.toml_value,",
            "    artifacts.concat_frames,",
            "    artifacts.artifact_hashes,",
            "    convergence.run_convergence_checks,",
            "    convergence.failed_convergence_record,",
            "    convergence.physical_requirements_status,",
            "    generation.generate_pair,",
            "    generation.build_eos,",
            "    generation.solve,",
            "    manifest.running_manifest,",
            "    manifest.terminal_status,",
            "    manifest.terminal_manifest,",
            "    manifest.failed_manifest,",
            "    preflight.validate_pair_experiment,",
            "    run_logs.worker_log_path,",
            "    run_logs.merge_worker_logs,",
            "    settings.numerical_presets_from_configuration,",
            "    settings.quark_parameters,",
            "    settings.resolved_numerical_settings,",
            ")",
            "assert all(callable(operation) for operation in operations)",
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


def test_runner_leaves_have_no_static_facade_imports():
    violations = []
    for path in RUNNER_MODULE_PATHS:
        module_name = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "src.physics.experiment_runner"
                    for alias in node.names
                ):
                    violations.append((module_name, node.lineno))
            elif isinstance(node, ast.ImportFrom):
                base_module = (
                    resolve_name(
                        "." * node.level + (node.module or ""),
                        "src.physics.runner",
                    )
                    if node.level
                    else node.module
                )
                imports_facade_name = base_module == ("src.physics.experiment_runner")
                imports_facade_module = base_module == "src.physics" and any(
                    alias.name == "experiment_runner" for alias in node.names
                )
                if imports_facade_name or imports_facade_module:
                    violations.append((module_name, node.lineno))

    assert violations == []
