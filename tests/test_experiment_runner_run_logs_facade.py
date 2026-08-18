from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.physics import experiment_runner
from src.physics.runner import run_logs


ROOT = Path(__file__).resolve().parents[1]


def test_run_logs_facade_preserves_exact_helper_signatures():
    assert callable(run_logs.worker_log_path)
    assert str(inspect.signature(experiment_runner._worker_log_path)) == (
        "(run_log_path: 'Path') -> 'Path'"
    )
    assert str(inspect.signature(experiment_runner._merge_worker_logs)) == (
        "(run_log_path: 'Path') -> 'None'"
    )


@pytest.mark.parametrize(
    "imports",
    (
        """
        from src.physics.runner import run_logs
        assert "src.physics.experiment_runner" not in sys.modules
        from src.physics import experiment_runner
        """,
        """
        from src.physics import experiment_runner
        from src.physics.runner import run_logs
        """,
    ),
    ids=("leaf-then-facade", "facade-then-leaf"),
)
def test_run_logs_leaf_and_facade_import_cleanly_in_both_orders(imports):
    script = "\n".join(
        (
            "import sys",
            textwrap.dedent(imports).strip(),
            "",
            "assert callable(experiment_runner._worker_log_path)",
            "assert callable(experiment_runner._merge_worker_logs)",
            "assert callable(run_logs.worker_log_path)",
            "assert callable(run_logs.merge_worker_logs)",
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


def test_run_logs_facade_uses_reloaded_leaf_functions():
    script = textwrap.dedent(
        """
        import importlib

        from src.physics import experiment_runner
        from src.physics.runner import run_logs

        importlib.reload(run_logs)
        run_log_path = object()
        worker_result = object()
        calls = []

        def worker(path_argument):
            assert path_argument is run_log_path
            calls.append("worker")
            return worker_result

        def merge(path_argument):
            assert path_argument is run_log_path
            calls.append("merge")
            return None

        run_logs.worker_log_path = worker
        run_logs.merge_worker_logs = merge

        assert experiment_runner._worker_log_path(run_log_path) is worker_result
        assert experiment_runner._merge_worker_logs(run_log_path) is None
        assert calls == ["worker", "merge"]
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
