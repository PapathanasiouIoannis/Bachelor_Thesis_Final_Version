from __future__ import annotations

import inspect
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from joblib import Parallel, delayed
from joblib.externals import cloudpickle

from src.physics import experiment_runner
from src.physics.runner import generation


ROOT = Path(__file__).resolve().parents[1]


def _raise_pair_generation_error_through_worker_facade():
    from src.physics import experiment_runner as worker_runner

    def worker_log_path(path):
        return path

    def configure_run_log(_path):
        return None

    def raise_from_sweep_point(*_args):
        raise worker_runner.PairGenerationError(
            "quark",
            "stellar_sequence",
            "escaped through loky",
        )

    original_dependencies = (
        worker_runner._worker_log_path,
        worker_runner.configure_run_log,
        worker_runner.SweepPoint,
    )
    try:
        worker_runner._worker_log_path = worker_log_path
        worker_runner.configure_run_log = configure_run_log
        worker_runner.SweepPoint = raise_from_sweep_point
        worker_runner._generate_pair({}, 0, 0.0, "configuration-hash", "unused.log")
    finally:
        (
            worker_runner._worker_log_path,
            worker_runner.configure_run_log,
            worker_runner.SweepPoint,
        ) = original_dependencies


def _assert_pair_generation_error_contract(error):
    assert type(error) is generation.PairGenerationError
    assert error.matter_type == "quark"
    assert error.stage == "stellar_sequence"
    assert error.args == ("synthetic reason",)
    assert str(error) == "synthetic reason"
    assert error.extra_context == {"attempt": 2}


def test_generation_facade_preserves_class_identity_and_exact_signatures():
    assert experiment_runner.PairGenerationError is generation.PairGenerationError
    assert generation.PairGenerationError.__module__ == (
        "src.physics.runner.generation"
    )
    assert str(inspect.signature(experiment_runner.PairGenerationError)) == (
        "(matter_type: 'str', stage: 'str', reason: 'str')"
    )
    assert str(inspect.signature(experiment_runner._generate_pair)) == (
        "(runtime: 'dict[str, Any]', sweep_index: 'int', amplitude: 'float', "
        "configuration_hash: 'str', run_log_path: 'str') -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(experiment_runner._build_eos)) == (
        "(runtime: 'dict[str, Any]', matter_type: 'str', amplitude: 'float', *, "
        "grid_points: 'int | None' = None)"
    )
    assert str(inspect.signature(experiment_runner._solve)) == (
        "(runtime: 'dict[str, Any]', eos, matter_type: 'str', *, "
        "n_points: 'int | None' = None, rtol: 'float | None' = None, "
        "atol: 'float | None' = None, "
        "enforce_physical_requirements: 'bool' = True) -> "
        "'tuple[list, dict, float]'"
    )


def test_pair_generation_error_round_trips_through_pickle_and_cloudpickle():
    error = experiment_runner.PairGenerationError(
        "quark",
        "stellar_sequence",
        "synthetic reason",
    )
    error.extra_context = {"attempt": 2}

    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(error, protocol=protocol))
        _assert_pair_generation_error_contract(restored)

    restored = cloudpickle.loads(cloudpickle.dumps(error))
    _assert_pair_generation_error_contract(restored)


def test_pair_generation_error_escapes_forced_loky_worker_with_exact_type():
    with pytest.raises(experiment_runner.PairGenerationError) as raised:
        Parallel(n_jobs=2, backend="loky")(
            [delayed(_raise_pair_generation_error_through_worker_facade)()]
        )

    assert type(raised.value) is experiment_runner.PairGenerationError
    assert raised.value.matter_type == "quark"
    assert raised.value.stage == "stellar_sequence"
    assert raised.value.args == ("escaped through loky",)
    assert str(raised.value) == "escaped through loky"


@pytest.mark.parametrize(
    "imports",
    (
        """
        from src.physics.runner import generation
        assert "src.physics.experiment_runner" not in sys.modules
        from src.physics import experiment_runner
        """,
        """
        from src.physics import experiment_runner
        from src.physics.runner import generation
        """,
    ),
    ids=("leaf-then-facade", "facade-then-leaf"),
)
def test_generation_leaf_and_facade_import_cleanly_in_both_orders(imports):
    script = "\n".join(
        (
            "import sys",
            textwrap.dedent(imports).strip(),
            "",
            "assert experiment_runner.PairGenerationError is "
            "generation.PairGenerationError",
            "assert callable(experiment_runner._generate_pair)",
            "assert callable(experiment_runner._build_eos)",
            "assert callable(experiment_runner._solve)",
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
