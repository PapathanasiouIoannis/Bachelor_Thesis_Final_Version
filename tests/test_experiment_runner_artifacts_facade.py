from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

from src.physics import experiment_runner
from src.physics.runner import artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_artifacts_facade_preserves_exact_helper_signatures():
    assert callable(artifacts.render_resolved_toml)
    assert str(inspect.signature(experiment_runner.render_resolved_toml)) == (
        "(runtime: 'dict[str, Any]') -> 'str'"
    )
    assert str(inspect.signature(experiment_runner._toml_value)) == (
        "(value: 'Any') -> 'str'"
    )
    assert str(inspect.signature(experiment_runner._concat_frames)) == (
        "(frames: 'list[pd.DataFrame]', columns) -> 'pd.DataFrame'"
    )
    assert str(inspect.signature(experiment_runner._artifact_hashes)) == (
        "(layout: 'RunLayout') -> 'dict[str, str]'"
    )


def test_artifacts_facade_uses_reloaded_leaf_and_live_named_helpers():
    script = textwrap.dedent(
        """
        import importlib

        from src.physics import experiment_runner
        from src.physics.runner import artifacts

        importlib.reload(artifacts)
        runtime = object()
        value = object()
        frames = [object()]
        columns = object()
        layout = object()
        value_result = object()
        render_result = object()
        concat_result = object()
        hash_result = object()
        calls = []

        def toml_value(value_argument):
            assert value_argument is value
            calls.append("value")
            return value_result

        def render(runtime_argument, *, value_renderer):
            assert runtime_argument is runtime
            assert value_renderer is facade_value_renderer
            calls.append("render")
            return render_result

        def concat(frames_argument, columns_argument):
            assert frames_argument is frames
            assert columns_argument is columns
            calls.append("concat")
            return concat_result

        def hashes(layout_argument, *, file_hasher):
            assert layout_argument is layout
            assert file_hasher is facade_file_hasher
            calls.append("hashes")
            return hash_result

        def facade_value_renderer(_value):
            raise AssertionError("the dispatch spy must not invoke the renderer")

        def facade_file_hasher(_path):
            raise AssertionError("the dispatch spy must not invoke the hasher")

        artifacts.toml_value = toml_value
        artifacts.render_resolved_toml = render
        artifacts.concat_frames = concat
        artifacts.artifact_hashes = hashes

        assert experiment_runner._toml_value(value) is value_result

        experiment_runner._toml_value = facade_value_renderer
        assert experiment_runner.render_resolved_toml(runtime) is render_result

        assert experiment_runner._concat_frames(frames, columns) is concat_result

        experiment_runner.file_sha256 = facade_file_hasher
        assert experiment_runner._artifact_hashes(layout) is hash_result
        assert calls == ["value", "render", "concat", "hashes"]
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
