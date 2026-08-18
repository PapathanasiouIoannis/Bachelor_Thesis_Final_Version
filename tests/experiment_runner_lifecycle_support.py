from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.eoslab_runtime import initialize_run_layout, write_json as runtime_write_json
from src.physics import experiment_runner


CONFIGURATION_HASH = "ab" * 32


class LifecycleHarness:
    def __init__(
        self,
        monkeypatch,
        tmp_path: Path,
        *,
        rejected: bool = False,
        convergence_check: str = "none",
        convergence_values: tuple[object, ...] = (),
        generation_error: BaseException | None = None,
        source_error: BaseException | None = None,
        log_error: BaseException | None = None,
        log_error_call: int = 1,
        artifact_error: BaseException | None = None,
    ) -> None:
        self.configuration = object()
        self.runs_root = tmp_path / "runs-root"
        self.layout = initialize_run_layout(tmp_path / "run")
        self.events: list[object] = []
        self.writes: list[tuple[Path, dict]] = []
        self.merge_calls: list[Path] = []
        self.merge_exceptions: list[BaseException | None] = []
        self.create_layout_calls: list[tuple] = []
        self.report_statuses: list[str] = []
        self.clock_calls = 0
        self.generation_error = generation_error
        self.source_error = source_error
        self.log_error = log_error
        self.log_error_call = log_error_call
        self.artifact_error = artifact_error
        self.log_calls = 0
        self.rejected = rejected
        self.runtime_configuration = {
            "schema_version": 1,
            "experiment_name": "lifecycle_probe",
            "workflow": "pair_sensitivity",
            "mode": "exploration",
            "numerical_settings": {"convergence_check": convergence_check},
            "execution": {
                "parallel_jobs": 1,
                "amplitudes_per_batch": 2,
            },
        }
        self.convergence = pd.DataFrame(
            {"passed": list(convergence_values)}
        )
        self.preflight_report = {
            "expected_curves": 2,
            "probe": "preflight-report",
        }
        self.preflight = SimpleNamespace(
            runtime_configuration=self.runtime_configuration,
            resolved=SimpleNamespace(config_hash=CONFIGURATION_HASH),
            sweep_points=[SimpleNamespace(index=0, amplitude=0.0)],
            to_dict=self._preflight_to_dict,
        )
        self._install(monkeypatch)

    def _install(self, monkeypatch) -> None:
        monkeypatch.setattr(
            experiment_runner,
            "validate_pair_experiment",
            self._validate_pair_experiment,
        )
        monkeypatch.setattr(
            experiment_runner,
            "_resolved_numerical_settings",
            self._resolved_numerical_settings,
        )
        monkeypatch.setattr(
            experiment_runner,
            "create_run_layout",
            self._create_run_layout,
        )
        monkeypatch.setattr(
            experiment_runner,
            "configure_run_log",
            self._configure_run_log,
        )
        monkeypatch.setattr(
            experiment_runner,
            "render_resolved_toml",
            self._render_resolved_toml,
        )
        monkeypatch.setattr(
            experiment_runner,
            "datetime",
            SimpleNamespace(now=self._now),
        )
        monkeypatch.setattr(
            experiment_runner,
            "source_tree_sha256",
            self._source_tree_sha256,
        )
        monkeypatch.setattr(experiment_runner, "git_revision", self._git_revision)
        monkeypatch.setattr(
            experiment_runner,
            "environment_metadata",
            self._environment_metadata,
        )
        monkeypatch.setattr(experiment_runner, "write_json", self._write_json)
        monkeypatch.setattr(
            experiment_runner,
            "LOGGER",
            SimpleNamespace(info=self._log_info),
        )
        monkeypatch.setattr(experiment_runner, "delayed", self._delayed)
        monkeypatch.setattr(experiment_runner, "Parallel", self._parallel)
        monkeypatch.setattr(experiment_runner, "_generate_pair", self._generate_pair)
        monkeypatch.setattr(
            experiment_runner,
            "_merge_worker_logs",
            self._merge_worker_logs,
        )
        monkeypatch.setattr(experiment_runner, "_concat_frames", self._concat_frames)
        monkeypatch.setattr(
            experiment_runner,
            "build_summary_table",
            self._build_summary_table,
        )
        monkeypatch.setattr(
            experiment_runner,
            "_run_convergence_checks",
            self._run_convergence_checks,
        )
        monkeypatch.setattr(
            experiment_runner,
            "create_standard_plots",
            self._create_standard_plots,
        )
        monkeypatch.setattr(
            experiment_runner,
            "write_markdown_report",
            self._write_markdown_report,
        )
        monkeypatch.setattr(
            experiment_runner,
            "build_causal_domain_table",
            self._build_causal_domain_table,
        )
        monkeypatch.setattr(
            experiment_runner,
            "_artifact_hashes",
            self._artifact_hashes,
        )
        monkeypatch.setattr(
            experiment_runner,
            "PAIR_INTERPRETATION",
            "lifecycle interpretation",
        )

    def _validate_pair_experiment(self, configuration):
        assert configuration is self.configuration
        self.events.append("validate")
        return self.preflight

    def _resolved_numerical_settings(self, runtime):
        self.events.append(("numerical", runtime))
        return {"eos_grid_points": 8}

    def _create_run_layout(
        self, experiment_name, configuration_hash, *, runs_root=None
    ):
        self.create_layout_calls.append(
            (experiment_name, configuration_hash, runs_root)
        )
        self.events.append("layout")
        assert experiment_name == "lifecycle_probe"
        assert configuration_hash == CONFIGURATION_HASH
        assert runs_root == self.runs_root
        return self.layout

    def _configure_run_log(self, path):
        assert path == self.layout.logs / "pipeline.log"
        self.events.append("configure-log")

    def _render_resolved_toml(self, runtime):
        self.events.append(("render", runtime))
        return "resolved lifecycle configuration\n"

    def _now(self, timezone_argument):
        assert timezone_argument is experiment_runner.timezone.utc
        self.clock_calls += 1
        value = f"time-{self.clock_calls}"
        self.events.append(("clock", value))
        return SimpleNamespace(isoformat=lambda: value)

    def _source_tree_sha256(self):
        self.events.append("source-hash")
        if self.source_error is not None:
            raise self.source_error
        return "source-hash"

    def _git_revision(self):
        self.events.append("git-revision")
        return "git-revision"

    def _environment_metadata(self):
        self.events.append("environment")
        return {"environment": "metadata"}

    def _preflight_to_dict(self):
        self.events.append("preflight-report")
        return copy.deepcopy(self.preflight_report)

    def _write_json(self, path, payload):
        snapshot = copy.deepcopy(payload)
        self.writes.append((path, snapshot))
        self.events.append(("write", snapshot["status"]))
        return runtime_write_json(path, payload)

    def _log_info(self, message, *arguments):
        self.log_calls += 1
        self.events.append(("log", message, arguments))
        if self.log_error is not None and self.log_calls == self.log_error_call:
            raise self.log_error

    def _delayed(self, function):
        self.events.append(("delayed", function))

        def bind(*args, **kwargs):
            return lambda: function(*args, **kwargs)

        return bind

    def _parallel(self, *, n_jobs, prefer, batch_size):
        self.events.append(("parallel", n_jobs, prefer, batch_size))

        def execute(tasks):
            return [task() for task in tasks]

        return execute

    def _generate_pair(
        self, runtime, sweep_index, amplitude, configuration_hash, run_log_path
    ):
        self.events.append(("generate", sweep_index, amplitude))
        assert runtime is not self.runtime_configuration
        assert configuration_hash == CONFIGURATION_HASH
        assert run_log_path == str(self.layout.logs / "pipeline.log")
        if self.generation_error is not None:
            raise self.generation_error
        rejection = (
            {
                "sweep_id": "A00000",
                "deformation_amplitude": amplitude,
                "matter_type": "hadronic",
                "stage": "stellar_sequence",
                "exception_type": "InjectedRejection",
                "reason": "injected rejection",
            }
            if self.rejected
            else None
        )
        return {
            "accepted": not self.rejected,
            "eos_frames": [object()],
            "stellar_frames": [] if self.rejected else [object()],
            "rejection": rejection,
        }

    def _merge_worker_logs(self, path):
        self.merge_calls.append(path)
        self.merge_exceptions.append(sys.exception())
        self.events.append("merge-logs")

    def _concat_frames(self, frames, columns):
        self.events.append(("concat", columns, len(frames)))
        if columns is experiment_runner.EOS_COLUMNS:
            return pd.DataFrame({"eos": [1]}) if frames else pd.DataFrame()
        assert columns is experiment_runner.STELLAR_COLUMNS
        return pd.DataFrame({"stellar": [1]}) if frames else pd.DataFrame()

    def _build_summary_table(self, stellar_curves):
        assert not stellar_curves.empty
        self.events.append("summary")
        return pd.DataFrame({"summary": [1]})

    def _run_convergence_checks(self, runtime, summary):
        self.events.append(("convergence", len(summary)))
        return self.convergence.copy()

    def _create_standard_plots(self, eos_tables, stellar_curves, summary, output):
        assert not eos_tables.empty
        self.events.append("plots")
        plot = output / "lifecycle.png"
        plot.write_text("plot", encoding="utf-8")
        return [plot]

    def _write_markdown_report(self, *args, run_status):
        report_path = args[5]
        assert report_path == self.layout.report
        self.report_statuses.append(run_status)
        self.events.append(("report", run_status))
        report_path.write_text(f"status: {run_status}\n", encoding="utf-8")

    def _build_causal_domain_table(self, eos_tables):
        assert not eos_tables.empty
        self.events.append("causal-domains")
        return SimpleNamespace(to_json=self._causal_domains_to_json)

    def _causal_domains_to_json(self, *, orient):
        assert orient == "records"
        self.events.append("causal-json")
        return '[{"domain":"core"}]'

    def _artifact_hashes(self, layout):
        assert layout is self.layout
        self.events.append("artifact-hashes")
        if self.artifact_error is not None:
            raise self.artifact_error
        return {"report.md": "artifact-hash"}

    def run(self, *, parallel_jobs=None):
        return experiment_runner.run_pair_experiment(
            self.configuration,
            parallel_jobs=parallel_jobs,
            runs_root=self.runs_root,
        )
