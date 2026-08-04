"""Runtime path and execution helpers shared by the pipeline entrypoints."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIX_MANIFEST_VERSION = "runtime-readiness-physics-ml-integrity-v1"


def _as_project_path(path: str | os.PathLike | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return _as_project_path(value) if value else None


def resolve_data_root(data_root: str | os.PathLike | None = None) -> Path:
    """Resolve the active data profile.

    Priority:
    1. explicit argument
    2. THESIS_DATA_ROOT
    3. legacy root data/ if it already contains generated runtime datasets
    4. data/1K profile
    """

    explicit = _as_project_path(data_root) or _env_path("THESIS_DATA_ROOT")
    if explicit is not None:
        return explicit

    root_data = PROJECT_ROOT / "data"
    legacy_markers = [
        root_data / "ml_ready_hadronic",
        root_data / "ml_ready_quark",
        root_data / "ml_tensors",
        root_data / "ml_tensors_perturb",
        root_data / "physics_test_dataset.parquet",
    ]
    if any(marker.exists() for marker in legacy_markers):
        return root_data.resolve()

    return (root_data / "1K").resolve()


@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    hadronic_ready_dir: Path
    quark_ready_dir: Path
    clean_tensor_dir: Path
    perturb_tensor_dir: Path
    physics_dataset: Path
    physics_temp_dir: Path
    outputs_root: Path
    outputs_perturb_root: Path
    plots_root: Path
    plots_perturb_root: Path


def runtime_paths(data_root: str | os.PathLike | None = None) -> RuntimePaths:
    data = resolve_data_root(data_root)
    legacy_root = data == (PROJECT_ROOT / "data").resolve()

    outputs_root = _env_path("THESIS_OUTPUT_ROOT")
    outputs_perturb_root = _env_path("THESIS_OUTPUTS_PERTURB_ROOT")
    plots_root = _env_path("THESIS_PLOTS_ROOT")
    plots_perturb_root = _env_path("THESIS_PLOTS_PERTURB_ROOT")

    if outputs_root is None:
        outputs_root = PROJECT_ROOT / "outputs" if legacy_root else data / "outputs"
    if outputs_perturb_root is None:
        outputs_perturb_root = PROJECT_ROOT / "outputs_perturb" if legacy_root else data / "outputs_perturb"
    if plots_root is None:
        plots_root = PROJECT_ROOT / "plots" if legacy_root else data / "plots"
    if plots_perturb_root is None:
        plots_perturb_root = PROJECT_ROOT / "plots_perturb" if legacy_root else data / "plots_perturb"

    return RuntimePaths(
        data_root=data,
        hadronic_ready_dir=data / "ml_ready_hadronic",
        quark_ready_dir=data / "ml_ready_quark",
        clean_tensor_dir=data / "ml_tensors",
        perturb_tensor_dir=data / "ml_tensors_perturb",
        physics_dataset=data / "physics_test_dataset.parquet",
        physics_temp_dir=data / "physics_temp",
        outputs_root=outputs_root.resolve(),
        outputs_perturb_root=outputs_perturb_root.resolve(),
        plots_root=plots_root.resolve(),
        plots_perturb_root=plots_perturb_root.resolve(),
    )


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=None, help="Data profile root, e.g. data/1K or data/10K.")
    parser.add_argument("--output-root", default=None, help="Clean ML artifact root.")
    parser.add_argument("--outputs-perturb-root", default=None, help="Perturbed ML artifact root.")
    parser.add_argument("--plots-root", default=None, help="Clean/physics plot root.")
    parser.add_argument("--plots-perturb-root", default=None, help="Perturbed plot root.")


def configure_runtime_from_args(args: argparse.Namespace) -> RuntimePaths:
    if getattr(args, "data_root", None):
        os.environ["THESIS_DATA_ROOT"] = str(_as_project_path(args.data_root))
    if getattr(args, "output_root", None):
        os.environ["THESIS_OUTPUT_ROOT"] = str(_as_project_path(args.output_root))
    if getattr(args, "outputs_perturb_root", None):
        os.environ["THESIS_OUTPUTS_PERTURB_ROOT"] = str(_as_project_path(args.outputs_perturb_root))
    if getattr(args, "plots_root", None):
        os.environ["THESIS_PLOTS_ROOT"] = str(_as_project_path(args.plots_root))
    if getattr(args, "plots_perturb_root", None):
        os.environ["THESIS_PLOTS_PERTURB_ROOT"] = str(_as_project_path(args.plots_perturb_root))
    if getattr(args, "fast", False) or getattr(args, "smoke_test", False):
        os.environ["THESIS_FAST"] = "1"
    return runtime_paths()


def require_paths(paths: Iterable[Path], context: str) -> None:
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(f"{context} missing required artifact(s):\n  - {formatted}")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def fast_enabled() -> bool:
    return os.environ.get("THESIS_FAST", "").strip().lower() in {"1", "true", "yes", "on"}


def optuna_trials(default: int) -> int:
    if fast_enabled():
        return env_int("THESIS_OPTUNA_TRIALS", min(default, 2))
    return env_int("THESIS_OPTUNA_TRIALS", default)


def train_epochs(default: int) -> int:
    if fast_enabled():
        return env_int("THESIS_EPOCHS", min(default, 3))
    return env_int("THESIS_EPOCHS", default)


def xgb_device_params(require_cuda: bool = False) -> dict:
    """Return XGBoost device kwargs.

    CPU is the default. CUDA is used only when explicitly requested and available.
    """

    requested = require_cuda or os.environ.get("THESIS_XGB_DEVICE", "").strip().lower() == "cuda"
    if not requested:
        return {}

    try:
        import torch
    except Exception as exc:  # pragma: no cover - defensive import guard
        raise RuntimeError("CUDA XGBoost was requested, but torch is unavailable for CUDA detection.") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA XGBoost was requested, but no CUDA device is available.")

    return {"device": "cuda"}


def write_run_manifest(
    output_dir: str | os.PathLike,
    component: str,
    data_root: str | os.PathLike | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / "run_manifest.json"
    payload = {
        "component": component,
        "data_root": str(resolve_data_root(data_root)),
        "fix_manifest_version": FIX_MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload.update(metadata)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return manifest_path
