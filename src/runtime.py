"""Runtime path and execution helpers shared by the pipeline entrypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIX_MANIFEST_VERSION = "controlled-eos-paired-sweep-ml-integrity-v3"


def _source_tree_sha256() -> str:
    """Fingerprint executable Python sources that define generated artifacts."""

    candidates = list((PROJECT_ROOT / "src").rglob("*.py"))
    candidates.extend((PROJECT_ROOT / "framework").rglob("*.py"))
    candidates.extend(PROJECT_ROOT.glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda item: item.as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        contents = path.read_bytes()
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _config_sha256() -> str:
    from src.config import CONFIG

    payload = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{context} missing required manifest: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{context} manifest is not a JSON object: {path}")
    return payload


def _tensor_lineage(tensor_dir: Path) -> dict[str, Any]:
    manifest_path = tensor_dir / "run_manifest.json"
    manifest = _read_json_object(manifest_path, "Tensor lineage")
    required = (
        "dataset_sha256",
        "tensor_input_sha256",
        "split_manifest_sha256",
        "approved_features",
        "preprocessing",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"Tensor manifest {manifest_path} is missing {missing}.")

    preprocessing = manifest["preprocessing"]
    if not isinstance(preprocessing, dict):
        raise ValueError(f"Tensor manifest {manifest_path} has invalid preprocessing metadata.")
    preprocessing_required = ("scaler_filename", "hpo_training_filename")
    preprocessing_missing = [
        key for key in preprocessing_required if key not in preprocessing
    ]
    if preprocessing_missing:
        raise ValueError(
            f"Tensor manifest {manifest_path} preprocessing is missing "
            f"{preprocessing_missing}."
        )

    dynamic_names = [
        str(preprocessing["scaler_filename"]),
        str(preprocessing["hpo_training_filename"]),
    ]
    if any(Path(name).name != name for name in dynamic_names):
        raise ValueError(
            f"Tensor manifest {manifest_path} contains a non-local artifact filename."
        )
    artifact_names = [
        "run_manifest.json",
        "train.parquet",
        "val.parquet",
        "test.parquet",
        "row_audit.parquet",
        "split_audit.parquet",
        *dynamic_names,
    ]
    artifact_paths = [tensor_dir / name for name in artifact_names]
    missing_artifacts = [str(path) for path in artifact_paths if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(
            "Tensor lineage is missing required artifact(s): "
            + ", ".join(missing_artifacts)
        )

    return {
        **{key: manifest[key] for key in required},
        "artifact_sha256": {
            name: _file_sha256(tensor_dir / name) for name in artifact_names
        },
    }


def artifact_lineage_path(artifact_path: str | os.PathLike) -> Path:
    artifact = Path(artifact_path)
    return artifact.with_name(f"{artifact.stem}.manifest.json")


def tensor_lineage_metadata(tensor_dir: str | os.PathLike) -> dict[str, Any]:
    """Expose the exact dataset/split/features identity for downstream manifests."""

    return _tensor_lineage(Path(tensor_dir))


def _selected_feature_lineage(
    tensor_lineage: dict[str, Any],
    selected_features: Iterable[str] | None,
) -> list[str]:
    approved = [str(feature) for feature in tensor_lineage["approved_features"]]
    selected = (
        approved
        if selected_features is None
        else [str(feature) for feature in selected_features]
    )
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Artifact lineage requires a non-empty unique feature list.")
    if any(feature not in approved for feature in selected):
        raise ValueError(
            f"Selected artifact features {selected} are not a subset of {approved}."
        )
    return selected


def write_artifact_lineage(
    artifact_path: str | os.PathLike,
    tensor_dir: str | os.PathLike,
    component: str,
    selected_features: Iterable[str] | None = None,
) -> Path:
    """Bind an HPO artifact to exact tensors, config, code, and file contents."""

    artifact = Path(artifact_path)
    if not artifact.exists():
        raise FileNotFoundError(f"Cannot fingerprint missing artifact: {artifact}")
    tensor_lineage = _tensor_lineage(Path(tensor_dir))
    payload = {
        "component": component,
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": _file_sha256(artifact),
        "tensor_lineage": tensor_lineage,
        "selected_features": _selected_feature_lineage(
            tensor_lineage, selected_features
        ),
        "source_tree_sha256": _source_tree_sha256(),
        "config_sha256": _config_sha256(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    lineage_path = artifact_lineage_path(artifact)
    with open(lineage_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return lineage_path


def require_artifact_lineage(
    artifact_path: str | os.PathLike,
    tensor_dir: str | os.PathLike,
    context: str,
    *,
    component: str | None = None,
    selected_features: Iterable[str] | None = None,
) -> None:
    """Fail closed if an HPO artifact is stale, edited, or from other tensors."""

    artifact = Path(artifact_path)
    if not artifact.exists():
        raise FileNotFoundError(f"{context} missing required artifact: {artifact}")
    lineage_path = artifact_lineage_path(artifact)
    lineage = _read_json_object(lineage_path, context)
    tensor_lineage = _tensor_lineage(Path(tensor_dir))
    expected = {
        "artifact_sha256": _file_sha256(artifact),
        "tensor_lineage": tensor_lineage,
        "selected_features": _selected_feature_lineage(
            tensor_lineage, selected_features
        ),
        "source_tree_sha256": _source_tree_sha256(),
        "config_sha256": _config_sha256(),
    }
    if component is not None:
        expected["component"] = component
    mismatches = [key for key, value in expected.items() if lineage.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"{context} refused stale or incompatible artifact {artifact}; "
            f"lineage mismatch in {mismatches}. Re-run HPO."
        )


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


def require_test_diagnostics_authorized() -> None:
    """Prevent exploratory utilities from repeatedly consuming held-out labels."""

    allowed = os.environ.get("THESIS_ALLOW_TEST_DIAGNOSTICS", "").strip().lower()
    if allowed not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "Held-out test diagnostics are locked. Run them only through the master "
            "pipeline with --run-test-diagnostics for a declared final analysis."
        )


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
        "source_tree_sha256": _source_tree_sha256(),
        "config_sha256": _config_sha256(),
        "git_revision": _git_revision(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload.update(metadata)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return manifest_path
