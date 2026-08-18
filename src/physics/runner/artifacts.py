"""Resolved-configuration and artifact-table helpers for pair experiments."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd


def render_resolved_toml(
    runtime: dict[str, Any],
    *,
    value_renderer: Callable[[Any], str],
) -> str:
    """Render the known resolved pair schema without a third-party TOML writer."""

    lines = [
        "# Resolved EoS Lab configuration. Generated automatically.",
        f"schema_version = {int(runtime['schema_version'])}",
        f"experiment_name = {value_renderer(runtime['experiment_name'])}",
        f"workflow = {value_renderer(runtime['workflow'])}",
        f"mode = {value_renderer(runtime['mode'])}",
    ]
    for section in (
        "hadronic_eos",
        "quark_eos",
        "deformation",
        "physical_requirements",
        "numerical_settings",
        "execution",
    ):
        lines.extend(("", f"[{section}]"))
        for key, value in runtime[section].items():
            lines.append(f"{key} = {value_renderer(value)}")
    lines.extend(
        (
            "",
            "# Derived amplitude values, catalog identifiers, provenance, and the",
            "# permitted interpretation are recorded in run_manifest.json.",
        )
    )
    return "\n".join(lines) + "\n"


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("Resolved TOML cannot contain a non-finite number.")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported resolved TOML value: {type(value).__name__}")


def concat_frames(frames: list[pd.DataFrame], columns) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).loc[:, list(columns)]


def artifact_hashes(
    layout: Any,
    *,
    file_hasher: Callable[[Any], str],
) -> dict[str, str]:
    paths = [
        layout.resolved_config,
        layout.data / "eos_tables.parquet",
        layout.data / "stellar_curves.parquet",
        layout.tables / "eos_summary.csv",
        layout.tables / "rejections.csv",
        layout.tables / "convergence.csv",
        layout.report,
        *sorted(layout.plots.glob("*.png")),
    ]
    return {
        str(path.relative_to(layout.root).as_posix()): file_hasher(path)
        for path in paths
        if path.is_file()
    }


__all__ = [
    "artifact_hashes",
    "concat_frames",
    "render_resolved_toml",
    "toml_value",
]
