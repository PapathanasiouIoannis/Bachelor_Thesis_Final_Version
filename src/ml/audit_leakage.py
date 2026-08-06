"""Audit tensor provenance, paired sweep grouping, scaling, and row balance."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.ml.dataset import APPROVED_OBSERVABLE_FEATURES
from src.runtime import add_runtime_args, configure_runtime_from_args, require_paths, runtime_paths


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AUDIT_LEAKAGE")


class DataLeakageError(Exception):
    """Raised when a serialized tensor violates a declared isolation invariant."""


def _load_required_splits(tensor_dir: Path, label: str):
    required = [
        tensor_dir / "train.parquet",
        tensor_dir / "val.parquet",
        tensor_dir / "test.parquet",
        tensor_dir / "split_audit.parquet",
        tensor_dir / "row_audit.parquet",
    ]
    require_paths(required, f"{label} leakage audit")
    tensors = {
        split: pd.read_parquet(tensor_dir / f"{split}.parquet", engine="pyarrow")
        for split in ("train", "val", "test")
    }
    split_audit = pd.read_parquet(
        tensor_dir / "split_audit.parquet", engine="pyarrow"
    )
    row_audit = pd.read_parquet(tensor_dir / "row_audit.parquet", engine="pyarrow")
    return tensors, split_audit, row_audit


def _audit_tensor_dir(tensor_dir: Path, label: str):
    tensors, sidecar, rows = _load_required_splits(tensor_dir, label)
    features = list(APPROVED_OBSERVABLE_FEATURES)
    expected_columns = set(features + ["Label"])
    for split, tensor in tensors.items():
        if set(tensor.columns) != expected_columns:
            raise DataLeakageError(
                f"[{label}] {split} exposes non-approved or missing tensor columns: "
                f"{sorted(set(tensor.columns) ^ expected_columns)}"
            )
        metadata = rows[rows["Split"] == split].reset_index(drop=True)
        if len(metadata) != len(tensor):
            raise DataLeakageError(
                f"[{label}] {split} tensor/row-audit lengths differ: "
                f"{len(tensor)} != {len(metadata)}"
            )
        if not np.array_equal(
            tensor["Label"].to_numpy(dtype=int), metadata["Label"].to_numpy(dtype=int)
        ):
            raise DataLeakageError(f"[{label}] {split} labels are not aligned to Row_ID metadata.")

    if rows["Row_ID"].duplicated().any():
        raise DataLeakageError(f"[{label}] Row_ID values are not globally unique.")
    required_sidecar = {
        "Curve_ID",
        "Sweep_ID",
        "Group_ID",
        "Perturb_A",
        "Label",
        "Split",
    }
    if not required_sidecar.issubset(sidecar.columns):
        raise DataLeakageError(
            f"[{label}] split_audit is missing {sorted(required_sidecar - set(sidecar.columns))}."
        )

    for group_column in ("Curve_ID", "Sweep_ID", "Group_ID"):
        split_counts = sidecar.groupby(group_column)["Split"].nunique()
        leaked = split_counts[split_counts > 1]
        if not leaked.empty:
            raise DataLeakageError(
                f"[{label}] {len(leaked)} {group_column} values cross splits."
            )

    paired_labels = sidecar.groupby("Sweep_ID")["Label"].agg(lambda values: set(values))
    controlled = bool(len(paired_labels) and all(labels == {0, 1} for labels in paired_labels))
    if controlled:
        amplitudes = sidecar.groupby("Sweep_ID")["Perturb_A"].agg(["min", "max"])
        if not np.allclose(amplitudes["min"], amplitudes["max"], atol=1e-10):
            raise DataLeakageError(f"[{label}] paired Sweep_ID members do not share A.")

        ordered = (
            sidecar[["Sweep_ID", "Perturb_A", "Split"]]
            .drop_duplicates()
            .sort_values("Perturb_A")
            .reset_index(drop=True)
        )
        for split in ("train", "val", "test"):
            positions = np.flatnonzero(ordered["Split"].to_numpy() == split)
            if len(positions) == 0 or np.any(np.diff(positions) != 1):
                raise DataLeakageError(
                    f"[{label}] {split} amplitudes are not one contiguous A block."
                )

    rows_per_curve = rows.groupby("Curve_ID").size()
    expected_rows = CONFIG["ML_MASS_GRID_POINTS"]
    if not (rows_per_curve == expected_rows).all():
        raise DataLeakageError(
            f"[{label}] curves do not all contribute exactly {expected_rows} rows."
        )

    train = tensors["train"]
    train_means = train[features].mean()
    train_stds = train[features].std()
    if not (train_means.abs() < 0.05).all() or not (
        (train_stds - 1.0).abs() < 0.05
    ).all():
        raise DataLeakageError(
            f"[{label}] train-only scaler check failed: "
            f"means={train_means.to_dict()}, stds={train_stds.to_dict()}"
        )

    logger.info(
        "[%s] passed: %d rows, %d curves, %d isolation groups.",
        label,
        len(rows),
        rows["Curve_ID"].nunique(),
        rows["Group_ID"].nunique(),
    )
    return sidecar, rows


def _audit_variant_alignment(clean_result, perturbed_result) -> None:
    clean_groups, clean_rows = clean_result
    noisy_groups, noisy_rows = perturbed_result
    group_columns = ["Curve_ID", "Sweep_ID", "Group_ID", "Label", "Split"]
    clean_group_map = clean_groups[group_columns].sort_values(group_columns).reset_index(drop=True)
    noisy_group_map = noisy_groups[group_columns].sort_values(group_columns).reset_index(drop=True)
    if not clean_group_map.equals(noisy_group_map):
        raise DataLeakageError("Clean and perturbed variants do not reuse identical group splits.")

    row_columns = ["Row_ID", "Curve_ID", "Sweep_ID", "Group_ID", "Label", "Split"]
    clean_row_map = clean_rows[row_columns].sort_values("Row_ID").reset_index(drop=True)
    noisy_row_map = noisy_rows[row_columns].sort_values("Row_ID").reset_index(drop=True)
    if not clean_row_map.equals(noisy_row_map):
        raise DataLeakageError("Clean and perturbed variants do not reuse identical latent rows.")
    logger.info("Clean/perturbed split and Row_ID alignment passed.")


def audit_leakage(data_root=None, include_clean=True, include_perturbed=False) -> None:
    paths = runtime_paths(data_root)
    clean_result = None
    perturbed_result = None
    if include_clean:
        clean_result = _audit_tensor_dir(paths.clean_tensor_dir, "clean")
    if include_perturbed:
        perturbed_result = _audit_tensor_dir(paths.perturb_tensor_dir, "perturbed")
    if clean_result is not None and perturbed_result is not None:
        _audit_variant_alignment(clean_result, perturbed_result)
    logger.info("AUDIT SUMMARY: requested tensor isolation checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit ML tensors for split leakage.")
    add_runtime_args(parser)
    parser.add_argument("--clean-only", action="store_true", help="audit only clean tensors")
    parser.add_argument(
        "--perturbed-only", action="store_true", help="audit only perturbed tensors"
    )
    parser.add_argument(
        "--include-perturbed", action="store_true", help="audit clean and perturbed tensors"
    )
    args = parser.parse_args()
    configure_runtime_from_args(args)

    if args.clean_only and args.perturbed_only:
        raise SystemExit("--clean-only and --perturbed-only are mutually exclusive.")
    audit_leakage(
        args.data_root,
        include_clean=not args.perturbed_only,
        include_perturbed=args.include_perturbed or args.perturbed_only,
    )
