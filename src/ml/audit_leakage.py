import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.runtime import add_runtime_args, configure_runtime_from_args, require_paths, runtime_paths


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AUDIT_LEAKAGE")


class DataLeakageError(Exception):
    pass


def _load_required_splits(tensor_dir: Path, label: str):
    required = [tensor_dir / "train.parquet", tensor_dir / "val.parquet", tensor_dir / "test.parquet", tensor_dir / "split_audit.parquet"]
    require_paths(required, f"{label} leakage audit")
    logger.info(f"[{label}] Loading datasets from {tensor_dir}...")
    return (
        pd.read_parquet(tensor_dir / "train.parquet", engine="pyarrow"),
        pd.read_parquet(tensor_dir / "val.parquet", engine="pyarrow"),
        pd.read_parquet(tensor_dir / "test.parquet", engine="pyarrow"),
        pd.read_parquet(tensor_dir / "split_audit.parquet", engine="pyarrow"),
    )


def _audit_tensor_dir(tensor_dir: Path, label: str) -> None:
    train_df, val_df, test_df, sidecar = _load_required_splits(tensor_dir, label)

    features = ["Mass", "Radius", "log10_Lambda"]
    missing_features = sorted(set(features + ["Label"]) - set(train_df.columns))
    if missing_features:
        raise DataLeakageError(f"[{label}] Missing required tensor columns: {missing_features}")

    logger.info(f"[{label}] Performing row-level intersection audit across Train, Val, Test...")
    train_hashes = set(pd.util.hash_pandas_object(train_df[features], index=False))
    val_hashes = set(pd.util.hash_pandas_object(val_df[features], index=False))
    test_hashes = set(pd.util.hash_pandas_object(test_df[features], index=False))

    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise DataLeakageError(
            f"[{label}] Row leakage detected: "
            f"Train-Val={len(train_val_overlap)}, Train-Test={len(train_test_overlap)}, Val-Test={len(val_test_overlap)}"
        )
    logger.info(f"[{label}] Row-level intersection audit passed.")

    logger.info(f"[{label}] Performing Curve_ID group disjointness audit...")
    required_sidecar_cols = {"Curve_ID", "Split"}
    if not required_sidecar_cols.issubset(sidecar.columns):
        raise DataLeakageError(f"[{label}] split_audit.parquet must contain {sorted(required_sidecar_cols)}.")

    sidecar = sidecar.drop_duplicates(subset=["Curve_ID", "Split"])
    split_counts = sidecar.groupby("Curve_ID")["Split"].nunique()
    leaked_ids = split_counts[split_counts > 1]
    if not leaked_ids.empty:
        sample = list(leaked_ids.head(10).index)
        raise DataLeakageError(f"[{label}] Curve_ID groups cross splits: {len(leaked_ids)} leaked curves. Sample: {sample}")

    train_groups = set(sidecar.loc[sidecar["Split"] == "train", "Curve_ID"])
    val_groups = set(sidecar.loc[sidecar["Split"] == "val", "Curve_ID"])
    test_groups = set(sidecar.loc[sidecar["Split"] == "test", "Curve_ID"])
    logger.info(
        f"[{label}] Curve_ID disjointness passed: "
        f"{len(train_groups)} train / {len(val_groups)} val / {len(test_groups)} test groups."
    )

    logger.info(f"[{label}] Checking duplicate feature rows with conflicting labels...")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    conflicting = all_df.groupby(features)["Label"].nunique()
    conflicting = conflicting[conflicting > 1]
    if len(conflicting) > 0:
        raise DataLeakageError(f"[{label}] Found {len(conflicting)} duplicate feature rows with conflicting labels.")
    logger.info(f"[{label}] Label consistency passed.")

    logger.info(f"[{label}] Verifying scaler appears fit on training data only...")
    train_means = train_df[features].mean()
    train_stds = train_df[features].std()
    mean_ok = (train_means.abs() < 0.05).all()
    std_ok = ((train_stds - 1.0).abs() < 0.05).all()
    if not mean_ok or not std_ok:
        raise DataLeakageError(
            f"[{label}] Scaler verification failed. "
            f"means={train_means.to_dict()}, stds={train_stds.to_dict()}"
        )

    logger.info(
        f"[{label}] Scaler verification passed: "
        f"max mean={float(train_means.abs().max()):.6f}, "
        f"max std delta={float((train_stds - 1.0).abs().max()):.6f}."
    )


def audit_leakage(data_root=None, include_clean=True, include_perturbed=False) -> None:
    paths = runtime_paths(data_root)
    if include_clean:
        _audit_tensor_dir(paths.clean_tensor_dir, "clean")
    if include_perturbed:
        _audit_tensor_dir(paths.perturb_tensor_dir, "perturbed")
    logger.info("AUDIT SUMMARY: No data leakage detected in requested tensor sets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit ML tensors for split leakage.")
    add_runtime_args(parser)
    parser.add_argument("--clean-only", action="store_true", help="audit only clean tensors")
    parser.add_argument("--perturbed-only", action="store_true", help="audit only perturbed tensors")
    parser.add_argument("--include-perturbed", action="store_true", help="audit clean and perturbed tensors")
    args = parser.parse_args()
    configure_runtime_from_args(args)

    if args.clean_only and args.perturbed_only:
        raise SystemExit("--clean-only and --perturbed-only are mutually exclusive.")
    audit_leakage(
        args.data_root,
        include_clean=not args.perturbed_only,
        include_perturbed=args.include_perturbed or args.perturbed_only,
    )
