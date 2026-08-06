"""Shared tensor writer with aligned row provenance and train-only scaling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.ml.dataset import APPROVED_OBSERVABLE_FEATURES, dataframe_fingerprint
from src.ml.splitting import (
    SPLITS,
    attach_split_assignments,
    build_split_manifest,
    persist_shared_manifest,
)
from src.runtime import write_run_manifest


logger = logging.getLogger(__name__)
SplitTransform = Callable[[pd.DataFrame, str], pd.DataFrame]
HPO_TRAIN_FILENAME = "train_unscaled.parquet"


def build_tensor_artifacts(
    latent_frame: pd.DataFrame,
    *,
    output_dir: Path,
    data_root: Path,
    component: str,
    scaler_filename: str,
    split_transform: SplitTransform | None = None,
    split_transform_metadata: dict[str, Any] | None = None,
) -> None:
    """Create scaled tensors while preserving group and row-level audit trails."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_hash = dataframe_fingerprint(latent_frame)
    manifest = build_split_manifest(latent_frame)
    split_hash = persist_shared_manifest(manifest, data_root / "split_manifest.parquet")
    assigned = attach_split_assignments(latent_frame, manifest)

    split_frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        subset = assigned[assigned["Split"] == split].copy().reset_index(drop=True)
        if split_transform is not None:
            subset = split_transform(subset, split)
        if subset.empty:
            raise RuntimeError(f"The {split} split is empty.")
        split_frames[split] = subset

    tensor_input_hash = dataframe_fingerprint(
        pd.concat([split_frames[split] for split in SPLITS], ignore_index=True)
    )

    feature_columns = list(APPROVED_OBSERVABLE_FEATURES)
    split_frames["train"][feature_columns + ["Label"]].to_parquet(
        output_dir / HPO_TRAIN_FILENAME,
        engine="pyarrow",
        index=False,
    )
    scaler = StandardScaler()
    scaler.fit(split_frames["train"][feature_columns])
    dump(scaler, output_dir / scaler_filename)

    row_sidecars = []
    curve_sidecars = []
    for split, subset in split_frames.items():
        scaled = pd.DataFrame(
            scaler.transform(subset[feature_columns]),
            columns=feature_columns,
        )
        labels = subset["Label"].astype("int32").reset_index(drop=True)
        pd.concat([scaled, labels], axis=1).to_parquet(
            output_dir / f"{split}.parquet",
            engine="pyarrow",
            index=False,
        )

        metadata_columns = [
            "Row_ID",
            "Curve_ID",
            "Sweep_ID",
            "Group_ID",
            "Perturb_A",
            "Baseline_Name",
            "Label",
        ]
        row_sidecars.append(subset[metadata_columns].assign(Split=split))
        curve_sidecars.append(
            subset[
                [
                    "Curve_ID",
                    "Sweep_ID",
                    "Group_ID",
                    "Perturb_A",
                    "Baseline_Name",
                    "Label",
                ]
            ]
            .drop_duplicates()
            .assign(Split=split)
        )

    row_audit = pd.concat(row_sidecars, ignore_index=True)
    split_audit = pd.concat(curve_sidecars, ignore_index=True)
    if row_audit["Row_ID"].duplicated().any():
        raise RuntimeError("Row_ID is not unique across tensor splits.")
    if split_audit.groupby("Group_ID")["Split"].nunique().max() != 1:
        raise RuntimeError("Group leakage detected before tensor serialization.")
    row_audit.to_parquet(output_dir / "row_audit.parquet", engine="pyarrow", index=False)
    split_audit.to_parquet(
        output_dir / "split_audit.parquet", engine="pyarrow", index=False
    )

    split_counts = {
        split: {
            "rows": int(len(split_frames[split])),
            "curves": int(split_frames[split]["Curve_ID"].nunique()),
            "sweep_groups": int(split_frames[split]["Group_ID"].nunique()),
        }
        for split in SPLITS
    }
    write_run_manifest(
        output_dir,
        component,
        data_root,
        {
            "experiment_scope": "APR-1 versus fixed-CFL4 model-pair discrimination",
            "dataset_sha256": dataset_hash,
            "tensor_input_sha256": tensor_input_hash,
            "split_manifest_sha256": split_hash,
            "approved_features": feature_columns,
            "preprocessing": {
                "transformer": "sklearn.preprocessing.StandardScaler",
                "outer_fit_scope": "train_only",
                "scaler_filename": scaler_filename,
                "hpo_training_filename": HPO_TRAIN_FILENAME,
                "hpo_fold_transform": "fit_on_inner_fit_only",
                "split_transform": split_transform_metadata or {"name": "none"},
            },
            "mass_grid": {
                "minimum_M_sun": CONFIG["ML_MASS_GRID_MIN"],
                "maximum_M_sun": CONFIG["ML_MASS_GRID_MAX"],
                "points_per_curve": CONFIG["ML_MASS_GRID_POINTS"],
            },
            "split_counts": split_counts,
        },
    )
    logger.info("Wrote %s tensors and aligned provenance to %s", component, output_dir)
