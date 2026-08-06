"""Deterministic group-safe train/validation/test assignment."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG


SPLITS = ("train", "val", "test")


def _split_counts(total: int) -> tuple[int, int, int]:
    if total < 5:
        raise ValueError("At least five independent groups are required for a 3-way split.")
    validation = max(1, int(round(total * 0.10)))
    test = max(1, int(round(total * 0.10)))
    train = total - validation - test
    if train < 3:
        raise ValueError("Split leaves fewer than three training groups.")
    return train, validation, test


def _is_paired_sweep(frame: pd.DataFrame) -> bool:
    label_sets = frame.groupby("Sweep_ID")["Label"].agg(lambda values: set(values))
    curves_per_label = frame.groupby(["Sweep_ID", "Label"])["Curve_ID"].nunique()
    return bool(
        len(label_sets) >= 5
        and all(labels == {0, 1} for labels in label_sets)
        and (curves_per_label == 1).all()
        and frame["Perturb_A"].notna().all()
    )


def _paired_blocked_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    group_table = (
        frame.groupby("Sweep_ID", as_index=False)
        .agg(Perturb_A=("Perturb_A", "first"))
        .sort_values(["Perturb_A", "Sweep_ID"])
        .reset_index(drop=True)
    )
    amplitude_spread = frame.groupby("Sweep_ID")["Perturb_A"].agg(
        lambda values: float(values.max() - values.min())
    )
    if not np.allclose(amplitude_spread, 0.0, atol=1e-10):
        raise ValueError("Paired Sweep_ID members do not share the same amplitude A.")

    train_count, validation_count, test_count = _split_counts(len(group_table))
    assignments = np.full(len(group_table), "train", dtype=object)
    assignments[:validation_count] = "val"
    assignments[validation_count + train_count :] = "test"
    group_table["Split"] = assignments
    group_table["Group_ID"] = group_table["Sweep_ID"].astype(str)
    group_table["Split_Strategy"] = "paired_contiguous_A_blocks"
    assert int((assignments == "test").sum()) == test_count
    return group_table[
        ["Group_ID", "Sweep_ID", "Perturb_A", "Split", "Split_Strategy"]
    ]


def _legacy_stratified_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    group_table = (
        frame.groupby("Curve_ID", as_index=False)
        .agg(
            Label=("Label", "first"),
            Sweep_ID=("Sweep_ID", "first"),
            Perturb_A=("Perturb_A", "first"),
        )
        .sort_values("Curve_ID")
        .reset_index(drop=True)
    )
    if frame.groupby("Curve_ID")["Label"].nunique().max() != 1:
        raise ValueError("A Curve_ID contains multiple class labels.")

    rng = np.random.default_rng(CONFIG["ML_RANDOM_SEED"])
    split_by_curve: dict[str, str] = {}
    for label, label_groups in group_table.groupby("Label", sort=True):
        del label
        curve_ids = label_groups["Curve_ID"].astype(str).to_numpy()
        train_count, validation_count, _ = _split_counts(len(curve_ids))
        curve_ids = rng.permutation(curve_ids)
        for curve_id in curve_ids[:validation_count]:
            split_by_curve[curve_id] = "val"
        for curve_id in curve_ids[validation_count : validation_count + train_count]:
            split_by_curve[curve_id] = "train"
        for curve_id in curve_ids[validation_count + train_count :]:
            split_by_curve[curve_id] = "test"

    group_table["Group_ID"] = group_table["Curve_ID"].astype(str)
    group_table["Split"] = group_table["Group_ID"].map(split_by_curve)
    group_table["Split_Strategy"] = "legacy_label_stratified_curve_groups"
    return group_table[
        ["Group_ID", "Sweep_ID", "Perturb_A", "Split", "Split_Strategy"]
    ]


def build_split_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Build paired A-block splits, with a stratified legacy fallback."""

    manifest = (
        _paired_blocked_manifest(frame)
        if _is_paired_sweep(frame)
        else _legacy_stratified_manifest(frame)
    )
    if manifest["Group_ID"].duplicated().any():
        raise RuntimeError("Split manifest contains duplicate Group_ID assignments.")
    if set(manifest["Split"]) != set(SPLITS):
        raise RuntimeError("Split manifest must contain train, val, and test groups.")
    return manifest.sort_values("Group_ID").reset_index(drop=True)


def manifest_fingerprint(manifest: pd.DataFrame) -> str:
    canonical = manifest.sort_values("Group_ID").reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def persist_shared_manifest(manifest: pd.DataFrame, path: Path) -> str:
    """Persist the deterministic assignment used by every data variant."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path, engine="pyarrow")
        if manifest_fingerprint(existing) != manifest_fingerprint(manifest):
            raise RuntimeError(
                f"Existing split manifest {path} does not match this dataset. "
                "Regenerate into a new data root or remove the stale manifest explicitly."
            )
    else:
        manifest.to_parquet(path, engine="pyarrow", index=False)
    return manifest_fingerprint(manifest)


def attach_split_assignments(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one group assignment to every row and verify no provenance loss."""

    paired = bool((manifest["Split_Strategy"] == "paired_contiguous_A_blocks").all())
    frame = frame.copy()
    frame["Group_ID"] = (
        frame["Sweep_ID"].astype(str) if paired else frame["Curve_ID"].astype(str)
    )
    split_map = manifest.set_index("Group_ID")["Split"]
    frame["Split"] = frame["Group_ID"].map(split_map)
    if frame["Split"].isna().any():
        missing = frame.loc[frame["Split"].isna(), "Group_ID"].unique()[:10]
        raise RuntimeError(f"Rows have no split assignment. Sample groups: {list(missing)}")
    if frame.groupby("Group_ID")["Split"].nunique().max() != 1:
        raise RuntimeError("A Group_ID crosses train/validation/test boundaries.")
    return frame
