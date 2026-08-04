"""Load training-only tensors with aligned groups for inner HPO cross-validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.config import CONFIG
from src.runtime import require_paths, tensor_lineage_metadata


def load_training_groups(
    tensor_dir: Path,
    features: list[str],
    context: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Read only training tensors and their row-aligned isolation groups."""

    lineage = tensor_lineage_metadata(tensor_dir)
    preprocessing = lineage["preprocessing"]
    train_path = tensor_dir / preprocessing["hpo_training_filename"]
    audit_path = tensor_dir / "row_audit.parquet"
    require_paths([train_path, audit_path], context)
    train = pd.read_parquet(train_path, engine="pyarrow")
    audit = pd.read_parquet(audit_path, engine="pyarrow")
    train_audit = audit[audit["Split"] == "train"].reset_index(drop=True)
    if len(train) != len(train_audit):
        raise RuntimeError(
            f"{context}: training tensor and row audit lengths do not match."
        )
    if not np.array_equal(
        train["Label"].to_numpy(dtype=int),
        train_audit["Label"].to_numpy(dtype=int),
    ):
        raise RuntimeError(f"{context}: training labels are not row-audit aligned.")
    missing = sorted(set(features + ["Label"]) - set(train.columns))
    if missing:
        raise KeyError(f"{context}: missing tensor columns {missing}.")
    return train[features], train["Label"], train_audit["Group_ID"]


def scale_inner_fold(
    features: pd.DataFrame | np.ndarray,
    fit_indices: np.ndarray,
    score_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing on an inner-fit fold and transform its score fold."""

    values = (
        features.to_numpy(dtype=float)
        if isinstance(features, pd.DataFrame)
        else np.asarray(features, dtype=float)
    )
    scaler = StandardScaler()
    fit = scaler.fit_transform(values[fit_indices])
    score = scaler.transform(values[score_indices])
    return fit, score


def grouped_cv_indices(groups: pd.Series, labels: pd.Series | np.ndarray):
    """Return deterministic, class-valid, group-isolated inner HPO folds.

    Controlled paired sweeps retain contiguous A blocks. Legacy single-class
    curve groups use stratified grouped folds instead of lexicographic blocks.
    """

    string_groups = groups.astype(str).reset_index(drop=True)
    label_values = pd.Series(np.asarray(labels, dtype=int)).reset_index(drop=True)
    if len(string_groups) != len(label_values):
        raise ValueError("Grouped HPO labels and groups are not row aligned.")
    if set(label_values.unique()) != {0, 1}:
        raise ValueError("Grouped HPO requires both binary classes in outer training.")

    group_label_sets = (
        pd.DataFrame({"Group_ID": string_groups, "Label": label_values})
        .groupby("Group_ID")["Label"]
        .agg(set)
    )
    paired_groups = bool(group_label_sets.map(lambda values: values == {0, 1}).all())
    indices: list[tuple[np.ndarray, np.ndarray]] = []

    if paired_groups:
        ordered_groups = np.asarray(sorted(group_label_sets.index))
        folds = min(CONFIG["ML_HPO_GROUP_FOLDS"], len(ordered_groups))
        if folds < 3:
            raise ValueError(
                "Grouped HPO requires at least three independent training groups."
            )
        group_array = string_groups.to_numpy()
        for score_groups in np.array_split(ordered_groups, folds):
            score_mask = np.isin(group_array, score_groups)
            indices.append((np.flatnonzero(~score_mask), np.flatnonzero(score_mask)))
    else:
        if not group_label_sets.map(len).eq(1).all():
            raise ValueError(
                "Legacy HPO groups must each contain exactly one class label."
            )
        groups_per_class = (
            group_label_sets.map(lambda values: next(iter(values))).value_counts()
        )
        folds = min(CONFIG["ML_HPO_GROUP_FOLDS"], int(groups_per_class.min()))
        if folds < 3:
            raise ValueError(
                "Legacy grouped HPO requires at least three curve groups per class."
            )
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=CONFIG["ML_RANDOM_SEED"],
        )
        placeholder = np.zeros((len(label_values), 1), dtype=float)
        indices.extend(
            splitter.split(placeholder, label_values.to_numpy(), string_groups)
        )

    group_array = string_groups.to_numpy()
    labels_array = label_values.to_numpy()
    for fit_indices, score_indices in indices:
        if len(score_indices) == 0 or len(fit_indices) == 0:
            raise RuntimeError("Grouped HPO produced an empty inner fold.")
        if set(labels_array[fit_indices]) != {0, 1} or set(
            labels_array[score_indices]
        ) != {0, 1}:
            raise RuntimeError("Grouped HPO produced a single-class inner fold.")
        if set(group_array[fit_indices]) & set(group_array[score_indices]):
            raise RuntimeError("Grouped HPO leaked a Group_ID across an inner fold.")
    return indices
