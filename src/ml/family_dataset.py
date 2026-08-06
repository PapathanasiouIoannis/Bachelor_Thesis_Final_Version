"""One-sample-per-curve representation for the multi-family pilot."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.ml.dataset import engineer_observables, resample_curves_to_common_mass_grid


CURVE_METADATA_COLUMNS = (
    "Curve_ID",
    "EoS_ID",
    "Variant_ID",
    "Sweep_ID",
    "Perturb_A",
    "Perturb_eps0",
    "Perturb_sigma",
    "Baseline_Name",
    "Family_Group_ID",
    "Parameter_Block_ID",
    "Model_Superfamily_ID",
    "Generation_Profile",
    "Gaussian_Grid_Max_Weight",
    "Primary_Citation_Available",
    "Exact_Formula_Primary_Verified",
    "Label",
)


def observable_mass_grid() -> np.ndarray:
    return np.linspace(
        CONFIG["ML_MASS_GRID_MIN"],
        CONFIG["ML_MASS_GRID_MAX"],
        CONFIG["ML_MASS_GRID_POINTS"],
    )


def _mass_token(mass: float) -> str:
    return f"{mass:.2f}".replace(".", "p")


def radius_feature_columns() -> list[str]:
    return [f"Radius_M{_mass_token(mass)}" for mass in observable_mass_grid()]


def tidal_feature_columns() -> list[str]:
    return [f"LogLambda_M{_mass_token(mass)}" for mass in observable_mass_grid()]


def curve_feature_columns(feature_set: str = "MRL") -> list[str]:
    """Return the explicit observable allowlist for a curve-level model."""

    normalized = feature_set.upper()
    if normalized == "MR":
        return radius_feature_columns()
    if normalized == "MRL":
        return [*radius_feature_columns(), *tidal_feature_columns()]
    raise ValueError(f"Unknown curve feature set {feature_set!r}; expected 'MR' or 'MRL'.")


def _constant_curve_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(CURVE_METADATA_COLUMNS) - set(frame.columns))
    if missing:
        raise KeyError(f"Family physics dataset is missing metadata: {missing}")
    if frame["Curve_ID"].isna().any():
        raise ValueError("Curve_ID contains missing values.")

    grouped = frame.groupby("Curve_ID", sort=True, dropna=False)
    varying = []
    for column in CURVE_METADATA_COLUMNS[1:]:
        counts = grouped[column].nunique(dropna=False)
        if (counts != 1).any():
            varying.append(column)
    if varying:
        raise ValueError(f"Curve-level metadata is non-constant: {varying}")

    metadata = grouped[list(CURVE_METADATA_COLUMNS[1:])].first().reset_index()
    metadata.insert(0, "Sample_ID", metadata["Curve_ID"].astype(str))
    if metadata["Sample_ID"].duplicated().any():
        raise RuntimeError("Curve-to-sample conversion produced duplicate Sample_ID values.")
    return metadata


def build_curve_samples(physics_rows: pd.DataFrame) -> pd.DataFrame:
    """Flatten each complete M-R-Lambda curve into exactly one ML sample.

    The common mass grid is implicit in the feature names. Mass itself is not a
    feature because it is identical for every sample.
    """

    metadata = _constant_curve_metadata(physics_rows)
    engineered = engineer_observables(physics_rows)
    resampled = resample_curves_to_common_mass_grid(engineered)
    resampled = resampled.sort_values(["Curve_ID", "Mass"]).reset_index(drop=True)

    expected_grid = observable_mass_grid()
    counts = resampled.groupby("Curve_ID").size()
    if not counts.eq(len(expected_grid)).all():
        raise RuntimeError("A resampled curve does not contain the complete mass grid.")
    observed_grids = resampled.groupby("Curve_ID")["Mass"].apply(
        lambda values: np.asarray(values, dtype=float)
    )
    if not all(np.allclose(values, expected_grid, atol=1e-12) for values in observed_grids):
        raise RuntimeError("Resampled curves do not share the exact locked mass grid.")

    resampled["Mass_Index"] = resampled.groupby("Curve_ID").cumcount()
    radii = resampled.pivot(index="Curve_ID", columns="Mass_Index", values="Radius")
    tides = resampled.pivot(
        index="Curve_ID", columns="Mass_Index", values="log10_Lambda"
    )
    radii.columns = radius_feature_columns()
    tides.columns = tidal_feature_columns()
    observables = pd.concat([radii, tides], axis=1).reset_index()

    samples = metadata.merge(
        observables,
        on="Curve_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(samples) != physics_rows["Curve_ID"].nunique():
        raise RuntimeError("Curve-to-sample conversion lost one or more curves.")
    features = curve_feature_columns("MRL")
    if samples[features].isna().any().any():
        raise RuntimeError("Curve-level observables contain missing values.")
    if not np.isfinite(samples[features].to_numpy(dtype=float)).all():
        raise RuntimeError("Curve-level observables contain non-finite values.")
    if set(samples["Label"].astype(int).unique()) != {0, 1}:
        raise ValueError("Curve-level dataset must contain both matter classes.")

    samples[features] = samples[features].astype("float32")
    samples["Label"] = samples["Label"].astype("int8")
    return samples.sort_values("Sample_ID").reset_index(drop=True)


def curve_dataset_fingerprint(
    samples: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
) -> str:
    """Fingerprint curve observables, labels, and split-critical provenance."""

    features = list(feature_columns or curve_feature_columns("MRL"))
    columns = [
        "Sample_ID",
        "Curve_ID",
        "EoS_ID",
        "Variant_ID",
        "Sweep_ID",
        "Family_Group_ID",
        "Parameter_Block_ID",
        "Generation_Profile",
        "Perturb_A",
        "Label",
        *features,
    ]
    missing = sorted(set(columns) - set(samples.columns))
    if missing:
        raise KeyError(f"Cannot fingerprint curve dataset; missing {missing}")
    canonical = samples[columns].sort_values("Sample_ID").reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()
