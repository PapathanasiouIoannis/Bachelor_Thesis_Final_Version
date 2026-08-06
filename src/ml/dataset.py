"""Observable-only dataset preparation shared by clean and noisy ML pipelines."""

from __future__ import annotations

import glob
import hashlib
import logging
import os

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from src.config import CONFIG


logger = logging.getLogger(__name__)
APPROVED_OBSERVABLE_FEATURES = ("Mass", "Radius", "log10_Lambda")


def _load_parquet_dir(directory: str, label: int, class_name: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No {class_name} parquet files found in {directory}")
    frame = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in files],
        ignore_index=True,
    )
    frame["Label"] = label
    return frame


def load_physics_rows(hadronic_dir: str, quark_dir: str) -> pd.DataFrame:
    """Load class files while retaining provenance strictly as metadata."""

    hadronic = _load_parquet_dir(hadronic_dir, 0, "hadronic")
    quark = _load_parquet_dir(quark_dir, 1, "quark")
    frame = pd.concat([hadronic, quark], ignore_index=True)
    if "Sweep_ID" not in frame.columns:
        logger.warning(
            "Legacy dataset has no Sweep_ID; split grouping will fall back to Curve_ID."
        )
        frame["Sweep_ID"] = frame["Curve_ID"].astype(str)
    if "Perturb_A" not in frame.columns:
        frame["Perturb_A"] = np.nan
    return frame


def engineer_observables(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the approved feature allowlist without global deduplication."""

    frame = frame.copy()
    required = {"Mass", "Radius", "Curve_ID", "Sweep_ID", "Label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Physics dataset is missing required columns: {missing}")

    if "Lambda" in frame.columns:
        frame = frame[frame["Lambda"] > 0.0].copy()
        frame["log10_Lambda"] = np.log10(frame["Lambda"])
    elif "LogLambda" in frame.columns:
        frame["log10_Lambda"] = frame["LogLambda"]
    elif "log10_Lambda" not in frame.columns:
        raise KeyError("Physics dataset has no Lambda, LogLambda, or log10_Lambda column.")

    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    frame.dropna(
        subset=[*APPROVED_OBSERVABLE_FEATURES, "Curve_ID", "Sweep_ID", "Label"],
        inplace=True,
    )
    frame = frame[(frame["Mass"] > 0.0) & (frame["Radius"] > 0.0)].copy()
    if frame.empty:
        raise ValueError("No finite positive observable rows remain after cleaning.")
    return frame


def _constant_metadata(curve: pd.DataFrame, column: str):
    values = curve[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(
            f"Curve {curve['Curve_ID'].iloc[0]} has non-constant metadata column {column}."
        )
    return values.iloc[0]


def resample_curves_to_common_mass_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every EoS exactly one row at each common observable mass."""

    mass_grid = np.linspace(
        CONFIG["ML_MASS_GRID_MIN"],
        CONFIG["ML_MASS_GRID_MAX"],
        CONFIG["ML_MASS_GRID_POINTS"],
    )
    records: list[dict] = []
    failures: list[str] = []

    for curve_id, curve in frame.groupby("Curve_ID", sort=True):
        curve = curve.sort_values("Mass").drop_duplicates(subset=["Mass"], keep="last")
        if len(curve) < 4:
            failures.append(f"{curve_id}: fewer than four unique mass points")
            continue
        mass = curve["Mass"].to_numpy(dtype=float)
        if mass[0] > mass_grid[0] or mass[-1] < mass_grid[-1]:
            failures.append(
                f"{curve_id}: mass support [{mass[0]:.4g}, {mass[-1]:.4g}] "
                f"does not cover [{mass_grid[0]:.4g}, {mass_grid[-1]:.4g}]"
            )
            continue

        radius_interp = PchipInterpolator(
            mass, curve["Radius"].to_numpy(dtype=float), extrapolate=False
        )
        lambda_interp = PchipInterpolator(
            mass, curve["log10_Lambda"].to_numpy(dtype=float), extrapolate=False
        )
        label = int(_constant_metadata(curve, "Label"))
        sweep_id = str(_constant_metadata(curve, "Sweep_ID"))
        amplitude = (
            float(_constant_metadata(curve, "Perturb_A"))
            if curve["Perturb_A"].notna().all()
            else np.nan
        )
        baseline_name = (
            str(_constant_metadata(curve, "Baseline_Name"))
            if "Baseline_Name" in curve.columns
            else "unknown"
        )

        radii = np.asarray(radius_interp(mass_grid), dtype=float)
        log_lambdas = np.asarray(lambda_interp(mass_grid), dtype=float)
        if np.any(~np.isfinite(radii)) or np.any(~np.isfinite(log_lambdas)):
            failures.append(f"{curve_id}: interpolation produced non-finite observables")
            continue
        for mass_index, (mass_value, radius, log_lambda) in enumerate(
            zip(mass_grid, radii, log_lambdas, strict=True)
        ):
            records.append(
                {
                    "Row_ID": f"{curve_id}:M{mass_index:03d}",
                    "Curve_ID": str(curve_id),
                    "Sweep_ID": sweep_id,
                    "Perturb_A": amplitude,
                    "Baseline_Name": baseline_name,
                    "Mass": float(mass_value),
                    "Radius": float(radius),
                    "log10_Lambda": float(log_lambda),
                    "Label": label,
                }
            )

    if failures:
        sample = "; ".join(failures[:8])
        raise ValueError(
            f"Common-mass resampling rejected {len(failures)} curves. {sample}"
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError("Common-mass resampling produced no rows.")
    expected = CONFIG["ML_MASS_GRID_POINTS"]
    counts = result.groupby("Curve_ID").size()
    if not (counts == expected).all():
        raise RuntimeError("Common-mass resampling did not equalize curve row counts.")
    return result


def load_and_preprocess(hadronic_dir: str, quark_dir: str) -> pd.DataFrame:
    """Load, clean, and equalize all curves on the common observable mass grid."""

    raw = load_physics_rows(hadronic_dir, quark_dir)
    engineered = engineer_observables(raw)
    return resample_curves_to_common_mass_grid(engineered)


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 fingerprint of latent rows and provenance."""

    columns = [
        "Row_ID",
        "Curve_ID",
        "Sweep_ID",
        "Perturb_A",
        "Mass",
        "Radius",
        "log10_Lambda",
        "Label",
    ]
    canonical = frame[columns].sort_values("Row_ID").reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()
