"""Shared sequence validation and row serialization for controlled EoS sweeps."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from framework.eos_sweep import FrameworkEos, SweepPoint
from src.config import CONFIG
from src.physics.feature_extraction import extract_features
from src.physics.solve_sequence import solve_sequence
from src.physics.verification import verify_eos_physical_validity


def _curve_identifier(
    label: int,
    framework_eos: FrameworkEos,
    sweep_point: SweepPoint,
) -> str:
    prefix = "H" if label == 0 else "Q"
    baseline_name = framework_eos.baseline_name
    baseline_token = re.sub(r"[^A-Za-z0-9]+", "-", baseline_name).strip("-")
    parameters = framework_eos.quark_parameters
    identity = "|".join(
        [
            str(label),
            baseline_name,
            f"{sweep_point.amplitude:.17g}",
            f"{framework_eos.deformation.epsilon0:.17g}",
            f"{framework_eos.deformation.sigma:.17g}",
            f"{parameters.bag_b:.17g}" if parameters else "0",
            f"{parameters.gap_delta:.17g}" if parameters else "0",
            f"{parameters.strange_mass:.17g}" if parameters else "0",
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{baseline_token}_{sweep_point.sweep_id}_{digest}"


def solve_and_validate_sequence(
    framework_eos: FrameworkEos,
    *,
    is_quark: bool,
) -> tuple[list, dict, float]:
    """Solve one EoS and enforce the same viability criteria for both classes."""

    curve, _, maximum_mass = solve_sequence(
        framework_eos.eos_callable,
        is_quark=is_quark,
        p_max_causal=framework_eos.p_max_causal,
        rtol=CONFIG["TOV_RTOL"],
        atol=CONFIG["TOV_ATOL"],
    )
    if not curve:
        raise RuntimeError("TOV integration returned no stable sequence points.")

    curve_array = np.asarray(curve, dtype=float)
    if curve_array[0, 0] > CONFIG["BH_LIMIT"]:
        raise RuntimeError(
            "TOV sequence did not trace the required low-mass branch "
            f"(first mass={curve_array[0, 0]:.6g} M_sun)."
        )
    if not verify_eos_physical_validity(curve_array):
        raise RuntimeError("TOV sequence has too few points for physical verification.")

    features = extract_features(curve_array, maximum_mass)
    if features is None or not np.isfinite(features["r_14"]):
        raise RuntimeError("Could not extract finite canonical 1.4-M_sun features.")

    common_mass_upper = min(CONFIG["H_M_MAX_UPPER"], CONFIG["Q_M_MAX_UPPER"])
    viability_failures = []
    if not CONFIG["M_MAX_LOWER_BOUND"] <= maximum_mass <= common_mass_upper:
        viability_failures.append(
            f"M_max={maximum_mass:.6g} outside "
            f"[{CONFIG['M_MAX_LOWER_BOUND']}, {common_mass_upper}] M_sun"
        )
    if not CONFIG["CONTROLLED_R14_MIN"] <= features["r_14"] <= CONFIG["CONTROLLED_R14_MAX"]:
        viability_failures.append(
            f"R_1.4={features['r_14']:.6g} outside "
            f"[{CONFIG['CONTROLLED_R14_MIN']}, {CONFIG['CONTROLLED_R14_MAX']}] km"
        )
    if viability_failures:
        raise RuntimeError("; ".join(viability_failures))

    return curve, features, float(maximum_mass)


def curve_to_dataframe(
    curve: list,
    features: dict,
    maximum_mass: float,
    framework_eos: FrameworkEos,
    sweep_point: SweepPoint,
    *,
    label: int,
) -> pd.DataFrame:
    """Serialize one accepted EoS curve with stable, auditable identifiers."""

    parameters = framework_eos.quark_parameters
    generation_seed = CONFIG["CONTROLLED_GENERATION_SEED"] + sweep_point.index
    curve_id = _curve_identifier(label, framework_eos, sweep_point)
    rows = []

    for point in curve:
        mass = float(point[0])
        if mass < CONFIG["M_MIN_SAVE"] or mass > maximum_mass:
            continue
        rows.append(
            {
                "Mass": mass,
                "Radius": float(point[1]),
                "Lambda": float(point[2]),
                "Label": int(label),
                "Curve_ID": curve_id,
                "Sweep_ID": sweep_point.sweep_id,
                "P_Central": float(point[3]),
                "Eps_Central": float(point[4]),
                "Eps_Surface": float(framework_eos.eps_surface),
                "CS2_Central": float(point[5]),
                "CS2_at_14": float(features["cs2_at_14"]),
                "Radius_14": float(features["r_14"]),
                "M_Max": float(maximum_mass),
                "Observationally_Viable": 1,
                "Slope14": float(features["slopes"].get(1.4, np.nan)),
                "Slope16": float(features["slopes"].get(1.6, np.nan)),
                "Slope18": float(features["slopes"].get(1.8, np.nan)),
                "Slope20": float(features["slopes"].get(2.0, np.nan)),
                "Bag_B": float(parameters.bag_b) if parameters else 0.0,
                "Gap_Delta": float(parameters.gap_delta) if parameters else 0.0,
                "Mass_Strange": float(parameters.strange_mass) if parameters else 0.0,
                "Generation_Seed": int(generation_seed),
                "Perturb_A": float(framework_eos.deformation.amplitude),
                "Perturb_eps0": float(framework_eos.deformation.epsilon0),
                "Perturb_sigma": float(framework_eos.deformation.sigma),
                "Baseline_Name": framework_eos.baseline_name,
            }
        )

    if not rows:
        raise RuntimeError("Accepted EoS produced no serializable stable-branch rows.")
    frame = pd.DataFrame.from_records(rows, columns=CONFIG["COLUMN_SCHEMA"])
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype("float32")
    frame["Label"] = frame["Label"].astype("int32")
    frame["Observationally_Viable"] = frame["Observationally_Viable"].astype("int8")
    return frame
