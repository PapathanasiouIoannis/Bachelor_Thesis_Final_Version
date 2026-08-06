"""Locked metadata and generation helpers for the one-week family pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from framework.eos_catalog import CFL_CATALOG, HADRONIC_CATALOG
from framework.eos_sweep import (
    GaussianDeformation,
    QuarkParameters,
    SweepPoint,
    admissible_amplitude_interval,
    build_hadronic_eos,
    build_quark_eos,
    cfl_baseline_grids,
    hadronic_baseline_grids,
    validate_sweep_within_interval,
)
from src.config import CONFIG
from src.physics.controlled_generation import (
    curve_to_dataframe,
    solve_and_validate_sequence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = Path(__file__).with_name("family_pilot_profile.json")
FAMILY_METADATA_COLUMNS = (
    "EoS_ID",
    "Variant_ID",
    "Family_Group_ID",
    "Parameter_Block_ID",
    "Model_Superfamily_ID",
    "Generation_Profile",
    "Gaussian_Grid_Max_Weight",
    "Primary_Citation_Available",
    "Exact_Formula_Primary_Verified",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_lookup() -> dict[str, Any]:
    return {
        entry.eos_id: entry
        for entry in (*HADRONIC_CATALOG, *CFL_CATALOG)
    }


def profile_entries(profile: dict) -> list[Any]:
    lookup = _catalog_lookup()
    requested = [*profile["hadronic_eos_ids"], *profile["quark_eos_ids"]]
    missing = sorted(set(requested) - set(lookup))
    if missing:
        raise ValueError(f"Family-pilot profile references unknown EoSs: {missing}")
    return [lookup[eos_id] for eos_id in requested]


def profile_sweep_points(profile: dict) -> list[SweepPoint]:
    amplitudes = np.asarray(profile["deformation"]["amplitudes"], dtype=float)
    if amplitudes.ndim != 1 or len(amplitudes) < 3:
        raise ValueError("Family pilot requires at least three shared amplitudes.")
    if np.any(~np.isfinite(amplitudes)) or np.any(np.diff(amplitudes) <= 0.0):
        raise ValueError("Family-pilot amplitudes must be finite and strictly increasing.")
    if not np.any(np.isclose(amplitudes, 0.0, atol=1e-14)):
        raise ValueError("Family-pilot amplitude grid must retain the A=0 control.")
    return [
        SweepPoint(index=index, amplitude=float(amplitude))
        for index, amplitude in enumerate(amplitudes)
    ]


def _validate_scan_evidence(profile: dict, entries: list[Any], points: list[SweepPoint]) -> None:
    scan_path = PROJECT_ROOT / profile["source_audit"]["amplitude_scan"]
    if not scan_path.is_file():
        raise FileNotFoundError(f"Family-pilot amplitude evidence is missing: {scan_path}")
    scan = pd.read_csv(scan_path)
    required_columns = {"eos_id", "amplitude", "acceptable", "reason"}
    missing = sorted(required_columns - set(scan.columns))
    if missing:
        raise ValueError(f"Amplitude evidence is missing columns: {missing}")

    evidence_rows = []
    for entry in entries:
        for point in points:
            matches = scan[
                scan["eos_id"].eq(entry.eos_id)
                & np.isclose(scan["amplitude"], point.amplitude, atol=1e-12)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Amplitude evidence must contain exactly one row for "
                    f"{entry.eos_id} at A={point.amplitude}; found {len(matches)}."
                )
            evidence_rows.append(matches.iloc[0])
    selected = pd.DataFrame(evidence_rows).reset_index(drop=True)

    acceptable = selected["acceptable"]
    if acceptable.dtype != bool:
        normalized = acceptable.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if normalized.isna().any():
            invalid = sorted(acceptable[normalized.isna()].astype(str).unique())
            raise ValueError(f"Amplitude evidence has invalid booleans: {invalid}")
        acceptable = normalized.astype(bool)
    if not acceptable.all():
        rejected = selected.loc[
            ~acceptable, ["eos_id", "amplitude", "reason"]
        ]
        raise ValueError(
            "Locked family-pilot profile includes rejected combinations: "
            f"{rejected.to_dict(orient='records')}"
        )


def load_family_pilot_profile(path: Path | None = None) -> dict:
    profile_path = (path or DEFAULT_PROFILE_PATH).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    required = {
        "profile_id",
        "claim_boundary",
        "source_audit",
        "deformation",
        "screens",
        "hadronic_eos_ids",
        "quark_eos_ids",
        "grouping",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Family-pilot profile is missing fields: {missing}")
    if len(profile["hadronic_eos_ids"]) != len(set(profile["hadronic_eos_ids"])):
        raise ValueError("Hadronic profile IDs are not unique.")
    if len(profile["quark_eos_ids"]) != len(set(profile["quark_eos_ids"])):
        raise ValueError("Quark profile IDs are not unique.")
    if set(profile["hadronic_eos_ids"]) & set(profile["quark_eos_ids"]):
        raise ValueError("An EoS ID appears in both matter classes.")
    if "PS" in profile["hadronic_eos_ids"]:
        raise ValueError("The uncited PS surrogate is forbidden in the locked profile.")

    entries = profile_entries(profile)
    points = profile_sweep_points(profile)
    eps0 = float(profile["deformation"]["epsilon0_mev_fm3"])
    sigma = float(profile["deformation"]["sigma_mev_fm3"])
    if eps0 != CONFIG["CONTROLLED_PERTURB_EPS0"] or sigma != CONFIG[
        "CONTROLLED_PERTURB_SIGMA"
    ]:
        raise ValueError("Family-pilot deformation must use the locked epsilon0 and sigma.")

    for entry in entries:
        if hasattr(entry, "bag_b_mev_fm3"):
            parameters = QuarkParameters(
                entry.bag_b_mev_fm3,
                entry.gap_delta_mev,
                entry.strange_mass_mev,
            )
            _, energy, cs2, _ = cfl_baseline_grids(parameters)
        else:
            if not entry.underlying_primary_url:
                raise ValueError(f"{entry.eos_id} has no underlying primary citation.")
            energy, cs2, _, _, _ = hadronic_baseline_grids(entry.eos_id)
        interval = admissible_amplitude_interval(energy, cs2, eps0, sigma)
        validate_sweep_within_interval(points, interval, entry.eos_id)

    _validate_scan_evidence(profile, entries, points)
    profile["profile_path"] = str(profile_path)
    profile["profile_sha256"] = _file_sha256(profile_path)
    return profile


def _entry_metadata(entry) -> dict[str, Any]:
    if hasattr(entry, "parameter_block_id"):
        return {
            "Family_Group_ID": entry.family_group_id,
            "Parameter_Block_ID": entry.parameter_block_id,
            "Model_Superfamily_ID": "Q_ANALYTIC_CFL_MIT_BAG",
            "Primary_Citation_Available": True,
            "Exact_Formula_Primary_Verified": True,
        }
    return {
        "Family_Group_ID": entry.family_group_id,
        "Parameter_Block_ID": "",
        "Model_Superfamily_ID": "H_REPOSITORY_SURROGATES",
        "Primary_Citation_Available": bool(entry.underlying_primary_url),
        "Exact_Formula_Primary_Verified": bool(entry.exact_formula_primary_verified),
    }


def generate_family_curve(entry, point: SweepPoint, profile: dict) -> pd.DataFrame:
    """Generate one fixed-baseline/A curve with immutable family metadata."""

    deformation = GaussianDeformation(
        point.amplitude,
        float(profile["deformation"]["epsilon0_mev_fm3"]),
        float(profile["deformation"]["sigma_mev_fm3"]),
    )
    is_quark = hasattr(entry, "bag_b_mev_fm3")
    if is_quark:
        parameters = QuarkParameters(
            entry.bag_b_mev_fm3,
            entry.gap_delta_mev,
            entry.strange_mass_mev,
        )
        eos = build_quark_eos(
            parameters,
            deformation,
            maximum_surface_energy_per_baryon=CONFIG["M_N"],
        )
        label = 1
    else:
        eos = build_hadronic_eos(entry.eos_id, deformation)
        label = 0

    screens = profile["screens"]
    curve, features, maximum_mass = solve_and_validate_sequence(
        eos,
        is_quark=is_quark,
        minimum_maximum_mass=float(screens["minimum_maximum_mass_msun"]),
        maximum_maximum_mass=float(screens["maximum_maximum_mass_msun"]),
        radius_14_bounds=tuple(float(value) for value in screens["radius_1p4_km"]),
    )
    frame = curve_to_dataframe(
        curve,
        features,
        maximum_mass,
        eos,
        point,
        label=label,
    )
    gaussian = np.exp(
        -0.5
        * ((eos.energy_density - deformation.epsilon0) / deformation.sigma) ** 2
    )
    metadata = _entry_metadata(entry)
    frame["EoS_ID"] = entry.eos_id
    frame["Variant_ID"] = f"{entry.eos_id}:{point.sweep_id}"
    frame["Generation_Profile"] = profile["profile_id"]
    frame["Gaussian_Grid_Max_Weight"] = float(np.max(gaussian))
    for column, value in metadata.items():
        frame[column] = value
    return frame


def validate_family_pilot_dataset(
    frame: pd.DataFrame,
    profile: dict,
) -> None:
    required = {*CONFIG["COLUMN_SCHEMA"], *FAMILY_METADATA_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Family-pilot dataset is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Family-pilot dataset is empty.")

    entries = profile_entries(profile)
    points = profile_sweep_points(profile)
    expected_ids = {entry.eos_id for entry in entries}
    expected_sweeps = {point.sweep_id: point.amplitude for point in points}
    if set(frame["EoS_ID"].unique()) != expected_ids:
        raise ValueError("Generated EoS IDs do not match the locked profile.")
    if set(frame["Sweep_ID"].unique()) != set(expected_sweeps):
        raise ValueError("Generated amplitude IDs do not match the locked profile.")
    if set(frame["Generation_Profile"].unique()) != {profile["profile_id"]}:
        raise ValueError("Generation profile metadata is not constant and locked.")

    combinations = frame.groupby(["EoS_ID", "Sweep_ID"])["Curve_ID"].nunique()
    if len(combinations) != len(expected_ids) * len(expected_sweeps):
        raise ValueError("Family-pilot dataset has missing EoS/amplitude combinations.")
    if not combinations.eq(1).all():
        raise ValueError("An EoS/amplitude combination has multiple Curve_ID values.")
    if frame.groupby("Curve_ID")[["EoS_ID", "Sweep_ID", "Label"]].nunique().max().max() != 1:
        raise ValueError("A Curve_ID crosses EoS, amplitude, or class identity.")
    if frame.groupby("Variant_ID")["Curve_ID"].nunique().max() != 1:
        raise ValueError("A Variant_ID maps to multiple Curve_ID values.")
    if frame.groupby("Curve_ID")["Variant_ID"].nunique().max() != 1:
        raise ValueError("A Curve_ID maps to multiple Variant_ID values.")

    entry_lookup = {entry.eos_id: entry for entry in entries}
    for eos_id, subset in frame.groupby("EoS_ID"):
        entry = entry_lookup[eos_id]
        expected_label = 1 if hasattr(entry, "bag_b_mev_fm3") else 0
        if set(subset["Label"].unique()) != {expected_label}:
            raise ValueError(f"{eos_id} has the wrong matter-class label.")
        if set(subset["Family_Group_ID"].unique()) != {entry.family_group_id}:
            raise ValueError(f"{eos_id} has incorrect family grouping metadata.")
        if set(subset["Sweep_ID"].unique()) != set(expected_sweeps):
            raise ValueError(f"{eos_id} does not have the complete shared A grid.")
        observed = subset.groupby("Sweep_ID")["Perturb_A"].first().to_dict()
        if any(
            not np.isclose(observed[sweep_id], amplitude, atol=1e-7)
            for sweep_id, amplitude in expected_sweeps.items()
        ):
            raise ValueError(f"{eos_id} stores an incorrect amplitude value.")

    support = frame.groupby("Curve_ID")["Mass"].agg(["min", "max"])
    mass_min, mass_max = profile["screens"]["observable_mass_grid_msun"]
    if not (support["min"] <= mass_min).all() or not (support["max"] >= mass_max).all():
        raise ValueError("One or more curves do not cover the locked observable mass grid.")
    curves_per_label = frame.groupby("Label")["Curve_ID"].nunique().to_dict()
    expected_per_label = len(points) * len(profile["hadronic_eos_ids"])
    if curves_per_label != {0: expected_per_label, 1: expected_per_label}:
        raise ValueError(
            f"Family-pilot curve balance failed: {curves_per_label}, "
            f"expected {expected_per_label} per class."
        )
