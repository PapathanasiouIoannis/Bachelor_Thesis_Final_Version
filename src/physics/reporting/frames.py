"""DataFrame serialization, validation, and summaries for EoS runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from framework.eos_sweep import tabulate_complete_eos
from src.physics.reporting.schemas import (
    CAUSAL_DOMAIN_COLUMNS,
    EOS_COLUMNS,
    STELLAR_COLUMNS,
    SUMMARY_COLUMNS,
    _STELLAR_NUMERIC_COLUMNS,
)


def serialize_eos_table(
    framework_eos: Any,
    matter_type: str,
    sweep_id: str,
    *,
    tabulate_eos=tabulate_complete_eos,
) -> pd.DataFrame:
    """Serialize the complete solved EoS domain before validity screening."""

    if matter_type not in {"hadronic", "quark"}:
        raise ValueError("matter_type must be 'hadronic' or 'quark'.")
    pressure, energy_density, sound_speed_squared, regions = tabulate_eos(framework_eos)
    catalog_identifier = framework_eos.catalog_identifier or framework_eos.baseline_name
    frame = pd.DataFrame(
        {
            "matter_type": matter_type,
            "baseline_name": catalog_identifier,
            "model_identifier": framework_eos.baseline_name,
            "sweep_id": sweep_id,
            "deformation_amplitude": framework_eos.deformation.amplitude,
            "pair_accepted": True,
            "eos_validation_passed": pd.NA,
            "eos_validation_reason": "not_checked",
            "eos_region": regions,
            "energy_density_mev_fm3": energy_density,
            "pressure_mev_fm3": pressure,
            "sound_speed_squared": sound_speed_squared,
            "causal_prefix_applied": framework_eos.discarded_suffix_points > 0,
            "discarded_suffix_points": framework_eos.discarded_suffix_points,
            "first_discarded_sound_speed_squared": (
                framework_eos.first_discarded_sound_speed_squared
            ),
            "causal_cutoff_pressure_mev_fm3": framework_eos.pressure[-1],
            "causal_cutoff_energy_density_mev_fm3": framework_eos.energy_density[-1],
        }
    )
    return frame.loc[:, EOS_COLUMNS]


def validate_eos_frame(frame: pd.DataFrame) -> None:
    """Reject a serialized EoS table that violates the declared physical checks."""

    if frame.empty:
        raise ValueError("The generated EoS table is empty.")
    _require_finite(
        frame,
        (
            "deformation_amplitude",
            "energy_density_mev_fm3",
            "pressure_mev_fm3",
            "sound_speed_squared",
        ),
    )
    if np.any(np.diff(frame["energy_density_mev_fm3"]) <= 0.0):
        raise ValueError(
            "The generated energy-density grid is not strictly increasing."
        )
    if np.any(np.diff(frame["pressure_mev_fm3"]) <= 0.0):
        raise ValueError("The generated pressure grid is not strictly increasing.")
    sound_speed = frame["sound_speed_squared"].to_numpy(dtype=float)
    if np.any((sound_speed <= 0.0) | (sound_speed > 1.0)):
        raise ValueError("The generated EoS violates 0 < c_s^2 <= 1.")


def eos_to_frame(
    framework_eos: Any,
    matter_type: str,
    sweep_id: str,
    *,
    serializer=serialize_eos_table,
    validator=validate_eos_frame,
) -> pd.DataFrame:
    """Serialize and validate the complete framework EoS table."""

    frame = serializer(framework_eos, matter_type, sweep_id)
    validator(frame)
    frame.loc[:, "eos_validation_passed"] = True
    frame.loc[:, "eos_validation_reason"] = "passed"
    return frame


def stellar_curve_to_frame(
    curve: list,
    framework_eos: Any,
    matter_type: str,
    sweep_id: str,
    curve_id: str,
) -> pd.DataFrame:
    """Serialize one turning-point-truncated stellar sequence."""

    rows = []
    catalog_identifier = framework_eos.catalog_identifier or framework_eos.baseline_name
    for mass, radius, tidal, pressure, energy, sound_speed, surface_energy in curve:
        rows.append(
            {
                "matter_type": matter_type,
                "baseline_name": catalog_identifier,
                "model_identifier": framework_eos.baseline_name,
                "sweep_id": sweep_id,
                "curve_id": curve_id,
                "deformation_amplitude": framework_eos.deformation.amplitude,
                "mass_msun": mass,
                "radius_km": radius,
                "tidal_deformability": tidal,
                "central_pressure_mev_fm3": pressure,
                "central_energy_density_mev_fm3": energy,
                "central_sound_speed_squared": sound_speed,
                "surface_energy_density_mev_fm3": surface_energy,
            }
        )
    frame = pd.DataFrame.from_records(rows, columns=STELLAR_COLUMNS)
    if frame.empty:
        raise ValueError("The accepted EoS produced no stellar sequence rows.")
    _require_finite(frame, _STELLAR_NUMERIC_COLUMNS)
    pressure = frame["central_pressure_mev_fm3"].to_numpy(dtype=float)
    mass = frame["mass_msun"].to_numpy(dtype=float)
    if np.any(np.diff(pressure) <= 0.0):
        raise ValueError("Central pressure is not strictly increasing on the sequence.")
    if np.any(np.diff(mass) < -1e-10):
        raise ValueError("The retained stellar branch is not monotonic in mass.")
    return frame


def summarize_stellar_curve(frame: pd.DataFrame) -> dict[str, Any]:
    """Return clear canonical observables for one complete curve."""

    if frame.empty or frame["curve_id"].nunique() != 1:
        raise ValueError("A stellar summary requires exactly one non-empty curve.")
    ordered = frame.sort_values("mass_msun")
    mass = ordered["mass_msun"].to_numpy(dtype=float)
    radius = ordered["radius_km"].to_numpy(dtype=float)
    tidal = ordered["tidal_deformability"].to_numpy(dtype=float)
    unique_mass, unique_indices = np.unique(mass, return_index=True)
    if len(unique_mass) < 4 or unique_mass[0] > 1.4 or unique_mass[-1] < 1.4:
        raise ValueError("The stable sequence does not bracket 1.4 solar masses.")
    radius_at_14 = float(PchipInterpolator(unique_mass, radius[unique_indices])(1.4))
    tidal_at_14 = float(PchipInterpolator(unique_mass, tidal[unique_indices])(1.4))
    first = ordered.iloc[0]
    return {
        "matter_type": str(first["matter_type"]),
        "baseline_name": str(first["baseline_name"]),
        "sweep_id": str(first["sweep_id"]),
        "deformation_amplitude": float(first["deformation_amplitude"]),
        "maximum_mass_msun": float(unique_mass[-1]),
        "radius_1p4_km": radius_at_14,
        "tidal_deformability_1p4": tidal_at_14,
        "turning_point_stability_estimate": True,
        "status": "accepted",
    }


def build_summary_table(
    stellar_curves: pd.DataFrame,
    *,
    summarizer=summarize_stellar_curve,
) -> pd.DataFrame:
    records = [
        summarizer(group)
        for _, group in stellar_curves.groupby("curve_id", sort=True)
    ]
    return pd.DataFrame.from_records(records, columns=SUMMARY_COLUMNS).sort_values(
        ["matter_type", "deformation_amplitude"], ignore_index=True
    )


def build_causal_domain_table(eos_tables: pd.DataFrame) -> pd.DataFrame:
    """Return one transparent causal-domain and validation record per EoS."""

    if eos_tables.empty:
        return pd.DataFrame(columns=CAUSAL_DOMAIN_COLUMNS)
    return (
        eos_tables.loc[:, CAUSAL_DOMAIN_COLUMNS]
        .drop_duplicates(ignore_index=True)
        .sort_values(["deformation_amplitude", "matter_type"], ignore_index=True)
    )


def _require_finite(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(
            f"Generated columns contain non-finite values: {list(columns)}"
        )


__all__ = [
    "build_causal_domain_table",
    "build_summary_table",
    "eos_to_frame",
    "serialize_eos_table",
    "stellar_curve_to_frame",
    "summarize_stellar_curve",
    "validate_eos_frame",
]
