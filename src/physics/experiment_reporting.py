"""Professional output tables, figures, and reports for controlled EoS runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.interpolate import PchipInterpolator

from framework.eos_sweep import tabulate_complete_eos
from src.physics.reporting.schemas import (
    CAUSAL_DOMAIN_COLUMNS,
    CAUSAL_DOMAIN_HEADINGS,
    CONVERGENCE_HEADINGS,
    EOS_COLUMNS,
    REJECTION_HEADINGS,
    STELLAR_COLUMNS,
    SUMMARY_COLUMNS,
    SUMMARY_HEADINGS,
    _STELLAR_NUMERIC_COLUMNS,
)


def serialize_eos_table(
    framework_eos: Any, matter_type: str, sweep_id: str
) -> pd.DataFrame:
    """Serialize the complete solved EoS domain before validity screening."""

    if matter_type not in {"hadronic", "quark"}:
        raise ValueError("matter_type must be 'hadronic' or 'quark'.")
    pressure, energy_density, sound_speed_squared, regions = tabulate_complete_eos(
        framework_eos
    )
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


def eos_to_frame(framework_eos: Any, matter_type: str, sweep_id: str) -> pd.DataFrame:
    """Serialize and validate the complete framework EoS table."""

    frame = serialize_eos_table(framework_eos, matter_type, sweep_id)
    validate_eos_frame(frame)
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


def build_summary_table(stellar_curves: pd.DataFrame) -> pd.DataFrame:
    records = [
        summarize_stellar_curve(group)
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


def create_standard_plots(
    eos_tables: pd.DataFrame,
    stellar_curves: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    """Create the four declared plot groups from persisted physical artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    amplitudes = np.sort(eos_tables["deformation_amplitude"].unique())
    normalizer = Normalize(vmin=float(amplitudes.min()), vmax=float(amplitudes.max()))
    colour_map = plt.get_cmap("viridis")
    saved: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, matter_type in zip(axes, ("hadronic", "quark"), strict=True):
        for _, curve in eos_tables[eos_tables["matter_type"] == matter_type].groupby(
            "sweep_id", sort=True
        ):
            amplitude = float(curve["deformation_amplitude"].iloc[0])
            accepted = bool(curve["pair_accepted"].iloc[0])
            axis.plot(
                curve["energy_density_mev_fm3"],
                curve["sound_speed_squared"],
                color=colour_map(normalizer(amplitude)),
                linewidth=1.0,
                alpha=1.0 if accepted else 0.55,
            )
        axis.set_title(f"{matter_type.capitalize()} EoS")
        axis.set_xlabel(r"Energy density $\epsilon$ [MeV fm$^{-3}$]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"Sound speed squared $c_s^2$")
    axes[0].set_ylim(0.0, 1.02)
    _add_amplitude_bar(figure, axes, colour_map, normalizer)
    saved.append(
        _save_figure(figure, output_directory / "sound_speed_vs_energy_density.png")
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, matter_type in zip(axes, ("hadronic", "quark"), strict=True):
        for _, curve in eos_tables[eos_tables["matter_type"] == matter_type].groupby(
            "sweep_id", sort=True
        ):
            amplitude = float(curve["deformation_amplitude"].iloc[0])
            accepted = bool(curve["pair_accepted"].iloc[0])
            axis.plot(
                curve["energy_density_mev_fm3"],
                curve["pressure_mev_fm3"],
                color=colour_map(normalizer(amplitude)),
                linewidth=1.0,
                alpha=1.0 if accepted else 0.55,
            )
        axis.set_title(f"{matter_type.capitalize()} EoS")
        axis.set_xlabel(r"Energy density $\epsilon$ [MeV fm$^{-3}$]")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(r"Pressure $P$ [MeV fm$^{-3}$]")
    _add_amplitude_bar(figure, axes, colour_map, normalizer)
    saved.append(
        _save_figure(figure, output_directory / "pressure_vs_energy_density.png")
    )

    if stellar_curves.empty or summary.empty:
        return saved

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for _, curve in stellar_curves.groupby("curve_id", sort=True):
        amplitude = float(curve["deformation_amplitude"].iloc[0])
        matter_type = str(curve["matter_type"].iloc[0])
        line_style = "-" if matter_type == "hadronic" else "--"
        colour = colour_map(normalizer(amplitude))
        axes[0].plot(curve["radius_km"], curve["mass_msun"], line_style, color=colour)
        axes[1].plot(
            curve["mass_msun"], curve["tidal_deformability"], line_style, color=colour
        )
    axes[0].set_xlabel("Radius [km]")
    axes[0].set_ylabel(r"Mass [$M_\odot$]")
    axes[0].set_title("Turning-point-truncated mass-radius curves")
    axes[1].set_xlabel(r"Mass [$M_\odot$]")
    axes[1].set_ylabel(r"Tidal deformability $\Lambda$")
    axes[1].set_yscale("log")
    axes[1].set_title("Tidal deformability")
    for axis in axes:
        axis.grid(alpha=0.25)
    _add_amplitude_bar(figure, axes, colour_map, normalizer)
    saved.append(_save_figure(figure, output_directory / "stable_stellar_curves.png"))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    observable_columns = (
        ("maximum_mass_msun", r"Maximum mass [$M_\odot$]"),
        ("radius_1p4_km", r"Radius at $1.4M_\odot$ [km]"),
        ("tidal_deformability_1p4", r"Tidal deformability at $1.4M_\odot$"),
    )
    for axis, (column, label) in zip(axes, observable_columns, strict=True):
        for matter_type, line_style in (("hadronic", "-o"), ("quark", "--s")):
            selected = summary[summary["matter_type"] == matter_type].sort_values(
                "deformation_amplitude"
            )
            axis.plot(
                selected["deformation_amplitude"],
                selected[column],
                line_style,
                markersize=3,
                label=matter_type.capitalize(),
            )
        axis.set_xlabel(r"Deformation amplitude $A$")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].legend()
    saved.append(
        _save_figure(figure, output_directory / "observables_vs_amplitude.png")
    )
    return saved


def write_markdown_report(
    eos_tables: pd.DataFrame,
    summary: pd.DataFrame,
    rejections: pd.DataFrame,
    convergence: pd.DataFrame,
    resolved_configuration: dict[str, Any],
    output_path: Path,
    *,
    run_status: str,
) -> Path:
    """Write a concise report with the experiment's scientific boundary."""

    deformation = resolved_configuration["deformation"]
    hadronic = resolved_configuration["hadronic_eos"]
    quark = resolved_configuration["quark_eos"]
    accepted = int(len(summary))
    rejected = int(len(rejections))
    requirements = resolved_configuration["physical_requirements"]
    execution = resolved_configuration["execution"]
    numerical = resolved_configuration["resolved_numerical_settings"]
    causal_domains = build_causal_domain_table(eos_tables)
    rejection_note = ""
    if (
        not rejections.empty
        and rejections["reason"]
        .astype(str)
        .str.contains("Maximum mass is not bracketed", regex=False)
        .any()
    ):
        rejection_note = (
            "At least one sequence reached the causal EoS boundary before a resolved "
            "post-peak mass decrease. Such an endpoint is not reported as a maximum "
            "mass."
        )
    elif (
        not rejections.empty
        and rejections["reason"]
        .astype(str)
        .str.contains("energy-density grid is not strictly increasing", regex=False)
        .any()
    ):
        rejection_note = (
            "The complete hadronic table contains a downward energy-density step "
            "inside the legacy crust fit. The runner preserved the diagnostic table "
            "and rejected the complete amplitude pair without smoothing the boundary."
        )
    interpretation = (
        "The retained stellar curves end at the first resolved mass turning point. "
        "This is a turning-point stability estimate and is not a radial-oscillation "
        "calculation."
        if not summary.empty
        else (
            "No stellar curve passed the preceding EoS and pair-level gates. If an "
            "EoS reaches stellar integration, it must still bracket a first mass "
            "turning point; that estimate is not a radial-oscillation calculation."
        )
    )
    lines = [
        "# Controlled EoS sensitivity report",
        "",
        "## Experiment",
        "",
        f"**Terminal run status: `{run_status}`**",
        "",
        f"This run compares one repository {hadronic['baseline']} surrogate with one analytic CFL "
        "MIT-bag baseline under the same additive Gaussian sound-speed deformation. "
        "It is a controlled model-pair sensitivity study, not a universal matter-phase classifier.",
        "",
        f"- Deformation centre, $\\epsilon_0$: {deformation['center_energy_density_mev_fm3']} MeV fm^-3",
        f"- Deformation width, $\\sigma$: {deformation['width_mev_fm3']} MeV fm^-3",
        f"- Bag constant, $B$: {quark['bag_constant_mev_fm3']} MeV fm^-3",
        f"- Pairing gap, $\\Delta$: {quark['pairing_gap_mev']} MeV",
        f"- Strange-quark mass, $m_s$: {quark['strange_quark_mass_mev']} MeV",
        f"- Random seed: {execution['random_seed']}",
        f"- Parallel jobs: {execution['parallel_jobs']}",
        f"- Amplitudes per worker batch: {execution['amplitudes_per_batch']}",
        f"- EoS grid points: {numerical['eos_grid_points']}",
        f"- Central-pressure points: {numerical['central_pressure_points']}",
        f"- TOV relative tolerance: {numerical['tov_relative_tolerance']}",
        f"- TOV absolute tolerance: {numerical['tov_absolute_tolerance']}",
        f"- Accepted curves: {accepted}",
        f"- Rejected amplitude pairs: {rejected}",
        "",
        "## Physical acceptance requirements",
        "",
        (
            "- Maximum mass: "
            f"{requirements['minimum_maximum_mass_msun']} to "
            f"{requirements['maximum_maximum_mass_msun']} $M_\\odot$"
        ),
        (
            "- Radius at $1.4M_\\odot$: "
            f"{requirements['radius_1p4_min_km']} to "
            f"{requirements['radius_1p4_max_km']} km"
        ),
        "",
        "## Canonical observables",
        "",
        _markdown_table(summary.loc[:, SUMMARY_COLUMNS], SUMMARY_HEADINGS),
        "",
        "## EoS validation and causal domains",
        "",
        (
            "The framework retains values only through the last causal, stable grid "
            "point and records any discarded suffix below. Values are not clipped or "
            "silently repaired. For hadronic models, the complete table also includes "
            "the crust domain used by the stellar solver."
        ),
        "",
        _markdown_table(causal_domains, CAUSAL_DOMAIN_HEADINGS),
        "",
        "## Rejected amplitude pairs",
        "",
        rejection_note,
        "",
        _markdown_table(rejections, REJECTION_HEADINGS),
        "",
        "## Numerical convergence",
        "",
        _markdown_table(convergence, CONVERGENCE_HEADINGS)
        if not convergence.empty
        else (
            "Convergence checks were disabled by this exploratory configuration."
            if resolved_configuration["numerical_settings"]["convergence_check"]
            == "none"
            else "No convergence results were produced."
        ),
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _require_finite(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    values = frame.loc[:, list(columns)].to_numpy(dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError(
            f"Generated columns contain non-finite values: {list(columns)}"
        )


def _add_amplitude_bar(figure, axes, colour_map, normalizer) -> None:
    scalar = plt.cm.ScalarMappable(norm=normalizer, cmap=colour_map)
    scalar.set_array([])
    figure.subplots_adjust(wspace=0.28, right=0.88)
    colour_axis = figure.add_axes((0.91, 0.15, 0.018, 0.7))
    figure.colorbar(
        scalar,
        cax=colour_axis,
        label=r"Deformation amplitude $A$",
    )


def _save_figure(figure, path: Path) -> Path:
    figure.subplots_adjust(wspace=0.28)
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)
    return path


def _markdown_table(frame: pd.DataFrame, headings: dict[str, str] | None = None) -> str:
    if frame.empty:
        return "No rows."
    display = frame.copy().rename(columns=headings or {})
    for column in display.select_dtypes(include=["float", "float32", "float64"]):
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])
