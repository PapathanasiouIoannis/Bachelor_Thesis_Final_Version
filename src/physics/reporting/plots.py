"""Matplotlib figures for persisted EoS experiment artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


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


__all__ = ["create_standard_plots"]
