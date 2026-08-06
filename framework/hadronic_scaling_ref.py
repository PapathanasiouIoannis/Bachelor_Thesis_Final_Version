"""Diagnostic plots for the same controlled APR-1 sweep used in production."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from framework.eos_sweep import GaussianDeformation, amplitude_grid, build_hadronic_eos
from src.config import CONFIG
from src.physics.solve_sequence import solve_sequence


def main() -> None:
    """Plot deterministic sound-speed and M-R diagnostics for APR-1 only."""

    output_dir = Path("plots") / "scaled_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_points = amplitude_grid(
        CONFIG["CONTROLLED_A_MIN"],
        CONFIG["CONTROLLED_A_MAX"],
        CONFIG["CONTROLLED_A_POINTS"],
    )

    figure_cs2, axis_cs2 = plt.subplots(figsize=(8, 6))
    figure_mr, axis_mr = plt.subplots(figsize=(8, 6))
    color_map = plt.get_cmap("coolwarm")
    normalization = plt.Normalize(
        sweep_points[0].amplitude, sweep_points[-1].amplitude
    )

    for point in sweep_points:
        deformation = GaussianDeformation(
            point.amplitude,
            CONFIG["CONTROLLED_PERTURB_EPS0"],
            CONFIG["CONTROLLED_PERTURB_SIGMA"],
        )
        eos = build_hadronic_eos(CONFIG["CONTROLLED_HADRONIC_BASELINE"], deformation)
        color = "black" if point.amplitude == 0.0 else color_map(normalization(point.amplitude))
        linewidth = 2.5 if point.amplitude == 0.0 else 1.0
        label = "A=0 control" if point.amplitude == 0.0 else None
        axis_cs2.plot(
            eos.energy_density,
            eos.sound_speed_squared,
            color=color,
            linewidth=linewidth,
            alpha=1.0 if point.amplitude == 0.0 else 0.55,
            label=label,
        )

        curve, _, _ = solve_sequence(
            eos.eos_callable,
            is_quark=False,
            p_max_causal=eos.p_max_causal,
            rtol=CONFIG["TOV_RTOL"],
            atol=CONFIG["TOV_ATOL"],
        )
        if curve:
            curve_array = np.asarray(curve)
            axis_mr.plot(
                curve_array[:, 1],
                curve_array[:, 0],
                color=color,
                linewidth=linewidth,
                alpha=1.0 if point.amplitude == 0.0 else 0.55,
                label=label,
            )

    scalar_map = plt.cm.ScalarMappable(norm=normalization, cmap=color_map)
    figure_cs2.colorbar(scalar_map, ax=axis_cs2, label="Gaussian amplitude A")
    figure_mr.colorbar(scalar_map, ax=axis_mr, label="Gaussian amplitude A")
    axis_cs2.axhline(1.0, color="red", linestyle="--", label="causality limit")
    axis_cs2.set(xlabel=r"Energy density $\epsilon$ [MeV/fm$^3$]", ylabel=r"$c_s^2$")
    axis_cs2.set_xlim(0.0, 1000.0)
    axis_cs2.set_ylim(0.0, 1.05)
    axis_cs2.legend()
    axis_cs2.grid(alpha=0.25)

    axis_mr.set(xlabel="Radius [km]", ylabel=r"Mass [$M_\odot$]")
    axis_mr.set_xlim(8.0, 16.0)
    axis_mr.set_ylim(0.1, 3.0)
    axis_mr.legend()
    axis_mr.grid(alpha=0.25)

    figure_cs2.tight_layout()
    figure_mr.tight_layout()
    figure_cs2.savefig(output_dir / "cs2_controlled_APR-1.pdf")
    figure_mr.savefig(output_dir / "MR_controlled_APR-1.pdf")
    plt.close(figure_cs2)
    plt.close(figure_mr)


if __name__ == "__main__":
    main()
