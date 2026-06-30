import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

from src.config import CONFIG
from src.utils.logger import get_logger
from src.visualize.style_config import COLORS, set_paper_style

logger = get_logger("VISUALIZATION")


def plot_core_physics(df: pd.DataFrame) -> None:
    """
    Condenses the plotting suite into two strictly physical curve plots:
    1. M-R Curves (Mass vs. Radius)
    2. Lambda Trajectories (LogLambda vs Mass & Radius)

    All curves are rendered smoothly using dense PchipInterpolator grids,
    with no topological backgrounds or ML colors.
    """
    set_paper_style()
    logger.info("\n--- Generating Core Physics Plots (Dense Curves) ---")

    fig_mr, ax_mr = plt.subplots(figsize=(8, 6))
    fig_lam_m, ax_lam_m = plt.subplots(figsize=(8, 6))
    fig_lam_r, ax_lam_r = plt.subplots(figsize=(8, 6))
    fig_cs2_pressure, ax_cs2_pressure = plt.subplots(figsize=(8, 6))
    fig_eos_bands, ax_eos_bands = plt.subplots(figsize=(8, 6))
    fig_mass_rho, ax_mass_rho = plt.subplots(figsize=(8, 6))

    MAX_CURVES = 100
    curve_ids = df["Curve_ID"].unique()
    if len(curve_ids) > MAX_CURVES:
        import random
        random.seed(42)
        sampled_ids = random.sample(list(curve_ids), MAX_CURVES)
        df = df[df["Curve_ID"].isin(sampled_ids)]
        logger.info(f"  > Downsampled from {len(curve_ids)} to {MAX_CURVES} curves to prevent MemoryError.")

    grouped = df.groupby("Curve_ID")
    n_curves = len(grouped)

    alpha_val = max(0.01, min(0.3, 500.0 / n_curves))
    lw_val = 0.5 if n_curves > 1000 else 1.0

    logger.info(
        f"  > Interpolating and plotting {n_curves} curves (alpha={alpha_val:.3f})..."
    )

    for _, group in tqdm(grouped, desc="Rendering Dense Physics Curves", leave=False):
        g = group.sort_values(by="Eps_Central")
        label = g["Label"].iloc[0]
        color = COLORS["Q_main"] if label == 1 else COLORS["H_main"]

        # only plot up to M_max to avoid the unstable collapsing branch
        m_max_iloc = g["Mass"].argmax()
        g_stable = g.iloc[:m_max_iloc + 1]

        # alpha fanning: scale alpha based on the maximum mass of the curve
        max_m_val = g_stable["Mass"].max()
        curve_alpha = max(0.05, min(0.6, alpha_val * (max_m_val / 2.0)**2))

        eps_vals, idx_unique = np.unique(
            g_stable["Eps_Central"].values, return_index=True
        )
        r_vals = g_stable["Radius"].values[idx_unique]
        m_vals = g_stable["Mass"].values[idx_unique]
        cs2_vals = g_stable["CS2_Central"].values[idx_unique]
        if "P_Central" in g_stable.columns:
            p_vals = g_stable["P_Central"].values[idx_unique]
        else:
            p_vals = np.zeros_like(r_vals)

        if "LogLambda" in g_stable.columns:
            lam_vals = g_stable["LogLambda"].values[idx_unique]
        else:
            lam_vals = np.log10(
                g_stable["Lambda"].replace(0, np.nan).values[idx_unique]
            )

        if len(eps_vals) > 3 and eps_vals[0] > 0:
            eps_grid = np.geomspace(eps_vals[0], eps_vals[-1], 500)
            r_grid = PchipInterpolator(eps_vals, r_vals)(eps_grid)
            m_grid = PchipInterpolator(eps_vals, m_vals)(eps_grid)
            cs2_grid = PchipInterpolator(eps_vals, cs2_vals)(eps_grid)
            p_grid = PchipInterpolator(eps_vals, p_vals)(eps_grid)
            lam_grid = PchipInterpolator(eps_vals, lam_vals)(eps_grid)
        else:
            eps_grid = eps_vals
            r_grid = r_vals
            m_grid = m_vals
            cs2_grid = cs2_vals
            p_grid = p_vals
            lam_grid = lam_vals

        m_max_idx = np.argmax(m_grid)
        m_grid = m_grid[:m_max_idx + 1]
        r_grid = r_grid[:m_max_idx + 1]
        lam_grid = lam_grid[:m_max_idx + 1]
        p_grid = p_grid[:m_max_idx + 1]
        cs2_grid = cs2_grid[:m_max_idx + 1]
        eps_grid = eps_grid[:m_max_idx + 1]

        # plot 1: M-R
        ax_mr.plot(
            r_grid, m_grid, color=color, alpha=curve_alpha, lw=lw_val, rasterized=True
        )

        # plot 2: Lambda-M
        ax_lam_m.plot(
            m_grid, lam_grid, color=color, alpha=curve_alpha, lw=lw_val, rasterized=True
        )
        
        # plot 3: Lambda-R
        ax_lam_r.plot(
            r_grid, lam_grid, color=color, alpha=curve_alpha, lw=lw_val, rasterized=True
        )

        # plot 4: CS2 vs Pressure
        ax_cs2_pressure.plot(
            p_grid, cs2_grid, color=color, alpha=curve_alpha, lw=lw_val, rasterized=True
        )

        # plot 5: EoS Bands
        if "P_Central" in g_stable.columns:
            ax_eos_bands.plot(
                eps_grid,
                p_grid,
                color=color,
                alpha=curve_alpha,
                lw=lw_val,
                rasterized=True,
            )

        # plot 6: Mass vs Central Density
        ax_mass_rho.plot(
            eps_grid, m_grid, color=color, alpha=curve_alpha, lw=lw_val, rasterized=True
        )

        # terminal causality / max mass marker
        if len(m_grid) > 0:
            ax_mr.plot(r_grid[-1], m_grid[-1], marker='X', color='red', markersize=4, alpha=0.5)
            ax_lam_m.plot(m_grid[-1], lam_grid[-1], marker='X', color='red', markersize=4, alpha=0.5)
            ax_lam_r.plot(r_grid[-1], lam_grid[-1], marker='X', color='red', markersize=4, alpha=0.5)
            ax_mass_rho.plot(eps_grid[-1], m_grid[-1], marker='X', color='red', markersize=4, alpha=0.5)

    legend_elements = [
        Line2D([0], [0], color=COLORS["H_main"], lw=2, label="Hadronic EoS"),
        Line2D([0], [0], color=COLORS["Q_main"], lw=2, label="Quark EoS"),
    ]

    # --- Formatting Plot 1: M-R ---
    ax_mr.set_xlim(CONFIG["PLOT_R_LIM"])
    ax_mr.set_ylim(CONFIG["PLOT_M_LIM"])
    ax_mr.set_xlabel(r"Radius $R$ [km]")
    ax_mr.set_ylabel(r"Mass $M$ [$M_{\odot}$]")
    ax_mr.set_title("Mass-Radius Trajectories")
    ax_mr.legend(handles=legend_elements, loc="upper right", framealpha=0.95)
    fig_mr.tight_layout()
    fig_mr.savefig(CONFIG["PLOT_CORE_MR"], dpi=400)
    plt.close(fig_mr)

    # --- Formatting Plot 2: Lambda vs M ---
    ax_lam_m.set_xlim(CONFIG["PLOT_M_LIM"])
    ax_lam_m.set_xlabel(r"Mass $M$ [$M_{\odot}$]")
    ax_lam_m.set_ylabel(r"$\log_{10}(\Lambda)$")
    ax_lam_m.set_title(r"$\Lambda$ vs Mass")
    ax_lam_m.legend(handles=legend_elements, loc="upper right", framealpha=0.95)
    fig_lam_m.tight_layout()
    fig_lam_m.savefig(CONFIG["PLOT_CORE_LAMBDA_M"], dpi=400)
    plt.close(fig_lam_m)

    # --- Formatting Plot 3: Lambda vs R ---
    ax_lam_r.set_xlim(CONFIG["PLOT_R_LIM"])
    ax_lam_r.set_xlabel(r"Radius $R$ [km]")
    ax_lam_r.set_ylabel(r"$\log_{10}(\Lambda)$")
    ax_lam_r.set_title(r"$\Lambda$ vs Radius")
    ax_lam_r.legend(handles=legend_elements, loc="upper right", framealpha=0.95)
    fig_lam_r.tight_layout()
    fig_lam_r.savefig(CONFIG["PLOT_CORE_LAMBDA_R"], dpi=400)
    plt.close(fig_lam_r)

    # --- Formatting Plot 4: CS2 vs Pressure ---
    ax_cs2_pressure.set_xscale("log")
    ax_cs2_pressure.set_ylim(CONFIG.get("PLOT_CS2_LIM", (0.0, 1.05)))
    ax_cs2_pressure.axhline(
        1.0 / 3.0, color="gray", linestyle="--", alpha=0.8, label="Conformal Limit"
    )
    ax_cs2_pressure.axhline(
        1.0, color="gray", linestyle="-", alpha=0.8, label="Causality Limit"
    )
    ax_cs2_pressure.set_xlabel(r"Pressure $P$ [MeV/fm$^3$]")
    ax_cs2_pressure.set_ylabel(r"Speed of Sound Squared $c_s^2$")
    ax_cs2_pressure.set_title("Speed of Sound vs Pressure")
    legend_cs2_pressure = legend_elements + [
        Line2D([0], [0], color="gray", linestyle="--", label="Conformal Limit"),
        Line2D([0], [0], color="gray", linestyle="-", label="Causality Limit"),
    ]
    ax_cs2_pressure.legend(
        handles=legend_cs2_pressure, loc="upper left", framealpha=0.95
    )
    fig_cs2_pressure.tight_layout()
    fig_cs2_pressure.savefig(CONFIG["PLOT_PHYSICS_CS2_VS_PRESSURE"], dpi=400)
    plt.close(fig_cs2_pressure)

    # --- Formatting Plot 5: EoS Bands ---
    ax_eos_bands.set_xscale("log")
    ax_eos_bands.set_yscale("log")
    ax_eos_bands.set_xlabel(r"Energy Density $\epsilon$ [MeV/fm$^3$]")
    ax_eos_bands.set_ylabel(r"Pressure $P$ [MeV/fm$^3$]")
    ax_eos_bands.set_title("Fundamental EoS Bands")
    ax_eos_bands.legend(handles=legend_elements, loc="lower right", framealpha=0.95)
    fig_eos_bands.tight_layout()
    fig_eos_bands.savefig(CONFIG["PLOT_PHYSICS_EOS_BANDS"], dpi=400)
    plt.close(fig_eos_bands)

    # --- Formatting Plot 6: Mass vs Central Density ---
    ax_mass_rho.set_xlim(CONFIG.get("PLOT_EPS_LIM", (0, 2500)))
    ax_mass_rho.set_ylim(CONFIG["PLOT_M_LIM"])
    ax_mass_rho.set_xlabel(r"Central Energy Density $\epsilon_c$ [MeV/fm$^3$]")
    ax_mass_rho.set_ylabel(r"Stellar Mass $M$ [$M_{\odot}$]")
    ax_mass_rho.set_title("Mass vs Central Density")
    ax_mass_rho.legend(handles=legend_elements, loc="lower right", framealpha=0.95)
    fig_mass_rho.tight_layout()
    fig_mass_rho.savefig(CONFIG["PLOT_PHYSICS_MASS_VS_RHO"], dpi=400)
    plt.close(fig_mass_rho)

    logger.info("done Saved Core Physics Plots.")
