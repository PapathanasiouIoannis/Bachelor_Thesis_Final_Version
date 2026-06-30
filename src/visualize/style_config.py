# src/visualize/style_config.py

"""
Centralized Matplotlib style configuration for figures.
Ensures uniform aesthetics, PDF vector embedding (Type 42), colorblind-safe
palettes, and correct LaTeX math formatting across all plots.
"""

import matplotlib.pyplot as plt

# ==========================================
# 1. COLOR PALETTE (Publication & Colorblind Safe)
# ==========================================
# optimized for high contrast, distinct monochrome printing, and visual clarity.
COLORS = {
    "H_main": "#0072B2",  # blue (Hadronic)
    "Q_main": "#D55E00",  # vermilion/Orange (Quark)
    "H_fade": "#56B4E9",  # light Blue for fills/confidence bands
    "Q_fade": "#E69F00",  # orange/Yellow for fills/confidence bands
    "Constraint": "#000000",  # black for observational limits (J0740, GW170817)
    "Guide": "#999999",  # gray for grids and reference lines
    "FalsePos": "#CC79A7",  # purplish Pink (for ML false positives)
    "FalseNeg": "#009E73",  # bluish Green (for ML false negatives)
}


def set_paper_style() -> None:
    """
    Configures Matplotlib global rcParams for ApJ/PRD/MNRAS quality plots.

    Configuration Details:
    - Font: Serif body (Times-like), Computer Modern for math ($...$).
    - Output: TrueType fonts (Type 42) embedded for journal PDF compliance.
    - DPI: 300 for high-resolution rasterized backdrops.
    - Ticks: Inward facing, present on all four sides.
    - Grid: Subtle dashed lines to guide the eye without cluttering.
    """
    import logging
    import warnings

    logging.getLogger("matplotlib.backends.backend_pdf").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")

    plt.rcParams.update(
        {
            # --- LAYOUT & RESOLUTION ---
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # --- FONTS & TEXT ---
            "font.family": "serif", 
            "font.serif": ["Computer Modern Roman", "Times New Roman"],
            "text.usetex": False,  # strictly enforce local TeX distribution for thesis
            "mathtext.fontset": "cm",  # computer Modern for LaTeX math
            "axes.formatter.use_mathtext": True,  # proper math formatting on axes
            "pdf.fonttype": 42,  # embed TrueType fonts (editable/compliant)
            "ps.fonttype": 42,
            # --- SIZES ---
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            # --- TICKS ---
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,  # ticks on all framing edges
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            # --- LINES & GEOMETRY ---
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 1.0,  # frame thickness
            # --- GRID ---
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": COLORS["Guide"],
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            # --- LEGEND ---
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": COLORS["Guide"],
            "legend.fancybox": False,  # square corners 
            "legend.loc": "best",
        }
    )
