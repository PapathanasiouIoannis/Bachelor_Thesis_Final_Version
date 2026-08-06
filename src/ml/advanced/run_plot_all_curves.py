import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.runtime import require_paths, runtime_paths


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ALL_CURVES")


def _load_ready_dir(directory):
    files = glob.glob(os.path.join(directory, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    return pd.concat([pd.read_parquet(path, engine="pyarrow") for path in files], ignore_index=True)


def plot_all_curves():
    logger.info("Initializing Raw EoS Curve Plotter...")
    paths = runtime_paths()
    require_paths([paths.hadronic_ready_dir, paths.quark_ready_dir], "Raw curve plotting")

    logger.info("Loading Hadronic data...")
    df_had = _load_ready_dir(str(paths.hadronic_ready_dir))
    logger.info("Loading Quark data...")
    df_quark = _load_ready_dir(str(paths.quark_ready_dir))

    max_curves = 3000
    rng = np.random.default_rng(42)
    had_curve_ids = df_had["Curve_ID"].unique()
    if len(had_curve_ids) > max_curves:
        selected_had = rng.choice(had_curve_ids, max_curves, replace=False)
        df_had = df_had[df_had["Curve_ID"].isin(selected_had)]
    quark_curve_ids = df_quark["Curve_ID"].unique()
    if len(quark_curve_ids) > max_curves:
        selected_quark = rng.choice(quark_curve_ids, max_curves, replace=False)
        df_quark = df_quark[df_quark["Curve_ID"].isin(selected_quark)]

    output_dir = paths.plots_root / "ml_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "all_curves_raw.png"

    logger.info("Drawing lines... this might take a moment.")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)

    logger.info("Plotting Hadronic curves...")
    for _, group in tqdm(df_had.groupby("Curve_ID"), desc="Hadronic", leave=False):
        group = group.sort_values("Mass")
        ax.plot(group["Radius"], group["Mass"], color="dodgerblue", alpha=0.3, linewidth=1)

    logger.info("Plotting Quark curves...")
    for _, group in tqdm(df_quark.groupby("Curve_ID"), desc="Quark", leave=False):
        group = group.sort_values("Mass")
        ax.plot(group["Radius"], group["Mass"], color="crimson", alpha=0.3, linewidth=1)

    import matplotlib.lines as mlines

    had_line = mlines.Line2D([], [], color="dodgerblue", linewidth=3, label=f"Hadronic ({len(df_had['Curve_ID'].unique())} curves)")
    quark_line = mlines.Line2D([], [], color="crimson", linewidth=3, label=f"Quark ({len(df_quark['Curve_ID'].unique())} curves)")
    ax.legend(handles=[had_line, quark_line], loc="best", fontsize=14)
    ax.set_title("Raw M-R EoS Curves", fontsize=18, pad=20)
    ax.set_xlabel("Radius (km)", fontsize=14)
    ax.set_ylabel(r"Mass ($M_\odot$)", fontsize=14)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 3.5)
    ax.grid(True, linestyle="--", alpha=0.2)

    logger.info(f"Saving high-resolution plot to {plot_path}...")
    plt.savefig(plot_path, bbox_inches="tight", facecolor="black")
    plt.close()
    plt.style.use("default")
    logger.info("Plotting complete!")


if __name__ == "__main__":
    plot_all_curves()
