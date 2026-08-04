import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter

from src.ml.advanced.common import FEATURES, clean_plot_dir, load_clean_scaler
from src.runtime import require_paths, runtime_paths


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DATASET_TOPOLOGY")


def run_dataset_topology():
    logger.info("Initializing Dataset Topology & Envelope Analysis...")
    paths = runtime_paths()
    tensor_dir = paths.clean_tensor_dir
    split_paths = [tensor_dir / "train.parquet", tensor_dir / "val.parquet", tensor_dir / "test.parquet"]
    require_paths(split_paths, "Dataset topology")

    logger.info("Loading full dataset (Train + Val + Test)...")
    full_df = pd.concat([pd.read_parquet(path, engine="pyarrow") for path in split_paths], ignore_index=True)
    scaler = load_clean_scaler()
    x_raw = scaler.inverse_transform(full_df[FEATURES].values)

    df_raw = pd.DataFrame(x_raw, columns=FEATURES)
    df_raw["Phase"] = ["Quark Star" if label == 1 else "Hadronic Star" for label in full_df["Label"].values]
    output_dir = clean_plot_dir()

    logger.info("Generating Probability Density (PD) plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    palette = {"Hadronic Star": "tab:blue", "Quark Star": "tab:red"}
    for i, feature in enumerate(FEATURES):
        sns.kdeplot(data=df_raw, x=feature, hue="Phase", fill=True, common_norm=False, palette=palette, alpha=0.5, ax=axes[i], linewidth=2)
        axes[i].set_title(f"Probability Density vs {feature}", fontsize=14)
        axes[i].set_ylabel("Probability Density")
        axes[i].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    pd_plot_path = output_dir / "probability_density_1D.pdf"
    plt.savefig(pd_plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PD plots to {pd_plot_path}")

    logger.info("Generating M-R Envelopes with 68% and 95% CIs...")
    plt.figure(figsize=(10, 8))
    sns.kdeplot(
        data=df_raw,
        x="Radius",
        y="Mass",
        hue="Phase",
        fill=True,
        alpha=0.4,
        palette=palette,
        levels=[0.05, 0.32, 1.0],
        common_norm=False,
    )

    xbins = np.linspace(8, 18, 150)
    ybins = np.linspace(0, 3.2, 150)
    had_df = df_raw[df_raw["Phase"] == "Hadronic Star"]
    quark_df = df_raw[df_raw["Phase"] == "Quark Star"]
    h_had, xedges, yedges = np.histogram2d(had_df["Radius"], had_df["Mass"], bins=[xbins, ybins])
    h_quark, _, _ = np.histogram2d(quark_df["Radius"], quark_df["Mass"], bins=[xbins, ybins])
    h_had_smooth = gaussian_filter(h_had, sigma=2.0)
    h_quark_smooth = gaussian_filter(h_quark, sigma=2.0)
    mask_had = h_had_smooth > (0.01 * h_had_smooth.max())
    mask_quark = h_quark_smooth > (0.01 * h_quark_smooth.max())
    overlap_mask = mask_had & mask_quark

    x_grid, y_grid = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    plt.contourf(x_grid, y_grid, overlap_mask, levels=[0.5, 1.5], colors=["#FFD700"], alpha=0.25)
    plt.contour(x_grid, y_grid, overlap_mask, levels=[0.5], colors=["#FFD700"], linewidths=2, linestyles="--")
    plt.title("Theoretical M-R 2D Envelopes (68% & 95% Density Contours)", fontsize=16)
    plt.xlabel("Radius (km)", fontsize=14)
    plt.ylabel(r"Mass ($M_\odot$)", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(8, 18)
    plt.ylim(0, 3.2)

    had_patch = mpatches.Patch(color="tab:blue", alpha=0.4, label="Hadronic Star (68/95% CI)")
    quark_patch = mpatches.Patch(color="tab:red", alpha=0.4, label="Quark Star (68/95% CI)")
    overlap_patch = mpatches.Patch(facecolor="#FFD700", alpha=0.25, edgecolor="#FFD700", linestyle="--", linewidth=2, label="Degeneracy Zone (Overlap)")
    plt.legend(handles=[had_patch, quark_patch, overlap_patch], loc="best", fontsize=11)

    mr_env_path = output_dir / "mr_envelopes_ci.pdf"
    plt.savefig(mr_env_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved M-R Envelopes to {mr_env_path}")
    logger.info("Dataset Topology & Envelope Analysis complete.")


if __name__ == "__main__":
    run_dataset_topology()
