import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap

from src.ml.advanced.common import clean_plot_dir, load_clean_test


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UMAP_TOPOLOGY")


def run_umap_analysis():
    _, X_test, y_test = load_clean_test()
    logger.info(f"Loaded Test Set Shape: {X_test.shape}. Commencing UMAP projection...")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_test)

    plot_path = clean_plot_dir() / "umap_topology.pdf"
    logger.info(f"Generating high-quality UMAP 2D Topology map to {plot_path}...")

    df_umap = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "Phase": ["Quark Star" if val == 1 else "Hadronic Star" for val in y_test],
        }
    )

    plt.figure(figsize=(10, 8))
    palette = {"Hadronic Star": "tab:blue", "Quark Star": "tab:red"}
    sns.kdeplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Phase", fill=True, alpha=0.5, palette=palette, levels=10, thresh=0.05)
    sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Phase", palette=palette, s=5, alpha=0.15, edgecolor=None, legend=False)
    plt.title("UMAP Projection of Equation of State Manifolds", fontsize=16)
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title="EoS Phase State", loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info("Topological Data Analysis complete.")


if __name__ == "__main__":
    run_umap_analysis()
