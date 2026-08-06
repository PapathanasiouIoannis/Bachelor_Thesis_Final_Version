import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap

from src.ml_perturb.advanced.common import FEATURE_SETS, load_perturb_test, perturb_plot_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("UMAP_PERTURB")


def run_umap_analysis():
    logger.info("Initializing UMAP Audit for Perturbed Pipeline...")
    plots_dir = perturb_plot_dir()
    palette = {"Hadronic Star": "tab:blue", "Quark Star": "tab:red"}

    for fset in FEATURE_SETS:
        _, x_test, y_test = load_perturb_test(fset)
        logger.info(f"[{fset}] Loaded Test Set Shape: {x_test.shape}. Commencing UMAP projection...")

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding = reducer.fit_transform(x_test)
        plot_path = plots_dir / f"umap_topology_{fset}.pdf"

        df_umap = pd.DataFrame(
            {
                "UMAP1": embedding[:, 0],
                "UMAP2": embedding[:, 1],
                "Phase": ["Quark Star" if val == 1 else "Hadronic Star" for val in y_test],
            }
        )

        plt.figure(figsize=(10, 8))
        sns.kdeplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Phase", fill=True, alpha=0.5, palette=palette, levels=10, thresh=0.05)
        sns.scatterplot(data=df_umap, x="UMAP1", y="UMAP2", hue="Phase", palette=palette, s=5, alpha=0.15, edgecolor=None, legend=False)
        plt.title(f"UMAP Projection of Equation of State Manifolds (Perturbed {fset})", fontsize=16)
        plt.xlabel("UMAP Dimension 1", fontsize=12)
        plt.ylabel("UMAP Dimension 2", fontsize=12)
        plt.legend(title="EoS Phase State", loc="best")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {plot_path}")


if __name__ == "__main__":
    run_umap_analysis()
