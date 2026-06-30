import os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UMAP_TOPOLOGY")

def run_umap_analysis():
    # 1. Data Ingestion (Strictly Pristine Test Set)
    DATA_DIR = "data"
    TENSOR_DIR = os.path.join(DATA_DIR, "ml_tensors")
    
    logger.info(f"Loading strictly pristine test tensor from {TENSOR_DIR}...")
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')

    X_test = test_df.drop(columns=['Label']).values
    y_test = test_df['Label'].values
    
    logger.info(f"Loaded Test Set Shape: {X_test.shape}. Commencing UMAP projection...")

    # 2. Dimensionality Reduction (UMAP)
    # n_neighbors=15, min_dist=0.1 are standard for balancing local and global structure
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_test)

    # 3. Visualization
    plots_dir = os.path.join("plots", "ml_advanced")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_path = os.path.join(plots_dir, "umap_topology.pdf")
    logger.info(f"Generating high-quality UMAP 2D Topology map to {plot_path}...")
    
    plt.figure(figsize=(10, 8))
    
    df_umap = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'Phase': ['Quark Star' if val == 1 else 'Hadronic Star' for val in y_test]
    })
    
    # 1. Density Contours (The main body of the data)
    sns.kdeplot(
        data=df_umap,
        x='UMAP1',
        y='UMAP2',
        hue='Phase',
        fill=True,
        alpha=0.5,
        palette={'Hadronic Star': 'tab:blue', 'Quark Star': 'tab:red'},
        levels=10,
        thresh=0.05
    )
    
    # 2. Faint Scatter Overlay (To show the boundaries/outliers)
    sns.scatterplot(
        data=df_umap,
        x='UMAP1', 
        y='UMAP2', 
        hue='Phase', 
        palette={'Hadronic Star': 'tab:blue', 'Quark Star': 'tab:red'},
        s=5, 
        alpha=0.15, 
        edgecolor=None,
        legend=False
    )

    plt.title('UMAP Projection of Equation of State Manifolds', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='EoS Phase State', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    logger.info("Topological Data Analysis complete.")

if __name__ == "__main__":
    run_umap_analysis()
