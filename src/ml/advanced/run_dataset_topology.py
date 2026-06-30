import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DATASET_TOPOLOGY")

def run_dataset_topology():
    logger.info("Initializing Dataset Topology & Envelope Analysis...")
    
    TENSOR_DIR = os.path.join("data", "ml_tensors")
    SCALER_PATH = os.path.join(TENSOR_DIR, "scaler.joblib")
    
    # 1. Load Data
    logger.info("Loading full dataset (Train + Val + Test)...")
    df_list = []
    for split in ["train.parquet", "val.parquet", "test.parquet"]:
        path = os.path.join(TENSOR_DIR, split)
        if os.path.exists(path):
            df_list.append(pd.read_parquet(path, engine='pyarrow'))
            
    if not df_list:
        logger.error("No parquet files found in data/ml_tensors/")
        return
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    features = ['Mass', 'Radius', 'log10_Lambda']
    X_scaled = full_df[features].values
    labels = full_df['Label'].values
    
    # 2. Inverse Transform to get Physical Values
    scaler = joblib.load(SCALER_PATH)
    X_raw = scaler.inverse_transform(X_scaled)
    
    df_raw = pd.DataFrame(X_raw, columns=features)
    df_raw['Phase'] = ['Quark Star' if l == 1 else 'Hadronic Star' for l in labels]
    
    output_dir = os.path.join("plots", "ml_advanced")
    os.makedirs(output_dir, exist_ok=True)
    
    # =====================================================================
    # PLOT 1: 1D Probability Density Functions (PD vs Parameter)
    # =====================================================================
    logger.info("Generating Probability Density (PD) plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    palette = {'Hadronic Star': 'tab:blue', 'Quark Star': 'tab:red'}
    
    for i, feature in enumerate(features):
        sns.kdeplot(data=df_raw, x=feature, hue='Phase', fill=True, common_norm=False, 
                    palette=palette, alpha=0.5, ax=axes[i], linewidth=2)
        axes[i].set_title(f'Probability Density vs {feature}', fontsize=14)
        axes[i].set_ylabel('Probability Density')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    pd_plot_path = os.path.join(output_dir, "probability_density_1D.pdf")
    plt.savefig(pd_plot_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PD plots to {pd_plot_path}")

    # =====================================================================
    # PLOT 2: M-R Envelopes with Confidence Intervals
    # =====================================================================
    logger.info("Generating M-R Envelopes with 68% and 95% CIs...")
    
    plt.figure(figsize=(10, 8))
    
    # Instead of manual 1D Mass binning (which causes artifacts due to TOV step density and curve snapbacks),
    # we use a robust 2D Kernel Density Estimate to draw the true 68% (1-sigma) and 95% (2-sigma) spatial contours.
    palette = {'Hadronic Star': 'tab:blue', 'Quark Star': 'tab:red'}
    
    sns.kdeplot(
        data=df_raw, 
        x='Radius', 
        y='Mass', 
        hue='Phase', 
        fill=True, 
        alpha=0.4, 
        palette=palette,
        levels=[0.05, 0.32, 1.0],  # 95% CI (0.05 to 1.0), 68% CI (0.32 to 1.0)
        common_norm=False
    )
    
    # Explicitly calculate and highlight the Overlap Region (Degeneracy Zone)
    xbins = np.linspace(8, 18, 150)
    ybins = np.linspace(0, 3.2, 150)
    
    had_df = df_raw[df_raw['Phase'] == 'Hadronic Star']
    quark_df = df_raw[df_raw['Phase'] == 'Quark Star']
    
    H_had, xedges, yedges = np.histogram2d(had_df['Radius'], had_df['Mass'], bins=[xbins, ybins])
    H_quark, _, _ = np.histogram2d(quark_df['Radius'], quark_df['Mass'], bins=[xbins, ybins])
    
    from scipy.ndimage import gaussian_filter
    H_had_smooth = gaussian_filter(H_had, sigma=2.0)
    H_quark_smooth = gaussian_filter(H_quark, sigma=2.0)
    
    # Define where both models have physically significant density (e.g., > 1% of their max density)
    mask_had = H_had_smooth > (0.01 * H_had_smooth.max())
    mask_quark = H_quark_smooth > (0.01 * H_quark_smooth.max())
    overlap_mask = mask_had & mask_quark
    
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
    
    # Highlight the overlap zone with a golden contour and dashed outline
    plt.contourf(X, Y, overlap_mask, levels=[0.5, 1.5], colors=['#FFD700'], alpha=0.25)
    plt.contour(X, Y, overlap_mask, levels=[0.5], colors=['#FFD700'], linewidths=2, linestyles='--')
    
    plt.title('Theoretical M-R 2D Envelopes (68% & 95% Density Contours)', fontsize=16)
    plt.xlabel('Radius (km)', fontsize=14)
    plt.ylabel(r'Mass ($M_\odot$)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Restrict axes to reasonable physical limits
    plt.xlim(8, 18)
    plt.ylim(0, 3.2)    
    # Custom legend to avoid duplicates
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Create custom legend entries
    had_patch = mpatches.Patch(color='tab:blue', alpha=0.4, label='Hadronic Star (68/95% CI)')
    quark_patch = mpatches.Patch(color='tab:red', alpha=0.4, label='Quark Star (68/95% CI)')
    overlap_patch = mpatches.Patch(facecolor='#FFD700', alpha=0.25, edgecolor='#FFD700', linestyle='--', linewidth=2, label='Degeneracy Zone (Overlap)')
    
    plt.legend(handles=[had_patch, quark_patch, overlap_patch], loc='best', fontsize=11)
    
    mr_env_path = os.path.join(output_dir, "mr_envelopes_ci.pdf")
    plt.savefig(mr_env_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved M-R Envelopes to {mr_env_path}")
    logger.info("Dataset Topology & Envelope Analysis complete.")

if __name__ == "__main__":
    run_dataset_topology()
