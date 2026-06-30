import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ALL_CURVES")

def plot_all_curves():
    logger.info("Initializing Raw EoS Curve Plotter...")
    
    DATA_DIR = "data"
    HADRONIC_DIR = os.path.join(DATA_DIR, "ml_ready_hadronic")
    QUARK_DIR = os.path.join(DATA_DIR, "ml_ready_quark")
    
    # 1. Load Data with Curve_ID intact
    logger.info("Loading Hadronic data...")
    had_files = glob.glob(os.path.join(HADRONIC_DIR, "*.parquet"))
    if not had_files:
        logger.error(f"No parquet files found in {HADRONIC_DIR}")
        return
    df_had = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in had_files], ignore_index=True)
    
    logger.info("Loading Quark data...")
    quark_files = glob.glob(os.path.join(QUARK_DIR, "*.parquet"))
    if not quark_files:
        logger.error(f"No parquet files found in {QUARK_DIR}")
        return
    df_quark = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in quark_files], ignore_index=True)

    # 2. Limit number of curves to prevent RAM/Viewer crash
    # PDF viewers will crash if you give them 100,000 vector lines. 
    # We will plot up to 3000 randomly selected curves from each class, rendering to a high-res PNG.
    MAX_CURVES = 3000
    
    had_curve_ids = df_had['Curve_ID'].unique()
    if len(had_curve_ids) > MAX_CURVES:
        logger.info(f"Subsampling Hadronic curves from {len(had_curve_ids)} down to {MAX_CURVES}...")
        np.random.seed(42)
        selected_had = np.random.choice(had_curve_ids, MAX_CURVES, replace=False)
        df_had = df_had[df_had['Curve_ID'].isin(selected_had)]
        
    quark_curve_ids = df_quark['Curve_ID'].unique()
    if len(quark_curve_ids) > MAX_CURVES:
        logger.info(f"Subsampling Quark curves from {len(quark_curve_ids)} down to {MAX_CURVES}...")
        np.random.seed(42)
        selected_quark = np.random.choice(quark_curve_ids, MAX_CURVES, replace=False)
        df_quark = df_quark[df_quark['Curve_ID'].isin(selected_quark)]

    # 3. Plotting
    output_dir = os.path.join("plots", "ml_advanced")
    os.makedirs(output_dir, exist_ok=True)
    # Using PNG because PDFs with thousands of lines become 500MB and crash computers
    plot_path = os.path.join(output_dir, "all_curves_raw.png") 
    
    logger.info("Drawing lines... this might take a moment.")
    
    # Use a dark background to make the overlapping lines glow
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    
    # Plot Hadronic (Blue)
    logger.info("Plotting Hadronic curves...")
    for curve_id, group in tqdm(df_had.groupby('Curve_ID'), desc="Hadronic", leave=False):
        # Sort by Mass to ensure lines draw cleanly from bottom to top
        group = group.sort_values('Mass')
        ax.plot(group['Radius'], group['Mass'], color='dodgerblue', alpha=0.3, linewidth=1)
        
    # Plot Quark (Red)
    logger.info("Plotting Quark curves...")
    for curve_id, group in tqdm(df_quark.groupby('Curve_ID'), desc="Quark", leave=False):
        group = group.sort_values('Mass')
        ax.plot(group['Radius'], group['Mass'], color='crimson', alpha=0.3, linewidth=1)

    # Custom invisible lines just for the legend
    import matplotlib.lines as mlines
    had_line = mlines.Line2D([], [], color='dodgerblue', linewidth=3, label=f'Hadronic ({len(df_had["Curve_ID"].unique())} curves)')
    quark_line = mlines.Line2D([], [], color='crimson', linewidth=3, label=f'Quark ({len(df_quark["Curve_ID"].unique())} curves)')
    ax.legend(handles=[had_line, quark_line], loc='best', fontsize=14)

    ax.set_title('Raw M-R EoS Curves', fontsize=18, pad=20)
    ax.set_xlabel('Radius (km)', fontsize=14)
    ax.set_ylabel(r'Mass ($M_\odot$)', fontsize=14)
    
    # Restrict axes to reasonable physical limits just in case there are crazy outliers
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 3.5)
    ax.grid(True, linestyle='--', alpha=0.2)

    logger.info(f"Saving high-resolution plot to {plot_path}...")
    plt.savefig(plot_path, bbox_inches='tight', facecolor='black')
    plt.close()
    
    # Reset plot style to default so it doesn't mess up future plots in the notebook
    plt.style.use('default')
    
    logger.info("Plotting complete!")

if __name__ == "__main__":
    plot_all_curves()
