import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.append(os.path.abspath("."))
try:
    from src.ml_perturb.data_pipeline import inject_observational_noise
except ImportError:
    def inject_observational_noise(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        np.random.seed(seed)
        noisy_df = df.copy()
        noisy_df['Mass'] = noisy_df['Mass'] + np.random.normal(0, 0.05 * noisy_df['Mass'])
        noisy_df['Radius'] = noisy_df['Radius'] + np.random.normal(0, 0.10 * noisy_df['Radius'])
        if 'Lambda' in noisy_df.columns:
            noisy_df['Lambda'] = np.abs(noisy_df['Lambda'] + np.random.normal(0, 0.20 * noisy_df['Lambda']))
            noisy_df['log10_Lambda'] = np.log10(np.clip(noisy_df['Lambda'], a_min=1e-10, a_max=None))
        elif 'LogLambda' in noisy_df.columns:
            noisy_df['log10_Lambda'] = noisy_df['LogLambda'] + np.random.normal(0, 0.086)
        return noisy_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PERTURB_VISUALIZER")

def load_all_curves(hadronic_dir, quark_dir):
    h_files = glob.glob(os.path.join(hadronic_dir, "*.parquet"))
    df_h = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in h_files], ignore_index=True) if h_files else pd.DataFrame()
    if not df_h.empty: df_h['Phase'] = 'Hadronic'

    q_files = glob.glob(os.path.join(quark_dir, "*.parquet"))
    df_q = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in q_files], ignore_index=True) if q_files else pd.DataFrame()
    if not df_q.empty: df_q['Phase'] = 'Quark'

    df = pd.concat([df_h, df_q], ignore_index=True)
    df = df[df['Radius'] > 0].copy()
    if 'Lambda' in df.columns:
        df = df[df['Lambda'] > 0]
        df['log10_Lambda'] = np.log10(df['Lambda'])
    elif 'LogLambda' in df.columns:
        df.rename(columns={'LogLambda': 'log10_Lambda'}, inplace=True)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Mass', 'Radius', 'log10_Lambda'], inplace=True)
    return df

def generate_individual_curve_plots(n_curves_per_phase=2, max_scatter_points_per_curve=600):
    logger.info("Generating Individual Curve Plot with specific error clouds...")
    
    clean_df = load_all_curves(os.path.join("data", "ml_ready_hadronic"), os.path.join("data", "ml_ready_quark"))
    if clean_df.empty: return

    # Randomly select curves
    np.random.seed(42)
    h_curves = np.random.choice(clean_df[clean_df['Phase'] == 'Hadronic']['Curve_ID'].unique(), 
                                min(n_curves_per_phase, clean_df[clean_df['Phase'] == 'Hadronic']['Curve_ID'].nunique()), 
                                replace=False)
    q_curves = np.random.choice(clean_df[clean_df['Phase'] == 'Quark']['Curve_ID'].unique(), 
                                min(n_curves_per_phase, clean_df[clean_df['Phase'] == 'Quark']['Curve_ID'].nunique()), 
                                replace=False)
    
    selected_curves = list(h_curves) + list(q_curves)
    sub_clean = clean_df[clean_df['Curve_ID'].isin(selected_curves)].copy()
    
    logger.info("Injecting noise to simulate error margins...")
    sub_noisy = inject_observational_noise(sub_clean)
    
    # Subsample noisy scatter for elegance
    subsampled_noisy_list = []
    for curve_id in selected_curves:
        c_df = sub_noisy[sub_noisy['Curve_ID'] == curve_id]
        if len(c_df) > max_scatter_points_per_curve:
            c_df = c_df.sample(n=max_scatter_points_per_curve, random_state=42)
        subsampled_noisy_list.append(c_df)
    sub_noisy_elegant = pd.concat(subsampled_noisy_list, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Use tab20 which has exactly 20 highly distinct colors
    colors = cm.tab20(np.linspace(0, 1, 20))
    curve_color_map = {}
    curve_style_map = {}
    curve_marker_map = {}
    
    for i, cid in enumerate(h_curves): 
        curve_color_map[cid] = colors[i]
        curve_style_map[cid] = '-'     # Solid line for Hadronic
        curve_marker_map[cid] = 'o'    # Circle scatter for Hadronic
        
    for i, cid in enumerate(q_curves): 
        curve_color_map[cid] = colors[10 + i]
        curve_style_map[cid] = '--'    # Dashed line for Quark
        curve_marker_map[cid] = '^'    # Triangle scatter for Quark

    # Plot Subplot 1: M-R
    ax = axes[0]
    for curve_id in selected_curves:
        color = curve_color_map[curve_id]
        lstyle = curve_style_map[curve_id]
        marker = curve_marker_map[curve_id]
        
        c_clean = sub_clean[sub_clean['Curve_ID'] == curve_id].sort_values('Mass')
        c_noisy = sub_noisy_elegant[sub_noisy_elegant['Curve_ID'] == curve_id]
        # Line
        ax.plot(c_clean['Radius'], c_clean['Mass'], color=color, linestyle=lstyle, linewidth=2.5, alpha=0.9, zorder=3)
        # Scatter
        ax.scatter(c_noisy['Radius'], c_noisy['Mass'], color=color, marker=marker, alpha=0.9, s=25, edgecolor='none', zorder=2)
        
    ax.set_title("Mass vs. Radius", fontsize=18)
    ax.set_xlabel("Radius (km)", fontsize=16)
    ax.set_ylabel(r"Mass ($M_\odot$)", fontsize=16)
    ax.set_xlim(left=5)  # Set lower limit to 5km
    ax.set_xlim(right=20)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # Plot Subplot 2: M-Lambda
    ax = axes[1]
    for curve_id in selected_curves:
        color = curve_color_map[curve_id]
        lstyle = curve_style_map[curve_id]
        marker = curve_marker_map[curve_id]
        
        c_clean = sub_clean[sub_clean['Curve_ID'] == curve_id].sort_values('Mass')
        c_noisy = sub_noisy_elegant[sub_noisy_elegant['Curve_ID'] == curve_id]
        ax.plot(c_clean['Mass'], c_clean['log10_Lambda'], color=color, linestyle=lstyle, linewidth=2.5, alpha=0.9, zorder=3)
        ax.scatter(c_noisy['Mass'], c_noisy['log10_Lambda'], color=color, marker=marker, alpha=0.35, s=25, edgecolor='none', zorder=2)
        
    ax.set_title(r"$\log_{10}(\Lambda)$ vs. Mass", fontsize=18)
    ax.set_xlabel(r"Mass ($M_\odot$)", fontsize=16)
    ax.set_ylabel(r"$\log_{10}(\Lambda)$", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', lw=2, label='Hadronic Curve (Solid)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.5, markersize=8, label='Hadronic Noise (Circles)'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, label='Quark Curve (Dashed)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black', alpha=0.5, markersize=8, label='Quark Noise (Triangles)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=14, frameon=False)
               
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_dir = os.path.join("plots_perturb", "ml_advanced")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "perturbation_effect_contours.pdf")
    
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Successfully generated new individual curve plots: {out_path}")

if __name__ == "__main__":
    generate_individual_curve_plots()
