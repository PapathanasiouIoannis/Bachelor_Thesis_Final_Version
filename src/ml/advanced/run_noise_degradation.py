import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.ml.advanced.common import clean_plot_dir, load_clean_mlp, load_clean_scaler, load_clean_xgb
from src.runtime import fast_enabled


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NOISE_DEGRADATION")


def run_noise_degradation():
    logger.info("Initializing Systematic Noise Degradation Analysis (XGBoost vs MLP)...")

    scaler = load_clean_scaler()
    xgb_model = load_clean_xgb()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = load_clean_mlp(input_dim=3, device=device)

    candidates = {
        "GW170817 (Hadronic)": {"M": 1.435, "R": 12.327, "L": 2.269},
        "HESS J1731-347 (Quark)": {"M": 0.779, "R": 10.285, "L": 2.822},
        "Boundary Star": {"M": 1.785, "R": 12.196, "L": 1.580},
    }

    noise_levels = np.linspace(0.0, 1.0, 11 if fast_enabled() else 51)
    n_samples = 500 if fast_enabled() else 5000
    rng = np.random.default_rng(42)

    xgb_results = {name: {"expected": [], "lb": [], "ub": []} for name in candidates}
    mlp_results = {name: {"expected": [], "lb": [], "ub": []} for name in candidates}

    logger.info("Running Monte Carlo sweeps...")
    for name, params in candidates.items():
        m_true, r_true, l_true = params["M"], params["R"], params["L"]
        logger.info(f"Processing Candidate {name} (M={m_true}, R={r_true}, L={l_true})...")
        for noise in tqdm(noise_levels, desc=f"Lambda Noise Sweep {name}", leave=False):
            raw_lambda_true = 10**l_true
            raw_lambda_err = raw_lambda_true * noise
            mass_samples = np.full(n_samples, m_true)
            radius_samples = np.full(n_samples, r_true)
            raw_lambda_samples = rng.normal(raw_lambda_true, raw_lambda_err, n_samples)
            raw_lambda_samples = np.clip(raw_lambda_samples, a_min=1e-10, a_max=None)
            lambda_samples = np.log10(raw_lambda_samples)

            x_mc = pd.DataFrame({"Mass": mass_samples, "Radius": radius_samples, "log10_Lambda": lambda_samples})
            x_mc_scaled = pd.DataFrame(scaler.transform(x_mc), columns=x_mc.columns)

            xgb_probs = xgb_model.predict_proba(x_mc_scaled)[:, 1]
            xgb_results[name]["expected"].append(np.mean(xgb_probs))
            xgb_results[name]["lb"].append(np.percentile(xgb_probs, 16.0))
            xgb_results[name]["ub"].append(np.percentile(xgb_probs, 84.0))

            with torch.no_grad():
                x_tensor = torch.FloatTensor(x_mc_scaled.values.copy()).to(device)
                mlp_probs = torch.sigmoid(mlp_model(x_tensor)).cpu().numpy().flatten()
            mlp_results[name]["expected"].append(np.mean(mlp_probs))
            mlp_results[name]["lb"].append(np.percentile(mlp_probs, 16.0))
            mlp_results[name]["ub"].append(np.percentile(mlp_probs, 84.0))

    logger.info("Generating degradation plots...")
    output_dir = clean_plot_dir()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True, sharex=True)
    noise_pct = noise_levels * 100

    for i, (name, stats) in enumerate(xgb_results.items()):
        ax = axes[0, i]
        expected = np.array(stats["expected"]) * 100
        lb = np.array(stats["lb"]) * 100
        ub = np.array(stats["ub"]) * 100
        ax.fill_between(noise_pct, lb, ub, color="blue", alpha=0.2, label="68% CI")
        ax.plot(noise_pct, expected, color="darkblue", linewidth=2, label="Expected Prob")
        ax.set_title(f"XGBoost | {name}")
        if i == 0:
            ax.set_ylabel("Uncalibrated CFL4 model score (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.4, linestyle="--")
        ax.legend(loc="best")

    for i, (name, stats) in enumerate(mlp_results.items()):
        ax = axes[1, i]
        expected = np.array(stats["expected"]) * 100
        lb = np.array(stats["lb"]) * 100
        ub = np.array(stats["ub"]) * 100
        ax.fill_between(noise_pct, lb, ub, color="green", alpha=0.2, label="68% CI")
        ax.plot(noise_pct, expected, color="darkgreen", linewidth=2, label="Expected Prob")
        ax.set_title(f"MLP | {name}")
        ax.set_xlabel(r"$\Lambda$ Observational Noise Level (%)")
        if i == 0:
            ax.set_ylabel("Uncalibrated CFL4 model score (%)")
        ax.grid(True, alpha=0.4, linestyle="--")
        ax.legend(loc="best")

    plt.suptitle(r"Pure $\Lambda$ Observational Noise Degradation (M, R perfectly known)", fontsize=18, y=1.02)
    plt.tight_layout()
    plot_path = output_dir / "noise_degradation_cones_comparison.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Analysis complete. Comparison degradation cones saved to {plot_path}")


if __name__ == "__main__":
    run_noise_degradation()
