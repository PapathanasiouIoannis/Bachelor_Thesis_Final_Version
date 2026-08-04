import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.ml_perturb.advanced.common import FEATURE_SETS, load_perturb_mlp, load_perturb_scaler, load_perturb_xgb, perturb_plot_dir
from src.runtime import fast_enabled


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MC_INFERENCE_PERTURB")


def run_mc_inference():
    logger.info("Initializing MC Observational Inference for Perturbed Pipeline...")

    n_samples = 1000 if fast_enabled() else 10000
    m_obs, m_err = 1.4, 0.05
    r_obs, r_err = 11.5, 0.5
    rng = np.random.default_rng(42)
    mass_samples = rng.normal(m_obs, m_err, n_samples)
    radius_samples = rng.normal(r_obs, r_err, n_samples)

    scaler = load_perturb_scaler()
    log10_lambda_mean = scaler.mean_[2]
    lambda_samples = np.full(n_samples, log10_lambda_mean)
    x_mc_3cols = pd.DataFrame({"Mass": mass_samples, "Radius": radius_samples, "log10_Lambda": lambda_samples})
    x_mc_scaled_3cols = pd.DataFrame(scaler.transform(x_mc_3cols), columns=x_mc_3cols.columns)
    output_dir = perturb_plot_dir()

    for fset, features in FEATURE_SETS.items():
        x_mc_scaled = x_mc_scaled_3cols[features]
        xgb_model = load_perturb_xgb(fset)
        xgb_probs = xgb_model.predict_proba(x_mc_scaled)[:, 1]
        xgb_expected = np.mean(xgb_probs)
        xgb_lb = np.percentile(xgb_probs, 2.5)
        xgb_ub = np.percentile(xgb_probs, 97.5)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = load_perturb_mlp(fset, input_dim=x_mc_scaled.shape[1], device=device)
        with torch.no_grad():
            outputs = mlp_model(torch.FloatTensor(x_mc_scaled.values.copy()).to(device))
            mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        mlp_expected = np.mean(mlp_probs)
        mlp_lb = np.percentile(mlp_probs, 2.5)
        mlp_ub = np.percentile(mlp_probs, 97.5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        scatter1 = ax1.scatter(mass_samples, radius_samples, c=xgb_probs, cmap="coolwarm", alpha=0.6, edgecolors="none", s=10)
        ax1.errorbar(m_obs, r_obs, xerr=m_err, yerr=r_err, fmt="k+", markersize=15, capsize=3, linewidth=1.5, label="Mean Obs +/- 1 sigma")
        ax1.set_xlabel(r"Mass ($M_\odot$)")
        ax1.set_ylabel("Radius (km)")
        ax1.set_title(f"XGBoost Topology (Perturbed {fset})\nExpected: {xgb_expected:.1%} [{xgb_lb:.1%} - {xgb_ub:.1%}]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")
        fig.colorbar(scatter1, ax=ax1, label="Probability (Quark Star)")

        scatter2 = ax2.scatter(mass_samples, radius_samples, c=mlp_probs, cmap="coolwarm", alpha=0.6, edgecolors="none", s=10)
        ax2.errorbar(m_obs, r_obs, xerr=m_err, yerr=r_err, fmt="k+", markersize=15, capsize=3, linewidth=1.5, label="Mean Obs +/- 1 sigma")
        ax2.set_xlabel(r"Mass ($M_\odot$)")
        ax2.set_ylabel("Radius (km)")
        ax2.set_title(f"MLP Topology (Perturbed {fset})\nExpected: {mlp_expected:.1%} [{mlp_lb:.1%} - {mlp_ub:.1%}]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")
        fig.colorbar(scatter2, ax=ax2, label="Probability (Quark Star)")

        plt.suptitle(f"Monte Carlo Observational Inference (N={n_samples})", fontsize=16)
        plt.tight_layout()
        plot_path = output_dir / f"mc_observational_noise_combined_{fset}.pdf"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {plot_path}")


if __name__ == "__main__":
    run_mc_inference()
