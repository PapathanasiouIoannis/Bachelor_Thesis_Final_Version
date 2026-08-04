import logging

import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.ml_perturb.advanced.common import FEATURE_SETS, load_perturb_mlp, load_perturb_test, load_perturb_xgb, perturb_plot_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CALIBRATION_PERTURB")


def run_calibration():
    logger.info("Initializing Calibration Audit for Perturbed Pipeline...")
    output_dir = perturb_plot_dir()

    for fset in FEATURE_SETS:
        _, x_test, y_test = load_perturb_test(fset)
        xgb_model = load_perturb_xgb(fset)
        xgb_probs = xgb_model.predict_proba(x_test)[:, 1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = load_perturb_mlp(fset, input_dim=x_test.shape[1], device=device)
        with torch.no_grad():
            outputs = mlp_model(torch.FloatTensor(x_test.copy()).to(device))
            mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()

        xgb_brier = brier_score_loss(y_test, xgb_probs)
        mlp_brier = brier_score_loss(y_test, mlp_probs)
        logger.info(f"[{fset}] XGBoost Brier Score: {xgb_brier:.5f}")
        logger.info(f"[{fset}] MLP Brier Score: {mlp_brier:.5f}")

        xgb_fraction, xgb_mean = calibration_curve(y_test, xgb_probs, n_bins=10, strategy="quantile")
        mlp_fraction, mlp_mean = calibration_curve(y_test, mlp_probs, n_bins=10, strategy="quantile")
        plot_path = output_dir / f"calibration_curve_{fset}.pdf"

        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated (Ideal)")
        plt.plot(xgb_mean, xgb_fraction, "s-", color="tab:blue", label=f"XGBoost (Brier={xgb_brier:.4f})", linewidth=2, markersize=8)
        plt.plot(mlp_mean, mlp_fraction, "o-", color="tab:red", label=f"MLP (Brier={mlp_brier:.4f})", linewidth=2, markersize=8)
        plt.xlabel("Mean uncalibrated model score", fontsize=14)
        plt.ylabel("Observed CFL4 fraction", fontsize=14)
        plt.title(f"Calibration Curve: XGBoost vs MLP (Perturbed {fset})", fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {plot_path}")


if __name__ == "__main__":
    run_calibration()
