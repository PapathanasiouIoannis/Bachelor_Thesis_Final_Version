import logging

import matplotlib.pyplot as plt
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.ml.advanced.common import clean_plot_dir, load_clean_mlp, load_clean_test, load_clean_xgb


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CALIBRATION")


def run_calibration():
    logger.info("Initializing Calibration Audit...")
    _, X_test_scaled, y_test = load_clean_test()

    xgb_model = load_clean_xgb()
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = load_clean_mlp(input_dim=X_test_scaled.shape[1], device=device)
    with torch.no_grad():
        outputs = mlp_model(torch.FloatTensor(X_test_scaled.copy()).to(device))
        mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()

    xgb_brier = brier_score_loss(y_test, xgb_probs)
    mlp_brier = brier_score_loss(y_test, mlp_probs)
    logger.info(f"XGBoost Brier Score: {xgb_brier:.5f}")
    logger.info(f"MLP Brier Score: {mlp_brier:.5f}")

    xgb_fraction, xgb_mean = calibration_curve(y_test, xgb_probs, n_bins=10, strategy="quantile")
    mlp_fraction, mlp_mean = calibration_curve(y_test, mlp_probs, n_bins=10, strategy="quantile")

    plot_path = clean_plot_dir() / "calibration_curve.pdf"
    logger.info(f"Generating Calibration Reliability Diagram to {plot_path}...")

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated (Ideal)")
    plt.plot(xgb_mean, xgb_fraction, "s-", color="tab:blue", label=f"XGBoost (Brier={xgb_brier:.4f})", linewidth=2, markersize=8)
    plt.plot(mlp_mean, mlp_fraction, "o-", color="tab:red", label=f"MLP (Brier={mlp_brier:.4f})", linewidth=2, markersize=8)
    plt.xlabel("Mean uncalibrated model score", fontsize=14)
    plt.ylabel("Observed CFL4 fraction", fontsize=14)
    plt.title("Calibration Curve (Reliability Diagram): XGBoost vs MLP", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info("Calibration Audit complete.")


if __name__ == "__main__":
    run_calibration()
