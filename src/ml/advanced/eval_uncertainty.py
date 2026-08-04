import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ml.advanced.common import FEATURES, clean_plot_dir, load_clean_mlp, load_clean_scaler, load_clean_test


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MC_DROPOUT")


def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def eval_uncertainty():
    scaler = load_clean_scaler()
    test_df, X_test_scaled, _ = load_clean_test()
    y_test = test_df["Label"].values
    X_test_raw = scaler.inverse_transform(X_test_scaled)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_clean_mlp(input_dim=len(FEATURES), device=device)
    model.eval()
    enable_dropout(model)

    n_iterations = 100
    logger.info(f"Performing Monte Carlo Dropout inference (N={n_iterations})...")
    X_test_tensor = torch.FloatTensor(X_test_scaled.copy()).to(device)
    predictions = []
    with torch.no_grad():
        for _ in range(n_iterations):
            preds = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
            predictions.append(preds)

    predictions = np.array(predictions)
    mean_preds = np.mean(predictions, axis=0)
    variance_preds = np.var(predictions, axis=0)

    plot_path = clean_plot_dir() / "uncertainty_calibration.pdf"
    logger.info(f"Generating Uncertainty vs. Compactness calibration plot to {plot_path}...")

    compactness = X_test_raw[:, 0] / X_test_raw[:, 1]
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(compactness, variance_preds, c=mean_preds, cmap="coolwarm", alpha=0.8, edgecolor="k")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Mean Predicted Probability (0=Hadronic, 1=Quark)")
    plt.title("Epistemic Uncertainty (MC Dropout) vs Compactness", fontsize=15)
    plt.xlabel("Compactness (C = M/R)", fontsize=12)
    plt.ylabel("Predictive Variance (Epistemic Uncertainty)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Uncertainty Quantification complete for {len(y_test)} test samples.")


if __name__ == "__main__":
    eval_uncertainty()
