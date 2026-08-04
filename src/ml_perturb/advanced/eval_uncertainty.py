import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ml_perturb.advanced.common import FEATURE_SETS, load_perturb_mlp, load_perturb_scaler, load_perturb_test, perturb_plot_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MC_DROPOUT_PERTURB")


def enable_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def eval_uncertainty():
    logger.info("Initializing MC Dropout Audit for Perturbed Pipeline...")
    scaler = load_perturb_scaler()
    plots_dir = perturb_plot_dir()

    for fset, features in FEATURE_SETS.items():
        test_df, x_test_scaled, _ = load_perturb_test(fset)
        x_test_for_inverse = test_df[["Mass", "Radius", "log10_Lambda"]].values
        x_test_raw = scaler.inverse_transform(x_test_for_inverse)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_perturb_mlp(fset, input_dim=x_test_scaled.shape[1], device=device)
        model.eval()
        enable_dropout(model)

        x_test_tensor = torch.FloatTensor(x_test_scaled.copy()).to(device)
        predictions = []
        with torch.no_grad():
            for _ in range(100):
                preds = torch.sigmoid(model(x_test_tensor)).cpu().numpy().flatten()
                predictions.append(preds)

        predictions = np.array(predictions)
        mean_preds = np.mean(predictions, axis=0)
        variance_preds = np.var(predictions, axis=0)
        compactness = x_test_raw[:, 0] / x_test_raw[:, 1]
        plot_path = plots_dir / f"uncertainty_calibration_{fset}.pdf"

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(compactness, variance_preds, c=mean_preds, cmap="coolwarm", alpha=0.8, edgecolor="k")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Mean uncalibrated score (0=APR-1, 1=CFL4)")
        plt.title(f"Epistemic Uncertainty vs Compactness (Perturbed {fset})", fontsize=15)
        plt.xlabel("Compactness (C = M/R)", fontsize=12)
        plt.ylabel("Predictive Variance (Epistemic Uncertainty)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {plot_path}")


if __name__ == "__main__":
    eval_uncertainty()
