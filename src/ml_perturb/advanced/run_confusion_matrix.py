import logging

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.ml_perturb.advanced.common import FEATURE_SETS, load_perturb_mlp, load_perturb_test, load_perturb_xgb, perturb_plot_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CONFUSION_MATRIX_PERTURB")


def plot_confusion_matrices(y_true, y_pred, title_prefix, plot_path):
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    labels = ["Hadronic (0)", "Quark (1)"]
    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="viridis", cbar=True, ax=ax1, xticklabels=labels, yticklabels=labels)
    ax1.set_title(f"{title_prefix} - Raw Counts", fontsize=14)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="viridis", cbar=True, ax=ax2, xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=1.0)
    ax2.set_title(f"{title_prefix} - Normalized (Row Probabilities)", fontsize=14)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def run_confusion_matrix():
    logger.info("Initializing Confusion Matrix Audit for Perturbed Pipeline...")
    plots_dir = perturb_plot_dir()

    for fset in FEATURE_SETS:
        logger.info(f"--- Evaluating Confusion Matrices for {fset} ---")
        _, x_test, y_test = load_perturb_test(fset)

        xgb_model = load_perturb_xgb(fset)
        xgb_preds = xgb_model.predict(x_test)
        plot_confusion_matrices(y_test, xgb_preds, f"XGBoost ({fset})", plots_dir / f"confusion_matrix_xgboost_{fset}.pdf")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = load_perturb_mlp(fset, input_dim=x_test.shape[1], device=device)
        with torch.no_grad():
            outputs = mlp_model(torch.FloatTensor(x_test.copy()).to(device))
            mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            mlp_preds = (mlp_probs >= 0.5).astype(int)
        plot_confusion_matrices(y_test, mlp_preds, f"MLP ({fset})", plots_dir / f"confusion_matrix_mlp_{fset}.pdf")

    logger.info("Perturbed confusion matrix audit complete.")


if __name__ == "__main__":
    run_confusion_matrix()
