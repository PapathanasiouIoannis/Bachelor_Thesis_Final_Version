import logging

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.ml.advanced.common import clean_plot_dir, load_clean_mlp, load_clean_test, load_clean_xgb


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CONFUSION_MATRIX_CLEAN")


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
    logger.info("Initializing Confusion Matrix Audit for Clean Pipeline...")
    _, X_test, y_test = load_clean_test()
    plots_dir = clean_plot_dir()

    logger.info("Evaluating XGBoost Confusion Matrix...")
    xgb_model = load_clean_xgb()
    xgb_preds = xgb_model.predict(X_test)
    plot_confusion_matrices(y_test, xgb_preds, "XGBoost", plots_dir / "confusion_matrix_xgboost.pdf")

    logger.info("Evaluating MLP Confusion Matrix...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = load_clean_mlp(input_dim=X_test.shape[1], device=device)
    with torch.no_grad():
        outputs = mlp_model(torch.FloatTensor(X_test.copy()).to(device))
        mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        mlp_preds = (mlp_probs >= 0.5).astype(int)
    plot_confusion_matrices(y_test, mlp_preds, "MLP", plots_dir / "confusion_matrix_mlp.pdf")
    logger.info("Confusion Matrix Audit complete.")


if __name__ == "__main__":
    run_confusion_matrix()
