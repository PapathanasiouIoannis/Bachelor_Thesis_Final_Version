import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.ml.advanced.common import clean_plot_dir, load_clean_mlp, load_clean_test, load_clean_xgb


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ROC_CURVE")


def run_roc():
    logger.info("Initializing ROC Curve Evaluation...")
    _, X_test_scaled, y_test = load_clean_test()

    xgb_model = load_clean_xgb()
    logger.info("Extracting XGBoost probabilistic predictions...")
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = load_clean_mlp(input_dim=X_test_scaled.shape[1], device=device)
    logger.info("Extracting MLP probabilistic predictions...")
    with torch.no_grad():
        outputs = mlp_model(torch.FloatTensor(X_test_scaled.copy()).to(device))
        mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()

    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
    mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_probs)
    xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
    mlp_roc_auc = auc(mlp_fpr, mlp_tpr)

    xgb_prec, xgb_rec, _ = precision_recall_curve(y_test, xgb_probs)
    mlp_prec, mlp_rec, _ = precision_recall_curve(y_test, mlp_probs)
    xgb_pr_auc = auc(xgb_rec, xgb_prec)
    mlp_pr_auc = auc(mlp_rec, mlp_prec)

    plot_path = clean_plot_dir() / "roc_pr_curves.pdf"
    logger.info(f"Generating ROC & PR Curves to {plot_path}...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.plot(xgb_fpr, xgb_tpr, color="tab:blue", lw=2, label=f"XGBoost (AUC = {xgb_roc_auc:.4f})")
    ax1.plot(mlp_fpr, mlp_tpr, color="tab:red", lw=2, label=f"PyTorch MLP (AUC = {mlp_roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--", label="Random Guess")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=14)
    ax1.set_ylabel("True Positive Rate", fontsize=14)
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=16)
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)

    baseline = np.sum(y_test == 1) / len(y_test)
    ax2.plot(xgb_rec, xgb_prec, color="tab:blue", lw=2, label=f"XGBoost (PR-AUC = {xgb_pr_auc:.4f})")
    ax2.plot(mlp_rec, mlp_prec, color="tab:red", lw=2, label=f"PyTorch MLP (PR-AUC = {mlp_pr_auc:.4f})")
    ax2.plot([0, 1], [baseline, baseline], color="gray", lw=2, linestyle="--", label=f"Baseline ({baseline:.2f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (True Positive Rate)", fontsize=14)
    ax2.set_ylabel("Precision (PPV)", fontsize=14)
    ax2.set_title("Precision-Recall Curve", fontsize=16)
    ax2.legend(loc="lower left", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info("ROC & PR Evaluation complete.")


if __name__ == "__main__":
    run_roc()
