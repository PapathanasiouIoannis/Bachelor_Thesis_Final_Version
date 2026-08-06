import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve

from src.ml.advanced.common import FEATURES, clean_plot_dir, load_clean_mlp, load_clean_test, load_clean_xgb


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FEATURE_IMPORTANCE_CLEAN")


def compute_pr_auc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


def run_permutation_importance(model, X_test, y_test, feature_names, base_score, is_mlp=False, device=None):
    importances = []
    for i, _ in enumerate(feature_names):
        X_shuffled = X_test.copy()
        rng = np.random.default_rng(42 + i)
        rng.shuffle(X_shuffled[:, i])
        if is_mlp:
            with torch.no_grad():
                probs = torch.sigmoid(model(torch.FloatTensor(X_shuffled.copy()).to(device))).cpu().numpy().flatten()
        else:
            probs = model.predict_proba(X_shuffled)[:, 1]
        importances.append(base_score - compute_pr_auc(y_test, probs))
    return importances


def plot_feature_importance(importances, feature_names, title, plot_path):
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(feature_names))
    plt.bar(x_pos, importances, color="tab:blue", align="center", alpha=0.8)
    plt.xticks(x_pos, feature_names, rotation=45, ha="right", fontsize=12)
    plt.ylabel("PR-AUC Drop", fontsize=12)
    plt.title(title, fontsize=15)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()


def run_feature_importance():
    logger.info("Initializing Permutation Feature Importance for Clean Pipeline...")
    _, X_test, y_test = load_clean_test()
    plots_dir = clean_plot_dir()

    logger.info("Evaluating XGBoost Feature Importance...")
    xgb_model = load_clean_xgb()
    xgb_base_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_base_score = compute_pr_auc(y_test, xgb_base_probs)
    xgb_importances = run_permutation_importance(xgb_model, X_test, y_test, FEATURES, xgb_base_score)
    plot_feature_importance(xgb_importances, FEATURES, "Permutation Feature Importance (XGBoost)", plots_dir / "feature_importance_xgboost.pdf")

    logger.info("Evaluating MLP Feature Importance...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = load_clean_mlp(input_dim=X_test.shape[1], device=device)
    with torch.no_grad():
        mlp_base_probs = torch.sigmoid(mlp_model(torch.FloatTensor(X_test.copy()).to(device))).cpu().numpy().flatten()
    mlp_base_score = compute_pr_auc(y_test, mlp_base_probs)
    mlp_importances = run_permutation_importance(mlp_model, X_test, y_test, FEATURES, mlp_base_score, is_mlp=True, device=device)
    plot_feature_importance(mlp_importances, FEATURES, "Permutation Feature Importance (MLP)", plots_dir / "feature_importance_mlp.pdf")
    logger.info("Feature importance complete.")


if __name__ == "__main__":
    run_feature_importance()
