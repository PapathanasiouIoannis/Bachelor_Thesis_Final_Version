import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve

from src.ml_perturb.advanced.common import FEATURE_SETS, load_perturb_mlp, load_perturb_test, load_perturb_xgb, perturb_plot_dir


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FEATURE_IMPORTANCE_PERTURB")


def compute_pr_auc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)


def run_permutation_importance(model, x_test, y_test, feature_names, base_score, is_mlp=False, device=None):
    importances = []
    for i, _ in enumerate(feature_names):
        x_shuffled = x_test.copy()
        rng = np.random.default_rng(42 + i)
        rng.shuffle(x_shuffled[:, i])
        if is_mlp:
            with torch.no_grad():
                probs = torch.sigmoid(model(torch.FloatTensor(x_shuffled.copy()).to(device))).cpu().numpy().flatten()
        else:
            probs = model.predict_proba(x_shuffled)[:, 1]
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
    logger.info("Initializing Permutation Feature Importance for Perturbed Pipeline...")
    plots_dir = perturb_plot_dir()

    for fset, feature_names in FEATURE_SETS.items():
        logger.info(f"--- Evaluating Feature Importance for {fset} ---")
        _, x_test, y_test = load_perturb_test(fset)

        xgb_model = load_perturb_xgb(fset)
        xgb_base_probs = xgb_model.predict_proba(x_test)[:, 1]
        xgb_base_score = compute_pr_auc(y_test, xgb_base_probs)
        xgb_importances = run_permutation_importance(xgb_model, x_test, y_test, feature_names, xgb_base_score)
        plot_feature_importance(xgb_importances, feature_names, f"Permutation Feature Importance - XGBoost ({fset})", plots_dir / f"feature_importance_xgboost_{fset}.pdf")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = load_perturb_mlp(fset, input_dim=x_test.shape[1], device=device)
        with torch.no_grad():
            mlp_base_probs = torch.sigmoid(mlp_model(torch.FloatTensor(x_test.copy()).to(device))).cpu().numpy().flatten()
        mlp_base_score = compute_pr_auc(y_test, mlp_base_probs)
        mlp_importances = run_permutation_importance(mlp_model, x_test, y_test, feature_names, mlp_base_score, is_mlp=True, device=device)
        plot_feature_importance(mlp_importances, feature_names, f"Permutation Feature Importance - MLP ({fset})", plots_dir / f"feature_importance_mlp_{fset}.pdf")

    logger.info("Perturbed feature importance complete.")


if __name__ == "__main__":
    run_feature_importance()
