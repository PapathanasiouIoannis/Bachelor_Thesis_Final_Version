import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

from src.ml.mlp_model import load_mlp_model
from src.runtime import fast_enabled, require_paths, runtime_paths
from src.utils.logger import get_logger


logger = get_logger("METRICS_TABLES")


def evaluate_model(y_test, y_prob, y_pred):
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    return pr_auc, f1, brier


def generate_performance_table():
    paths = runtime_paths()
    tables_dir = paths.outputs_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    test_tensor_path = paths.clean_tensor_dir / "test.parquet"
    xgb_model_path = paths.outputs_root / "xgboost" / "xgboost_weights.json"
    mlp_model_path = paths.outputs_root / "mlp" / "mlp_weights.pth"
    mlp_params_path = paths.outputs_root / "mlp_best_params.json"
    require_paths([test_tensor_path, xgb_model_path, mlp_model_path, mlp_params_path], "Performance table generation")

    logger.info(f"Loading test dataset from {test_tensor_path}...")
    df_test = pd.read_parquet(test_tensor_path)
    x_test_scaled = df_test.drop(columns=["Label"]).values
    y_test = df_test["Label"].values
    metrics_data = []

    logger.info("Evaluating Baseline XGBoost...")
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(xgb_model_path)
    y_prob_xgb = model_xgb.predict_proba(x_test_scaled)[:, 1]
    y_pred_xgb = model_xgb.predict(x_test_scaled)
    pr_auc, f1, brier = evaluate_model(y_test, y_prob_xgb, y_pred_xgb)
    metrics_data.append({"Model": "Baseline XGBoost", "PR-AUC": pr_auc, "F1-Score": f1, "Brier Score": brier})

    logger.info("Evaluating PyTorch MLP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mlp = load_mlp_model(mlp_params_path, mlp_model_path, x_test_scaled.shape[1], device)
    with torch.no_grad():
        logits = model_mlp(torch.FloatTensor(x_test_scaled.copy()).to(device))
        y_prob_mlp = torch.sigmoid(logits).cpu().numpy().flatten()
    y_pred_mlp = (y_prob_mlp >= 0.5).astype(int)
    pr_auc, f1, brier = evaluate_model(y_test, y_prob_mlp, y_pred_mlp)
    metrics_data.append({"Model": "PyTorch MLP", "PR-AUC": pr_auc, "F1-Score": f1, "Brier Score": brier})

    logger.info("Evaluating MC Dropout.")
    model_mlp.train()
    preds = []
    iterations = 10 if fast_enabled() else 100
    with torch.no_grad():
        for _ in range(iterations):
            logits = model_mlp(torch.FloatTensor(x_test_scaled.copy()).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.append(probs)
    y_prob_mc = np.mean(preds, axis=0)
    y_pred_mc = (y_prob_mc >= 0.5).astype(int)
    pr_auc, f1, brier = evaluate_model(y_test, y_prob_mc, y_pred_mc)
    metrics_data.append({"Model": "MC Dropout", "PR-AUC": pr_auc, "F1-Score": f1, "Brier Score": brier})

    df_metrics = pd.DataFrame(metrics_data)
    logger.info("Generating fully formatted LaTeX table...")
    styler = df_metrics.style.format({"PR-AUC": "{:.4f}", "F1-Score": "{:.4f}", "Brier Score": "{:.4f}"}).hide(axis="index")
    latex_str = styler.to_latex(
        environment="table",
        caption="Comprehensive test set evaluation metrics across baseline and advanced probabilistic models.",
        label="tab:model_performance",
        position="htbp",
        hrules=True,
    )
    latex_str = latex_str.replace("\\begin{table}[htbp]", "\\begin{table}[htbp]\n\\centering")

    out_path = tables_dir / "model_performance.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    logger.info(f"Successfully exported LaTeX table to {out_path}")


if __name__ == "__main__":
    generate_performance_table()
