import json

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src.ml.metrics import binary_metrics
from src.ml.mlp_model import load_mlp_model
from src.runtime import (
    fast_enabled,
    require_paths,
    require_test_diagnostics_authorized,
    runtime_paths,
)
from src.utils.logger import get_logger


logger = get_logger("METRICS_TABLES")


def evaluate_model(y_test, y_prob, threshold):
    return binary_metrics(y_test, y_prob, threshold)


def generate_performance_table():
    require_test_diagnostics_authorized()
    paths = runtime_paths()
    tables_dir = paths.outputs_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    test_tensor_path = paths.clean_tensor_dir / "test.parquet"
    xgb_model_path = paths.outputs_root / "xgboost" / "xgboost_weights.json"
    mlp_model_path = paths.outputs_root / "mlp" / "mlp_weights.pth"
    mlp_params_path = paths.outputs_root / "mlp_best_params.json"
    xgb_metrics_path = paths.outputs_root / "xgboost" / "metrics.json"
    mlp_metrics_path = paths.outputs_root / "mlp" / "metrics.json"
    require_paths(
        [
            test_tensor_path,
            xgb_model_path,
            mlp_model_path,
            mlp_params_path,
            xgb_metrics_path,
            mlp_metrics_path,
        ],
        "Performance table generation",
    )
    with open(xgb_metrics_path, "r", encoding="utf-8") as handle:
        xgb_threshold = float(json.load(handle)["Decision-Threshold"])
    with open(mlp_metrics_path, "r", encoding="utf-8") as handle:
        mlp_threshold = float(json.load(handle)["Decision-Threshold"])

    logger.info(f"Loading test dataset from {test_tensor_path}...")
    df_test = pd.read_parquet(test_tensor_path)
    x_test_scaled = df_test.drop(columns=["Label"]).values
    y_test = df_test["Label"].values
    metrics_data = []

    logger.info("Evaluating Baseline XGBoost...")
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(xgb_model_path)
    y_prob_xgb = model_xgb.predict_proba(x_test_scaled)[:, 1]
    xgb_metrics = evaluate_model(y_test, y_prob_xgb, xgb_threshold)
    metrics_data.append({"Model": "Baseline XGBoost", **xgb_metrics})

    logger.info("Evaluating PyTorch MLP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mlp = load_mlp_model(mlp_params_path, mlp_model_path, x_test_scaled.shape[1], device)
    with torch.no_grad():
        logits = model_mlp(torch.FloatTensor(x_test_scaled.copy()).to(device))
        y_prob_mlp = torch.sigmoid(logits).cpu().numpy().flatten()
    mlp_metrics = evaluate_model(y_test, y_prob_mlp, mlp_threshold)
    metrics_data.append({"Model": "PyTorch MLP", **mlp_metrics})

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
    mc_metrics = evaluate_model(y_test, y_prob_mc, mlp_threshold)
    metrics_data.append({"Model": "MC Dropout", **mc_metrics})

    df_metrics = pd.DataFrame(metrics_data)
    logger.info("Generating fully formatted LaTeX table...")
    metric_columns = [column for column in df_metrics.columns if column != "Model"]
    styler = df_metrics.style.format(
        {column: "{:.4f}" for column in metric_columns}
    ).hide(axis="index")
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
