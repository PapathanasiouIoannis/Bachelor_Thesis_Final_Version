# src/visualize/generate_tables.py

import os
import torch
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, brier_score_loss

from src.config import CONFIG
from src.utils.logger import get_logger
from src.ml.train_mlp import PhysicsMLP

logger = get_logger("METRICS_TABLES")

def evaluate_model(y_test, y_prob, y_pred):
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    return pr_auc, f1, brier

def generate_performance_table():
    os.makedirs("outputs/tables", exist_ok=True)
    
    logger.info("Loading test dataset from data/ml_tensors/test.parquet...")
    test_tensor_path = "data/ml_tensors/test.parquet"
    if not os.path.exists(test_tensor_path):
        logger.error(f"Test tensor missing at {test_tensor_path}! Crashing as per strict requirements.")
        raise FileNotFoundError(f"{test_tensor_path} not found.")
        
    df_test = pd.read_parquet(test_tensor_path)
    X_test_scaled = df_test.drop(columns=["Label"]).values
    y_test = df_test["Label"].values
    
    metrics_data = []
    
    # 1. XGBoost
    xgb_model_path = "models/xgboost_weights.json"
    if not os.path.exists(xgb_model_path):
        logger.error(f"XGBoost model missing at {xgb_model_path}! Crashing.")
        raise FileNotFoundError(xgb_model_path)
    
    logger.info("Evaluating Baseline XGBoost...")
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(xgb_model_path)
    
    y_prob_xgb = model_xgb.predict_proba(X_test_scaled)[:, 1]
    y_pred_xgb = model_xgb.predict(X_test_scaled)
    pr_auc, f1, brier = evaluate_model(y_test, y_prob_xgb, y_pred_xgb)
    metrics_data.append({"Model": "Baseline XGBoost", "PR-AUC": pr_auc, "F1-Score": f1, "Brier Score": brier})
    
    # 2. PyTorch MLP
    mlp_model_path = "models/mlp_weights.pth"
    if not os.path.exists(mlp_model_path):
        logger.error(f"MLP weights missing at {mlp_model_path}! Crashing.")
        raise FileNotFoundError(mlp_model_path)
        
    logger.info("Evaluating PyTorch MLP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mlp = PhysicsMLP(input_dim=X_test_scaled.shape[1]).to(device)
    model_mlp.load_state_dict(torch.load(mlp_model_path, map_location=device))
    
    model_mlp.eval()
    with torch.no_grad():
        y_prob_mlp = model_mlp(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().flatten()
    y_pred_mlp = (y_prob_mlp >= 0.5).astype(int)
    pr_auc, f1, brier = evaluate_model(y_test, y_prob_mlp, y_pred_mlp)
    metrics_data.append({"Model": "PyTorch MLP", "PR-AUC": pr_auc, "F1-Score": f1, "Brier Score": brier})
    
    # 3. MC Dropout
    logger.info("Evaluating MC Dropout (BNN)...")
    model_mlp.train() # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(100):
            preds.append(model_mlp(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy().flatten())
    y_prob_mc = np.mean(preds, axis=0)
    y_pred_mc = (y_prob_mc >= 0.5).astype(int)
    pr_auc, f1, brier = evaluate_model(y_test, y_prob_mc, y_pred_mc)
    metrics_data.append({"Model": "MC Dropout (BNN)", "PR-AUC": pr_auc, "F1-Score": f1, "Brier Score": brier})
    
    # Output to Table
    df_metrics = pd.DataFrame(metrics_data)
    
    logger.info("Generating fully formatted LaTeX table...")
    styler = df_metrics.style.format({
        "PR-AUC": "{:.3f}",
        "F1-Score": "{:.3f}",
        "Brier Score": "{:.3f}"
    }).hide(axis="index")
    
    latex_str = styler.to_latex(
        environment="table",
        caption="Comprehensive test set evaluation metrics across baseline and advanced probabilistic models.",
        label="tab:model_performance",
        position="htbp",
        hrules=True,
    )
    
    latex_str = latex_str.replace("\\begin{table}[htbp]", "\\begin{table}[htbp]\n\\centering")
    
    out_path = "outputs/tables/model_performance.tex"
    with open(out_path, "w") as f:
        f.write(latex_str)
        
    logger.info(f"Successfully exported LaTeX table to {out_path}")

if __name__ == "__main__":
    generate_performance_table()
