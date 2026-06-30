import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CALIBRATION_PERTURB")

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate):
        super(DynamicMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def run_calibration():
    logger.info("Initializing Calibration Audit for Perturbed Pipeline...")
    
    TENSOR_DIR = os.path.join("data", "ml_tensors_perturb")
    
    for fset in ["MR", "MRL"]:
        if not os.path.exists(TENSOR_DIR):
            logger.warning(f"Skipping {fset}, tensor dir not found.")
            continue
            
        logger.info(f"Loading {fset} test tensor from {TENSOR_DIR}...")
        test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
        
        if fset == "MR":
            X_test_scaled = test_df.drop(columns=['Label', 'log10_Lambda']).values
        else:
            X_test_scaled = test_df.drop(columns=['Label']).values
        y_test = test_df['Label'].values
        input_dim = X_test_scaled.shape[1]
        
        # XGBoost
        xgb_model_path = os.path.join("outputs_perturb", f"xgboost_{fset}", "xgboost_weights.json")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)
        xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        # MLP
        mlp_params_path = os.path.join("outputs_perturb", f"mlp_{fset}_best_params.json")
        with open(mlp_params_path, "r") as f:
            best_params = json.load(f)
        hidden_sizes = eval(best_params['hidden_layer_sizes'])
        dropout_rate = best_params['dropout_rate']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = DynamicMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
        mlp_model_path = os.path.join("outputs_perturb", f"mlp_{fset}", "mlp_weights.pth")
        mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device, weights_only=True))
        mlp_model.eval()
        
        with torch.no_grad():
            outputs = mlp_model(torch.FloatTensor(X_test_scaled).to(device))
            mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
        xgb_brier = brier_score_loss(y_test, xgb_probs)
        mlp_brier = brier_score_loss(y_test, mlp_probs)
        logger.info(f"[{fset}] XGBoost Brier Score: {xgb_brier:.5f}")
        logger.info(f"[{fset}] MLP Brier Score: {mlp_brier:.5f}")
        
        xgb_fraction_of_positives, xgb_mean_predicted_value = calibration_curve(y_test, xgb_probs, n_bins=10, strategy='quantile')
        mlp_fraction_of_positives, mlp_mean_predicted_value = calibration_curve(y_test, mlp_probs, n_bins=10, strategy='quantile')
        
        output_dir = os.path.join("plots_perturb", "ml_advanced")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"calibration_curve_{fset}.pdf")
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated (Ideal)")
        plt.plot(xgb_mean_predicted_value, xgb_fraction_of_positives, "s-", color="tab:blue", label=f"XGBoost (Brier={xgb_brier:.4f})", linewidth=2, markersize=8)
        plt.plot(mlp_mean_predicted_value, mlp_fraction_of_positives, "o-", color="tab:red", label=f"MLP (Brier={mlp_brier:.4f})", linewidth=2, markersize=8)
        plt.xlabel("Mean Predicted Probability", fontsize=14)
        plt.ylabel("Fraction of Positives (Empirical Probability)", fontsize=14)
        plt.title(f"Calibration Curve: XGBoost vs MLP (Perturbed {fset})", fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {plot_path}")

if __name__ == "__main__":
    run_calibration()
