import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CONFUSION_MATRIX_CLEAN")

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

def plot_confusion_matrices(y_true, y_pred, title_prefix, plot_path):
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = ['Hadronic (0)', 'Quark (1)']
    
    # Raw Counts
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='viridis', cbar=True, ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_title(f"{title_prefix} - Raw Counts", fontsize=14)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='viridis', cbar=True, ax=ax2,
                xticklabels=labels, yticklabels=labels, vmin=0.0, vmax=1.0)
    ax2.set_title(f"{title_prefix} - Normalized (Row Probabilities)", fontsize=14)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def run_confusion_matrix():
    logger.info("Initializing Confusion Matrix Audit for Clean Pipeline...")
    
    TENSOR_DIR = os.path.join("data", "ml_tensors")
    if not os.path.exists(TENSOR_DIR):
        logger.error(f"Tensor directory not found: {TENSOR_DIR}")
        return
        
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
    X_test = test_df.drop(columns=['Label']).values
    y_test = test_df['Label'].values
    input_dim = X_test.shape[1]
    
    plots_dir = os.path.join("plots", "ml_advanced")
    os.makedirs(plots_dir, exist_ok=True)
    
    # --- XGBoost ---
    logger.info("Evaluating XGBoost Confusion Matrix...")
    xgb_model_path = os.path.join("outputs", "xgboost", "xgboost_weights.json")
    if os.path.exists(xgb_model_path):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)
        xgb_preds = xgb_model.predict(X_test)
        
        plot_path_xgb = os.path.join(plots_dir, "confusion_matrix_xgboost.pdf")
        plot_confusion_matrices(y_test, xgb_preds, "XGBoost", plot_path_xgb)
        logger.info(f"Saved XGBoost Confusion Matrix to {plot_path_xgb}")
    else:
        logger.warning("XGBoost weights not found.")
        
    # --- MLP ---
    logger.info("Evaluating MLP Confusion Matrix...")
    mlp_params_path = os.path.join("outputs", "mlp_best_params.json")
    mlp_model_path = os.path.join("outputs", "mlp", "mlp_weights.pth")
    if os.path.exists(mlp_params_path) and os.path.exists(mlp_model_path):
        with open(mlp_params_path, "r") as f:
            best_params = json.load(f)
            
        hidden_sizes = eval(best_params['hidden_layer_sizes'])
        dropout_rate = best_params['dropout_rate']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = DynamicMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
        mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device, weights_only=True))
        mlp_model.eval()
        
        with torch.no_grad():
            outputs = mlp_model(torch.FloatTensor(X_test).to(device))
            mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            mlp_preds = (mlp_probs >= 0.5).astype(int)
            
        plot_path_mlp = os.path.join(plots_dir, "confusion_matrix_mlp.pdf")
        plot_confusion_matrices(y_test, mlp_preds, "MLP", plot_path_mlp)
        logger.info(f"Saved MLP Confusion Matrix to {plot_path_mlp}")
    else:
        logger.warning("MLP weights or params not found.")

if __name__ == "__main__":
    run_confusion_matrix()
