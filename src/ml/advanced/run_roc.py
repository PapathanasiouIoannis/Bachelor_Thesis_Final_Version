import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ROC_CURVE")

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
        # No Sigmoid here because we use BCEWithLogitsLoss during training
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def run_roc():
    logger.info("Initializing ROC Curve Evaluation...")
    
    TENSOR_DIR = os.path.join("data", "ml_tensors")
    logger.info(f"Loading pristine test tensor from {TENSOR_DIR}...")
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
    
    X_test_scaled = test_df.drop(columns=['Label']).values
    y_test = test_df['Label'].values
    
    # 1. XGBoost Inference
    xgb_model_path = os.path.join("outputs", "xgboost", "xgboost_weights.json")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_model_path)
    
    logger.info("Extracting XGBoost probabilistic predictions...")
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    # 2. MLP Inference
    mlp_params_path = os.path.join("outputs", "mlp_best_params.json")
    with open(mlp_params_path, "r") as f:
        best_params = json.load(f)
        
    hidden_sizes = eval(best_params['hidden_layer_sizes'])
    dropout_rate = best_params['dropout_rate']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_model = DynamicMLP(input_dim=3, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
    mlp_model_path = os.path.join("outputs", "mlp", "mlp_weights.pth")
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device, weights_only=True))
    mlp_model.eval()
    
    logger.info("Extracting MLP probabilistic predictions...")
    with torch.no_grad():
        outputs = mlp_model(torch.FloatTensor(X_test_scaled).to(device))
        mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
    # Calculate ROC Curves
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
    xgb_roc_auc = auc(xgb_fpr, xgb_tpr)
    
    mlp_fpr, mlp_tpr, _ = roc_curve(y_test, mlp_probs)
    mlp_roc_auc = auc(mlp_fpr, mlp_tpr)
    
    # Calculate PR Curves for secondary plot
    xgb_prec, xgb_rec, _ = precision_recall_curve(y_test, xgb_probs)
    xgb_pr_auc = auc(xgb_rec, xgb_prec)
    
    mlp_prec, mlp_rec, _ = precision_recall_curve(y_test, mlp_probs)
    mlp_pr_auc = auc(mlp_rec, mlp_prec)

    # 4. Visualization
    output_dir = os.path.join("plots", "ml_advanced")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "roc_pr_curves.pdf")
    
    logger.info(f"Generating ROC & PR Curves to {plot_path}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot ROC
    ax1.plot(xgb_fpr, xgb_tpr, color='tab:blue', lw=2, label=f'XGBoost (AUC = {xgb_roc_auc:.4f})')
    ax1.plot(mlp_fpr, mlp_tpr, color='tab:red', lw=2, label=f'PyTorch MLP (AUC = {mlp_roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=16)
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot PR
    baseline = np.sum(y_test == 1) / len(y_test)
    ax2.plot(xgb_rec, xgb_prec, color='tab:blue', lw=2, label=f'XGBoost (PR-AUC = {xgb_pr_auc:.4f})')
    ax2.plot(mlp_rec, mlp_prec, color='tab:red', lw=2, label=f'PyTorch MLP (PR-AUC = {mlp_pr_auc:.4f})')
    ax2.plot([0, 1], [baseline, baseline], color='gray', lw=2, linestyle='--', label=f'Baseline ({baseline:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (True Positive Rate)', fontsize=14)
    ax2.set_ylabel('Precision (PPV)', fontsize=14)
    ax2.set_title('Precision-Recall Curve', fontsize=16)
    ax2.legend(loc="lower left", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    logger.info("ROC & PR Evaluation complete.")

if __name__ == "__main__":
    run_roc()
