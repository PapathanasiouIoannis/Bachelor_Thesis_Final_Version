import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FEATURE_IMPORTANCE_PERTURB")

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

def compute_pr_auc(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def run_permutation_importance(model, X_test, y_test, feature_names, base_score, is_mlp=False, device=None):
    importances = []
    
    for i, feature in enumerate(feature_names):
        X_shuffled = X_test.copy()
        np.random.shuffle(X_shuffled[:, i])
        
        if is_mlp:
            with torch.no_grad():
                outputs = model(torch.FloatTensor(X_shuffled).to(device))
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        else:
            probs = model.predict_proba(X_shuffled)[:, 1]
            
        shuffled_score = compute_pr_auc(y_test, probs)
        drop_in_score = base_score - shuffled_score
        importances.append(drop_in_score)
        
    return importances

def plot_feature_importance(importances, feature_names, title, plot_path):
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(feature_names))
    
    plt.bar(x_pos, importances, color='tab:blue', align='center', alpha=0.8)
    plt.xticks(x_pos, feature_names, rotation=45, ha='right', fontsize=12)
    plt.ylabel('PR-AUC Drop', fontsize=12)
    plt.title(title, fontsize=15)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

def run_feature_importance():
    logger.info("Initializing Permutation Feature Importance for Perturbed Pipeline...")
    
    TENSOR_DIR = os.path.join("data", "ml_tensors_perturb")
    if not os.path.exists(TENSOR_DIR):
        logger.error(f"Tensor directory not found: {TENSOR_DIR}")
        return
        
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
    
    plots_dir = os.path.join("plots_perturb", "ml_advanced")
    os.makedirs(plots_dir, exist_ok=True)
    
    for fset in ["MR", "MRL"]:
        logger.info(f"--- Evaluating Feature Importance for {fset} ---")
        
        if fset == "MR":
            feature_names = ['Mass', 'Radius']
            X_test = test_df.drop(columns=['Label', 'log10_Lambda']).values
        else:
            feature_names = ['Mass', 'Radius', 'log10_Lambda']
            X_test = test_df.drop(columns=['Label']).values
            
        y_test = test_df['Label'].values
        input_dim = X_test.shape[1]
        
        # --- XGBoost ---
        xgb_model_path = os.path.join("outputs_perturb", f"xgboost_{fset}", "xgboost_weights.json")
        if os.path.exists(xgb_model_path):
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_model_path)
            xgb_base_probs = xgb_model.predict_proba(X_test)[:, 1]
            xgb_base_score = compute_pr_auc(y_test, xgb_base_probs)
            
            xgb_importances = run_permutation_importance(xgb_model, X_test, y_test, feature_names, xgb_base_score)
            plot_path_xgb = os.path.join(plots_dir, f"feature_importance_xgboost_{fset}.pdf")
            plot_feature_importance(xgb_importances, feature_names, f"Permutation Feature Importance - XGBoost ({fset})", plot_path_xgb)
            logger.info(f"[{fset}] Saved XGBoost Feature Importance to {plot_path_xgb}")
        else:
            logger.warning(f"[{fset}] XGBoost weights not found.")
            
        # --- MLP ---
        mlp_params_path = os.path.join("outputs_perturb", f"mlp_{fset}_best_params.json")
        mlp_model_path = os.path.join("outputs_perturb", f"mlp_{fset}", "mlp_weights.pth")
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
                mlp_base_outputs = mlp_model(torch.FloatTensor(X_test).to(device))
                mlp_base_probs = torch.sigmoid(mlp_base_outputs).cpu().numpy().flatten()
                mlp_base_score = compute_pr_auc(y_test, mlp_base_probs)
                
            mlp_importances = run_permutation_importance(mlp_model, X_test, y_test, feature_names, mlp_base_score, is_mlp=True, device=device)
            plot_path_mlp = os.path.join(plots_dir, f"feature_importance_mlp_{fset}.pdf")
            plot_feature_importance(mlp_importances, feature_names, f"Permutation Feature Importance - MLP ({fset})", plot_path_mlp)
            logger.info(f"[{fset}] Saved MLP Feature Importance to {plot_path_mlp}")
        else:
            logger.warning(f"[{fset}] MLP weights or params not found.")

if __name__ == "__main__":
    run_feature_importance()
