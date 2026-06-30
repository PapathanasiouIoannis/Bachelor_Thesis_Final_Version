import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MC_INFERENCE_PERTURB")

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

def run_mc_inference():
    logger.info("Initializing MC Observational Inference for Perturbed Pipeline...")
    
    N = 10000
    M_obs = 1.4
    M_err = 0.05
    R_obs = 11.5
    R_err = 0.5
    
    mass_samples = np.random.normal(M_obs, M_err, N)
    radius_samples = np.random.normal(R_obs, R_err, N)
    
    for fset in ["MR", "MRL"]:
        scaler_path = os.path.join("data", "ml_tensors_perturb", "scaler_perturb.joblib")
        if not os.path.exists(scaler_path):
            continue
        scaler = joblib.load(scaler_path)
        
        log10_lambda_mean = scaler.mean_[2]
        lambda_samples = np.full(N, log10_lambda_mean)
        X_mc_3cols = pd.DataFrame({
            'Mass': mass_samples,
            'Radius': radius_samples,
            'log10_Lambda': lambda_samples
        })
        
        X_mc_scaled_3cols = pd.DataFrame(scaler.transform(X_mc_3cols), columns=X_mc_3cols.columns)
        
        if fset == "MRL":
            X_mc_scaled = X_mc_scaled_3cols
        else:
            X_mc_scaled = X_mc_scaled_3cols.drop(columns=['log10_Lambda'])
            
        input_dim = X_mc_scaled.shape[1]
        
        xgb_model_path = os.path.join("outputs_perturb", f"xgboost_{fset}", "xgboost_weights.json")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)
        xgb_probs = xgb_model.predict_proba(X_mc_scaled)[:, 1]
        
        xgb_expected = np.mean(xgb_probs)
        xgb_lb = np.percentile(xgb_probs, 2.5)
        xgb_ub = np.percentile(xgb_probs, 97.5)
        
        mlp_params_path = os.path.join("outputs_perturb", f"mlp_{fset}_best_params.json")
        with open(mlp_params_path, "r") as f:
            best_params = json.load(f)
        hidden_sizes = eval(best_params['hidden_layer_sizes'])
        dropout_rate = best_params['dropout_rate']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = DynamicMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
        mlp_model_path = os.path.join("outputs_perturb", f"mlp_{fset}", "mlp_weights.pth")
        mlp_model.load_state_dict(torch.load(mlp_model_path, weights_only=True))
        mlp_model.eval()
        
        with torch.no_grad():
            outputs = mlp_model(torch.FloatTensor(X_mc_scaled.values.copy()).to(device))
            mlp_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
        mlp_expected = np.mean(mlp_probs)
        mlp_lb = np.percentile(mlp_probs, 2.5)
        mlp_ub = np.percentile(mlp_probs, 97.5)
        
        output_dir = os.path.join("plots_perturb", "ml_advanced")
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        scatter1 = ax1.scatter(mass_samples, radius_samples, c=xgb_probs, cmap='coolwarm', alpha=0.6, edgecolors='none', s=10)
        ax1.errorbar(M_obs, R_obs, xerr=M_err, yerr=R_err, fmt='k+', markersize=15, capsize=3, linewidth=1.5, label='Mean Obs ± 1σ')
        ax1.set_xlabel(r'Mass ($M_\odot$)')
        ax1.set_ylabel('Radius (km)')
        ax1.set_title(f'XGBoost Topology (Perturbed {fset})\nExpected: {xgb_expected:.1%} [{xgb_lb:.1%} - {xgb_ub:.1%}]')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        fig.colorbar(scatter1, ax=ax1, label='Probability (Quark Star)')
        
        scatter2 = ax2.scatter(mass_samples, radius_samples, c=mlp_probs, cmap='coolwarm', alpha=0.6, edgecolors='none', s=10)
        ax2.errorbar(M_obs, R_obs, xerr=M_err, yerr=R_err, fmt='k+', markersize=15, capsize=3, linewidth=1.5, label='Mean Obs ± 1σ')
        ax2.set_xlabel(r'Mass ($M_\odot$)')
        ax2.set_ylabel('Radius (km)')
        ax2.set_title(f'MLP Topology (Perturbed {fset})\nExpected: {mlp_expected:.1%} [{mlp_lb:.1%} - {mlp_ub:.1%}]')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        fig.colorbar(scatter2, ax=ax2, label='Probability (Quark Star)')
        
        plt.suptitle(f'Monte Carlo Observational Inference (N={N})', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"mc_observational_noise_combined_{fset}.pdf")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
if __name__ == "__main__":
    run_mc_inference()
