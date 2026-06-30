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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NOISE_DEGRADATION")

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
        # Removed Sigmoid here because we use BCEWithLogitsLoss during training
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def load_mlp_model():
    params_path = os.path.join("outputs", "mlp_best_params.json")
    with open(params_path, "r") as f:
        best_params = json.load(f)
    
    hidden_sizes = eval(best_params['hidden_layer_sizes'])
    dropout_rate = best_params['dropout_rate']
    
    model = DynamicMLP(input_dim=3, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate)
    model_path = os.path.join("outputs", "mlp", "mlp_weights.pth")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def run_noise_degradation():
    logger.info("Initializing Phase 9: Systematic Noise Degradation Analysis (XGBoost vs MLP)...")
    
    # Load Scaler and Models
    scaler_path = os.path.join("data", "ml_tensors", "scaler.joblib")
    xgb_model_path = os.path.join("outputs", "xgboost", "xgboost_weights.json")
        
    scaler = joblib.load(scaler_path)
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_model_path)
    
    mlp_model = load_mlp_model()
    
    # 1. Define Candidates (Real-World Matches)
    candidates = {
        "GW170817 (Hadronic)": {"M": 1.435, "R": 12.327, "L": 2.269},
        "HESS J1731-347 (Quark)": {"M": 0.779, "R": 10.285, "L": 2.822},
        "Boundary Star": {"M": 1.785, "R": 12.196, "L": 1.580}
    }
    
    # 2. Define Noise Sweep for Lambda ONLY (0% to 100%)
    noise_levels = np.linspace(0.0, 1.0, 51) # 51 steps (every 2%)
    N_samples = 5000
    
    xgb_results = {name: {"expected": [], "lb": [], "ub": []} for name in candidates.keys()}
    mlp_results = {name: {"expected": [], "lb": [], "ub": []} for name in candidates.keys()}
    
    logger.info("Running Monte Carlo sweeps...")
    for name, params in candidates.items():
        M_true = params["M"]
        R_true = params["R"]
        L_true = params["L"]
        
        logger.info(f"Processing Candidate {name} (M={M_true}, R={R_true}, L={L_true})...")
        for noise in tqdm(noise_levels, desc=f"Lambda Noise Sweep {name}", leave=False):
            # Perfect measurement of M and R
            M_err = 0.0
            R_err = 0.0
            
            # Mathematical Fix: Telescope noise applies to raw Lambda, NOT log10(Lambda).
            raw_lambda_true = 10 ** L_true
            raw_lambda_err = raw_lambda_true * noise
            
            mass_samples = np.random.normal(M_true, M_err, N_samples)
            radius_samples = np.random.normal(R_true, R_err, N_samples)
            raw_lambda_samples = np.random.normal(raw_lambda_true, raw_lambda_err, N_samples)
            
            mass_samples = np.clip(mass_samples, a_min=0.01, a_max=None)
            radius_samples = np.clip(radius_samples, a_min=1.0, a_max=None)
            raw_lambda_samples = np.clip(raw_lambda_samples, a_min=1e-10, a_max=None)
            
            lambda_samples = np.log10(raw_lambda_samples)
            
            c_samples = mass_samples / radius_samples
            
            X_mc = pd.DataFrame({
                'Mass': mass_samples,
                'Radius': radius_samples,
                'log10_Lambda': lambda_samples
            })
            
            X_mc_scaled = pd.DataFrame(scaler.transform(X_mc), columns=X_mc.columns)
            
            # XGBoost Inference
            xgb_probs = xgb_model.predict_proba(X_mc_scaled)[:, 1]
            xgb_results[name]["expected"].append(np.mean(xgb_probs))
            xgb_results[name]["lb"].append(np.percentile(xgb_probs, 16.0))
            xgb_results[name]["ub"].append(np.percentile(xgb_probs, 84.0))
            
            # MLP Inference
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_mc_scaled.values)
                outputs = mlp_model(X_tensor)
                mlp_probs = torch.sigmoid(outputs).numpy().flatten()
                
            mlp_results[name]["expected"].append(np.mean(mlp_probs))
            mlp_results[name]["lb"].append(np.percentile(mlp_probs, 16.0))
            mlp_results[name]["ub"].append(np.percentile(mlp_probs, 84.0))
            
    # 3. Visualization
    logger.info("Generating degradation plots...")
    output_dir = os.path.join("plots", "ml_advanced")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True, sharex=True)
    noise_pct = noise_levels * 100
    
    # Plot XGBoost (Row 0)
    for i, (name, stats) in enumerate(xgb_results.items()):
        ax = axes[0, i]
        expected = np.array(stats["expected"]) * 100
        lb = np.array(stats["lb"]) * 100
        ub = np.array(stats["ub"]) * 100
        
        ax.fill_between(noise_pct, lb, ub, color='blue', alpha=0.2, label='68% CI')
        ax.plot(noise_pct, expected, color='darkblue', linewidth=2, label='Expected Prob')
        
        ax.set_title(f"XGBoost | {name}")
        if i == 0:
            ax.set_ylabel('Quark Phase Probability (%)')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(loc='best')
        
    # Plot MLP (Row 1)
    for i, (name, stats) in enumerate(mlp_results.items()):
        ax = axes[1, i]
        expected = np.array(stats["expected"]) * 100
        lb = np.array(stats["lb"]) * 100
        ub = np.array(stats["ub"]) * 100
        
        ax.fill_between(noise_pct, lb, ub, color='green', alpha=0.2, label='68% CI')
        ax.plot(noise_pct, expected, color='darkgreen', linewidth=2, label='Expected Prob')
        
        ax.set_title(f"MLP | {name}")
        ax.set_xlabel(r'$\Lambda$ Observational Noise Level (%)')
        if i == 0:
            ax.set_ylabel('Quark Phase Probability (%)')
        
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.legend(loc='best')
        
    plt.suptitle(r"Pure $\Lambda$ Observational Noise Degradation (M, R perfectly known)", fontsize=18, y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "noise_degradation_cones_comparison.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Analysis complete. Comparison degradation cones saved to {plot_path}")

if __name__ == "__main__":
    run_noise_degradation()
