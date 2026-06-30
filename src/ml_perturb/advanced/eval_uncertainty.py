import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MC_DROPOUT_PERTURB")

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

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def eval_uncertainty():
    logger.info("Initializing MC Dropout Audit for Perturbed Pipeline...")
    
    TENSOR_DIR = os.path.join("data", "ml_tensors_perturb")
    
    for fset in ["MR", "MRL"]:
        if not os.path.exists(TENSOR_DIR):
            continue
            
        SCALER_PATH = os.path.join(TENSOR_DIR, "scaler_perturb.joblib")
        scaler = joblib.load(SCALER_PATH)
        
        test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
        
        if fset == "MR":
            X_test_scaled = test_df.drop(columns=['Label', 'log10_Lambda']).values
            # For MR, we need to artificially inject the lambda column to inverse_transform
            # Wait, scaler_perturb.joblib is fit on [Mass, Radius, log10_Lambda].
            # So inverse_transform requires 3 columns.
            X_test_for_inverse = test_df.drop(columns=['Label']).values
            X_test_raw = scaler.inverse_transform(X_test_for_inverse)
        else:
            X_test_scaled = test_df.drop(columns=['Label']).values
            X_test_raw = scaler.inverse_transform(X_test_scaled)
        
        input_dim = X_test_scaled.shape[1]
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        PARAMS_PATH = os.path.join("outputs_perturb", f"mlp_{fset}_best_params.json")
        MODEL_PATH = os.path.join("outputs_perturb", f"mlp_{fset}", "mlp_weights.pth")
        
        with open(PARAMS_PATH, "r") as f:
            best_params = json.load(f)
        hidden_sizes = eval(best_params['hidden_layer_sizes'])
        dropout_rate = best_params['dropout_rate']
        
        model = DynamicMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        
        model.eval()
        enable_dropout(model)
        
        N_ITERATIONS = 100
        X_test_tensor = X_test_tensor.to(device)
        predictions = []
        
        with torch.no_grad():
            for i in range(N_ITERATIONS):
                outputs = model(X_test_tensor)
                preds = torch.sigmoid(outputs).cpu().numpy().flatten()
                predictions.append(preds)
                
        predictions = np.array(predictions)
        mean_preds = np.mean(predictions, axis=0)
        variance_preds = np.var(predictions, axis=0)
        
        plots_dir = os.path.join("plots_perturb", "ml_advanced")
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"uncertainty_calibration_{fset}.pdf")
        
        plt.figure(figsize=(10, 6))
        compactness = X_test_raw[:, 0] / X_test_raw[:, 1]
        
        scatter = plt.scatter(compactness, variance_preds, c=mean_preds, cmap='coolwarm', alpha=0.8, edgecolor='k')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Mean Predicted Probability (0=Hadronic, 1=Quark)')
        
        plt.title(f'Epistemic Uncertainty vs Compactness (Perturbed {fset})', fontsize=15)
        plt.xlabel('Compactness (C = M/R)', fontsize=12)
        plt.ylabel('Predictive Variance (Epistemic Uncertainty)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    eval_uncertainty()
