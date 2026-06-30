import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import joblib
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MC_DROPOUT")

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

def enable_dropout(model):
    """ Function to explicitly enable dropout layers during inference """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def eval_uncertainty():
    DATA_DIR = "data"
    TENSOR_DIR = os.path.join(DATA_DIR, "ml_tensors")
    MODEL_PATH = os.path.join("outputs", "mlp", "mlp_weights.pth")
    PARAMS_PATH = os.path.join("outputs", "mlp_best_params.json")
    
    SCALER_PATH = os.path.join(TENSOR_DIR, "scaler.joblib")
    
    features = ['Mass', 'Radius', 'log10_Lambda']
    
    # 1. Load the pre-fitted scaler from data pipeline
    logger.info("Loading fitted scaler...")
    scaler = joblib.load(SCALER_PATH)

    # 2. Ingest Pristine Test Set (These tensors are already scaled)
    logger.info("Loading strictly pristine test tensor...")
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
    X_test_scaled = test_df[features].values
    y_test = test_df['Label'].values
    
    # Get true physical values for plotting the axes later
    X_test_raw = scaler.inverse_transform(X_test_scaled)
    
    X_test_tensor = torch.FloatTensor(X_test_scaled)

    # 3. Model Ingestion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)
        
    hidden_sizes = eval(best_params['hidden_layer_sizes'])
    dropout_rate = best_params['dropout_rate']
    
    model = DynamicMLP(input_dim=len(features), hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Fatal: Saved PyTorch weights not found at {MODEL_PATH}. Cannot proceed without training weights.")
        return
        
    logger.info(f"Loading trained weights from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    
    # 4. Monte Carlo Inference
    model.eval()
    enable_dropout(model) # force dropout active
    
    N_ITERATIONS = 100
    logger.info(f"Performing Monte Carlo Dropout inference (N={N_ITERATIONS})...")
    
    X_test_tensor = X_test_tensor.to(device)
    predictions = []
    
    with torch.no_grad():
        for i in range(N_ITERATIONS):
            outputs = model(X_test_tensor)
            preds = torch.sigmoid(outputs).cpu().numpy().flatten()
            predictions.append(preds)
            
    predictions = np.array(predictions) # shape: (100, N_test_samples)
    
    mean_preds = np.mean(predictions, axis=0)
    variance_preds = np.var(predictions, axis=0) # epistemic uncertainty
    
    # 5. Visualization
    plots_dir = os.path.join("plots", "ml_advanced")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, "uncertainty_calibration.pdf")
    
    logger.info(f"Generating Uncertainty vs. Compactness calibration plot to {plot_path}...")
    
    plt.figure(figsize=(10, 6))
    
    # Recalculate Compactness (C = M/R) for plotting
    compactness = X_test_raw[:, 0] / X_test_raw[:, 1]
    
    scatter = plt.scatter(compactness, variance_preds, c=mean_preds, cmap='coolwarm', alpha=0.8, edgecolor='k')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Mean Predicted Probability (0=Hadronic, 1=Quark)')
    
    plt.title('Epistemic Uncertainty (MC Dropout) vs Compactness', fontsize=15)
    plt.xlabel('Compactness (C = M/R)', fontsize=12)
    plt.ylabel('Predictive Variance (Epistemic Uncertainty)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    logger.info("Uncertainty Quantification module complete.")

if __name__ == "__main__":
    eval_uncertainty()
