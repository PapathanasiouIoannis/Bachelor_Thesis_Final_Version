import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import streamlit as st

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
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

# Streamlit Page Config
st.set_page_config(page_title="Astrophysical EoS Inference Engine", layout="wide")
st.title("Astrophysical Multi-Messenger Inference Engine")
st.markdown("A rigorous Monte Carlo integration engine for predicting Quark phase topologies based on noisy telescope observables. Supports X-ray and Gravitational Wave marginalization.")

# Sidebar Configuration
st.sidebar.header("Observatory Mode")
obs_mode = st.sidebar.radio(
    "Select Available Data:",
    ["NICER X-ray (Mass & Radius)", "LIGO Gravitational Wave (Mass & Λ)", "Multi-Messenger (Mass, Radius & Λ)"]
)

st.sidebar.header("Telescope Observables")
M_obs = st.sidebar.number_input("Mean Mass (M_sun)", min_value=0.01, max_value=10.0, value=1.40, step=0.01)
M_err = st.sidebar.number_input("Mass 1-sigma Uncertainty", min_value=0.001, max_value=1.0, value=0.05, step=0.01)

if "Radius" in obs_mode:
    R_obs = st.sidebar.number_input("Mean Radius (km)", min_value=1.0, max_value=100.0, value=11.50, step=0.1)
    R_err = st.sidebar.number_input("Radius 1-sigma Uncertainty", min_value=0.01, max_value=3.0, value=0.5, step=0.05)

if "Λ" in obs_mode:
    L_obs = st.sidebar.number_input("Mean log10(Λ)", min_value=0.0, max_value=5.0, value=2.50, step=0.1)
    L_err = st.sidebar.number_input("log10(Λ) 1-sigma Uncertainty", min_value=0.01, max_value=2.0, value=0.2, step=0.05)

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox("Select Inference Model", ["Optimized XGBoost", "Optimized MLP"])

if st.sidebar.button("Run Monte Carlo Inference", type="primary"):
    # OOD Guardrail
    if M_obs < 0.10 or M_obs > 2.99:
        st.error("Warning: Mass input resides outside the generated physical manifold (0.10 - 2.99 M_sun). Inference aborted to prevent uncalibrated extrapolation.")
        st.stop()
    if "Radius" in obs_mode and (R_obs < 4.22 or R_obs > 42.02):
        st.error("Warning: Radius input resides outside the generated physical manifold (4.22 - 42.02 km). Inference aborted to prevent uncalibrated extrapolation.")
        st.stop()
        
    with st.spinner("Generating N=5,000 MC samples and running inference..."):
        N = 5000
        
        try:
            scaler = joblib.load(os.path.join("data", "ml_tensors", "scaler.joblib"))
        except Exception as e:
            st.error(f"Failed to load standard scaler: {e}")
            st.stop()
            
        # 1. Generate Gaussian Samples based on Observatory Mode
        mass_samples = np.random.normal(M_obs, M_err, N)
        
        if obs_mode == "NICER X-ray (Mass & Radius)":
            radius_samples = np.random.normal(R_obs, R_err, N)
            c_samples = mass_samples / radius_samples
            lambda_samples = np.full(N, scaler.mean_[3])  # Marginalize Lambda
            
        elif obs_mode == "LIGO Gravitational Wave (Mass & Λ)":
            lambda_samples = np.random.normal(L_obs, L_err, N)
            radius_samples = np.full(N, scaler.mean_[1])  # Marginalize Radius
            c_samples = np.full(N, scaler.mean_[2])       # Marginalize Compactness directly
            
        else: # Multi-Messenger
            radius_samples = np.random.normal(R_obs, R_err, N)
            lambda_samples = np.random.normal(L_obs, L_err, N)
            c_samples = mass_samples / radius_samples
            
        X_mc = pd.DataFrame({
            'Mass': mass_samples,
            'Radius': radius_samples,
            'C': c_samples,
            'log10_Lambda': lambda_samples
        })
        
        X_mc_scaled = pd.DataFrame(scaler.transform(X_mc), columns=X_mc.columns)
        
        # 3. Model Inference
        if model_choice == "Optimized XGBoost":
            model_path = os.path.join("outputs", "xgboost", "xgboost_weights.json")
            if not os.path.exists(model_path):
                st.error("XGBoost weights not found.")
                st.stop()
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(model_path)
            probs = xgb_model.predict_proba(X_mc_scaled)[:, 1]
            
        elif model_choice == "Optimized MLP":
            params_path = os.path.join("outputs", "mlp_best_params.json")
            model_path = os.path.join("outputs", "mlp", "mlp_weights.pth")
            if not os.path.exists(params_path) or not os.path.exists(model_path):
                st.error("MLP weights or params not found.")
                st.stop()
                
            with open(params_path, "r") as f:
                best_params = json.load(f)
            hidden_sizes = eval(best_params['hidden_layer_sizes'])
            dropout_rate = best_params['dropout_rate']
            
            device = torch.device("cpu")
            mlp_model = DynamicMLP(input_dim=4, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
            mlp_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            mlp_model.eval()
            
            with torch.no_grad():
                probs = mlp_model(torch.FloatTensor(X_mc_scaled.values.copy()).to(device)).cpu().numpy().flatten()
                
        # Calculate statistics
        expected_prob = np.mean(probs)
        lower_bound = np.percentile(probs, 2.5)
        upper_bound = np.percentile(probs, 97.5)
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Quark Probability", f"{expected_prob:.2%}")
        col2.metric("Lower Bound (95% CI)", f"{lower_bound:.2%}")
        col3.metric("Upper Bound (95% CI)", f"{upper_bound:.2%}")
        
        # 4. Render Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Choose X-axis based on Observatory mode
        if obs_mode == "LIGO Gravitational Wave (Mass & Λ)":
            x_vals = mass_samples
            y_vals = lambda_samples
            x_label = r'Mass ($M_\odot$)'
            y_label = r'$\log_{10}(\Lambda)$'
            y_obs, y_err = L_obs, L_err
        else:
            x_vals = mass_samples
            y_vals = radius_samples
            x_label = r'Mass ($M_\odot$)'
            y_label = 'Radius (km)'
            y_obs, y_err = R_obs, R_err
            
        scatter = ax.scatter(x_vals, y_vals, c=probs, cmap='coolwarm', alpha=0.6, edgecolors='none', s=10)
        ax.errorbar(M_obs, y_obs, xerr=M_err, yerr=y_err, fmt='k+', markersize=15, capsize=3, linewidth=1.5, label='Mean Obs ± 1σ')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{obs_mode} Topology Map (N={N}) - {model_choice}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Predicted Probability (Quark Phase)')
        
        st.pyplot(fig)
