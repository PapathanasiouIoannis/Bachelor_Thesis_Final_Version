import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------
st.set_page_config(
    page_title="Astrophysical EoS Inference Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background-color: #0b0f19;
            color: #e2e8f0;
        }
        
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 800;
            color: #00f2fe;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1.1rem !important;
            font-weight: 400;
            color: #94a3b8;
        }
        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: transform 0.2s ease-in-out;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(0, 242, 254, 0.3);
        }
        
        .title-gradient {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 0rem;
        }
        .subtitle {
            color: #94a3b8;
            font-weight: 300;
            font-size: 1.2rem;
            margin-bottom: 2.5rem;
        }
        
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: #0b0f19;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 242, 254, 0.3);
            color: #fff;
        }
        
        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-gradient">Astrophysical Multi-Messenger Inference Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A rigorous Monte Carlo engine for predicting Equation of State topologies using deep learning.</div>', unsafe_allow_html=True)

# -----------------------------------------
# 2. PyTorch Model Definition
# -----------------------------------------
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

# -----------------------------------------
# 3. Cached Resource Loaders
# -----------------------------------------
@st.cache_resource(show_spinner="Loading Standard Scaler...")
def load_scaler():
    return joblib.load(os.path.join("data", "ml_tensors", "scaler.joblib"))

@st.cache_resource(show_spinner="Loading XGBoost Core...")
def load_xgboost():
    model_path = os.path.join("outputs", "xgboost", "xgboost_weights.json")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

@st.cache_resource(show_spinner="Loading Deep PyTorch Core...")
def load_mlp():
    params_path = os.path.join("outputs", "mlp_best_params.json")
    model_path = os.path.join("outputs", "mlp", "mlp_weights.pth")
    with open(params_path, "r") as f:
        best_params = json.load(f)
    hidden_sizes = eval(best_params['hidden_layer_sizes'])
    dropout_rate = best_params['dropout_rate']
    
    device = torch.device("cpu")
    model = DynamicMLP(input_dim=3, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

@st.cache_data(show_spinner="Loading Theoretical EoS Manifold...")
def load_background_manifold(_scaler):
    # Load ALL available data to show the true, dense physical background manifold
    df_list = []
    for split in ["train.parquet", "val.parquet", "test.parquet"]:
        path = os.path.join("data", "ml_tensors", split)
        if os.path.exists(path):
            df_list.append(pd.read_parquet(path, engine='pyarrow'))
            
    if not df_list:
        return pd.DataFrame()
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    features = ['Mass', 'Radius', 'log10_Lambda']
    X_scaled = full_df[features].values
    X_raw = _scaler.inverse_transform(X_scaled)
    df_raw = pd.DataFrame(X_raw, columns=features)
    df_raw['Phase_Class'] = np.where(full_df['Label'] == 1, "Theoretical Quark EoS", "Theoretical Hadronic EoS")
    
    # User specifically requested ALL points, so we remove the random downsampling
    return df_raw

if not os.path.exists(os.path.join("data", "ml_tensors", "scaler.joblib")):
    st.error("🚨 **Fatal Error:** Could not find `scaler.joblib`. Please ensure the ML pipeline has been run.")
    st.stop()
    
scaler = load_scaler()
bg_manifold = load_background_manifold(scaler)

# -----------------------------------------
# 4. Sidebar UI (Telemetry & Control)
# -----------------------------------------
with st.sidebar:
    st.markdown("### 🔭 Observatory Mode")
    obs_mode = st.radio(
        "Select Available Data stream:",
        ["NICER X-ray (Mass & Radius)", "LIGO Gravitational Wave (Mass & Λ)", "Multi-Messenger (Mass, Radius & Λ)"],
        label_visibility="collapsed"
    )
    
    st.markdown("### 📡 Telemetry Input")
    M_obs = st.number_input("Mean Mass (M_sun)", min_value=0.01, max_value=10.0, value=1.40, step=0.01)
    M_err = st.number_input("Mass ±1σ Error", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
    
    if "Radius" in obs_mode:
        R_obs = st.number_input("Mean Radius (km)", min_value=1.0, max_value=100.0, value=11.50, step=0.1)
        R_err = st.number_input("Radius ±1σ Error", min_value=0.01, max_value=3.0, value=0.5, step=0.05)
    else:
        R_obs, R_err = None, None
        
    if "Λ" in obs_mode:
        L_obs = st.number_input("Mean log10(Λ)", min_value=0.0, max_value=5.0, value=2.50, step=0.1)
        L_err = st.number_input("log10(Λ) ±1σ Error", min_value=0.01, max_value=2.0, value=0.2, step=0.05)
    else:
        L_obs, L_err = None, None
        
    st.markdown("### 🧠 AI Core")
    model_choice = st.selectbox("Inference Model", ["Optimized XGBoost", "Optimized MLP"], label_visibility="collapsed")
    
    st.markdown("---")
    run_btn = st.button("🚀 Execute Monte Carlo", use_container_width=True)

# -----------------------------------------
# 5. Core Execution Logic
# -----------------------------------------
if run_btn:
    if M_obs < 0.10 or M_obs > 2.99:
        st.error("🚨 **Out of Distribution:** Mass input resides outside the generated physical manifold (0.10 - 2.99 M_sun). Inference aborted.")
        st.stop()
    if "Radius" in obs_mode and (R_obs < 4.22 or R_obs > 42.02):
        st.error("🚨 **Out of Distribution:** Radius input resides outside the generated physical manifold (4.22 - 42.02 km). Inference aborted.")
        st.stop()

    with st.spinner(f"Running highly-parallelized Monte Carlo Inference using {model_choice}..."):
        N = 5000
        
        mass_samples = np.random.normal(M_obs, M_err, N)
        
        if obs_mode == "NICER X-ray (Mass & Radius)":
            radius_samples = np.random.normal(R_obs, R_err, N)
            lambda_samples = np.full(N, scaler.mean_[2]) 
            
        elif obs_mode == "LIGO Gravitational Wave (Mass & Λ)":
            lambda_samples = np.random.normal(L_obs, L_err, N)
            radius_samples = np.full(N, scaler.mean_[1])  
            
        else:
            radius_samples = np.random.normal(R_obs, R_err, N)
            lambda_samples = np.random.normal(L_obs, L_err, N)
            
        X_mc = pd.DataFrame({
            'Mass': mass_samples,
            'Radius': radius_samples,
            'log10_Lambda': lambda_samples
        })
        
        X_mc_core = X_mc.copy()
        X_mc_core.columns = ['Mass', 'Radius', 'log10_Lambda']
        X_mc_scaled = pd.DataFrame(scaler.transform(X_mc_core), columns=X_mc_core.columns)
        
        if model_choice == "Optimized XGBoost":
            xgb_model = load_xgboost()
            probs = xgb_model.predict_proba(X_mc_scaled)[:, 1]
        else:
            mlp_model = load_mlp()
            with torch.no_grad():
                probs = mlp_model(torch.FloatTensor(X_mc_scaled.values.copy())).cpu().numpy().flatten()
                
        X_mc['Quark_Probability'] = probs
        X_mc['Phase_Class'] = np.where(probs >= 0.5, "MC Observation (Quark Phase)", "MC Observation (Hadronic Phase)")
        
        expected_prob = np.mean(probs)
        lower_bound = np.percentile(probs, 2.5)
        upper_bound = np.percentile(probs, 97.5)
        
        # Calculate Reliability based on CI Spread
        ci_spread = upper_bound - lower_bound
        if ci_spread <= 0.10:
            rel_status = "Very High"
        elif ci_spread <= 0.30:
            rel_status = "Moderate"
        else:
            rel_status = "Low (Boundary Straddle)"
            
        st.markdown("### 📊 Inference Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Quark Probability", f"{expected_prob:.2%}")
        m2.metric("Lower Bound (95% CI)", f"{lower_bound:.2%}")
        m3.metric("Upper Bound (95% CI)", f"{upper_bound:.2%}")
        m4.metric("Inference Reliability", rel_status, f"{ci_spread:.1%} variance", delta_color="inverse")
        
        # -----------------------------------------
        # 6. Plotly Interactive Visualization
        # -----------------------------------------
        st.markdown("<br>### 🌌 Topological Phase Manifold", unsafe_allow_html=True)
        
        fig = go.Figure()

        # 1. Plot the distinct background EoS universe
        fig.add_trace(go.Scatter(
            x=bg_manifold['Radius'] if obs_mode != "LIGO Gravitational Wave (Mass & Λ)" else bg_manifold['log10_Lambda'],
            y=bg_manifold['Mass'],
            mode='markers',
            marker=dict(
                color=np.where(bg_manifold['Phase_Class'] == "Theoretical Quark EoS", '#ff4b4b', '#1f77b4'),
                size=4,
                opacity=0.3
            ),
            name="Theoretical EoS Manifold",
            hoverinfo='skip'
        ))

        # 2. Add the glowing MC Observational Measurement (transparent)
        fig.add_trace(go.Scatter(
            x=X_mc['Radius'] if obs_mode != "LIGO Gravitational Wave (Mass & Λ)" else X_mc['log10_Lambda'],
            y=X_mc['Mass'],
            mode='markers',
            marker=dict(
                color=probs,
                colorscale='curl',
                size=5,
                opacity=0.4,
                showscale=True,
                colorbar=dict(title="Quark Phase %")
            ),
            name="Telescope Measurement (MC)",
            hovertemplate="<b>Mass:</b> %{y:.2f}<br><b>X-Axis:</b> %{x:.2f}<br><b>Prob:</b> %{marker.color:.2%}<extra></extra>"
        ))
        
        x_title = 'Radius (km)' if obs_mode != "LIGO Gravitational Wave (Mass & Λ)" else 'log10(Λ)'

        if obs_mode == "Multi-Messenger (Mass, Radius & Λ)":
            fig = go.Figure()
            
            # 3D Background
            fig.add_trace(go.Scatter3d(
                x=bg_manifold['Radius'],
                y=bg_manifold['log10_Lambda'],
                z=bg_manifold['Mass'],
                mode='markers',
                marker=dict(
                    color=np.where(bg_manifold['Phase_Class'] == "Theoretical Quark EoS", '#ff4b4b', '#1f77b4'),
                    size=3,
                    opacity=0.25
                ),
                name="Theoretical EoS Manifold",
                hoverinfo='skip'
            ))

            # 3D MC Ellipsoid points (transparent)
            fig.add_trace(go.Scatter3d(
                x=X_mc['Radius'],
                y=X_mc['log10_Lambda'],
                z=X_mc['Mass'],
                mode='markers',
                marker=dict(
                    color=probs,
                    colorscale='curl',
                    size=4,
                    opacity=0.35,
                    showscale=True,
                    colorbar=dict(title="Quark Phase %")
                ),
                name="3D Observation Ellipsoid",
                hovertemplate="<b>R:</b> %{x:.2f}<br><b>Λ:</b> %{y:.2f}<br><b>M:</b> %{z:.2f}<extra></extra>"
            ))
            
            # 3D 1-sigma bounding surface (perimeter curve)
            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 40)
            x_ell = R_obs + R_err * np.outer(np.cos(u), np.sin(v))
            y_ell = L_obs + L_err * np.outer(np.sin(u), np.sin(v))
            z_ell = M_obs + M_err * np.outer(np.ones_like(u), np.cos(v))

            fig.add_trace(go.Surface(
                x=x_ell, y=y_ell, z=z_ell,
                opacity=0.15,
                colorscale=[[0, '#00f2fe'], [1, '#00f2fe']],
                showscale=False,
                name='1σ Observation Perimeter',
                hoverinfo='skip'
            ))
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='Radius (km)',
                    yaxis_title='log10(Λ)',
                    zaxis_title='Mass (M_sun)',
                    xaxis=dict(gridcolor='#334155', backgroundcolor='rgba(0,0,0,0)'),
                    yaxis=dict(gridcolor='#334155', backgroundcolor='rgba(0,0,0,0)'),
                    zaxis=dict(gridcolor='#334155', backgroundcolor='rgba(0,0,0,0)')
                ),
                height=700
            )
        else:
            # Add 2D bounding contour based on the theoretical 1-sigma observing ellipse
            t = np.linspace(0, 2 * np.pi, 100)
            if obs_mode != "LIGO Gravitational Wave (Mass & Λ)":
                x_ell = R_obs + R_err * np.cos(t)
                y_ell = M_obs + M_err * np.sin(t)
            else:
                x_ell = L_obs + L_err * np.cos(t)
                y_ell = M_obs + M_err * np.sin(t)
                
            fig.add_trace(go.Scatter(
                x=x_ell, y=y_ell, mode='lines',
                line=dict(color='#00f2fe', width=3, dash='solid'),
                name='1σ Observation Perimeter',
                hoverinfo='skip'
            ))
            
            # 2D Plot layout
            fig.update_layout(
                xaxis=dict(title=x_title, showgrid=True, gridcolor='#334155', zeroline=False),
                yaxis=dict(title='Mass (M_sun)', showgrid=True, gridcolor='#334155', zeroline=False),
                height=600
            )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # -----------------------------------------
        # 7. AI Explainability (SHAP Waterfall)
        # -----------------------------------------
        with st.expander("🧠 AI Decision Explainability (SHAP)"):
            st.markdown("This module uses the core **XGBoost AI** to mathematically break down exactly how your Mean Observation influenced the final Probability prediction. *(Note: SHAP values are calculated in log-odds margin space. A final positive `f(x)` strongly predicts a Quark Star).*")
            
            # Construct the precise Mean Observation
            r_val = R_obs if R_obs is not None else scaler.mean_[1]
            l_val = L_obs if L_obs is not None else scaler.mean_[2]
            
            mean_obs_df = pd.DataFrame({
                'Mass': [M_obs],
                'Radius': [r_val],
                'log10_Lambda': [l_val]
            })
            
            mean_scaled = pd.DataFrame(scaler.transform(mean_obs_df), columns=mean_obs_df.columns)
            
            xgb_core = load_xgboost()
            explainer = shap.TreeExplainer(xgb_core)
            shap_values = explainer(mean_scaled)
            
            # Fix the UI: Replace the scaled data with raw physical values for the plot labels
            shap_values.data = np.round(mean_obs_df.values, 3)
            
            plt.style.use('dark_background')
            fig_shap = plt.figure(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            
            # Make the plot fully transparent to match the sleek Streamlit dark theme
            fig_shap = plt.gcf()
            fig_shap.patch.set_alpha(0.0)
            ax = plt.gca()
            ax.patch.set_alpha(0.0)
            
            st.pyplot(fig_shap, transparent=True)
            plt.clf()
            plt.style.use('default')
        
        csv = X_mc.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Monte Carlo Tensor (CSV)",
            data=csv,
            file_name=f'MC_{model_choice.replace(" ", "_")}_results.csv',
            mime='text/csv',
        )

else:
    st.info("👈 Enter observatory telemetry parameters in the sidebar and click **Execute Monte Carlo** to begin.")
