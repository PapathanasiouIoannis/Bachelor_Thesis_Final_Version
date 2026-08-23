import os
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from src.ml.mlp_model import load_mlp_model
from src.runtime import runtime_paths

PATHS = runtime_paths()

# -----------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------
st.set_page_config(
    page_title="Compact-star synthetic classifier — baseline demonstration",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
        
        h1 {
            color: #4facfe;
            letter-spacing: -0.02em;
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

        @media (max-width: 700px) {
            h1 {
                font-size: 2rem !important;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.65rem !important;
            }
            div[data-testid="metric-container"] {
                padding: 12px;
            }
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("Compact-star synthetic classifier")
st.caption(
    "Baseline demonstration · restricted APR-1-surrogate versus fixed-CFL4 comparison"
)
st.info(
    "This independent post-thesis demonstration explores classifier scores inside a "
    "restricted synthetic model comparison. Its outputs are not observational measurements, "
    "calibrated astrophysical probabilities, composition posteriors, or evidence about the "
    "interior of any real star."
)
with st.expander("Scope, assumptions, and responsibility"):
    st.markdown(
        """
- The retained classifier distinguishes the repository's analytic **APR-1 hadronic surrogate**
  from one **fixed-CFL4 benchmark** within this experiment.
- Input uncertainty is represented by independent Gaussian samples. Real NICER or LIGO
  posterior structure, covariance, source dependence, selection effects, and an astrophysical
  class prior are not included.
- The displayed score range describes variation under the selected sampling assumptions. It is
  not a confidence interval for stellar composition.
- SHAP describes how the fitted classifier responds to its inputs; it does not establish physical
  truth or observational validity.

Developed by **Ioannis Papathanasiou** as an independent post-thesis extension, under the
supervision of **Charalampos Moustakidis** and **Theodoros Diakonidis**. Codex assisted with
software development; scientific interpretation and responsibility remain with Ioannis
Papathanasiou.
"""
    )


# -----------------------------------------
# 2. Cached Resource Loaders
# -----------------------------------------
@st.cache_resource(show_spinner="Loading Standard Scaler...")
def load_scaler():
    return joblib.load(PATHS.clean_tensor_dir / "scaler.joblib")


@st.cache_resource(show_spinner="Loading XGBoost Core...")
def load_xgboost():
    model_path = PATHS.outputs_root / "xgboost" / "xgboost_weights.json"
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


@st.cache_resource(show_spinner="Loading Deep PyTorch Core...")
def load_mlp():
    device = torch.device("cpu")
    return load_mlp_model(
        PATHS.outputs_root / "mlp_best_params.json",
        PATHS.outputs_root / "mlp" / "mlp_weights.pth",
        3,
        device,
    )


@st.cache_data(show_spinner="Loading restricted synthetic model space...")
def load_background_manifold(_scaler):
    # Load ALL available data to show the true, dense physical background manifold
    df_list = []
    for split in ["train.parquet", "val.parquet", "test.parquet"]:
        path = PATHS.clean_tensor_dir / split
        if os.path.exists(path):
            df_list.append(pd.read_parquet(path, engine="pyarrow"))

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)

    features = ["Mass", "Radius", "log10_Lambda"]
    X_scaled = full_df[features].values
    X_raw = _scaler.inverse_transform(X_scaled)
    df_raw = pd.DataFrame(X_raw, columns=features)
    df_raw["Model_Class"] = np.where(
        full_df["Label"] == 1,
        "fixed-CFL4 benchmark",
        "APR-1 hadronic surrogate",
    )

    # User specifically requested ALL points, so we remove the random downsampling
    return df_raw


if not os.path.exists(PATHS.clean_tensor_dir / "scaler.joblib"):
    st.error(
        "🚨 **Fatal Error:** Could not find `scaler.joblib`. Please ensure the ML pipeline has been run."
    )
    st.stop()

scaler = load_scaler()
bg_manifold = load_background_manifold(scaler)

# -----------------------------------------
# 4. Sidebar UI (synthetic inputs and classifier controls)
# -----------------------------------------
with st.sidebar:
    st.markdown("**Synthetic input mode**")
    obs_mode = st.radio(
        "Select available synthetic inputs",
        ["Mass–radius inputs", "Mass–tidal inputs", "Mass–radius–tidal inputs"],
    )
    uses_radius = obs_mode != "Mass–tidal inputs"
    uses_lambda = obs_mode != "Mass–radius inputs"

    st.markdown("**Input assumptions**")
    M_obs = st.number_input(
        "Mean mass (solar masses)",
        min_value=0.01,
        max_value=10.0,
        value=1.40,
        step=0.01,
    )
    M_err = st.number_input(
        "Mass Gaussian σ (solar masses)",
        min_value=0.001,
        max_value=1.0,
        value=0.05,
        step=0.01,
    )

    if uses_radius:
        R_obs = st.number_input(
            "Mean Radius (km)", min_value=1.0, max_value=100.0, value=11.50, step=0.1
        )
        R_err = st.number_input(
            "Radius Gaussian σ (km)",
            min_value=0.01,
            max_value=3.0,
            value=0.5,
            step=0.05,
        )
    else:
        R_obs, R_err = None, None

    if uses_lambda:
        L_obs = st.number_input(
            "Mean log10(Λ)", min_value=0.0, max_value=5.0, value=2.50, step=0.1
        )
        L_err = st.number_input(
            "log10(Λ) Gaussian σ", min_value=0.01, max_value=2.0, value=0.2, step=0.05
        )
    else:
        L_obs, L_err = None, None

    st.markdown("**Classifier**")
    model_choice = st.selectbox(
        "Classifier model", ["Optimized XGBoost", "Optimized MLP"]
    )

    st.markdown("---")
    run_btn = st.button("Run synthetic sampling", use_container_width=True)

# -----------------------------------------
# 5. Core Execution Logic
# -----------------------------------------
if run_btn:
    if M_obs < 0.10 or M_obs > 2.99:
        st.error(
            "Mass lies outside the retained synthetic support (0.10–2.99 solar masses). Sampling stopped."
        )
        st.stop()
    if uses_radius and (R_obs < 4.22 or R_obs > 42.02):
        st.error(
            "Radius lies outside the retained synthetic support (4.22–42.02 km). Sampling stopped."
        )
        st.stop()

    with st.spinner(
        f"Sampling the restricted synthetic classifier with {model_choice}..."
    ):
        N = 5000

        mass_samples = np.random.normal(M_obs, M_err, N)

        if obs_mode == "Mass–radius inputs":
            radius_samples = np.random.normal(R_obs, R_err, N)
            lambda_samples = np.full(N, scaler.mean_[2])

        elif obs_mode == "Mass–tidal inputs":
            lambda_samples = np.random.normal(L_obs, L_err, N)
            radius_samples = np.full(N, scaler.mean_[1])

        else:
            radius_samples = np.random.normal(R_obs, R_err, N)
            lambda_samples = np.random.normal(L_obs, L_err, N)

        X_mc = pd.DataFrame(
            {
                "Mass": mass_samples,
                "Radius": radius_samples,
                "log10_Lambda": lambda_samples,
            }
        )

        X_mc_core = X_mc.copy()
        X_mc_core.columns = ["Mass", "Radius", "log10_Lambda"]
        X_mc_scaled = pd.DataFrame(
            scaler.transform(X_mc_core), columns=X_mc_core.columns
        )

        if model_choice == "Optimized XGBoost":
            xgb_model = load_xgboost()
            probs = xgb_model.predict_proba(X_mc_scaled)[:, 1]
        else:
            mlp_model = load_mlp()
            with torch.no_grad():
                logits = mlp_model(torch.FloatTensor(X_mc_scaled.values.copy()))
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

        X_mc["Fixed_CFL4_Model_Score"] = probs

        expected_prob = np.mean(probs)
        lower_bound = np.percentile(probs, 2.5)
        upper_bound = np.percentile(probs, 97.5)

        score_range_width = upper_bound - lower_bound

        st.markdown("## Classifier-score summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean fixed-CFL4 score", f"{expected_prob:.2%}")
        m2.metric("Central 95% range — lower", f"{lower_bound:.2%}")
        m3.metric("Central 95% range — upper", f"{upper_bound:.2%}")
        m4.metric("Score-range width", f"{score_range_width:.2%}")
        st.caption(
            "These are uncalibrated classifier scores under the selected independent Gaussian "
            "sampling model. The central range is not a confidence interval or composition posterior."
        )

        # -----------------------------------------
        # 6. Plotly Interactive Visualization
        # -----------------------------------------
        st.markdown("## Restricted synthetic model space")

        fig = go.Figure()

        # 1. Plot the distinct background EoS universe
        fig.add_trace(
            go.Scatter(
                x=bg_manifold["Radius"] if uses_radius else bg_manifold["log10_Lambda"],
                y=bg_manifold["Mass"],
                mode="markers",
                marker=dict(
                    color=np.where(
                        bg_manifold["Model_Class"] == "fixed-CFL4 benchmark",
                        "#ff4b4b",
                        "#1f77b4",
                    ),
                    size=4,
                    opacity=0.3,
                ),
                name="Restricted synthetic training support",
                hoverinfo="skip",
            )
        )

        # 2. Add the synthetic input samples.
        fig.add_trace(
            go.Scatter(
                x=X_mc["Radius"] if uses_radius else X_mc["log10_Lambda"],
                y=X_mc["Mass"],
                mode="markers",
                marker=dict(
                    color=probs,
                    colorscale="curl",
                    size=5,
                    opacity=0.4,
                    showscale=True,
                    colorbar=dict(title="fixed-CFL4 score"),
                ),
                name="Synthetic input samples",
                hovertemplate="<b>Mass:</b> %{y:.2f}<br><b>X-axis:</b> %{x:.2f}<br><b>Model score:</b> %{marker.color:.2%}<extra></extra>",
            )
        )

        x_title = "Radius (km)" if uses_radius else "log10(Λ)"

        if obs_mode == "Mass–radius–tidal inputs":
            fig = go.Figure()

            # 3D Background (Sub-sampled to prevent WebGL crash on 400k+ points)
            bg_sample_3d = (
                bg_manifold.sample(n=10000, random_state=42)
                if len(bg_manifold) > 10000
                else bg_manifold
            )

            fig.add_trace(
                go.Scatter3d(
                    x=bg_sample_3d["Radius"],
                    y=bg_sample_3d["log10_Lambda"],
                    z=bg_sample_3d["Mass"],
                    mode="markers",
                    marker=dict(
                        color=np.where(
                            bg_sample_3d["Model_Class"] == "fixed-CFL4 benchmark",
                            "#ff4b4b",
                            "#1f77b4",
                        ),
                        size=3,
                        opacity=0.25,
                    ),
                    name="Restricted synthetic training support",
                    hoverinfo="skip",
                )
            )

            # 3D synthetic samples
            fig.add_trace(
                go.Scatter3d(
                    x=X_mc["Radius"],
                    y=X_mc["log10_Lambda"],
                    z=X_mc["Mass"],
                    mode="markers",
                    marker=dict(
                        color=probs,
                        colorscale="curl",
                        size=4,
                        opacity=0.35,
                        showscale=True,
                        colorbar=dict(title="fixed-CFL4 score"),
                    ),
                    name="Synthetic input samples",
                    hovertemplate="<b>R:</b> %{x:.2f}<br><b>Λ:</b> %{y:.2f}<br><b>M:</b> %{z:.2f}<extra></extra>",
                )
            )

            # 3D surface for the selected independent Gaussian scales
            u = np.linspace(0, 2 * np.pi, 40)
            v = np.linspace(0, np.pi, 40)
            x_ell = R_obs + R_err * np.outer(np.cos(u), np.sin(v))
            y_ell = L_obs + L_err * np.outer(np.sin(u), np.sin(v))
            z_ell = M_obs + M_err * np.outer(np.ones_like(u), np.cos(v))

            fig.add_trace(
                go.Surface(
                    x=x_ell,
                    y=y_ell,
                    z=z_ell,
                    opacity=0.15,
                    colorscale=[[0, "#00f2fe"], [1, "#00f2fe"]],
                    showscale=False,
                    name="Selected Gaussian 1σ surface",
                    hoverinfo="skip",
                )
            )

            fig.update_layout(
                scene=dict(
                    xaxis_title="Radius (km)",
                    yaxis_title="log10(Λ)",
                    zaxis_title="Mass (solar masses)",
                    xaxis=dict(gridcolor="#334155", backgroundcolor="rgba(0,0,0,0)"),
                    yaxis=dict(gridcolor="#334155", backgroundcolor="rgba(0,0,0,0)"),
                    zaxis=dict(gridcolor="#334155", backgroundcolor="rgba(0,0,0,0)"),
                ),
                height=700,
            )
        else:
            # Add a 2D contour for the selected independent Gaussian scales
            t = np.linspace(0, 2 * np.pi, 100)
            if uses_radius:
                x_ell = R_obs + R_err * np.cos(t)
                y_ell = M_obs + M_err * np.sin(t)
            else:
                x_ell = L_obs + L_err * np.cos(t)
                y_ell = M_obs + M_err * np.sin(t)

            fig.add_trace(
                go.Scatter(
                    x=x_ell,
                    y=y_ell,
                    mode="lines",
                    line=dict(color="#00f2fe", width=3, dash="solid"),
                    name="Selected Gaussian 1σ contour",
                    hoverinfo="skip",
                )
            )

            # 2D Plot layout
            fig.update_layout(
                xaxis=dict(
                    title=x_title, showgrid=True, gridcolor="#334155", zeroline=False
                ),
                yaxis=dict(
                    title="Mass (solar masses)",
                    showgrid=True,
                    gridcolor="#334155",
                    zeroline=False,
                ),
                height=600,
            )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------
        # 7. Reference XGBoost score explanation (SHAP waterfall)
        # -----------------------------------------
        with st.expander("Reference XGBoost score explanation (SHAP)"):
            st.markdown(
                "The summary above follows the classifier selected in the sidebar. This separate "
                "reference view always explains the retained XGBoost model at the mean synthetic "
                "input. SHAP values are calculated in log-odds margin space. A positive `f(x)` "
                "favors the fixed-CFL4 benchmark over the APR-1 surrogate inside this restricted "
                "experiment; it does not establish physical truth."
            )

            # Construct the precise Mean Observation
            r_val = R_obs if R_obs is not None else scaler.mean_[1]
            l_val = L_obs if L_obs is not None else scaler.mean_[2]

            mean_obs_df = pd.DataFrame(
                {"Mass": [M_obs], "Radius": [r_val], "log10_Lambda": [l_val]}
            )

            mean_scaled = pd.DataFrame(
                scaler.transform(mean_obs_df), columns=mean_obs_df.columns
            )

            xgb_core = load_xgboost()
            explainer = shap.TreeExplainer(xgb_core)
            shap_values = explainer(mean_scaled)

            # Fix the UI: Replace the scaled data with raw physical values for the plot labels
            shap_values.data = np.round(mean_obs_df.values, 3)

            plt.style.use("dark_background")
            fig_shap = plt.figure(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)

            # Make the plot fully transparent to match the sleek Streamlit dark theme
            fig_shap = plt.gcf()
            fig_shap.patch.set_alpha(0.0)
            ax = plt.gca()
            ax.patch.set_alpha(0.0)

            st.pyplot(fig_shap, transparent=True)
            plt.clf()
            plt.style.use("default")

        csv = X_mc.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download synthetic samples and model scores (CSV)",
            data=csv,
            file_name=f"synthetic_scores_{model_choice.replace(' ', '_')}.csv",
            mime="text/csv",
        )

else:
    st.info(
        "Choose synthetic input assumptions in the sidebar and select **Run synthetic sampling** to begin."
    )
