import os

import pandas as pd
import plotly.graph_objects as go

from src.utils.logger import get_logger
from src.visualize.style_config import COLORS

logger = get_logger("VISUAL_3D")


def plot_interactive_3d(df: pd.DataFrame) -> None:
    """
    Creates an interactive 3D scatter plot of the EoS manifolds.
    X = Radius (R)
    Y = Mass (M)
    Z = Central Pressure (P_c)

    Sub-samples a maximum of 500 Hadronic and 500 Quark stars to keep the HTML lightweight.
    """
    logger.info("\n--- Generating Interactive 3D Manifold Plot ---")

    # separate manifolds
    df_h = df[df["Label"] == 0].copy()
    df_q = df[df["Label"] == 1].copy()

    # strict sub-sampling: randomly select max 500 points for each manifold
    if len(df_h) > 500:
        df_h = df_h.sample(n=500, random_state=42)
    if len(df_q) > 500:
        df_q = df_q.sample(n=500, random_state=42)

    logger.info(f"Plotting {len(df_h)} Hadronic and {len(df_q)} Quark stars.")

    fig = go.Figure()

    # hadronic Manifold
    fig.add_trace(
        go.Scatter3d(
            x=df_h["Radius"],
            y=df_h["Mass"],
            z=df_h["P_Central"],
            mode="markers",
            marker=dict(
                size=4,
                color=COLORS.get("H_main", "#0072B2"),
                opacity=0.7,
                line=dict(width=0),
            ),
            name="Hadronic Stars",
        )
    )

    # quark Manifold
    fig.add_trace(
        go.Scatter3d(
            x=df_q["Radius"],
            y=df_q["Mass"],
            z=df_q["P_Central"],
            mode="markers",
            marker=dict(
                size=4,
                color=COLORS.get("Q_main", "#D55E00"),
                opacity=0.7,
                line=dict(width=0),
            ),
            name="Quark Stars",
        )
    )

    # aesthetics
    fig.update_layout(
        template="plotly_dark",
        title="3D EoS Manifold: Radius vs Mass vs Central Pressure",
        scene=dict(
            xaxis_title="Radius (km)",
            yaxis_title="Mass (M_sun)",
            zaxis_title="Central Pressure (MeV/fm^3)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # ensure output directory exists and save
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig_interactive_3d_manifold.html")

    fig.write_html(out_path)
    logger.info(f"done Saved 3D Interactive Plot to {out_path}")
