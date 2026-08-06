"""Shared helpers for clean advanced ML evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
import xgboost as xgb

from src.ml.mlp_model import load_mlp_model
from src.runtime import require_paths, require_test_diagnostics_authorized, runtime_paths


FEATURES = ["Mass", "Radius", "log10_Lambda"]


def clean_runtime():
    return runtime_paths()


def clean_plot_dir() -> Path:
    paths = clean_runtime()
    output_dir = paths.plots_root / "ml_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_clean_test():
    require_test_diagnostics_authorized()
    paths = clean_runtime()
    test_path = paths.clean_tensor_dir / "test.parquet"
    require_paths([test_path], "Advanced clean evaluation")
    test_df = pd.read_parquet(test_path, engine="pyarrow")
    return test_df, test_df[FEATURES].values, test_df["Label"].values


def load_clean_scaler():
    paths = clean_runtime()
    scaler_path = paths.clean_tensor_dir / "scaler.joblib"
    require_paths([scaler_path], "Advanced clean evaluation scaler")
    return joblib.load(scaler_path)


def load_clean_xgb():
    paths = clean_runtime()
    model_path = paths.outputs_root / "xgboost" / "xgboost_weights.json"
    require_paths([model_path], "Advanced clean XGBoost evaluation")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def load_clean_mlp(input_dim: int, device: torch.device):
    paths = clean_runtime()
    params_path = paths.outputs_root / "mlp_best_params.json"
    weights_path = paths.outputs_root / "mlp" / "mlp_weights.pth"
    require_paths([params_path, weights_path], "Advanced clean MLP evaluation")
    return load_mlp_model(params_path, weights_path, input_dim, device)
