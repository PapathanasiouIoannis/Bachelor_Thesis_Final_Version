"""Shared helpers for perturbed advanced ML evaluation scripts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
import xgboost as xgb

from src.ml.mlp_model import load_mlp_model
from src.runtime import require_paths, runtime_paths


FEATURE_SETS = {"MR": ["Mass", "Radius"], "MRL": ["Mass", "Radius", "log10_Lambda"]}


def perturb_runtime():
    return runtime_paths()


def perturb_plot_dir() -> Path:
    paths = perturb_runtime()
    output_dir = paths.plots_perturb_root / "ml_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_perturb_test(feature_set: str):
    paths = perturb_runtime()
    test_path = paths.perturb_tensor_dir / "test.parquet"
    require_paths([test_path], f"Perturbed advanced evaluation {feature_set}")
    test_df = pd.read_parquet(test_path, engine="pyarrow")
    features = FEATURE_SETS[feature_set]
    return test_df, test_df[features].values, test_df["Label"].values


def load_perturb_scaler():
    paths = perturb_runtime()
    scaler_path = paths.perturb_tensor_dir / "scaler_perturb.joblib"
    require_paths([scaler_path], "Perturbed advanced evaluation scaler")
    return joblib.load(scaler_path)


def load_perturb_xgb(feature_set: str):
    paths = perturb_runtime()
    model_path = paths.outputs_perturb_root / f"xgboost_{feature_set}" / "xgboost_weights.json"
    require_paths([model_path], f"Perturbed XGBoost evaluation {feature_set}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def load_perturb_mlp(feature_set: str, input_dim: int, device: torch.device):
    paths = perturb_runtime()
    params_path = paths.outputs_perturb_root / f"mlp_{feature_set}_best_params.json"
    weights_path = paths.outputs_perturb_root / f"mlp_{feature_set}" / "mlp_weights.pth"
    require_paths([params_path, weights_path], f"Perturbed MLP evaluation {feature_set}")
    return load_mlp_model(params_path, weights_path, input_dim, device)
