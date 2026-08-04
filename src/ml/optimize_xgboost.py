import argparse
import json
import logging
import os

import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve

from src.runtime import (
    add_runtime_args,
    configure_runtime_from_args,
    optuna_trials,
    require_paths,
    runtime_paths,
    write_run_manifest,
    xgb_device_params,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OPT_XGBOOST")


def load_data(data_root=None):
    paths = runtime_paths(data_root)
    tensor_dir = paths.clean_tensor_dir
    require_paths([tensor_dir / "train.parquet", tensor_dir / "val.parquet"], "XGBoost optimization")
    logger.info(f"Loading tensors from {tensor_dir}...")

    train_df = pd.read_parquet(tensor_dir / "train.parquet", engine="pyarrow")
    val_df = pd.read_parquet(tensor_dir / "val.parquet", engine="pyarrow")

    return train_df.drop(columns=["Label"]), train_df["Label"], val_df.drop(columns=["Label"]), val_df["Label"]


def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight, xgb_device):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_estimators": 1000,
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    param.update(xgb_device)

    model = xgb.XGBClassifier(**param, use_label_encoder=False, early_stopping_rounds=50, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    return auc(recall, precision)


def run_optimization(data_root=None, use_cuda_xgb=False):
    paths = runtime_paths(data_root)
    X_train, y_train, X_val, y_val = load_data(paths.data_root)

    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    scale_pos_weight = count_hadronic / count_quark if count_quark > 0 else 1.0
    xgb_device = xgb_device_params(use_cuda_xgb)

    logger.info("Initializing Optuna Study for XGBoost PR-AUC maximization...")
    study = optuna.create_study(direction="maximize")
    trials = optuna_trials(50)
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight, xgb_device), n_trials=trials)

    logger.info(f"Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    out_path = paths.outputs_root / "xgboost_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    write_run_manifest(paths.outputs_root, "clean_xgboost_optimization", paths.data_root)
    logger.info(f"Saved optimized hyperparameters to {out_path}")

    plots_dir = paths.plots_root / "ml_optimization"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        vis.plot_optimization_history(study).write_image(plots_dir / "xgboost_opt_history.pdf")
        vis.plot_parallel_coordinate(study).write_image(plots_dir / "xgboost_parallel_coordinate.pdf")
        logger.info(f"Saved Optuna visualizations to {plots_dir}")
    except Exception as e:
        logger.warning(f"Could not save visualizations (make sure kaleido/plotly are installed): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize clean XGBoost classifier.")
    add_runtime_args(parser)
    parser.add_argument("--use-cuda-xgb", action="store_true", help="request CUDA-backed XGBoost")
    parser.add_argument("--fast", action="store_true", help="use readiness-sized HPO defaults")
    args = parser.parse_args()
    configure_runtime_from_args(args)
    run_optimization(args.data_root, args.use_cuda_xgb)
