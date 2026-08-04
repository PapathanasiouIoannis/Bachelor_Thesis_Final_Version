import argparse
import json
import logging

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
logger = logging.getLogger("PERTURB_OPT_XGBOOST")


def features_for(feature_set):
    if feature_set == "MR":
        return ["Mass", "Radius"]
    if feature_set == "MRL":
        return ["Mass", "Radius", "log10_Lambda"]
    raise ValueError("Invalid feature set.")


def load_data(feature_set="MR", data_root=None):
    paths = runtime_paths(data_root)
    tensor_dir = paths.perturb_tensor_dir
    require_paths([tensor_dir / "train.parquet", tensor_dir / "val.parquet"], f"Perturbed XGBoost optimization {feature_set}")
    logger.info(f"Loading noisy tensors from {tensor_dir} for Feature Set: {feature_set}...")

    train_df = pd.read_parquet(tensor_dir / "train.parquet", engine="pyarrow")
    val_df = pd.read_parquet(tensor_dir / "val.parquet", engine="pyarrow")
    features = features_for(feature_set)
    return train_df[features], train_df["Label"], val_df[features], val_df["Label"]


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


def run_optimization(feature_set, data_root=None, use_cuda_xgb=False):
    paths = runtime_paths(data_root)
    X_train, y_train, X_val, y_val = load_data(feature_set, paths.data_root)

    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    scale_pos_weight = count_hadronic / count_quark if count_quark > 0 else 1.0
    xgb_device = xgb_device_params(use_cuda_xgb)

    logger.info(f"Initializing Optuna Study for XGBoost [{feature_set}] PR-AUC maximization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight, xgb_device), n_trials=optuna_trials(30))

    logger.info(f"[{feature_set}] Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"[{feature_set}] Best Params: {study.best_params}")

    paths.outputs_perturb_root.mkdir(parents=True, exist_ok=True)
    out_path = paths.outputs_perturb_root / f"xgboost_{feature_set}_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    write_run_manifest(paths.outputs_perturb_root, f"perturbed_xgboost_{feature_set}_optimization", paths.data_root)
    logger.info(f"Saved {feature_set} optimized hyperparameters to {out_path}")

    plots_dir = paths.plots_perturb_root / "ml_optimization"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        vis.plot_optimization_history(study).write_image(plots_dir / f"xgboost_opt_history_{feature_set}.pdf")
        vis.plot_parallel_coordinate(study).write_image(plots_dir / f"xgboost_parallel_coordinate_{feature_set}.pdf")
    except Exception as e:
        logger.warning(f"Could not save visualizations: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize perturbed XGBoost classifiers.")
    add_runtime_args(parser)
    parser.add_argument("--use-cuda-xgb", action="store_true", help="request CUDA-backed XGBoost")
    parser.add_argument("--fast", action="store_true", help="use readiness-sized HPO defaults")
    args = parser.parse_args()
    configure_runtime_from_args(args)
    for feature_set in ["MR", "MRL"]:
        logger.info(f"=== Optimizing XGBoost for {feature_set} ===")
        run_optimization(feature_set, args.data_root, args.use_cuda_xgb)
