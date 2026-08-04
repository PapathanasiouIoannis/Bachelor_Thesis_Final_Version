import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import auc, f1_score, precision_recall_curve

from src.runtime import (
    add_runtime_args,
    configure_runtime_from_args,
    fast_enabled,
    require_paths,
    runtime_paths,
    write_run_manifest,
    xgb_device_params,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RUN_XGBOOST")


def load_data(data_root=None):
    paths = runtime_paths(data_root)
    tensor_dir = paths.clean_tensor_dir
    require_paths(
        [tensor_dir / "train.parquet", tensor_dir / "val.parquet", tensor_dir / "test.parquet"],
        "Final XGBoost training",
    )
    logger.info(f"Loading tensors from {tensor_dir}...")

    train_df = pd.read_parquet(tensor_dir / "train.parquet", engine="pyarrow")
    val_df = pd.read_parquet(tensor_dir / "val.parquet", engine="pyarrow")
    test_df = pd.read_parquet(tensor_dir / "test.parquet", engine="pyarrow")

    return (
        train_df.drop(columns=["Label"]),
        train_df["Label"],
        val_df.drop(columns=["Label"]),
        val_df["Label"],
        test_df.drop(columns=["Label"]),
        test_df["Label"],
    )


def main(data_root=None, use_cuda_xgb=False):
    paths = runtime_paths(data_root)
    output_dir = paths.outputs_root / "xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(paths.data_root)

    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    scale_pos_weight = count_hadronic / count_quark if count_quark > 0 else 1.0
    logger.info(f"Dynamically calculated scale_pos_weight: {scale_pos_weight:.4f}")

    params_path = paths.outputs_root / "xgboost_best_params.json"
    require_paths([params_path], "Final XGBoost training")
    with open(params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    logger.info(f"Loaded best params: {best_params}")

    model_params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_estimators": 1000,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    model_params.update(xgb_device_params(use_cuda_xgb))
    model_params.update(best_params)

    model = xgb.XGBClassifier(**model_params)
    logger.info("Training final XGBoost model with Early Stopping...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    logger.info("Evaluating on strictly held-out Test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    metrics = {"PR-AUC": float(pr_auc), "F1-Score": float(f1)}

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Final Test Metrics: PR-AUC={pr_auc:.4f}, F1={f1:.4f}")

    model_path = output_dir / "xgboost_weights.json"
    model.save_model(model_path)
    write_run_manifest(output_dir, "clean_xgboost_final", paths.data_root)
    logger.info(f"Model saved to {model_path}")

    if fast_enabled():
        logger.info("Fast mode enabled; skipping SHAP summary plot.")
        return

    logger.info("Generating SHAP Summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_dir / "shap_summary.pdf", bbox_inches="tight")
    plt.close()

    logger.info("XGBoost Final Execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final clean XGBoost classifier.")
    add_runtime_args(parser)
    parser.add_argument("--use-cuda-xgb", action="store_true", help="request CUDA-backed XGBoost")
    parser.add_argument("--fast", action="store_true", help="skip heavyweight optional plots")
    args = parser.parse_args()
    configure_runtime_from_args(args)
    main(args.data_root, args.use_cuda_xgb)
