import argparse
import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from src.ml.metrics import binary_metrics, select_macro_f1_threshold
from src.runtime import (
    add_runtime_args,
    configure_runtime_from_args,
    require_artifact_lineage,
    require_paths,
    runtime_paths,
    tensor_lineage_metadata,
    write_run_manifest,
    xgb_device_params,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PERTURB_RUN_XGBOOST")


def features_for(feature_set):
    if feature_set == "MR":
        return ["Mass", "Radius"]
    if feature_set == "MRL":
        return ["Mass", "Radius", "log10_Lambda"]
    raise ValueError("Invalid feature set.")


def load_data(feature_set="MR", data_root=None):
    paths = runtime_paths(data_root)
    tensor_dir = paths.perturb_tensor_dir
    require_paths(
        [tensor_dir / "train.parquet", tensor_dir / "val.parquet", tensor_dir / "test.parquet"],
        f"Perturbed XGBoost final training {feature_set}",
    )
    train_df = pd.read_parquet(tensor_dir / "train.parquet", engine="pyarrow")
    val_df = pd.read_parquet(tensor_dir / "val.parquet", engine="pyarrow")
    test_df = pd.read_parquet(tensor_dir / "test.parquet", engine="pyarrow")

    features = features_for(feature_set)
    return train_df[features], train_df["Label"], val_df[features], val_df["Label"], test_df[features], test_df["Label"]


def train_and_evaluate(feature_set, data_root=None, use_cuda_xgb=False):
    paths = runtime_paths(data_root)
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(feature_set, paths.data_root)

    param_path = paths.outputs_perturb_root / f"xgboost_{feature_set}_best_params.json"
    require_paths([param_path], f"Perturbed XGBoost final training {feature_set}")
    require_artifact_lineage(
        param_path,
        paths.perturb_tensor_dir,
        f"Perturbed XGBoost final training {feature_set}",
        component=f"perturbed_xgboost_{feature_set}_hpo_parameters",
        selected_features=features_for(feature_set),
    )
    with open(param_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    logger.info(f"Loaded {feature_set} optimized params: {best_params}")

    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    scale_pos_weight = count_hadronic / count_quark if count_quark > 0 else 1.0

    model_params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_estimators": 1000,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "use_label_encoder": False,
        "n_jobs": -1,
    }
    model_params.update(xgb_device_params(use_cuda_xgb))
    model_params.update(best_params)

    model = xgb.XGBClassifier(**model_params)
    logger.info(f"Training XGBoost Final Model [{feature_set}] on Noisy Tensors with Early Stopping...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    out_dir = paths.outputs_perturb_root / f"xgboost_{feature_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "xgboost_weights.json")

    logger.info(f"Evaluating XGBoost Final Model [{feature_set}] on Test Set...")
    validation_probabilities = model.predict_proba(X_val)[:, 1]
    threshold = select_macro_f1_threshold(y_val, validation_probabilities)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = binary_metrics(y_test, y_pred_proba, threshold)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    np.save(out_dir / "test_probs.npy", y_pred_proba)
    np.save(out_dir / "test_labels.npy", y_test.values)
    write_run_manifest(
        out_dir,
        f"perturbed_xgboost_{feature_set}_final",
        paths.data_root,
        {
            "decision_threshold_selected_on_validation": threshold,
            "selected_features": features_for(feature_set),
            "tensor_lineage": tensor_lineage_metadata(paths.perturb_tensor_dir),
        },
    )

    logger.info(f"[{feature_set}] Metrics saved to {out_dir}")
    logger.info("[%s] Test metrics: %s", feature_set, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final perturbed XGBoost classifiers.")
    add_runtime_args(parser)
    parser.add_argument("--use-cuda-xgb", action="store_true", help="request CUDA-backed XGBoost")
    args = parser.parse_args()
    configure_runtime_from_args(args)
    for feature_set in ["MR", "MRL"]:
        logger.info(f"=== Running Final XGBoost for {feature_set} ===")
        train_and_evaluate(feature_set, args.data_root, args.use_cuda_xgb)
