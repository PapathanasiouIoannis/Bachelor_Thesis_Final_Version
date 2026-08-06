import argparse
import json
import logging

import numpy as np
import optuna
import optuna.visualization as vis
import xgboost as xgb
from sklearn.metrics import auc, precision_recall_curve

from src.config import CONFIG
from src.ml.hpo_groups import (
    grouped_cv_indices,
    load_training_groups,
    scale_inner_fold,
)
from src.runtime import (
    add_runtime_args,
    configure_runtime_from_args,
    fast_enabled,
    optuna_trials,
    runtime_paths,
    write_artifact_lineage,
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
    features = features_for(feature_set)
    logger.info(
        f"Loading noisy training-only tensors and groups from {tensor_dir} "
        f"for Feature Set: {feature_set}..."
    )
    return load_training_groups(
        tensor_dir, features, f"Perturbed XGBoost grouped optimization {feature_set}"
    )


def objective(trial, X_train, y_train, groups, xgb_device):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": CONFIG["ML_RANDOM_SEED"],
        "n_estimators": 1000,
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    param.update(xgb_device)

    scores = []
    for fold_index, (fit_idx, score_idx) in enumerate(
        grouped_cv_indices(groups, y_train)
    ):
        X_fit, X_score = scale_inner_fold(X_train, fit_idx, score_idx)
        y_fit = y_train.iloc[fit_idx]
        quark_count = int((y_fit == 1).sum())
        scale_pos_weight = int((y_fit == 0).sum()) / quark_count if quark_count else 1.0
        model = xgb.XGBClassifier(
            **param,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model.fit(
            X_fit,
            y_fit,
            eval_set=[(X_score, y_train.iloc[score_idx])],
            verbose=False,
        )
        probabilities = model.predict_proba(X_score)[:, 1]
        precision, recall, _ = precision_recall_curve(
            y_train.iloc[score_idx], probabilities
        )
        scores.append(auc(recall, precision))
        trial.report(float(np.mean(scores)), fold_index)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return float(np.mean(scores))


def run_optimization(feature_set, data_root=None, use_cuda_xgb=False):
    paths = runtime_paths(data_root)
    X_train, y_train, groups = load_data(feature_set, paths.data_root)
    xgb_device = xgb_device_params(use_cuda_xgb)

    logger.info(
        f"Initializing grouped inner-CV XGBoost study [{feature_set}] "
        "(validation/test remain locked)..."
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CONFIG["ML_RANDOM_SEED"]),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, groups, xgb_device),
        n_trials=optuna_trials(30),
    )

    logger.info(f"[{feature_set}] Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"[{feature_set}] Best Params: {study.best_params}")

    paths.outputs_perturb_root.mkdir(parents=True, exist_ok=True)
    out_path = paths.outputs_perturb_root / f"xgboost_{feature_set}_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    write_artifact_lineage(
        out_path,
        paths.perturb_tensor_dir,
        f"perturbed_xgboost_{feature_set}_hpo_parameters",
        features_for(feature_set),
    )
    write_run_manifest(
        paths.outputs_perturb_root,
        f"perturbed_xgboost_{feature_set}_optimization",
        paths.data_root,
        {
            "hpo_scope": "training-only grouped inner cross-validation",
            "selected_features": features_for(feature_set),
        },
    )
    logger.info(f"Saved {feature_set} optimized hyperparameters to {out_path}")

    if fast_enabled():
        logger.info("Fast mode enabled; skipping Optuna visualization export.")
        return

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
