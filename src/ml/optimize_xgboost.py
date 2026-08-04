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
logger = logging.getLogger("OPT_XGBOOST")


def load_data(data_root=None):
    paths = runtime_paths(data_root)
    tensor_dir = paths.clean_tensor_dir
    logger.info(f"Loading training-only tensors and groups from {tensor_dir}...")
    return load_training_groups(
        tensor_dir,
        ["Mass", "Radius", "log10_Lambda"],
        "XGBoost grouped optimization",
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


def run_optimization(data_root=None, use_cuda_xgb=False):
    paths = runtime_paths(data_root)
    features = ["Mass", "Radius", "log10_Lambda"]
    X_train, y_train, groups = load_data(paths.data_root)
    xgb_device = xgb_device_params(use_cuda_xgb)

    logger.info("Initializing grouped inner-CV Optuna study (validation/test remain locked)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CONFIG["ML_RANDOM_SEED"]),
    )
    trials = optuna_trials(50)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, groups, xgb_device),
        n_trials=trials,
    )

    logger.info(f"Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    out_path = paths.outputs_root / "xgboost_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    write_artifact_lineage(
        out_path,
        paths.clean_tensor_dir,
        "clean_xgboost_hpo_parameters",
        features,
    )
    write_run_manifest(
        paths.outputs_root,
        "clean_xgboost_optimization",
        paths.data_root,
        {
            "hpo_scope": "training-only grouped inner cross-validation",
            "selected_features": features,
        },
    )
    logger.info(f"Saved optimized hyperparameters to {out_path}")

    if fast_enabled():
        logger.info("Fast mode enabled; skipping Optuna visualization export.")
        return

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
