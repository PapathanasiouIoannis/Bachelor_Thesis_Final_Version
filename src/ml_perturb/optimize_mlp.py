import argparse
import ast
import json
import logging

import numpy as np
import optuna
import optuna.visualization as vis
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

from src.config import CONFIG
from src.ml.hpo_groups import (
    grouped_cv_indices,
    load_training_groups,
    scale_inner_fold,
)
from src.ml.mlp_model import DynamicMLP
from src.runtime import add_runtime_args, configure_runtime_from_args, fast_enabled, optuna_trials, runtime_paths, train_epochs, write_artifact_lineage, write_run_manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PERTURB_OPT_MLP")


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
    x_train, y_train, groups = load_training_groups(
        tensor_dir, features, f"Perturbed MLP grouped optimization {feature_set}"
    )
    return x_train.values, y_train.values, groups.reset_index(drop=True)


def objective(trial, X_train, y_train, groups, device):
    hidden_layer_sizes_str = trial.suggest_categorical(
        "hidden_layer_sizes",
        ["[64, 32]", "[128, 64, 32]", "[256, 128, 64]", "[128, 128]", "[64, 64, 32]"],
    )
    hidden_sizes = ast.literal_eval(hidden_layer_sizes_str)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    fold_scores = []
    for fold_index, (fit_idx, score_idx) in enumerate(
        grouped_cv_indices(groups, y_train)
    ):
        X_fit, X_score = scale_inner_fold(X_train, fit_idx, score_idx)
        torch.manual_seed(CONFIG["ML_RANDOM_SEED"] + fold_index)
        model = DynamicMLP(
            input_dim=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
        ).to(device)
        y_fit = y_train[fit_idx]
        count_quark = np.sum(y_fit == 1)
        pos_weight = torch.tensor(
            [np.sum(y_fit == 0) / count_quark if count_quark > 0 else 1.0]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        fit_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_fit),
                torch.FloatTensor(y_fit.copy()).unsqueeze(1),
            ),
            batch_size=1024,
            shuffle=True,
        )
        score_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_score),
                torch.FloatTensor(y_train[score_idx].copy()).unsqueeze(1),
            ),
            batch_size=1024,
            shuffle=False,
        )

        best_fold_score = 0.0
        epochs_no_improve = 0
        for _ in range(train_epochs(100)):
            model.train()
            for x_batch, y_batch in fit_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            model.eval()
            predictions = []
            with torch.no_grad():
                for x_batch, _ in score_loader:
                    predictions.extend(
                        torch.sigmoid(model(x_batch.to(device))).cpu().numpy()
                    )
            precision, recall, _ = precision_recall_curve(
                y_train[score_idx], predictions
            )
            fold_score = auc(recall, precision)
            if fold_score > best_fold_score:
                best_fold_score = fold_score
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break

        fold_scores.append(best_fold_score)
        trial.report(float(np.mean(fold_scores)), fold_index)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return float(np.mean(fold_scores))


def run_optimization(feature_set, data_root=None):
    paths = runtime_paths(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for {feature_set}")

    X_train_scaled, y_train, groups = load_data(feature_set, paths.data_root)
    logger.info(
        f"Initializing grouped inner-CV MLP study [{feature_set}] "
        "(validation/test remain locked)..."
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CONFIG["ML_RANDOM_SEED"]),
    )
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, y_train, groups, device),
        n_trials=optuna_trials(30),
    )

    logger.info(f"[{feature_set}] Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"[{feature_set}] Best Params: {study.best_params}")

    paths.outputs_perturb_root.mkdir(parents=True, exist_ok=True)
    out_path = paths.outputs_perturb_root / f"mlp_{feature_set}_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    write_artifact_lineage(
        out_path,
        paths.perturb_tensor_dir,
        f"perturbed_mlp_{feature_set}_hpo_parameters",
        features_for(feature_set),
    )
    write_run_manifest(
        paths.outputs_perturb_root,
        f"perturbed_mlp_{feature_set}_optimization",
        paths.data_root,
        {
            "hpo_scope": "training-only grouped inner cross-validation",
            "selected_features": features_for(feature_set),
        },
    )
    logger.info(f"Saved optimized hyperparameters to {out_path}")

    if fast_enabled():
        logger.info("Fast mode enabled; skipping Optuna visualization export.")
        return

    plots_dir = paths.plots_perturb_root / "ml_optimization"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        vis.plot_optimization_history(study).write_image(plots_dir / f"mlp_opt_history_{feature_set}.pdf")
        vis.plot_parallel_coordinate(study).write_image(plots_dir / f"mlp_parallel_coordinate_{feature_set}.pdf")
    except Exception as e:
        logger.warning(f"Could not save visualizations: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize perturbed MLP classifiers.")
    add_runtime_args(parser)
    parser.add_argument("--fast", action="store_true", help="use readiness-sized HPO defaults")
    args = parser.parse_args()
    configure_runtime_from_args(args)
    for feature_set in ["MR", "MRL"]:
        logger.info(f"=== Optimizing MLP for {feature_set} ===")
        run_optimization(feature_set, args.data_root)
