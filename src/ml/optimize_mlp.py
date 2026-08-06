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
from src.runtime import (
    add_runtime_args,
    configure_runtime_from_args,
    fast_enabled,
    optuna_trials,
    runtime_paths,
    train_epochs,
    write_artifact_lineage,
    write_run_manifest,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OPT_MLP")


def load_and_preprocess_data(data_root=None):
    paths = runtime_paths(data_root)
    tensor_dir = paths.clean_tensor_dir
    features = ["Mass", "Radius", "log10_Lambda"]
    logger.info(f"Loading training-only tensors and groups from {tensor_dir}...")
    x_train, y_train, groups = load_training_groups(
        tensor_dir, features, "MLP grouped optimization"
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


def run_optimization(data_root=None):
    paths = runtime_paths(data_root)
    features = ["Mass", "Radius", "log10_Lambda"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X_train_scaled, y_train, groups = load_and_preprocess_data(paths.data_root)

    logger.info("Initializing grouped inner-CV Optuna study (validation/test remain locked)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CONFIG["ML_RANDOM_SEED"]),
    )
    study.optimize(
        lambda trial: objective(trial, X_train_scaled, y_train, groups, device),
        n_trials=optuna_trials(30),
    )

    logger.info(f"Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    out_path = paths.outputs_root / "mlp_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    write_artifact_lineage(
        out_path,
        paths.clean_tensor_dir,
        "clean_mlp_hpo_parameters",
        features,
    )
    write_run_manifest(
        paths.outputs_root,
        "clean_mlp_optimization",
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
        vis.plot_optimization_history(study).write_image(plots_dir / "mlp_opt_history.pdf")
        vis.plot_parallel_coordinate(study).write_image(plots_dir / "mlp_parallel_coordinate.pdf")
        logger.info(f"Saved Optuna visualizations to {plots_dir}")
    except Exception as e:
        logger.warning(f"Could not save visualizations (make sure kaleido/plotly are installed): {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize clean MLP classifier.")
    add_runtime_args(parser)
    parser.add_argument("--fast", action="store_true", help="use readiness-sized HPO defaults")
    args = parser.parse_args()
    configure_runtime_from_args(args)
    run_optimization(args.data_root)
