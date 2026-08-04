import argparse
import ast
import json
import logging

import numpy as np
import optuna
import optuna.visualization as vis
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

from src.ml.mlp_model import DynamicMLP
from src.runtime import add_runtime_args, configure_runtime_from_args, optuna_trials, require_paths, runtime_paths, train_epochs, write_run_manifest


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
    require_paths([tensor_dir / "train.parquet", tensor_dir / "val.parquet"], f"Perturbed MLP optimization {feature_set}")
    logger.info(f"Loading noisy tensors from {tensor_dir} for Feature Set: {feature_set}...")

    train_df = pd.read_parquet(tensor_dir / "train.parquet", engine="pyarrow")
    val_df = pd.read_parquet(tensor_dir / "val.parquet", engine="pyarrow")
    features = features_for(feature_set)
    return train_df[features].values, train_df["Label"].values, val_df[features].values, val_df["Label"].values


def objective(trial, X_train, y_train, X_val, y_val, device):
    hidden_layer_sizes_str = trial.suggest_categorical(
        "hidden_layer_sizes",
        ["[64, 32]", "[128, 64, 32]", "[256, 128, 64]", "[128, 128]", "[64, 64, 32]"],
    )
    hidden_sizes = ast.literal_eval(hidden_layer_sizes_str)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    model = DynamicMLP(input_dim=X_train.shape[1], hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    pos_weight = torch.tensor([count_hadronic / count_quark if count_quark > 0 else 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(torch.FloatTensor(X_train.copy()), torch.FloatTensor(y_train.copy()).unsqueeze(1))
    val_dataset = TensorDataset(torch.FloatTensor(X_val.copy()), torch.FloatTensor(y_val.copy()).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    best_val_pr_auc = 0.0
    patience = 10
    epochs_no_improve = 0

    for epoch in range(train_epochs(100)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                probs = torch.sigmoid(model(X_batch.to(device)))
                val_preds.extend(probs.cpu().numpy())

        precision, recall, _ = precision_recall_curve(y_val, val_preds)
        val_pr_auc = auc(recall, precision)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        trial.report(val_pr_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if epochs_no_improve >= patience:
            break

    return best_val_pr_auc


def run_optimization(feature_set, data_root=None):
    paths = runtime_paths(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for {feature_set}")

    X_train_scaled, y_train, X_val_scaled, y_val = load_data(feature_set, paths.data_root)
    logger.info(f"Initializing Optuna Study for MLP [{feature_set}] PR-AUC maximization...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, device), n_trials=optuna_trials(30))

    logger.info(f"[{feature_set}] Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"[{feature_set}] Best Params: {study.best_params}")

    paths.outputs_perturb_root.mkdir(parents=True, exist_ok=True)
    out_path = paths.outputs_perturb_root / f"mlp_{feature_set}_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=4)
    write_run_manifest(paths.outputs_perturb_root, f"perturbed_mlp_{feature_set}_optimization", paths.data_root)
    logger.info(f"Saved optimized hyperparameters to {out_path}")

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
