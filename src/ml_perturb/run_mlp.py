import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, classification_report, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

from src.ml.mlp_model import build_mlp, load_mlp_params
from src.runtime import add_runtime_args, configure_runtime_from_args, require_paths, runtime_paths, train_epochs, write_run_manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PERTURB_RUN_MLP")


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
        f"Perturbed MLP final training {feature_set}",
    )
    train_df = pd.read_parquet(tensor_dir / "train.parquet", engine="pyarrow")
    val_df = pd.read_parquet(tensor_dir / "val.parquet", engine="pyarrow")
    test_df = pd.read_parquet(tensor_dir / "test.parquet", engine="pyarrow")

    features = features_for(feature_set)
    return train_df[features].values, train_df["Label"].values, val_df[features].values, val_df["Label"].values, test_df[features].values, test_df["Label"].values


def train_and_evaluate(feature_set, data_root=None):
    paths = runtime_paths(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{feature_set}] Using device: {device}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(feature_set, paths.data_root)

    param_path = paths.outputs_perturb_root / f"mlp_{feature_set}_best_params.json"
    require_paths([param_path], f"Perturbed MLP final training {feature_set}")
    best_params = load_mlp_params(param_path)
    logger.info(f"Loaded {feature_set} optimized params: {best_params}")

    model = build_mlp(input_dim=X_train.shape[1], params=best_params, device=device)

    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    pos_weight = torch.tensor([count_hadronic / count_quark if count_quark > 0 else 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train.copy()), torch.FloatTensor(y_train.copy()).unsqueeze(1)), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val.copy()), torch.FloatTensor(y_val.copy()).unsqueeze(1)), batch_size=1024, shuffle=False)

    patience = 15
    epochs_no_improve = 0
    best_val_loss = float("inf")
    best_model_state = None

    logger.info(f"Training Final MLP [{feature_set}] on Noisy Tensors with Early Stopping (patience={patience})...")
    for epoch in range(train_epochs(150)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logger.info(f"[{feature_set}] Early stopping triggered at epoch {epoch + 1}")
            break

    if best_model_state is None:
        raise RuntimeError(f"[{feature_set}] MLP training completed without a valid best model state.")
    model.load_state_dict(best_model_state)

    out_dir = paths.outputs_perturb_root / f"mlp_{feature_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "mlp_weights.pth")

    logger.info(f"Evaluating Final MLP [{feature_set}] on Test Set...")
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test.copy()).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_tensor)).cpu().numpy().squeeze()

    y_pred = (probs > 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    rep = classification_report(y_test, y_pred, target_names=["Hadronic", "Quark"], output_dict=True)
    metrics = {"PR-AUC": float(pr_auc), "F1-Score": float(rep["macro avg"]["f1-score"])}

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    np.save(out_dir / "test_probs.npy", probs)
    np.save(out_dir / "test_labels.npy", y_test)
    write_run_manifest(out_dir, f"perturbed_mlp_{feature_set}_final", paths.data_root)

    logger.info(f"[{feature_set}] Metrics saved to {out_dir}")
    logger.info(f"[{feature_set}] PR-AUC: {pr_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final perturbed MLP classifiers.")
    add_runtime_args(parser)
    parser.add_argument("--fast", action="store_true", help="use readiness-sized training defaults")
    args = parser.parse_args()
    configure_runtime_from_args(args)
    for feature_set in ["MR", "MRL"]:
        logger.info(f"=== Running Final MLP for {feature_set} ===")
        train_and_evaluate(feature_set, args.data_root)
