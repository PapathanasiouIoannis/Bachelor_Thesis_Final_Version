import argparse
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import auc, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

from src.ml.mlp_model import build_mlp, load_mlp_params
from src.runtime import add_runtime_args, configure_runtime_from_args, require_paths, runtime_paths, train_epochs, write_run_manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RUN_MLP")


def load_and_preprocess_data(data_root=None):
    paths = runtime_paths(data_root)
    tensor_dir = paths.clean_tensor_dir
    require_paths(
        [tensor_dir / "train.parquet", tensor_dir / "val.parquet", tensor_dir / "test.parquet"],
        "Final MLP training",
    )
    logger.info(f"Loading tensors from {tensor_dir}...")

    train_df = pd.read_parquet(tensor_dir / "train.parquet", engine="pyarrow")
    val_df = pd.read_parquet(tensor_dir / "val.parquet", engine="pyarrow")
    test_df = pd.read_parquet(tensor_dir / "test.parquet", engine="pyarrow")

    features = ["Mass", "Radius", "log10_Lambda"]
    return (
        train_df[features].values,
        train_df["Label"].values,
        val_df[features].values,
        val_df["Label"].values,
        test_df[features].values,
        test_df["Label"].values,
        len(features),
    )


def main(data_root=None):
    paths = runtime_paths(data_root)
    output_dir = paths.outputs_root / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, input_dim = load_and_preprocess_data(paths.data_root)

    params_path = paths.outputs_root / "mlp_best_params.json"
    require_paths([params_path], "Final MLP training")
    best_params = load_mlp_params(params_path)
    logger.info(f"Loaded best params: {best_params}")

    hidden_sizes = best_params["hidden_layer_sizes"]
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mlp(input_dim=input_dim, params=best_params, device=device)

    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    pos_weight = torch.tensor([count_hadronic / count_quark if count_quark > 0 else 1.0]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train.copy()), torch.FloatTensor(y_train.copy()).unsqueeze(1)), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val.copy()), torch.FloatTensor(y_val.copy()).unsqueeze(1)), batch_size=1024, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test.copy()), torch.FloatTensor(y_test.copy()).unsqueeze(1)), batch_size=1024, shuffle=False)

    patience = 15
    epochs_no_improve = 0
    best_val_loss = float("inf")
    best_model_state = None
    history = {"train_loss": [], "val_loss": [], "val_pr_auc": []}

    logger.info("Training final MLP with Early Stopping on Validation set...")
    for epoch in range(train_epochs(100)):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())

        val_loss /= len(val_loader.dataset)
        precision, recall, _ = precision_recall_curve(y_val, val_preds)
        val_pr_auc = auc(recall, precision)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_pr_auc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_model_state is None:
        raise RuntimeError("MLP training completed without producing a valid best model state.")
    model.load_state_dict(best_model_state)

    logger.info("Evaluating on strictly held-out Test set...")
    model.eval()
    test_preds_proba = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            probs = torch.sigmoid(model(X_batch.to(device)))
            test_preds_proba.extend(probs.cpu().numpy())

    y_pred_proba = np.array(test_preds_proba).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    precision_t, recall_t, _ = precision_recall_curve(y_test, y_pred_proba)
    test_pr_auc = auc(recall_t, precision_t)
    f1 = f1_score(y_test, y_pred)

    metrics = {"PR-AUC": float(test_pr_auc), "F1-Score": float(f1)}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Final Test Metrics: PR-AUC={test_pr_auc:.4f}, F1={f1:.4f}")

    model_path = output_dir / "mlp_weights.pth"
    torch.save(model.state_dict(), model_path)
    write_run_manifest(output_dir, "clean_mlp_final", paths.data_root)
    logger.info(f"Model saved to {model_path}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(history["train_loss"], color="tab:red", label="Train Loss")
    ax1.plot(history["val_loss"], color="darkred", linestyle="--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.legend(loc="center right")

    ax2 = ax1.twinx()
    ax2.set_ylabel("PR-AUC", color="tab:blue")
    ax2.plot(history["val_pr_auc"], color="tab:blue", label="Val PR-AUC")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.title(f"MLP Final Training History (hidden={hidden_sizes}, dropout={dropout_rate})")
    plt.savefig(output_dir / "training_history.pdf", bbox_inches="tight")
    plt.close()

    logger.info("MLP Final Execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final clean MLP classifier.")
    add_runtime_args(parser)
    parser.add_argument("--fast", action="store_true", help="use readiness-sized training defaults")
    args = parser.parse_args()
    configure_runtime_from_args(args)
    main(args.data_root)
