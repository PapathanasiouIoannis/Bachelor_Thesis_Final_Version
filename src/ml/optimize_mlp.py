import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import optuna
import logging
from sklearn.metrics import precision_recall_curve, auc
import optuna.visualization as vis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OPT_MLP")

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate):
        super(DynamicMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = size
            
        layers.append(nn.Linear(in_dim, 1))
        # Removed Sigmoid here because we will use BCEWithLogitsLoss which is numerically safer
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def load_and_preprocess_data():
    TENSOR_DIR = os.path.join("data", "ml_tensors")
    logger.info(f"Loading tensors from {TENSOR_DIR}...")
    
    train_df = pd.read_parquet(os.path.join(TENSOR_DIR, "train.parquet"), engine='pyarrow')
    val_df = pd.read_parquet(os.path.join(TENSOR_DIR, "val.parquet"), engine='pyarrow')
    
    features = ['Mass', 'Radius', 'log10_Lambda']
    
    X_train_raw = train_df[features].values
    y_train = train_df['Label'].values
    
    X_val_raw = val_df[features].values
    y_val = val_df['Label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    return X_train_scaled, y_train, X_val_scaled, y_val

def objective(trial, X_train, y_train, X_val, y_val, device):
    hidden_layer_sizes_str = trial.suggest_categorical("hidden_layer_sizes", [
        "[64, 32]", 
        "[128, 64, 32]", 
        "[256, 128, 64]",
        "[128, 128]",
        "[64, 64, 32]"
    ])
    hidden_sizes = eval(hidden_layer_sizes_str)
    
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    
    model = DynamicMLP(input_dim=X_train.shape[1], hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
    
    # Calculate pos_weight for BCEWithLogitsLoss (imbalanced datasets)
    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    pos_weight = torch.tensor([count_hadronic / count_quark if count_quark > 0 else 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(torch.FloatTensor(X_train.copy()), torch.FloatTensor(y_train.copy()).unsqueeze(1))
    val_dataset = TensorDataset(torch.FloatTensor(X_val.copy()), torch.FloatTensor(y_val.copy()).unsqueeze(1))

    # Increased batch_size to 1024 for massive datasets
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    epochs = 100
    best_val_pr_auc = 0.0
    patience = 10
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                outputs = model(X_batch.to(device))
                probs = torch.sigmoid(outputs)
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

def run_optimization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    X_train_scaled, y_train, X_val_scaled, y_val = load_and_preprocess_data()

    logger.info("Initializing Optuna Study for MLP PR-AUC maximization...")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, device), n_trials=30)

    logger.info(f"Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "mlp_best_params.json")
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Saved optimized hyperparameters to {out_path}")

    # Visualizations
    plots_dir = os.path.join("plots", "ml_optimization")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        fig_hist = vis.plot_optimization_history(study)
        fig_hist.write_image(os.path.join(plots_dir, "mlp_opt_history.pdf"))
        
        fig_para = vis.plot_parallel_coordinate(study)
        fig_para.write_image(os.path.join(plots_dir, "mlp_parallel_coordinate.pdf"))
        logger.info(f"Saved Optuna visualizations to {plots_dir}")
    except Exception as e:
        logger.warning(f"Could not save visualizations (make sure kaleido/plotly are installed): {e}")

if __name__ == "__main__":
    run_optimization()
